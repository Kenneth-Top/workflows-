import requests
import re
import json
import pandas as pd
from datetime import datetime
import os
import time

# === 配置 ===
DATA_FILE = "history_database.csv"
MODELS_API = "https://openrouter.ai/api/v1/models"

# 公共 Session（复用连接池，避免每次请求都创建新 Session）
SESSION = requests.Session()
SESSION.trust_env = False
SESSION.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
})


def fetch_all_model_ids():
    """从 OpenRouter API 自动获取所有可用模型的 id 列表"""
    print("🔍 正在从 OpenRouter API 获取模型列表...")
    try:
        resp = SESSION.get(MODELS_API, timeout=30)
        resp.raise_for_status()
        models = resp.json().get("data", [])
        # 按 created 倒序排列（最新的在前）
        models.sort(key=lambda m: m.get("created", 0), reverse=True)
        ids = [m["id"] for m in models]
        print(f"✅ 发现 {len(ids)} 个模型")
        return ids
    except Exception as e:
        print(f"❌ 获取模型列表失败: {e}")
        return []


def fetch_analytics(model_id):
    """从模型页面 HTML 中提取 analytics 数据"""
    url = f"https://openrouter.ai/{model_id}"
    try:
        resp = SESSION.get(url, timeout=20)
        if resp.status_code != 200:
            return None

        match = re.search(r'\\?"analytics\\?":\s*(\[\{.*?\}\])', resp.text)
        if match:
            raw = match.group(1).replace('\\"', '"').replace('\\\\', '\\')
            return json.loads(raw)
    except Exception as e:
        print(f"  ❌ 错误 {model_id}: {e}")
    return None


def update_database():
    # 1. 读取旧数据
    columns = ['Date', 'Model', 'Prompt', 'Completion', 'Reasoning', 'Total_Tokens']
    if os.path.exists(DATA_FILE):
        try:
            df_old = pd.read_csv(DATA_FILE)
            df_old['Date'] = pd.to_datetime(df_old['Date'])
        except Exception:
            df_old = pd.DataFrame(columns=columns)
    else:
        df_old = pd.DataFrame(columns=columns)

    # 2. 自动获取模型列表
    all_models = fetch_all_model_ids()
    if not all_models:
        print("⚠️ 无法获取模型列表，终止")
        return

    # 识别新模型（CSV 中尚未出现的模型）
    existing_models = set(df_old['Model'].unique()) if not df_old.empty else set()
    new_models = [m for m in all_models if m not in existing_models]
    if new_models:
        print(f"🆕 发现 {len(new_models)} 个新模型:")
        for m in new_models[:10]:  # 最多显示10个
            print(f"   + {m}")
        if len(new_models) > 10:
            print(f"   ... 及另外 {len(new_models) - 10} 个")

    # 3. 批量爬取 analytics 数据
    new_records = []
    today_str = datetime.utcnow().strftime("%Y-%m-%d")  # OpenRouter 使用 UTC 日期

    for i, model in enumerate(all_models):
        print(f"🚀 [{i+1}/{len(all_models)}] 正在抓取: {model}")
        data = fetch_analytics(model)
        if not data:
            continue

        for record in data:
            # 过滤当天未结算数据（当天统计不完整，会导致数值偏低）
            record_date_str = record['date'][:10]  # "2026-02-13 00:00:00" -> "2026-02-13"
            if record_date_str == today_str:
                continue

            p = (record.get('total_prompt_tokens') or 0) / 1e9
            c = (record.get('total_completion_tokens') or 0) / 1e9
            r = (record.get('total_native_tokens_reasoning') or 0) / 1e9
            t = p + c  # Total = Prompt + Completion (OpenAI 标准)

            new_records.append({
                'Date': datetime.strptime(record['date'], "%Y-%m-%d %H:%M:%S"),
                'Model': model,
                'Prompt': round(p, 6),
                'Completion': round(c, 6),
                'Reasoning': round(r, 6),
                'Total_Tokens': round(t, 6)
            })
        time.sleep(1)

    if not new_records:
        print("⚠️ 本次未抓取到任何数据")
        return

    df_new = pd.DataFrame(new_records)

    # 4. 增量合并 (Upsert)：Date + Model 为唯一键
    df_combined = pd.concat([df_old, df_new])
    df_combined = df_combined.sort_values('Date').drop_duplicates(
        subset=['Date', 'Model'], keep='last'
    )

    # 5. 保存
    df_combined.to_csv(DATA_FILE, index=False)
    print(f"✅ 数据库更新完成！当前总记录数: {len(df_combined)}")


if __name__ == "__main__":
    if "HTTP_PROXY" in os.environ:
        del os.environ["HTTP_PROXY"]
    update_database()
