import requests
import re
import json
import pandas as pd
from datetime import datetime
import os
import time

# === 配置 ===
DATA_FILE = "history_database.csv"
# ================= 批量配置区域 =================
MODELS = [
    "deepseek/deepseek-r1",
    "deepseek/deepseek-v3.2",
    "moonshotai/kimi-k2-thinking",
    "moonshotai/kimi-k2.5",
    "z-ai/glm-4.7",
    "z-ai/glm-4.7-flash",
    "z-ai/glm-4.6v",
    "minimax/minimax-m2.1",
    "minimax/minimax-m2-her",
    "qwen/qwen3-coder-next",
    "qwen/qwen3-embedding-8b",
    "qwen/qwen3-embedding-4b",
    "anthropic/claude-opus-4.6",
    "anthropic/claude-opus-4.5",
    "google/gemini-3-pro-preview",
    "google/gemini-3-flash-preview",
    "google/gemini-3-pro-image-preview"
    "openai/gpt-5.2-codex",
    "openai/gpt-5.2",
    "x-ai/grok-4.1-fast"
]
# ===========================================

def fetch_data(model_id):
    url = f"https://openrouter.ai/{model_id}"
    print(f"🚀 正在抓取: {model_id} ...")
    
    try:
        session = requests.Session()
        session.trust_env = False
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        
        response = session.get(url, headers=headers, timeout=20)
        if response.status_code != 200: return None
        
        # 提取 JSON
        match = re.search(r'\\?"analytics\\?":\s*(\[\{.*?\}\])', response.text)
        if match:
            raw = match.group(1).replace('\\"', '"').replace('\\\\', '\\')
            return json.loads(raw)
    except Exception as e:
        print(f"❌ 错误 {model_id}: {e}")
    return None

def update_database():
    # 1. 读取旧数据
    if os.path.exists(DATA_FILE):
        try:
            df_old = pd.read_csv(DATA_FILE)
            df_old['Date'] = pd.to_datetime(df_old['Date'])
        except:
            df_old = pd.DataFrame(columns=['Date', 'Model', 'Prompt', 'Completion', 'Reasoning', 'Total_Tokens'])
    else:
        df_old = pd.DataFrame(columns=['Date', 'Model', 'Prompt', 'Completion', 'Reasoning', 'Total_Tokens'])

    new_records = []

    # 2. 爬取新数据
    for model in MODELS:
        data = fetch_data(model)
        if not data: continue
        
        for record in data:
            # === 数据清洗与单位转换 (Billion) ===
            # 使用 or 0 防止 None 值报错
            p = (record.get('total_prompt_tokens') or 0) / 1e9
            c = (record.get('total_completion_tokens') or 0) / 1e9
            r = (record.get('total_native_tokens_reasoning') or 0) / 1e9
            
            # Total = Prompt + Completion (OpenAI 标准)
            t = p + c
            
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
    
    # 3. 增量合并 (Upsert)
    # 合并新旧数据
    df_combined = pd.concat([df_old, df_new])
    
    # 去重逻辑：Date + Model 是唯一键
    # keep='last' 确保保留最新抓取的数据（如果 OpenRouter 更新了当天的统计）
    df_combined = df_combined.sort_values('Date').drop_duplicates(subset=['Date', 'Model'], keep='last')
    
    # 4. 保存
    df_combined.to_csv(DATA_FILE, index=False)
    print(f"✅ 数据库更新完成！当前总记录数: {len(df_combined)}")

if __name__ == "__main__":
    if "HTTP_PROXY" in os.environ: del os.environ["HTTP_PROXY"]
    update_database()

