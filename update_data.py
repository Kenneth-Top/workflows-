import requests
import re
import json
import pandas as pd
from datetime import datetime
import os
import time

# === 配置 ===
DATA_FILE = "history_database.csv"  # 我们的“数据库”文件
MODELS = [
    "moonshotai/kimi-k2-thinking",
    "moonshotai/kimi-k2.5",
    "deepseek/deepseek-v3.2",
    "minimax/minimax-m2.1",
    "x-ai/grok-4.1-fast",
    "openai/gpt-5.1",
    # 在这里添加更多模型...
]

def fetch_data(model_id):
    """抓取单个模型数据 (包含代理修复)"""
    url = f"https://openrouter.ai/{model_id}"
    print(f"🚀 正在抓取: {model_id} ...")
    
    try:
        session = requests.Session()
        session.trust_env = False  # 绕过系统代理，防止报错
        headers = {"User-Agent": "Mozilla/5.0"}
        
        response = session.get(url, headers=headers, timeout=20)
        if response.status_code != 200: return None
        
        match = re.search(r'\\?"analytics\\?":\s*(\[\{.*?\}\])', response.text)
        if match:
            raw = match.group(1).replace('\\"', '"').replace('\\\\', '\\')
            return json.loads(raw)
    except Exception as e:
        print(f"❌ 错误 {model_id}: {e}")
    return None

def update_database():
    # 1. 读取现有数据库 (如果存在)
    if os.path.exists(DATA_FILE):
        print("📂 读取现有历史数据...")
        df_old = pd.read_csv(DATA_FILE)
        df_old['Date'] = pd.to_datetime(df_old['Date'])
    else:
        print("📂 初始化新数据库...")
        df_old = pd.DataFrame(columns=['Date', 'Model', 'Prompt', 'Completion', 'Reasoning', 'Total_Tokens'])

    new_records = []

    # 2. 爬取最新数据
    for model in MODELS:
        data = fetch_data(model)
        if not data: continue
        
        for record in data:
            # 数据清洗与单位转换 (Billion)
            p = record.get('total_prompt_tokens', 0) / 1e9
            c = record.get('total_completion_tokens', 0) / 1e9
            r = record.get('total_native_tokens_reasoning', 0) / 1e9
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
    
    # 3. 核心逻辑：增量合并 (Upsert)
    # 我们将旧数据和新数据合并
    df_combined = pd.concat([df_old, df_new])
    
    # 关键：如果有重复的 (Date, Model)，保留最新的那一条（df_new 的）
    # 这样既能保留历史，又能更新“昨天”不完整的数据
    df_combined = df_combined.sort_values('Date').drop_duplicates(subset=['Date', 'Model'], keep='last')
    
    # 4. 保存回 CSV
    df_combined.to_csv(DATA_FILE, index=False)
    print(f"✅ 数据库更新完成！当前总记录数: {len(df_combined)}")

if __name__ == "__main__":
    # 清理环境变量防止代理干扰
    if "HTTP_PROXY" in os.environ: del os.environ["HTTP_PROXY"]
    update_database()