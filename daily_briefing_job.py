import os
import pandas as pd
import requests
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import smtplib
import datetime
import re

DATA_FILE = "history_database.csv"
PRICING_FILE = "openrouter_pricing_provider_records.csv"
BENCHMARK_FILE = "openrouter_benchmark_records.csv"
LMARENA_FILE = "lmarena_leaderboard_records.csv"

def load_csv(path):
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
        return df
    except Exception:
        return None

def build_db_context(df, df_price, df_bench, df_lmarena):
    context_parts = []
    
    if df is not None and not df.empty:
        # 只取近7天数据作为大盘摘要
        seven_days_ago = df['Date'].max() - pd.Timedelta(days=7)
        recent_df = df[df['Date'] >= seven_days_ago]
        top_models = recent_df.groupby('Model')['Total_Tokens'].sum().nlargest(15)
        context_parts.append(f"### 近7天 Token 消耗排名前 15 模型 (单位: 10亿)\n{top_models.to_string()}")

    if df_price is not None and not df_price.empty:
        latest_date = df_price['Date'].max()
        recent_price = df_price[df_price['Date'] == latest_date]
        context_parts.append(f"### 最新 API 定价采样 (日期: {latest_date.strftime('%Y-%m-%d')})\n列: Model, Provider, Input_Price, Output_Price\n{recent_price.head(30).to_string(index=False)}")

    if df_lmarena is not None and not df_lmarena.empty:
        latest_date = df_lmarena['Date'].max()
        recent_arena = df_lmarena[df_lmarena['Date'] == latest_date]
        context_parts.append(f"### 最新 LMArena 综合跑分排名 (日期: {latest_date.strftime('%Y-%m-%d')})\n{recent_arena.head(20)[['Model', 'Score_text', 'Rank_Overall', 'Rank_Coding']].to_string(index=False)}")
    
    return '\n\n'.join(context_parts)

def fetch_search_context():
    # 使用 Bing 搜索最新的行业资讯，供给 AI
    try:
        import urllib.parse
        import requests as _s_req
        from bs4 import BeautifulSoup
        
        h={'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        r=_s_req.get('https://www.bing.com/search?q='+urllib.parse.quote("最新大模型 发布 降价 测评 动态 OpenAI DeepSeek Anthropic"), headers=h, timeout=8)
        s=BeautifulSoup(r.text, 'html.parser')
        results = [{'title':li.find('h2').text, 'body':li.find('p').text if li.find('p') else ''} for li in s.find_all('li', class_='b_algo')[:5] if li.find('h2')]
        
        if results:
            context = "【今日最新网络搜索参考资讯】\n"
            for r in results:
                context += f"- {r.get('title', '')}: {r.get('body', '')}\n"
            return context
    except Exception as e:
        print(f"搜索失败: {e}")
    return ""

def main():
    print("Loading databases...")
    df = load_csv(DATA_FILE)
    df_price = load_csv(PRICING_FILE)
    df_bench = load_csv(BENCHMARK_FILE)
    df_lmarena = load_csv(LMARENA_FILE)
    
    db_context = build_db_context(df, df_price, df_bench, df_lmarena)
    search_context = fetch_search_context()
    
    # 优先使用配置的魔塔社区模型
    api_key = os.environ.get("MODELSCOPE_API_KEY", "")
    if not api_key:
        print("未配置 MODELSCOPE_API_KEY，将尝试使用 OpenRouter")
        api_key = os.environ.get("OPENROUTER_API_KEY", "")
        base_url = "https://openrouter.ai/api/v1"
        model = "anthropic/claude-3.5-sonnet"
    else:
        base_url = "https://api-inference.modelscope.cn/v1"
        model = "MiniMax/MiniMax-M2.5"  # 用户要求默认 Minimax-M2.5
        
    print(f"Using Model {model} on {base_url}...")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    prompt = f"""你是一个顶尖的 AI 行业投资顾问兼分析师。
请你根据以下数据库上下文摘要与今日最新搜索资讯，撰写一份【每日大模型行业情报简报】。
要求：
1. **优先覆盖核心厂商**：重点分析 OpenAI、Anthropic、Google、Moonshot (Kimi)、DeepSeek、阿里、腾讯、百度的最新模型和动态。
2. **聚焦变化与动量**：筛选出近期用量激增、跑分爬升或价格骤降的核心模型。过滤掉长尾或无变化的模型。
3. **多模态专区**：单独开辟一个章节，如果有包含 Vision 等多模态评测变动，请给出分析。
4. **客观事实与数据**：保持投资研究研报的严谨风格，必须引用上下文中的具体数据支撑您的观点，并结合近期资讯做事实归因。
5. **排版要求**：全篇使用 Markdown 格式，结构清晰，适合在邮件中阅读。

【# 数据库最新情况】
{db_context}

{search_context}
"""
    
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": prompt}, {"role": "user", "content": "请生成今天的每日简报。"}],
        "max_tokens": 4000,
        "temperature": 0.2
    }
    
    try:
        resp = requests.post(f"{base_url}/chat/completions", headers=headers, json=payload, timeout=300)
        resp.raise_for_status()
        briefing_text = resp.json()['choices'][0]['message']['content']
        
        # 移除部分模型喜欢带的 <think> 标签
        briefing_text = re.sub(r'<think>.*?</think>', '', briefing_text, flags=re.DOTALL).strip()
    except Exception as e:
        print(f"AI API 请求失败: {e}")
        if resp:
            print(resp.text)
        return

    today_str = datetime.datetime.now().strftime('%Y-%m-%d')
    output_filename = f"briefing_cache/daily_briefing_{today_str}.md"
    os.makedirs("briefing_cache", exist_ok=True)
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(briefing_text)
    print(f"Saved briefing to {output_filename}")
    
    # 覆盖一个固定的最新版，便于前端读取
    with open("briefing_cache/daily_briefing_latest.md", "w", encoding="utf-8") as f:
        f.write(briefing_text)
        
    # --- 邮件发送逻辑 ---
    sender = os.environ.get("SMTP_SENDER", "")
    password = os.environ.get("SMTP_PASSWORD", "")
    server = os.environ.get("SMTP_SERVER", "smtp.qq.com")  # 默认 QQ 邮箱 SMTP
    port = int(os.environ.get("SMTP_PORT", "465"))
    recipients_str = os.environ.get("SMTP_RECIPIENTS", "799399681@qq.com")
    
    if sender and password and recipients_str:
        recipients = [email.strip() for email in recipients_str.split(',')]
        msg = MIMEMultipart()
        msg['From'] = f"AI Briefing <{sender}>"
        msg['To'] = ", ".join(recipients)
        msg['Subject'] = f"【AI 行业情报简报】{today_str}"
        
        # 将 Markdown 转换为极简 HTML 以便邮件渲染 (可选用更复杂的库，此处用简单的 text/plain)
        msg.attach(MIMEText(briefing_text, 'plain', 'utf-8'))
        
        try:
            print(f"Sending email from {sender} to {recipients} via {server}:{port}...")
            if port == 465:
                # SSL
                with smtplib.SMTP_SSL(server, port) as smtp:
                    smtp.login(sender, password)
                    smtp.send_message(msg)
            else:
                # TLS
                with smtplib.SMTP(server, port) as smtp:
                    smtp.starttls()
                    smtp.login(sender, password)
                    smtp.send_message(msg)
            print("Email sent successfully!")
        except Exception as e:
            print(f"Email failed to send: {e}")
    else:
        print("SMTP 配置不完整，跳过发送邮件。")

if __name__ == "__main__":
    main()
