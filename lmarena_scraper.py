"""
LMARENA (Arena) 排行榜数据爬虫
数据源: https://arena.ai/leaderboard (Overview 页面)
包含：8 个 ELO 排行榜 + Arena Overview (综合排名表)
"""
import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
import re
from datetime import datetime

OUTPUT_FILE = "lmarena_leaderboard_records.csv"

# 8 个排行榜的表格索引 → 列名映射
LEADERBOARD_MAP = {
    0: 'Score_text',
    1: 'Score_code',
    2: 'Score_vision',
    3: 'Score_text_to_image',
    4: 'Score_image_edit',
    5: 'Score_search',
    6: 'Score_text_to_video',
    7: 'Score_image_to_video',
}

# Arena Overview 表的维度列名
OVERVIEW_COLS = ['Overall', 'Expert', 'Hard_Prompts', 'Coding', 
                 'Math', 'Creative_Writing', 'Instruction_Following', 'Longer_Query']

def clean_model_name(raw_name):
    """清理模型名称 — 去除厂商前缀粘连 (如 'Anthropicclaude-opus-4-6' -> 'claude-opus-4-6')"""
    # arena.ai 的 HTML 中厂商 icon 的文本会粘连在模型名前面
    prefixes = ['Anthropic', 'Google', 'OpenAI', 'xAI', 'Meta', 'Mistral', 
                'Alibaba', 'DeepSeek', 'Zhipu', 'Baidu', 'ByteDance', 'Cohere',
                'Reka', 'AI21', 'Together', 'Nvidia', 'Amazon', 'Apple', 'Microsoft',
                'Tencent', 'Stability', 'Ideogram', 'Midjourney', 'Black Forest Labs',
                'Luma', 'Runway', 'Perplexity', 'You.com', 'Kling', 'Minimax',
                'Step', 'Moonshot', '01.AI', 'InternLM', 'Yi', 'Nous']
    name = raw_name.strip()
    for prefix in prefixes:
        if name.startswith(prefix) and len(name) > len(prefix):
            rest = name[len(prefix):]
            # 确保去掉前缀后的第一个字符是小写或特殊字符（说明是粘连的）
            if rest[0].islower() or rest[0] in '-_./0123456789':
                name = rest
                break
    return name

def fetch_arena_leaderboard():
    """从 arena.ai 获取排行榜数据"""
    url = 'https://arena.ai/leaderboard'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
    }
    
    print("正在从 arena.ai 获取最新排行榜...")
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    html = resp.text
    print(f"  页面大小: {len(html)} bytes")
    
    soup = BeautifulSoup(html, 'html.parser')
    tables = soup.find_all('table')
    print(f"  找到 {len(tables)} 个表格")
    
    if len(tables) < 9:
        print(f"  [警告] 预期 9 个表格,实际 {len(tables)} 个!")
        return None
    
    today = datetime.now().strftime('%Y-%m-%d')
    all_data = {}  # model_name -> row dict
    
    # --- 解析 8 个 ELO 排行榜 (表格 0-7) ---
    for table_idx, col_name in LEADERBOARD_MAP.items():
        table = tables[table_idx]
        rows = table.find_all('tr')[1:]  # 跳过表头
        count = 0
        for row in rows:
            cells = row.find_all(['td', 'th'])
            if len(cells) >= 4:
                model = clean_model_name(cells[1].get_text(strip=True))
                try:
                    score = int(cells[2].get_text(strip=True).replace(',', ''))
                except ValueError:
                    continue
                try:
                    votes = int(cells[3].get_text(strip=True).replace(',', ''))
                except ValueError:
                    votes = 0
                
                if model not in all_data:
                    all_data[model] = {'Date': today, 'Model': model}
                all_data[model][col_name] = score
                all_data[model][f'Votes_{col_name.replace("Score_", "")}'] = votes
                count += 1
        print(f"  {col_name}: {count} 个模型")
    
    # --- 解析 Arena Overview 表 (表格 8) ---
    overview_table = tables[8]
    overview_rows = overview_table.find_all('tr')[1:]  # 跳过表头
    overview_count = 0
    for row in overview_rows:
        cells = row.find_all(['td', 'th'])
        if len(cells) >= 2:
            model = clean_model_name(cells[0].get_text(strip=True))
            # 去掉 "610/610" 等后缀
            model = re.sub(r'\d+/\d+$', '', model).strip()
            
            if model not in all_data:
                all_data[model] = {'Date': today, 'Model': model}
            
            # Overall 到 Longer_Query 的排名
            for i, col in enumerate(OVERVIEW_COLS):
                col_idx = i + 1  # 第 0 列是模型名
                if col_idx < len(cells):
                    try:
                        rank_val = int(cells[col_idx].get_text(strip=True))
                        all_data[model][f'Rank_{col}'] = rank_val
                    except ValueError:
                        pass
            overview_count += 1
    print(f"  Arena Overview: {overview_count} 个模型")
    
    # 构建 DataFrame
    df_new = pd.DataFrame(list(all_data.values()))
    print(f"  总计: {len(df_new)} 个唯一模型")
    
    return df_new

def main():
    df_new = fetch_arena_leaderboard()
    if df_new is None or df_new.empty:
        print("未获取到数据。")
        return
    
    # 合并历史数据
    if os.path.exists(OUTPUT_FILE):
        df_old = pd.read_csv(OUTPUT_FILE)
        # 只保留今天之前的数据
        today = datetime.now().strftime('%Y-%m-%d')
        df_old = df_old[df_old['Date'] != today]
        df = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df = df_new
    
    df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    
    # 数据预览
    print(f"\n数据预览 (前5行):")
    print(df_new.head().to_string())
    
    # ELO 分数列统计
    score_cols = [c for c in df_new.columns if c.startswith('Score_')]
    rank_cols = [c for c in df_new.columns if c.startswith('Rank_')]
    print(f"\nELO 分数列: {score_cols}")
    print(f"排名列: {rank_cols}")
    for c in score_cols + rank_cols:
        print(f"  {c}: {df_new[c].notna().sum()} 非空")
    
    print(f"\nLMARENA 数据更新完成: {OUTPUT_FILE} ({len(df)} 条记录)")

if __name__ == "__main__":
    main()
