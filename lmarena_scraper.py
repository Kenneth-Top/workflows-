"""
LMARENA (Arena) 排行榜数据爬虫
数据源: https://arena.ai/leaderboard (Overview 页面) 及各个具体赛道页面
包含：8 个 ELO 排行榜 + Arena Overview (综合排名表) 的全量数据
"""
import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
import re
from datetime import datetime

OUTPUT_FILE = "lmarena_leaderboard_records.csv"

# 8 个子品类路由 -> 列名映射
LEADERBOARD_URLS = {
    'text': 'Score_text',
    'code': 'Score_code',
    'vision': 'Score_vision',
    'text-to-image': 'Score_text_to_image',
    'image-edit': 'Score_image_edit',
    'search': 'Score_search',
    'text-to-video': 'Score_text_to_video',
    'image-to-video': 'Score_image_to_video',
}

# Arena Overview 表的维度列名
OVERVIEW_COLS = ['Overall', 'Expert', 'Hard_Prompts', 'Coding', 
                 'Math', 'Creative_Writing', 'Instruction_Following', 'Longer_Query']

def clean_model_name(raw_name_or_tag):
    """清理模型名称 — 去除厂商前缀粘连并处理 a 标签"""
    if hasattr(raw_name_or_tag, 'find'):
        # On subpages, the model name might be inside an <a> tag
        a_tag = raw_name_or_tag.find('a')
        if a_tag:
            name = a_tag.get_text(strip=True)
        else:
            name = raw_name_or_tag.get_text(strip=True)
    else:
        name = str(raw_name_or_tag).strip()
    
    # arena.ai 的 HTML 中厂商 icon 的文本会粘连在模型名前面
    prefixes = ['Anthropic', 'Google', 'OpenAI', 'xAI', 'Meta', 'Mistral', 
                'Alibaba', 'DeepSeek', 'Zhipu', 'Baidu', 'ByteDance', 'Cohere',
                'Reka', 'AI21', 'Together', 'Nvidia', 'Amazon', 'Apple', 'Microsoft',
                'Tencent', 'Stability', 'Ideogram', 'Midjourney', 'Black Forest Labs',
                'Luma', 'Runway', 'Perplexity', 'You.com', 'Kling', 'Minimax',
                'Step', 'Moonshot', '01.AI', 'InternLM', 'Yi', 'Nous']
    for prefix in prefixes:
        if name.startswith(prefix) and len(name) > len(prefix):
            rest = name[len(prefix):]
            if rest[0].islower() or rest[0] in '-_.0123456789':
                name = rest
                break
                
    # 去除 Overview 中带有的类似 /610 后缀
    name = re.sub(r'\d+/\d+$', '', name).strip()
    return name

def fetch_arena_leaderboard():
    """从 arena.ai 获取全量排行榜数据"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
    }
    
    today = datetime.now().strftime('%Y-%m-%d')
    all_data = {}  # model_name -> row dict
    
    # === 1. 构建 Overview (以获取完整排行模型库) ===
    print("正在从 arena.ai/leaderboard 获取 Overview 全面排名...")
    try:
        resp = requests.get('https://arena.ai/leaderboard', headers=headers, timeout=30)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, 'html.parser')
        tables = soup.find_all('table')
        
        if len(tables) >= 9:
            overview_table = tables[8]
            overview_rows = overview_table.find_all('tr')[1:]
            for row in overview_rows:
                cells = row.find_all(['td', 'th'])
                if len(cells) >= 2:
                    model = clean_model_name(cells[0])
                    if model not in all_data:
                        all_data[model] = {'Date': today, 'Model': model}
                    
                    for i, col in enumerate(OVERVIEW_COLS):
                        col_idx = i + 1
                        if col_idx < len(cells):
                            try:
                                rank_val = int(cells[col_idx].get_text(strip=True))
                                all_data[model][f'Rank_{col}'] = rank_val
                            except ValueError:
                                pass
            print(f"  [成功] Overview: {len(all_data)} 个模型入库")
        else:
            print("  [警告] 未能找到第 9 个榜单结构（Overview）")
    except Exception as e:
        print(f"获取 Overview 异常: {e}")

    # === 2. 构建 8大分项赛道 (以获取全量 ELO 打分) ===
    for path, col_name in LEADERBOARD_URLS.items():
        url = f'https://arena.ai/leaderboard/{path}'
        print(f"正在拉取大类打分榜单: /{path} ...")
        try:
            r = requests.get(url, headers=headers, timeout=30)
            if r.status_code != 200:
                print(f"  [失败] HTTP {r.status_code}")
                continue
            
            s = BeautifulSoup(r.text, 'html.parser')
            table = s.find('table')
            if not table:
                print(f"  [缺失] 页面不含表格")
                continue
            
            count = 0
            for row in table.find_all('tr')[1:]:
                cells = row.find_all(['td', 'th'])
                if len(cells) >= 5:
                    model = clean_model_name(cells[2])
                    
                    score_str = cells[3].get_text(strip=True)
                    score_match = re.search(r'^(\d+)', score_str)
                    if score_match:
                        score = int(score_match.group(1))
                    else:
                        score = None
                    
                    try:
                        votes = int(cells[4].get_text(strip=True).replace(',', ''))
                    except ValueError:
                        votes = 0

                    if score is not None:
                        if model not in all_data:
                            all_data[model] = {'Date': today, 'Model': model}
                        all_data[model][col_name] = score
                        all_data[model][f'Votes_{col_name.replace("Score_", "")}'] = votes
                        count += 1
            print(f"  [成功] {col_name}: {count} 款参评分数挂载")
        except Exception as e:
            print(f"  [异常] 爬取 /{path} 时发生错误: {e}")

    df_new = pd.DataFrame(list(all_data.values()))
    print(f"\n全量拼装检索完毕，总计覆盖 {len(df_new)} 个唯一模型\n")
    return df_new

def main():
    df_new = fetch_arena_leaderboard()
    if df_new is None or df_new.empty:
        print("未获取到数据，请检查网络或网站结构是否变更。")
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
    print(f"数据总预览 (前5行):")
    print(df_new.head().to_string())
    
    # ELO 分数列统计
    score_cols = [c for c in df_new.columns if c.startswith('Score_')]
    rank_cols = [c for c in df_new.columns if c.startswith('Rank_')]
    print(f"\nELO 分数列检查: {score_cols}")
    print(f"排名列检查: {rank_cols}")
    for c in score_cols + rank_cols:
        print(f"  {c}: {df_new[c].notna().sum()} 条有效信息")
    
    print(f"\nLMARENA 数据管道全链路聚合更新完成: {OUTPUT_FILE} ({len(df)} 条记录)")

if __name__ == "__main__":
    main()
