"""
LMARENA (Chatbot Arena) 排行榜数据爬虫
数据源: https://lmarena.ai/leaderboard (官网页面内嵌 JSON)
输出: lmarena_leaderboard_records.csv
"""
import requests
import re
import json
import pandas as pd
from datetime import datetime
import os

OUTPUT_FILE = "lmarena_leaderboard_records.csv"


def fetch_lmarena_from_website():
    """从 LMARENA 官网页面提取最新排行榜数据"""
    print("正在从 lmarena.ai 获取最新排行榜...")
    
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    })
    
    try:
        resp = session.get("https://lmarena.ai/leaderboard", timeout=30)
        resp.raise_for_status()
        html = resp.text
        print(f"  页面大小: {len(html)} bytes")
    except Exception as e:
        print(f"获取页面失败: {e}")
        return None
    
    # 从 Next.js 流式 script 标签中提取 initialModels JSON
    script_tags = re.findall(r'<script[^>]*>(.*?)</script>', html, re.DOTALL)
    
    for s in script_tags:
        if 'initialModels' not in s:
            continue
        
        # 解码 Next.js push 格式
        push_match = re.search(r'self\.__next_f\.push\(\[1,"(.*?)"\]\)', s, re.DOTALL)
        if not push_match:
            continue
        
        raw = push_match.group(1)
        decoded = raw.replace('\\"', '"').replace('\\\\', '\\').replace('\\n', '\n')
        
        # 提取 initialModels JSON 数组
        idx = decoded.find('"initialModels":[')
        if idx == -1:
            continue
        
        arr_start = decoded.find('[', idx)
        bracket_count = 0
        arr_end = arr_start
        for j in range(arr_start, len(decoded)):
            if decoded[j] == '[':
                bracket_count += 1
            elif decoded[j] == ']':
                bracket_count -= 1
                if bracket_count == 0:
                    arr_end = j + 1
                    break
        
        models_json = decoded[arr_start:arr_end]
        
        try:
            models = json.loads(models_json)
            print(f"  成功解析 {len(models)} 个模型")
            return models
        except json.JSONDecodeError as e:
            print(f"  JSON 解析失败: {e}")
            return None
    
    print("  未找到 initialModels 数据")
    return None


def build_dataframe(models):
    """从模型列表构建 DataFrame"""
    if not models:
        return pd.DataFrame()
    
    today = datetime.now().strftime("%Y-%m-%d")
    rows = []
    
    # 所有可能的 modality
    all_modalities = set()
    for m in models:
        rbm = m.get('rankByModality', {})
        all_modalities.update(rbm.keys())
    
    for m in models:
        rank = m.get('rank')
        if rank is None:
            continue
        
        # 过滤掉 rank 为极大值（9007199254740991 = JS MAX_SAFE_INTEGER，表示未排名）
        rbm = m.get('rankByModality', {})
        
        # 只保留有实际排名的 modality (非 MAX_SAFE_INTEGER)
        valid_modalities = {k: v for k, v in rbm.items() if v < 9007199254740000}
        
        if not valid_modalities and rank >= 9007199254740000:
            continue
        
        row = {
            'Date': today,
            'Model': m.get('displayName') or m.get('name', ''),
            'Organization': m.get('organization', ''),
            'Overall_Rank': rank if rank < 9007199254740000 else None,
        }
        
        # 添加各 modality 的排名
        for mod in sorted(all_modalities):
            mod_rank = rbm.get(mod)
            row[f'Rank_{mod}'] = mod_rank if mod_rank and mod_rank < 9007199254740000 else None
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # 按 Overall_Rank 排序
    if 'Overall_Rank' in df.columns:
        df = df.sort_values('Overall_Rank', na_position='last')
    
    print(f"  有效模型: {len(df)}, 维度: {sorted(all_modalities)}")
    return df


def update_database(new_df, file_name=OUTPUT_FILE):
    """增量更新 CSV"""
    if new_df.empty:
        print("无数据可更新。")
        return
    
    if os.path.exists(file_name):
        existing_df = pd.read_csv(file_name)
        current_date = new_df['Date'].iloc[0]
        existing_df = existing_df[existing_df['Date'] != current_date]
        final_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        final_df = new_df
    
    final_df.to_csv(file_name, index=False)
    print(f"LMARENA 数据更新完成: {file_name} ({len(final_df)} 条记录)")


def main():
    models = fetch_lmarena_from_website()
    if not models:
        return
    df = build_dataframe(models)
    update_database(df)


if __name__ == "__main__":
    main()
