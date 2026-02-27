"""
LMARENA 排行榜数据爬虫 (9 个排行榜 + ELO 分数)
数据源: https://lmarena-ai.com/ (包含完整 ELO 分数的聚合页面)
输出: lmarena_leaderboard_records.csv

HTML 结构:
  <div class="leaderboard-card">
    <div class="card-header"><h3>LMArena文本排行榜</h3></div>
    <ol>
      <li><span class="rank">1</span><span class="model-name">Gemini-3-Pro</span><span class="model-score">1491</span></li>
      ...
    </ol>
  </div>
"""
import requests
import pandas as pd
from datetime import datetime
from bs4 import BeautifulSoup
import os
import re

OUTPUT_FILE = "lmarena_leaderboard_records.csv"

# H3 索引 (1-9) 对应的 CSV 列名（索引 0 是"评测原理"不是排行榜）
H3_INDEX_TO_COL = {
    1: 'Score_text',       # 文本排行榜
    2: 'Score_vision',     # 视觉排行榜
    3: 'Score_webdev',     # 网页开发排行榜
    4: 'Score_image_gen',  # 文生图排行榜
    5: 'Score_image_edit', # 图像编辑排行榜
    6: 'Score_search',     # 搜索排行榜
    7: 'Score_text_video', # 文生视频排行榜
    8: 'Score_img_video',  # 图生视频排行榜
    9: 'Overall_Rank',     # 综合排行榜（排名，非分数）
}

SCORE_COLS = [
    'Score_text', 'Score_vision', 'Score_webdev',
    'Score_image_gen', 'Score_image_edit', 'Score_search',
    'Score_text_video', 'Score_img_video', 'Overall_Rank',
]


def fetch_lmarena_data():
    """从 lmarena-ai.com 解析 9 个排行榜数据"""
    print("正在从 lmarena-ai.com 获取最新排行榜...")

    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    })

    try:
        resp = session.get("https://lmarena-ai.com/", timeout=30)
        resp.raise_for_status()
        html = resp.text
        print(f"  页面大小: {len(html)} bytes")
    except Exception as e:
        print(f"获取页面失败: {e}")
        return None

    soup = BeautifulSoup(html, 'html.parser')
    h3_tags = soup.find_all('h3')
    print(f"  找到 {len(h3_tags)} 个 h3 标签")

    all_models = {}  # model_name -> {col: value, ...}

    for h3_idx, col_name in H3_INDEX_TO_COL.items():
        if h3_idx >= len(h3_tags):
            print(f"  [跳过] h3 索引 {h3_idx} 超出范围")
            continue

        h3 = h3_tags[h3_idx]

        # 找到包含该 h3 的 leaderboard-card div
        card = h3.find_parent('div', class_='leaderboard-card')
        if not card:
            # 备用：向上找任何包含 li 的父 div
            card = h3.find_parent('div', class_=True)
            while card:
                if len(card.find_all('li')) > 3:
                    break
                card = card.find_parent('div', class_=True)

        if not card:
            print(f"  [警告] 未找到 {col_name} 的数据容器")
            continue

        lis = card.find_all('li')
        count = 0

        for li in lis:
            name_span = li.find('span', class_='model-name')
            score_span = li.find('span', class_='model-score')
            rank_span = li.find('span', class_=re.compile(r'^rank'))

            if not name_span:
                continue

            model_name = name_span.text.strip()
            if not model_name:
                continue

            if model_name not in all_models:
                all_models[model_name] = {}

            if col_name == 'Overall_Rank' and rank_span:
                # 综合排行榜用排名
                try:
                    all_models[model_name][col_name] = int(rank_span.text.strip())
                except ValueError:
                    pass
            elif score_span:
                # 其他排行榜用 ELO 分数
                try:
                    all_models[model_name][col_name] = int(score_span.text.strip())
                except ValueError:
                    pass

            count += 1

        print(f"  {col_name}: {count} 个模型")

    print(f"  总计: {len(all_models)} 个唯一模型")
    return all_models


def build_dataframe(all_models):
    """从解析结果构建 DataFrame"""
    if not all_models:
        return pd.DataFrame()

    today = datetime.now().strftime("%Y-%m-%d")
    rows = []

    for model_name, scores in all_models.items():
        row = {'Date': today, 'Model': model_name}
        for col in SCORE_COLS:
            row[col] = scores.get(col)
        rows.append(row)

    df = pd.DataFrame(rows)

    # 按文本 ELO 分数降序排序
    if 'Score_text' in df.columns:
        df = df.sort_values('Score_text', ascending=False, na_position='last')

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
    all_models = fetch_lmarena_data()
    if not all_models:
        return
    df = build_dataframe(all_models)
    if not df.empty:
        print(f"\n数据预览 (前5行):")
        print(df.head().to_string())
        print(f"\n各列非空计数:")
        print(df.notna().sum())
    update_database(df)


if __name__ == "__main__":
    main()
