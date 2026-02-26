"""
LMARENA (Chatbot Arena) 排行榜数据爬虫
数据源: https://github.com/nakasyou/lmarena-history (每日更新的 JSON 镜像)
输出: lmarena_leaderboard_records.csv
"""
import requests
import pandas as pd
from datetime import datetime
import os
import json

LMARENA_JSON_URL = "https://raw.githubusercontent.com/nakasyou/lmarena-history/main/output/scores.json"
OUTPUT_FILE = "lmarena_leaderboard_records.csv"

# 我们关注的核心维度 (LMARENA 的 text 类别)
CATEGORIES_OF_INTEREST = ['overall', 'coding', 'math', 'creative_writing', 'hard_6', 'chinese', 'english']


def fetch_lmarena_scores():
    """从 GitHub 拉取 LMARENA 历史分数 JSON"""
    print("🔍 正在从 LMARENA History 镜像获取排行榜数据...")
    try:
        resp = requests.get(LMARENA_JSON_URL, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        dates = sorted(data.keys())
        print(f"✅ 数据获取成功: {len(dates)} 个快照日期 ({dates[0]} ~ {dates[-1]})")
        return data
    except Exception as e:
        print(f"❌ 获取 LMARENA 数据失败: {e}")
        return None


def build_leaderboard_dataframe(scores_data):
    """
    从 JSON 构建 DataFrame
    输出格式: Date | Model | Category | ELO_Score
    仅提取最新日期的数据 (避免 CSV 过大)
    """
    if not scores_data:
        return pd.DataFrame()

    # 取最新日期
    latest_date_key = sorted(scores_data.keys())[-1]
    latest_data = scores_data[latest_date_key]

    # 格式化日期: 20250522 -> 2025-05-22
    date_str = f"{latest_date_key[:4]}-{latest_date_key[4:6]}-{latest_date_key[6:]}"
    print(f"📅 提取日期: {date_str}")

    rows = []

    # 提取 text 维度下的各类别
    text_data = latest_data.get('text', {})
    for category in CATEGORIES_OF_INTEREST:
        cat_scores = text_data.get(category, {})
        for model_name, elo_score in cat_scores.items():
            rows.append({
                'Date': date_str,
                'Model': model_name,
                'Category': category,
                'ELO_Score': round(float(elo_score), 2)
            })

    # 额外提取 vision 维度的 overall (如果存在)
    vision_data = latest_data.get('vision', {})
    vision_overall = vision_data.get('overall', {})
    for model_name, elo_score in vision_overall.items():
        rows.append({
            'Date': date_str,
            'Model': model_name,
            'Category': 'vision_overall',
            'ELO_Score': round(float(elo_score), 2)
        })

    df = pd.DataFrame(rows)
    print(f"📊 总记录数: {len(df)}, 涉及模型 {df['Model'].nunique()} 个, 类别 {df['Category'].nunique()} 个")
    return df


def update_leaderboard_database(new_df, file_name=OUTPUT_FILE):
    """增量更新 CSV 数据库"""
    if new_df.empty:
        print("⚠️ 无数据可更新。")
        return

    if os.path.exists(file_name):
        existing_df = pd.read_csv(file_name)
        current_date = new_df['Date'].iloc[0]

        # 移除旧的同日数据，用新数据替换
        existing_df = existing_df[existing_df['Date'] != current_date]
        final_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        final_df = new_df

    final_df.sort_values(by=['Date', 'Category', 'ELO_Score'], ascending=[True, True, False], inplace=True)
    final_df.to_csv(file_name, index=False)
    print(f"✅ LMARENA 排行榜数据库更新完成! 保存至 {file_name}")
    print(f"   总记录: {len(final_df)}, 日期范围: {final_df['Date'].nunique()} 天")


def main():
    scores_data = fetch_lmarena_scores()
    if not scores_data:
        return

    df = build_leaderboard_dataframe(scores_data)
    update_leaderboard_database(df)


if __name__ == "__main__":
    main()
