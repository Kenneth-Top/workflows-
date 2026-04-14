"""
utils/analytics.py — 核心分析逻辑
包含：
  1. detect_fast_ramp  — 新模型上量过快监测
  2. detect_slope_drop — 新模型斜率骤降监测
  3. 辅助函数：计算累计斜率、提取历史基准等

所有函数为纯计算逻辑，不包含任何 Streamlit UI 调用。
"""

import pandas as pd
import numpy as np
from config import (
    NEW_MODEL_LOOKBACK_DAYS,
    RAMP_OBSERVATION_WINDOW,
    RAMP_PERCENTILE_THRESHOLD,
    SLOPE_DROP_THRESHOLD,
    SLOPE_MIN_DAYS,
)


def _get_new_models(df: pd.DataFrame, lookback_days: int) -> pd.DataFrame:
    """
    找出最近 lookback_days 天内首次出现的模型。
    返回 DataFrame: [Display_Name, first_date]
    """
    latest_date = df["Date"].max()
    cutoff = latest_date - pd.Timedelta(days=lookback_days)

    first_dates = df.groupby("Display_Name")["Date"].min().reset_index()
    first_dates.columns = ["Display_Name", "first_date"]

    new_models = first_dates[first_dates["first_date"] >= cutoff]
    return new_models


def _calc_early_slope(df: pd.DataFrame, model_name: str, window: int) -> float | None:
    """
    计算指定模型在前 window 天内的「日均累计斜率」:
        slope = cumulative_tokens_at_day_N / N

    如果模型上线不满 window 天，则用其实际天数对齐：
        slope = cumulative_tokens_at_day_K / K

    返回 None 表示数据不足（< 1 天）。
    """
    m_df = df[df["Display_Name"] == model_name].sort_values("Date").copy()
    if m_df.empty:
        return None

    start_date = m_df.iloc[0]["Date"]
    # 去掉最后一天的不完整数据（与原有逻辑一致）
    if len(m_df) > 1:
        m_df = m_df.iloc[:-1]
    if m_df.empty:
        return None

    m_df["day_num"] = (m_df["Date"] - start_date).dt.days
    # 截取到观察窗口
    m_df_window = m_df[m_df["day_num"] <= window]
    if m_df_window.empty:
        return None

    cum_tokens = m_df_window["Total_Tokens"].sum()
    actual_days = m_df_window["day_num"].max()
    if actual_days <= 0:
        # 只有第 0 天（上线当天），无法计算斜率
        return cum_tokens  # 当天总量视为 1 天的斜率

    return cum_tokens / actual_days


def _calc_global_baseline(df: pd.DataFrame, window: int, percentile: float) -> float:
    """
    计算整个数据库中所有模型在前 window 天的日均斜率的第 `percentile` 百分位。
    用于判断新模型的早期上量是否比全库 (100-percentile)% 的模型都快。
    返回百分位阈值 (float)。如果数据不足，返回 inf。
    """
    all_models = df["Display_Name"].unique()

    slopes = []
    for model in all_models:
        s = _calc_early_slope(df, model, window)
        if s is not None and s > 0:
            slopes.append(s)

    if not slopes:
        return float("inf")

    return float(np.percentile(slopes, percentile))



def detect_fast_ramp(df: pd.DataFrame) -> dict:
    """
    检测最近 NEW_MODEL_LOOKBACK_DAYS 天内新上线的模型，
    其前 RAMP_OBSERVATION_WINDOW 天的日均累计斜率是否超过历史 P95。

    返回:
        {
          "triggered": [
              {
                "model": str,
                "online_date": str,
                "observation_days": int,
                "current_slope": float,
                "threshold": float,
                "exceed_pct": float,       # 超出幅度百分比
              }, ...
          ],
          "threshold": float,
          "percentile": int,
          "window": int,
        }
    """
    new_models = _get_new_models(df, NEW_MODEL_LOOKBACK_DAYS)
    if new_models.empty:
        return {"triggered": [], "threshold": None, "percentile": RAMP_PERCENTILE_THRESHOLD, "window": RAMP_OBSERVATION_WINDOW}

    new_names = new_models["Display_Name"].tolist()
    baseline = _calc_global_baseline(df, RAMP_OBSERVATION_WINDOW, RAMP_PERCENTILE_THRESHOLD)

    triggered = []
    for _, row in new_models.iterrows():
        model = row["Display_Name"]
        first_date = row["first_date"]
        slope = _calc_early_slope(df, model, RAMP_OBSERVATION_WINDOW)
        if slope is None:
            continue

        # 计算实际观察天数
        m_df = df[df["Display_Name"] == model].sort_values("Date")
        if len(m_df) > 1:
            m_df = m_df.iloc[:-1]
        if m_df.empty:
            continue
        obs_days = (m_df.iloc[-1]["Date"] - first_date).days
        obs_days = max(obs_days, 1)

        if slope > baseline:
            exceed_pct = ((slope - baseline) / baseline) * 100 if baseline > 0 else float("inf")
            triggered.append({
                "model": model,
                "online_date": first_date.strftime("%Y-%m-%d"),
                "observation_days": min(obs_days, RAMP_OBSERVATION_WINDOW),
                "current_slope": round(slope, 6),
                "threshold": round(baseline, 6),
                "exceed_pct": round(exceed_pct, 1),
            })

    # 按超出幅度降序排列
    triggered.sort(key=lambda x: x["exceed_pct"], reverse=True)

    return {
        "triggered": triggered,
        "threshold": round(baseline, 6) if baseline != float("inf") else None,
        "percentile": RAMP_PERCENTILE_THRESHOLD,
        "window": RAMP_OBSERVATION_WINDOW,
    }


def detect_slope_drop(df: pd.DataFrame) -> dict:
    """
    检测最近 NEW_MODEL_LOOKBACK_DAYS 天内新上线的模型，
    其「最新一天的累计斜率」相比「前一天的累计斜率」是否骤降。

    斜率定义:
        slope(T) = cumulative_tokens_up_to_T / (T 距上线天数)

    判定:
        V_drop = (slope_latest - slope_prev) / slope_prev
        若 V_drop <= -SLOPE_DROP_THRESHOLD  →  触发报警

    返回:
        {
          "triggered": [
              {
                "model": str,
                "online_date": str,
                "slope_prev": float,
                "slope_latest": float,
                "drop_pct": float,            # 负值，如 -35.2
                "threshold": float,            # 配置阈值如 0.30
              }, ...
          ],
          "threshold": float,
        }
    """
    new_models = _get_new_models(df, NEW_MODEL_LOOKBACK_DAYS)
    if new_models.empty:
        return {"triggered": [], "threshold": SLOPE_DROP_THRESHOLD}

    triggered = []
    for _, row in new_models.iterrows():
        model = row["Display_Name"]
        first_date = row["first_date"]

        m_df = df[df["Display_Name"] == model].sort_values("Date").copy()
        # 去掉最后一天不完整数据
        if len(m_df) > 1:
            m_df = m_df.iloc[:-1]
        if len(m_df) < SLOPE_MIN_DAYS:
            continue

        m_df["day_num"] = (m_df["Date"] - first_date).dt.days
        m_df["cum_tokens"] = m_df["Total_Tokens"].cumsum()

        # 取最后两个数据点
        latest_row = m_df.iloc[-1]
        prev_row = m_df.iloc[-2]

        day_latest = latest_row["day_num"]
        day_prev = prev_row["day_num"]

        if day_latest <= 0 or day_prev <= 0:
            continue

        slope_latest = latest_row["cum_tokens"] / day_latest
        slope_prev = prev_row["cum_tokens"] / day_prev

        if slope_prev <= 0:
            continue

        v_drop = (slope_latest - slope_prev) / slope_prev

        if v_drop <= -SLOPE_DROP_THRESHOLD:
            triggered.append({
                "model": model,
                "online_date": first_date.strftime("%Y-%m-%d"),
                "slope_prev": round(slope_prev, 6),
                "slope_latest": round(slope_latest, 6),
                "drop_pct": round(v_drop * 100, 1),
                "threshold": SLOPE_DROP_THRESHOLD,
            })

    # 按降幅从大到小排序
    triggered.sort(key=lambda x: x["drop_pct"])

    return {"triggered": triggered, "threshold": SLOPE_DROP_THRESHOLD}
