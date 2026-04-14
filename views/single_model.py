"""
pages/single_model.py — 单模型用量页面
单模型每日详情 + 趋势分析 + 日期筛选
"""

import streamlit as st
import pandas as pd
import altair as alt

from utils.helpers import get_dataset_download


def render(df: pd.DataFrame):
    """渲染单模型用量页面"""
    st.subheader("单模型每日详情趋势")

    all_model_names = sorted(df["Display_Name"].unique().tolist())
    selected_name = st.selectbox("选择模型", all_model_names)

    m_df_full = df[df["Display_Name"] == selected_name].sort_values("Date").copy()

    if m_df_full.empty:
        st.warning("该模型暂无数据。")
        return

    min_date = m_df_full["Date"].min().date()
    max_date = m_df_full["Date"].max().date()
    st.success(f"📅 **{selected_name}** 数据收录区间: {min_date} 至 {max_date}")

    # 日期筛选
    col_filter1, col_filter2 = st.columns([1, 3])
    with col_filter1:
        date_range = st.date_input(
            "🔎 筛选时间段",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )

    if len(date_range) == 2:
        start_filter, end_filter = date_range
        mask = (m_df_full["Date"].dt.date >= start_filter) & (m_df_full["Date"].dt.date <= end_filter)
        m_df = m_df_full.loc[mask]
    else:
        m_df = m_df_full

    if m_df.empty:
        st.warning("⚠️ 所选时间段内无数据。")
        return

    # 核心指标卡片
    latest = m_df.iloc[-1]
    c1, c2, c3 = st.columns(3)
    c1.metric("选段最新日期", latest["Date"].strftime("%Y-%m-%d"))
    c2.metric("当日消耗", f"{latest['Total_Tokens']:.4f} B")
    if latest.get("Reasoning", 0) > 0 and latest.get("Completion", 0) > 0:
        ratio = (latest["Reasoning"] / latest["Completion"]) * 100
        c3.metric("Reasoning 占比", f"{ratio:.1f}%")
    else:
        c3.metric("Prompt Tokens", f"{latest['Prompt']:.4f} B")

    # 折线图
    chart = (
        alt.Chart(m_df)
        .mark_line(point=True)
        .encode(
            x=alt.X("Date", title="日期", axis=alt.Axis(format="%m-%d")),
            y=alt.Y("Total_Tokens", title="Token (Billion)"),
            tooltip=["Date", "Total_Tokens", "Prompt", "Completion"],
        )
    )
    st.altair_chart(chart, width="stretch")

    # 数据表格
    display_cols = ["Date", "Total_Tokens", "Prompt", "Completion", "Reasoning"]
    valid_cols = [c for c in display_cols if c in m_df.columns]
    st.dataframe(
        m_df[valid_cols].sort_values("Date", ascending=False).style.format({"Total_Tokens": "{:.4f}"}),
        width="stretch",
    )

    data, fname, mime, label = get_dataset_download(m_df[valid_cols], f"{selected_name}_daily")
    st.download_button(label=label, data=data, file_name=fname, mime=mime)
