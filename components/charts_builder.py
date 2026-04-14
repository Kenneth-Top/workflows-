"""
components/charts_builder.py — 可复用的 Altair 图表构建函数
每个函数接收准备好的 DataFrame，返回 Altair Chart 对象或直接 st.altair_chart。
"""

import altair as alt
import streamlit as st
import pandas as pd


def cumulative_growth_chart(df_plot: pd.DataFrame, max_day: int) -> alt.LayerChart:
    """
    多模型累计增长折线图。
    df_plot 列: Model, Day, Date, Cumulative_Tokens
    """
    base = alt.Chart(df_plot).encode(
        x=alt.X(
            "Day",
            title="上线天数 (Daily)",
            scale=alt.Scale(domain=[0, max_day + 2], clamp=True),
            axis=alt.Axis(labelFontSize=16, titleFontSize=18, grid=True),
        ),
        y=alt.Y(
            "Cumulative_Tokens",
            title="累计 Token (Billion)",
            axis=alt.Axis(labelFontSize=16, titleFontSize=18),
        ),
        color=alt.Color(
            "Model",
            title="模型名称",
            scale=alt.Scale(scheme="tableau10"),
            legend=alt.Legend(orient="bottom"),
        ),
        tooltip=["Model", "Day", "Date", "Cumulative_Tokens"],
    )
    chart = (base.mark_line(strokeWidth=3) + base.mark_circle(size=60)).properties(height=600)
    return chart


def single_model_daily_chart(m_df: pd.DataFrame) -> alt.Chart:
    """
    单模型每日 Token 消耗折线图。
    m_df 列: Date, Total_Tokens, Prompt, Completion
    """
    chart = (
        alt.Chart(m_df)
        .mark_line(point=True)
        .encode(
            x=alt.X("Date", title="日期", axis=alt.Axis(format="%m-%d")),
            y=alt.Y("Total_Tokens", title="Token (Billion)"),
            tooltip=["Date", "Total_Tokens", "Prompt", "Completion"],
        )
    )
    return chart
