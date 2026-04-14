"""
pages/cumulative_compare.py — 累计用量对比页面
核心页面：多模型累计增长对比 + 预警面板
"""

import streamlit as st
import pandas as pd

from components.custom_multiselect import model_multiselect
from components.charts_builder import cumulative_growth_chart
from components.alert_panels import render_fast_ramp_panel, render_slope_drop_panel
from utils.analytics import detect_fast_ramp, detect_slope_drop
from utils.helpers import get_dataset_download


def render(df: pd.DataFrame):
    """渲染累计用量对比页面"""
    st.subheader("多模型累计增长对比")

    all_model_names = sorted(df["Display_Name"].unique().tolist())

    # ── 自定义多选控件 ──
    selected_names = model_multiselect(
        all_models=all_model_names,
        default_models=all_model_names[:3] if len(all_model_names) >= 3 else all_model_names,
        key_prefix="cum",
    )

    if selected_names:
        plot_data = []
        max_day_plot = 0

        # 显示每个模型的收录起始日
        cols = st.columns(min(len(selected_names), 6))
        for idx, name in enumerate(selected_names):
            m_df_temp = df[df["Display_Name"] == name].sort_values("Date")
            if not m_df_temp.empty:
                s_date = m_df_temp.iloc[0]["Date"].strftime("%Y-%m-%d")
                cols[idx % len(cols)].caption(f"📅 **{name}**: {s_date}")

        # 构建绘图数据
        for name in selected_names:
            m_df = df[df["Display_Name"] == name].sort_values("Date").copy()
            m_df["Cum_Tokens"] = m_df["Total_Tokens"].cumsum()
            if len(m_df) > 1:
                m_df = m_df.iloc[:-1]
            if m_df.empty:
                continue

            start_date = m_df.iloc[0]["Date"]
            current_max_day = (m_df.iloc[-1]["Date"] - start_date).days
            if current_max_day > max_day_plot:
                max_day_plot = current_max_day

            for _, row in m_df.iterrows():
                day_num = (row["Date"] - start_date).days
                plot_data.append({
                    "Model": name,
                    "Day": day_num,
                    "Date": row["Date"].strftime("%Y-%m-%d"),
                    "Cumulative_Tokens": row["Cum_Tokens"],
                })

        if plot_data:
            df_plot = pd.DataFrame(plot_data)
            chart = cumulative_growth_chart(df_plot, max_day_plot)
            st.altair_chart(chart, width="stretch")

            # 数值明细
            st.markdown("### 📅 累计数值明细")
            df_pivot = df_plot.pivot_table(index="Day", columns="Model", values="Cumulative_Tokens")
            st.dataframe(df_pivot.style.format("{:.4f} B"), width="stretch")

            data, fname, mime, label = get_dataset_download(df_pivot.reset_index(), "cumulative_growth")
            st.download_button(label=label, data=data, file_name=fname, mime=mime)

    # ── 预警面板区域 ──
    st.divider()
    st.markdown("## 🔔 新模型监测预警")

    col_alert1, col_alert2 = st.columns(2)
    with col_alert1:
        ramp_result = detect_fast_ramp(df)
        render_fast_ramp_panel(ramp_result)
    with col_alert2:
        drop_result = detect_slope_drop(df)
        render_slope_drop_panel(drop_result)
