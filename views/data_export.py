"""
pages/data_export.py — 数据导出页面
全量下载 + 按模型筛选查看原始数据
"""

import streamlit as st
import pandas as pd

from utils.helpers import get_dataset_download


def render(df: pd.DataFrame):
    """渲染数据导出页面"""
    st.subheader("数据库原始数据")

    # 全量下载
    st.markdown("#### 💾 全量数据下载")
    data, fname, mime, label = get_dataset_download(df, "full_history_database")
    st.download_button(label=label, data=data, file_name=fname, mime=mime)

    st.divider()

    # 按模型查看
    all_model_names = sorted(df["Display_Name"].unique().tolist())
    check_name = st.selectbox("选择要检查的模型:", all_model_names)
    filtered_df = df[df["Display_Name"] == check_name].sort_values("Date", ascending=False)

    st.dataframe(
        filtered_df.style.format({
            "Prompt": "{:.6f} B",
            "Completion": "{:.6f} B",
            "Reasoning": "{:.6f} B",
            "Total_Tokens": "{:.6f} B",
        }),
        width="stretch",
    )
