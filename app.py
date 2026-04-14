"""
app.py — 应用入口与路由
职责：
  1. 页面级配置 (set_page_config)
  2. 数据加载
  3. 侧边栏导航 (基于 config.PAGE_VISIBILITY 过滤)
  4. 页面路由分发
"""

import streamlit as st
import pandas as pd

from config import PAGE_VISIBILITY
from utils.data_services import load_all

# ── 页面配置 (必须在最前) ──
st.set_page_config(page_title="LLM 数据看板", layout="wide")
st.title("LLM 数据看板")

# ── 加载数据 ──
data = load_all()
df = data["df"]
error = data["error"]
df_price = data["df_price"]
df_bench = data["df_bench"]
df_lmarena = data["df_lmarena"]

if error and df is None:
    st.error(error)
    st.stop()

# ── 侧边栏导航 ──
st.sidebar.title("导航")

# 仅展示 PAGE_VISIBILITY 中标记为 True 的页面
visible_pages = [name for name, visible in PAGE_VISIBILITY.items() if visible]
page = st.sidebar.radio("选择分析视图", visible_pages)

# 数据概览面板
st.sidebar.divider()
st.sidebar.markdown("#### 数据概览")
if df is not None:
    all_model_names = df["Display_Name"].unique()
    st.sidebar.metric("追踪模型数", len(all_model_names))
    st.sidebar.caption(
        f"数据区间: {df['Date'].min().strftime('%Y-%m-%d')} ~ {df['Date'].max().strftime('%Y-%m-%d')}"
    )
if df_price is not None:
    st.sidebar.metric("定价模型数", df_price["Model"].nunique())
    st.sidebar.caption(f"定价更新至: {df_price['Date'].max().strftime('%Y-%m-%d')}")
if df_bench is not None:
    bench_models = [c for c in df_bench.columns if c not in ["Date", "Metric"]]
    st.sidebar.metric("跑分模型数", len(bench_models))
    st.sidebar.caption(f"跑分更新至: {df_bench['Date'].max().strftime('%Y-%m-%d')}")
if df_lmarena is not None:
    st.sidebar.metric("LMARENA 模型数", df_lmarena["Model"].nunique())
    st.sidebar.caption(f"LMARENA 更新至: {df_lmarena['Date'].max().strftime('%Y-%m-%d')}")

# ── 页面路由 ──
if page == "累计用量对比":
    from views.cumulative_compare import render
    render(df)

elif page == "单模型用量":
    from views.single_model import render
    render(df)

elif page == "数据导出":
    from views.data_export import render
    render(df)

# ── 隐藏页面（保留逻辑，按需恢复即可） ──
# 如果通过修改 config.PAGE_VISIBILITY 恢复了某个页面，在此添加 elif 分支即可。
# 原始完整逻辑已保存在 app_backup.py 中。