"""
utils/data_services.py — 数据加载与缓存层
所有 CSV 文件的读取集中在此，返回干净的 DataFrame。
页面层不直接接触文件 I/O。
"""

import os
import pandas as pd
import streamlit as st
from config import DATA_FILE, PRICING_FILE, BENCHMARK_FILE, LMARENA_FILE


@st.cache_data(ttl=600)
def load_data():
    """加载 Token 消耗主表，返回 (df, error_msg)"""
    if not os.path.exists(DATA_FILE):
        return None, f"❌ 找不到文件 `{DATA_FILE}`，请等待爬虫运行。"
    try:
        df = pd.read_csv(DATA_FILE)
        if df.empty:
            return None, "CSV 文件为空"
        df["Date"] = pd.to_datetime(df["Date"])
        df["Display_Name"] = df["Model"].apply(lambda x: x.split("/")[-1] if "/" in x else x)
        return df, None
    except Exception as e:
        return None, str(e)


@st.cache_data(ttl=600)
def load_pricing_data():
    """加载 OpenRouter 定价数据"""
    if not os.path.exists(PRICING_FILE):
        return None
    try:
        df_price = pd.read_csv(PRICING_FILE)
        df_price["Date"] = pd.to_datetime(df_price["Date"])
        return df_price
    except Exception:
        return None


@st.cache_data(ttl=600)
def load_benchmark_data():
    """加载 Artificial Analysis Benchmark 跑分"""
    if not os.path.exists(BENCHMARK_FILE):
        return None
    try:
        df_bench = pd.read_csv(BENCHMARK_FILE)
        df_bench["Date"] = pd.to_datetime(df_bench["Date"])
        return df_bench
    except Exception:
        return None


@st.cache_data(ttl=600)
def load_lmarena_data():
    """加载 Arena 竞技排行数据"""
    if not os.path.exists(LMARENA_FILE):
        return None
    try:
        df = pd.read_csv(LMARENA_FILE)
        df["Date"] = pd.to_datetime(df["Date"])
        df["Score_text"] = pd.to_numeric(df["Score_text"], errors="coerce")
        df["Score_text"] = df["Score_text"].fillna(1000)
        return df
    except Exception:
        return None


def load_all():
    """一次性加载所有数据源，返回字典"""
    df, error = load_data()
    df_price = load_pricing_data()
    df_bench = load_benchmark_data()
    df_lmarena = load_lmarena_data()
    return {
        "df": df,
        "error": error,
        "df_price": df_price,
        "df_bench": df_bench,
        "df_lmarena": df_lmarena,
    }
