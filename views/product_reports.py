"""
views/product_reports.py — AI 产品测评页面
从本地 investment-report 目录发现可用产品、嵌入 HTML 报告，并展示 JSON 样本概览。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from config import PRODUCT_REPORT_ROOT


@dataclass(frozen=True)
class ProductReport:
    vendor: str
    product: str
    folder: Path
    reports: tuple[Path, ...]
    datasets: tuple[Path, ...]


def _version_number(path: Path) -> int:
    match = re.search(r"_v(\d+)", path.stem)
    return int(match.group(1)) if match else -1


def _ranked_files(files: list[Path]) -> tuple[Path, ...]:
    return tuple(
        sorted(
            files,
            key=lambda item: (_version_number(item), item.stat().st_mtime),
            reverse=True,
        )
    )


def _product_label(folder: Path) -> tuple[str, str]:
    name = folder.name.lower()
    if "hailuo" in name:
        return "MiniMax", "海螺 Hailuo"
    if "minimax" in name and "m2" in name:
        return "MiniMax", "MiniMax M 系列"
    return folder.name.replace("-", " ").title(), folder.name


def _candidate_json(path: Path) -> bool:
    stem = path.stem.lower()
    if "chunk" in stem or "irrelevant" in stem or "raw" in stem:
        return False
    return "final_dataset" in stem or stem.endswith("_merged") or "translated" in stem


@st.cache_data(ttl=600, show_spinner=False)
def discover_reports(root: str) -> list[ProductReport]:
    root_path = Path(root).expanduser()
    if not root_path.exists():
        return []

    products: list[ProductReport] = []
    for folder in sorted([item for item in root_path.iterdir() if item.is_dir()]):
        if folder.name.startswith("_"):
            continue

        reports = _ranked_files(
            [
                path
                for path in folder.rglob("*.html")
                if "template" not in path.stem.lower()
            ]
        )[:1]
        datasets = _ranked_files([path for path in folder.rglob("*.json") if _candidate_json(path)])[:1]

        if not reports and not datasets:
            continue

        vendor, product = _product_label(folder)
        products.append(
            ProductReport(
                vendor=vendor,
                product=product,
                folder=folder,
                reports=reports,
                datasets=datasets,
            )
        )
    return products


@st.cache_data(ttl=600, show_spinner=False)
def read_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8", errors="ignore")


@st.cache_data(ttl=600, show_spinner=False)
def load_dataset(path: str) -> pd.DataFrame:
    with Path(path).open("r", encoding="utf-8") as file:
        payload: Any = json.load(file)

    records = payload if isinstance(payload, list) else payload.get("data", [])
    df = pd.DataFrame(records)
    if df.empty:
        return df

    if "t" in df.columns:
        df["t"] = pd.to_datetime(df["t"], errors="coerce")
    return df


def _summary_metrics(df: pd.DataFrame) -> None:
    total = len(df)
    source_count = df["source"].nunique() if "source" in df.columns else 0
    latest_date = "未知"
    if "t" in df.columns and df["t"].notna().any():
        latest_date = df["t"].max().strftime("%Y-%m-%d")

    c1, c2, c3 = st.columns(3)
    c1.metric("样本数", f"{total:,}")
    c2.metric("来源数", source_count if source_count else "未知")
    c3.metric("最近样本", latest_date)


def _filter_df(df: pd.DataFrame) -> pd.DataFrame:
    filtered = df.copy()
    filter_cols = [
        ("sentiment", "情绪"),
        ("region_bucket", "区域"),
        ("source", "来源"),
        ("narrative_bucket", "叙事主题"),
        ("user_group_4", "用户类型"),
    ]

    active_cols = [(col, label) for col, label in filter_cols if col in filtered.columns]
    if not active_cols:
        return filtered

    cols = st.columns(min(3, len(active_cols)))
    for index, (col, label) in enumerate(active_cols):
        options = sorted([item for item in filtered[col].dropna().unique().tolist() if item != ""])
        selected = cols[index % len(cols)].multiselect(label, options, default=options)
        if selected:
            filtered = filtered[filtered[col].isin(selected)]
    return filtered


def _value_chart(df: pd.DataFrame, column: str, title: str) -> None:
    if column not in df.columns:
        return

    counts = (
        df[column]
        .explode()
        .dropna()
        .astype(str)
        .loc[lambda series: series.str.len() > 0]
        .value_counts()
        .head(12)
    )
    if counts.empty:
        return

    st.markdown(f"#### {title}")
    st.bar_chart(counts)


def _render_dataset(df: pd.DataFrame) -> None:
    if df.empty:
        st.info("这个数据文件暂时没有可展示记录。")
        return

    _summary_metrics(df)
    filtered = _filter_df(df)

    chart_cols = st.columns(2)
    with chart_cols[0]:
        _value_chart(filtered, "sentiment", "情绪分布")
        _value_chart(filtered, "scene_tags", "场景标签")
    with chart_cols[1]:
        _value_chart(filtered, "narrative_bucket", "叙事主题")
        _value_chart(filtered, "competitor_tags", "竞品提及")

    st.markdown("#### 样本明细")
    preferred_cols = [
        "t",
        "handle",
        "sentiment",
        "region_bucket",
        "narrative_bucket",
        "scenario_tags",
        "scene_tags",
        "praise_tags",
        "complaint_tags",
        "competitor_tags",
        "translated",
        "u",
    ]
    visible_cols = [col for col in preferred_cols if col in filtered.columns]
    if not visible_cols:
        visible_cols = filtered.columns.tolist()

    st.dataframe(filtered[visible_cols], width="stretch", hide_index=True)
    st.download_button(
        "下载当前筛选结果 CSV",
        data=filtered[visible_cols].to_csv(index=False).encode("utf-8-sig"),
        file_name="ai_product_report_samples.csv",
        mime="text/csv",
    )


def render() -> None:
    st.subheader("AI 产品测评")
    st.caption("选择厂商和产品后，可以查看现成 HTML 报告，也可以下钻到测评样本数据。")

    root = Path(PRODUCT_REPORT_ROOT).expanduser()
    products = discover_reports(str(root))
    if not products:
        st.error(f"未找到产品测评报告目录：`{root}`")
        st.info("可以设置环境变量 `PRODUCT_REPORT_ROOT` 指向 investment-report 目录。")
        return

    vendor_options = sorted({item.vendor for item in products})
    selected_vendor = st.selectbox("选择厂商", vendor_options)
    vendor_products = [item for item in products if item.vendor == selected_vendor]

    product_labels = {item.product: item for item in vendor_products}
    selected_product = st.selectbox("选择产品", list(product_labels.keys()))
    product = product_labels[selected_product]

    report_path = None
    dataset_path = None
    selector_cols = st.columns(2)

    with selector_cols[0]:
        if product.reports:
            report_options = {path.name: path for path in product.reports}
            report_path = report_options[st.selectbox("选择 HTML 报告", list(report_options.keys()))]
        else:
            st.info("该产品暂无 HTML 报告。")

    with selector_cols[1]:
        if product.datasets:
            dataset_options = {path.name: path for path in product.datasets}
            dataset_path = dataset_options[st.selectbox("选择数据文件", list(dataset_options.keys()))]
        else:
            st.info("该产品暂无 JSON 样本数据。")

    report_tab, data_tab = st.tabs(["HTML 报告", "数据概览"])

    with report_tab:
        if report_path is None:
            st.info("请选择一个 HTML 报告。")
        else:
            html = read_text(str(report_path))
            st.download_button(
                "下载 HTML 报告",
                data=html.encode("utf-8"),
                file_name=report_path.name,
                mime="text/html",
            )
            components.html(html, height=920, scrolling=True)

    with data_tab:
        if dataset_path is None:
            st.info("请选择一个 JSON 数据文件。")
        else:
            df = load_dataset(str(dataset_path))
            _render_dataset(df)
