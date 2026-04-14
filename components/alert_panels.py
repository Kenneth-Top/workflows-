"""
components/alert_panels.py — 预警面板 UI 渲染组件

接收 analytics.py 的计算结果字典，渲染为 Streamlit 卡片。
本模块仅负责展示，不包含计算逻辑。
"""

import streamlit as st
import pandas as pd


def render_fast_ramp_panel(result: dict):
    """
    渲染「新模型上量过快监测」面板。

    Parameters
    ----------
    result : detect_fast_ramp() 的返回字典
    """
    st.markdown("### 🚀 新模型上量过快监测")
    st.caption(
        f"规则：上线不超过 30 天的新模型，前 {result['window']} 天日均累计斜率超过全库 P{result['percentile']}（比 {result['percentile']}% 的模型都快）即触发。"
    )

    if result["threshold"] is not None:
        st.info(f"📊 全库基准（P{result['percentile']}）：{result['threshold']:.6f} B/天")

    triggered = result["triggered"]
    if not triggered:
        st.success("✅ 无上量过快的新模型")
        return

    rows = []
    for item in triggered:
        rows.append({
            "模型": item["model"],
            "上线日期": item["online_date"],
            "观察天数": item["observation_days"],
            "当前斜率 (B/天)": item["current_slope"],
            "全库阈值 (B/天)": item["threshold"],
            "超出幅度": f"+{item['exceed_pct']}%",
        })

    df_display = pd.DataFrame(rows)
    st.dataframe(
        df_display.style.format({
            "当前斜率 (B/天)": "{:.6f}",
            "全库阈值 (B/天)": "{:.6f}",
        }),
        width="stretch",
        hide_index=True,
    )


def render_slope_drop_panel(result: dict):
    """
    渲染「新模型斜率骤降提醒」面板。

    Parameters
    ----------
    result : detect_slope_drop() 的返回字典
    """
    st.markdown("### 📉 新模型斜率骤降提醒")
    st.caption(
        f"规则：最近 30 天内新模型，最新一天累计斜率相比前一天降幅 ≥ {result['threshold']*100:.0f}% 即触发。"
    )

    triggered = result["triggered"]
    if not triggered:
        st.success("✅ 近期无上量斜率骤降的新模型")
        return

    rows = []
    for item in triggered:
        rows.append({
            "模型": item["model"],
            "上线日期": item["online_date"],
            "前一天斜率 (B/天)": item["slope_prev"],
            "最新斜率 (B/天)": item["slope_latest"],
            "降幅": f"{item['drop_pct']}%",
            "触发阈值": f"-{item['threshold']*100:.0f}%",
        })

    df_display = pd.DataFrame(rows)
    st.dataframe(
        df_display.style.format({
            "前一天斜率 (B/天)": "{:.6f}",
            "最新斜率 (B/天)": "{:.6f}",
        }),
        width="stretch",
        hide_index=True,
    )
