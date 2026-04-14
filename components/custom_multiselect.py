"""
components/custom_multiselect.py — 基于 st.popover 的高级多选控件

特点:
  - 折叠态只显示 "已选 X / 总 Y 个模型"
  - 展开后有搜索框
  - 支持多选，勾选后弹窗不自动关闭
  - 底部 "全选" / "清空" / "恢复默认" 按钮
"""

import streamlit as st


def model_multiselect(
    all_models: list[str],
    default_models: list[str] | None = None,
    key_prefix: str = "ms",
) -> list[str]:
    """
    渲染一个基于 st.popover 的模型多选控件。

    Parameters
    ----------
    all_models : 所有可选的模型名列表
    default_models : 默认选中的模型列表 (None = 前 3 个)
    key_prefix : session_state key 前缀，避免多处调用冲突

    Returns
    -------
    选中的模型名列表
    """
    if not all_models:
        st.info("暂无可选模型。")
        return []

    all_models = sorted(all_models)
    if default_models is None:
        default_models = all_models[:3] if len(all_models) >= 3 else all_models[:]

    # ── 初始化 session_state ──
    state_key = f"{key_prefix}_selected"
    if state_key not in st.session_state:
        st.session_state[state_key] = list(default_models)

    selected: list = st.session_state[state_key]
    total = len(all_models)
    count = len(selected)

    # ── Popover 控件 ──
    with st.popover(f"🔽 已选 {count} / {total} 个模型", width="stretch"):
        # 搜索框
        search = st.text_input("🔍 搜索模型", value="", key=f"{key_prefix}_search", placeholder="输入关键词过滤…")

        # 批量操作
        col_a, col_b, col_c = st.columns(3)
        if col_a.button("✅ 全选", key=f"{key_prefix}_all", width="stretch"):
            st.session_state[state_key] = list(all_models)
            st.rerun()
        if col_b.button("🗑️ 清空", key=f"{key_prefix}_clear", width="stretch"):
            st.session_state[state_key] = []
            st.rerun()
        if col_c.button("↩️ 恢复默认", key=f"{key_prefix}_reset", width="stretch"):
            st.session_state[state_key] = list(default_models)
            st.rerun()

        st.divider()

        # 过滤
        filtered = [m for m in all_models if search.lower() in m.lower()] if search else all_models

        # 分页：每次最多展示 50 个避免渲染过重
        PAGE_SIZE = 50
        page_key = f"{key_prefix}_page"
        if page_key not in st.session_state:
            st.session_state[page_key] = 0

        total_filtered = len(filtered)
        max_page = max(0, (total_filtered - 1) // PAGE_SIZE)
        current_page = min(st.session_state[page_key], max_page)

        start = current_page * PAGE_SIZE
        end = min(start + PAGE_SIZE, total_filtered)
        page_items = filtered[start:end]

        if total_filtered > PAGE_SIZE:
            st.caption(f"显示 {start+1}-{end} / {total_filtered}    (共 {max_page+1} 页)")
            pcol1, pcol2 = st.columns(2)
            if pcol1.button("⬅️ 上一页", key=f"{key_prefix}_prev", disabled=(current_page == 0)):
                st.session_state[page_key] = current_page - 1
                st.rerun()
            if pcol2.button("➡️ 下一页", key=f"{key_prefix}_next", disabled=(current_page >= max_page)):
                st.session_state[page_key] = current_page + 1
                st.rerun()

        # 勾选列表
        for model in page_items:
            checked = model in selected
            new_val = st.checkbox(model, value=checked, key=f"{key_prefix}_cb_{model}")
            if new_val and model not in selected:
                selected.append(model)
            elif not new_val and model in selected:
                selected.remove(model)

        # 写回
        st.session_state[state_key] = selected

    return list(st.session_state[state_key])
