"""
utils/helpers.py — 与业务无关的纯函数工具集
包含：模型名标准化、模糊匹配、深度推理判断、Excel/CSV 导出等。
"""

import re
import io
import pandas as pd


# ── 深度推理模型判断 ────────────────────────────────────
def is_reasoning_model(model_name: str) -> bool:
    """基于模型命名规则粗略判断是否为深度推理模型"""
    if not isinstance(model_name, str):
        return False
    name_lower = model_name.lower()
    reasoning_keywords = ["reasoning", "o1", "o3", "r1", "qwq"]
    return any(kw in name_lower for kw in reasoning_keywords)


# ── Token 化与相似度 ───────────────────────────────────
def _tokenize_model_name(name: str) -> set:
    """将模型名拆为 token 集合，用于模糊匹配"""
    n = name.lower()
    if "/" in n:
        n = n.split("/")[-1]
    n = re.sub(r"\s*\(.*?\)", "", n)
    tokens = set(re.split(r"[\s\-_.]+", n.strip()))
    tokens.discard("")
    return tokens


def _jaccard_similarity(set_a: set, set_b: set) -> float:
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def normalize_model_name(name: str) -> str:
    """统一消除厂商前缀，使不同数据源中的同款模型能合并"""
    if not isinstance(name, str):
        return str(name)
    n = name.lower()
    if "/" in n:
        n = n.split("/")[-1]
    mapping = {
        "deepseek v3": "deepseek-chat",
        "deepseek-v3": "deepseek-chat",
    }
    for key, val in mapping.items():
        if key in n:
            return val
    n = re.sub(r"\s*\(.*?\)", "", n).strip()
    n = n.replace(" ", "-").replace(".", "-")
    return n


def fuzzy_match_model(target_norm: str, candidate_names: list, threshold: float = 0.55) -> list:
    """在候选模型名列表中，用 Token 化 Jaccard 匹配找出与 target_norm 相似的名字"""
    target_tokens = _tokenize_model_name(target_norm)
    matched_with_scores = []
    for cand in candidate_names:
        cand_tokens = _tokenize_model_name(cand)
        sim = _jaccard_similarity(target_tokens, cand_tokens)
        if sim >= threshold:
            matched_with_scores.append((cand, sim))
    matched_with_scores.sort(key=lambda x: x[1], reverse=True)
    return [m[0] for m in matched_with_scores]


# ── Excel / CSV 下载数据构建 ───────────────────────────
def get_dataset_download(df: pd.DataFrame, filename_prefix: str):
    """
    返回 (data_bytes, file_name, mime_type, button_label)，
    优先生成 xlsx，openpyxl 不可用时降级为 CSV。
    """
    try:
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="Sheet1")
        data = output.getvalue()
        file_name = f"{filename_prefix}.xlsx"
        mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        label = "📥 下载 Excel 文件 (.xlsx)"
    except ImportError:
        data = df.to_csv(index=False).encode("utf-8-sig")
        file_name = f"{filename_prefix}.csv"
        mime = "text/csv"
        label = "📥 下载 CSV 文件 (Excel兼容)"
    return data, file_name, mime, label
