"""
config.py — 全局配置中心
包含：文件路径、Feature Flag、告警阈值、AI 提供商配置、页面注册表。
所有可调参数集中于此，页面与组件仅读取，不硬编码。
"""

import os
import streamlit as st

# ── 数据文件路径 ──────────────────────────────────────────
DATA_FILE = "history_database.csv"
PRICING_FILE = "openrouter_pricing_provider_records.csv"
BENCHMARK_FILE = "openrouter_benchmark_records.csv"
LMARENA_FILE = "lmarena_leaderboard_records.csv"

# ── Feature Flag：页面可见性 ────────────────────────────
# True = 在侧边栏导航中展示；False = 隐藏但保留代码逻辑
PAGE_VISIBILITY = {
    "累计用量对比": True,
    "单模型用量":   True,
    "数据导出":     True,
    # 以下页面隐藏，需要时改为 True 即可恢复
    "AI 查询":       False,
    "每日简报":      False,
    "T+N 日用量对比": False,
    "供应商定价":     False,
    "基准测试":       False,
    "单模型深度分析": False,
}

# ── 累计用量对比页：默认选中模型 ────────────────────────
DEFAULT_SELECTED_MODELS: list[str] = []  # 留空 = 默认选前 3 个

# ── 新模型告警阈值 ─────────────────────────────────────
# 面板 1: 新模型上量过快监测
NEW_MODEL_LOOKBACK_DAYS = 30       # 「新模型」的定义窗口（最近 N 天内首次出现）
RAMP_OBSERVATION_WINDOW = 7        # 早期观察窗口天数（前 n 天）
RAMP_PERCENTILE_THRESHOLD = 95     # 历史基准百分位（P95）

# 面板 2: 新模型斜率骤降提醒
SLOPE_DROP_THRESHOLD = 0.30        # 斜率降幅 ≥ 30% 触发告警
SLOPE_MIN_DAYS = 2                 # 模型至少需要 2 天数据才能计算斜率变化

# ── AI 密钥读取 ────────────────────────────────────────
def get_api_key(key_name: str) -> str:
    try:
        if key_name in st.secrets:
            return st.secrets[key_name]
    except Exception:
        pass
    return os.environ.get(key_name, "")

OPENROUTER_KEY = get_api_key("OPENROUTER_API_KEY")
GOOGLE_KEY = get_api_key("GOOGLE_API_KEY")
MODELSCOPE_KEY = get_api_key("MODELSCOPE_API_KEY")

# ── AI 提供商配置 ──────────────────────────────────────
AI_PROVIDERS = {
    "OpenRouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "key": OPENROUTER_KEY,
        "models": {
            "GLM-4.5-Air (免费)": "z-ai/glm-4.5-air:free",
            "Gemini 3 Flash (OpenRouter版)": "google/gemini-3-flash-preview",
            "Claude Haiku 4.5": "anthropic/claude-haiku-4.5",
        },
    },
    "Google AI Studio": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "key": GOOGLE_KEY,
        "models": {
            "Gemini 2.5 Flash": "gemini-2.5-flash",
            "Gemini 2.5 Flash Lite": "gemini-2.5-flash-lite",
            "Gemini 3.0 flash preview": "gemini-3-flash-preview",
        },
    },
    "魔塔社区 (ModelScope)": {
        "base_url": "https://api-inference.modelscope.cn/v1",
        "key": MODELSCOPE_KEY,
        "models": {
            "GLM-5": "ZhipuAI/GLM-5",
            "Minimax-M2.5": "MiniMax/MiniMax-M2.5",
            "Kimi-K2.5": "moonshotai/Kimi-K2.5",
            "Qwen-3.5": "Qwen/Qwen3.5-397B-A17B",
        },
    },
}
