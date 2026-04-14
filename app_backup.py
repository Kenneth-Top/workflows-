import streamlit as st
import pandas as pd
import altair as alt
import os
import io
import json

# === 1. 基础配置 ===
st.set_page_config(page_title="LLM 数据看板", layout="wide")
DATA_FILE = "history_database.csv"
PRICING_FILE = "openrouter_pricing_provider_records.csv"
BENCHMARK_FILE = "openrouter_benchmark_records.csv"
LMARENA_FILE = "lmarena_leaderboard_records.csv"

# AI 秘钥配置 (优先从 Streamlit Secrets 安全读取，无配置文件则退回环境变量)
def get_api_key(key_name):
    try:
        if key_name in st.secrets:
            return st.secrets[key_name]
    except Exception:
        pass
    return os.environ.get(key_name, "")

OPENROUTER_KEY = get_api_key("OPENROUTER_API_KEY")
GOOGLE_KEY = get_api_key("GOOGLE_API_KEY")
MODELSCOPE_KEY = get_api_key("MODELSCOPE_API_KEY")

# 定义 AI 提供商配置
AI_PROVIDERS = {
    "OpenRouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "key": OPENROUTER_KEY,
        "models": {
            "GLM-4.5-Air (免费)": "z-ai/glm-4.5-air:free",
            "Gemini 3 Flash (OpenRouter版)": "google/gemini-3-flash-preview",
            "Claude Haiku 4.5": "anthropic/claude-haiku-4.5",
        }
    },
    "Google AI Studio": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "key": GOOGLE_KEY,
        "models": {
            "Gemini 2.5 Flash": "gemini-2.5-flash",
            "Gemini 2.5 Flash Lite":"gemini-2.5-flash-lite",
            "Gemini 3.0 flash preview": "gemini-3-flash-preview",
        }
    },
    "魔塔社区 (ModelScope)": {
        "base_url": "https://api-inference.modelscope.cn/v1",
        "key": MODELSCOPE_KEY,
        "models": {
            "GLM-5": "ZhipuAI/GLM-5",
            "Minimax-M2.5": "MiniMax/MiniMax-M2.5",
            "Kimi-K2.5": "moonshotai/Kimi-K2.5",
            "Qwen-3.5": "Qwen/Qwen3.5-397B-A17B",
        }
    }
}

# 页面标题
st.title("LLM 数据看板")

# 定义页面名称常量
NAV_AI_QUERY = "AI 查询"
NAV_DAILY_BRIEF = "每日简报"
NAV_TN_DAILY = "T+N 日用量对比"
NAV_CUMULATIVE_COMPARE = "累计用量对比"
NAV_DETAIL_DAILY = "单模型用量"
NAV_RAW_DATA = "数据导出"
NAV_PRICING = "供应商定价"
NAV_BENCHMARK = "基准测试"
NAV_SINGLE_MODEL = "单模型深度分析"

# === 2. 工具函数 ===

def is_reasoning_model(model_name: str) -> bool:
    """基于模型命名规则进行粗略判断是否为深度推理模型"""
    if not isinstance(model_name, str): return False
    name_lower = model_name.lower()
    reasoning_keywords = ['reasoning', 'o1', 'o3', 'r1', 'qwq']
    for kw in reasoning_keywords:
        if kw in name_lower:
            return True
    return False

import re as _re_global

def _tokenize_model_name(name: str) -> set:
    """将模型名拆为 token 集合，用于模糊匹配"""
    n = name.lower()
    # 去掉厂商前缀
    if '/' in n:
        n = n.split('/')[-1]
    # 去掉括号内修饰词，如 (Reasoning), (Oct '24), (Non-reasoning)
    n = _re_global.sub(r'\s*\(.*?\)', '', n)
    # 按 空格、横线、下划线、点号 分割
    tokens = set(_re_global.split(r'[\s\-_.]+', n.strip()))
    tokens.discard('')
    return tokens

def _jaccard_similarity(set_a: set, set_b: set) -> float:
    """计算两个集合的 Jaccard 相似度"""
    if not set_a or not set_b:
        return 0.0
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union)

def normalize_model_name(name: str) -> str:
    """统一消除厂商前缀和无用的大小写，使不同数据源中的同款模型能合并"""
    if not isinstance(name, str): return str(name)
    n = name.lower()
    # 移除诸如 'anthropic/', 'google/' 等前缀
    if '/' in n:
        n = n.split('/')[-1]
    
    # 仅保留极端特例的硬映射（完全不同命名的情况）
    mapping = {
        'deepseek v3': 'deepseek-chat',
        'deepseek-v3': 'deepseek-chat',
    }
    
    for key, val in mapping.items():
        if key in n:
            return val
            
    # 去除多余括号如 (Reasoning) 等干扰词，保留核心 slug
    n = _re_global.sub(r'\s*\(.*?\)', '', n).strip()
    # 统一点号与空格为横线，确保版本号对齐（如 4.6 -> 4-6）
    n = n.replace(' ', '-').replace('.', '-')
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
    
    # 按得分从高到低排序，确保 matched[0] 是最像的
    matched_with_scores.sort(key=lambda x: x[1], reverse=True)
    return [m[0] for m in matched_with_scores]

@st.cache_data(ttl=600)
def load_data():
    if not os.path.exists(DATA_FILE):
        return None, f"❌ 找不到文件 `{DATA_FILE}`，请等待爬虫运行。"
    try:
        # Load Token Data
        df = pd.read_csv(DATA_FILE)
        if df.empty: return None, "CSV 文件为空"
        df['Date'] = pd.to_datetime(df['Date'])
        
        # 名称清洗：去掉 '/' 前面的厂商名
        df['Display_Name'] = df['Model'].apply(lambda x: x.split('/')[-1] if '/' in x else x)
        
        return df, None
    except Exception as e:
        return None, str(e)

@st.cache_data(ttl=600)
def load_pricing_data():
    if not os.path.exists(PRICING_FILE):
        return None
    try:
        df_price = pd.read_csv(PRICING_FILE)
        df_price['Date'] = pd.to_datetime(df_price['Date'])
        return df_price
    except Exception:
        return None

@st.cache_data(ttl=600)
def load_benchmark_data():
    if not os.path.exists(BENCHMARK_FILE):
        return None
    try:
        df_bench = pd.read_csv(BENCHMARK_FILE)
        df_bench['Date'] = pd.to_datetime(df_bench['Date'])
        return df_bench
    except Exception:
        return None

@st.cache_data(ttl=600)
def load_lmarena_data():
    if not os.path.exists(LMARENA_FILE): return None
    try:
        df = pd.read_csv(LMARENA_FILE)
        df['Date'] = pd.to_datetime(df['Date'])
        # 为没有绝对分数的模型预填充 1000 以确保后续条形排序和展示高度
        df['Score_text'] = pd.to_numeric(df['Score_text'], errors='coerce')
        df['Score_text'] = df['Score_text'].fillna(1000)
        return df
    except Exception:
        return None

# Excel/CSV 智能导出函数
def get_dataset_download(df, filename_prefix):
    try:
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Sheet1')
        data = output.getvalue()
        file_name = f"{filename_prefix}.xlsx"
        mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        label = "📥 下载 Excel 文件 (.xlsx)"
    except ImportError:
        data = df.to_csv(index=False).encode('utf-8-sig')
        file_name = f"{filename_prefix}.csv"
        mime = "text/csv"
        label = "📥 下载 CSV 文件 (Excel兼容)"
    
    return data, file_name, mime, label

df, error = load_data()
df_price = load_pricing_data()
df_bench = load_benchmark_data()
df_lmarena = load_lmarena_data()

if error and not (df_price is not None or df_bench is not None):
    st.error(error)
    st.stop()

# === 3. 侧边栏导航 ===
st.sidebar.title("导航")
page = st.sidebar.radio("选择分析视图", [
    NAV_AI_QUERY,
    NAV_DAILY_BRIEF,
    NAV_SINGLE_MODEL,
    NAV_TN_DAILY,
    NAV_CUMULATIVE_COMPARE,
    NAV_DETAIL_DAILY,
    NAV_PRICING,
    NAV_BENCHMARK,
    NAV_RAW_DATA
])

all_model_names = df['Display_Name'].unique() if df is not None else []
all_pricing_models = df_price['Model'].unique() if df_price is not None else []
all_benchmark_models = [c for c in df_bench.columns if c not in ['Date', 'Metric']] if df_bench is not None else []

# 数据概览面板
st.sidebar.divider()
st.sidebar.markdown("#### 数据概览")
if df is not None:
    st.sidebar.metric("追踪模型数", len(all_model_names))
    st.sidebar.caption(f"数据区间: {df['Date'].min().strftime('%Y-%m-%d')} ~ {df['Date'].max().strftime('%Y-%m-%d')}")
if df_price is not None:
    st.sidebar.metric("定价模型数", len(all_pricing_models))
    st.sidebar.caption(f"定价更新至: {df_price['Date'].max().strftime('%Y-%m-%d')}")
if df_bench is not None:
    st.sidebar.metric("跑分模型数", len(all_benchmark_models))
    st.sidebar.caption(f"跑分更新至: {df_bench['Date'].max().strftime('%Y-%m-%d')}")
if df_lmarena is not None:
    st.sidebar.metric("LMARENA 模型数", df_lmarena['Model'].nunique())
    st.sidebar.caption(f"LMARENA 更新至: {df_lmarena['Date'].max().strftime('%Y-%m-%d')}")

# ========================================================
# 页面 0: AI 智能查询
# ========================================================
if page == NAV_AI_QUERY:
    st.subheader("AI 数据分析助手")
    
    # 顶部控制区：提供商与模型选择
    col_p1, col_p2, col_p3 = st.columns([1, 1, 1])
    with col_p1:
        provider_name = st.selectbox("选择 AI 服务商:", list(AI_PROVIDERS.keys()), index=2) # 默认选 魔塔社区
        provider_cfg = AI_PROVIDERS[provider_name]
    with col_p2:
        mq_model_list = list(provider_cfg["models"].keys())
        mq_default_idx = 0
        if "GLM-5" in mq_model_list:
            mq_default_idx = mq_model_list.index("GLM-5")
        selected_model_label = st.selectbox("选择模型:", mq_model_list, index=mq_default_idx)
        AI_MODEL = provider_cfg["models"][selected_model_label]
    with col_p3:
        st.write("") # 占位
        st.write("")
        enable_web_search = st.toggle("🌐 启用联网搜索", value=False)

    api_key = provider_cfg["key"]
    if not api_key:
        api_key = st.text_input(f"请输入 {provider_name} API Key:", type="password")
    else:
        # 构建数据库上下文摘要
        @st.cache_data(ttl=600)
        def build_db_context(_df, _df_price, _df_bench, _df_lmarena):
            context_parts = []
            
            if _df is not None and not _df.empty:
                # 仅展示用量最高的前 50 个模型，避免上下文爆炸
                top_models = _df.groupby('Display_Name')['Total_Tokens'].sum().nlargest(50).index.tolist()
                display_names = sorted(top_models)
                context_parts.append(f"""### Token 消耗数据 (变量名: df)
- 列: Date, Model, Prompt, Completion, Reasoning, Total_Tokens, Display_Name
- 记录数: {len(_df)}, 日期范围: {_df['Date'].min().strftime('%Y-%m-%d')} ~ {_df['Date'].max().strftime('%Y-%m-%d')}
- Token 单位: Billion (10亿)
- **Top 50 常用模型列表**: {', '.join(display_names)} (如果你需要的模型不在列表中，可以直接尝试匹配其 Display_Name)""")

            if _df_price is not None and not _df_price.empty:
                price_models = sorted(_df_price['Model'].dropna().unique().tolist())
                context_parts.append(f"""### 定价数据 (变量名: df_price)
- 列: Date, Model, Input_Price_1M, Output_Price_1M
- 价格单位: $/1M Tokens
- **全部定价模型列表**: {', '.join(price_models)}""")

            if _df_bench is not None and not _df_bench.empty:
                context_parts.append(f"""### Benchmark 跑分 (变量名: df_bench)
- 结构: 宽表，每行是一个 Metric，每列是一个模型名
- Metric: {', '.join(_df_bench['Metric'].unique()[:8])}""")

            if _df_lmarena is not None and not _df_lmarena.empty:
                score_cols = [c for c in _df_lmarena.columns if c.startswith('Score_')]
                context_parts.append(f"""### Arena 竞技排行 (变量名: df_lmarena)
- 数据源: arena.ai (原 LMARENA)
- 8 个 ELO 排行榜: {', '.join(c.replace('Score_','') for c in score_cols)}
- 模型示例 (前20): {', '.join(_df_lmarena['Model'].dropna().unique().tolist()[:20])}""")
            
            return '\n\n'.join(context_parts)
        
        db_context = build_db_context(df, df_price, df_bench, df_lmarena)
        
        # 动态注入联网强力规则
        web_search_rules = ""
        if enable_web_search:
            web_search_rules = """
## 🌐 联网搜索规范（最高警戒）
你当前已启用联网功能。你**仅能使用网络信息来解释数据趋势背后的“外部原因”**（如：查阅某天模型用量暴增是否因为降价/某平台免费、发新版或突发新闻）。
**严禁**用网络上的公开数据来修改、替代或伪造本地数据库（df, df_price等）中的数值。代码绘制的图表和输出的具体 Token 数据，必须 **100% 严格来源于本地数据库**！
"""
        
        # 【极致强化】专业 TMT 投资分析与可视化指令
        SYSTEM_PROMPT = f"""你是负责数据可视化的 TMT 投资分析师。你的所有回答必须严格基于提供的本地变量。

### [ALERT] 绝对禁令（违者报错）
1. **禁止重新加载数据**：`df`, `df_price`, `df_bench`, `df_lmarena` 已经在内存中。**严禁**写任何 `pd.read_csv` 或 `open` 代码！
2. **禁止任何 XML 标签**：严禁输出 `<parameter>`, `<tool_call>`, `[tool_call]`, `<invoke>` 等标签。
3. **禁止解释代码**：不要在回复中用文字解释 Python 代码是如何写的，直接输出结论和代码块。
4. **禁止使用 display() 或 fig.show()**：当前是 Streamlit 运行环境，绝对禁止使用 `display(fig)` 或 `fig.show()`。你必须使用 `st.plotly_chart(fig, use_container_width=True)` 或 `st.dataframe()` 来渲染和展示图表与数据！

### [#] 数据库上下文
{db_context}

{web_search_rules}

### [#] 绘图与分析规范（全景多维矩阵）
1. **强制三维图表连发**：无论用户询问单模型还是多模型，你的 ```python 块必须**同时渲染 3 个维度的图表**：
   - 图表 1：Tokens 消耗热度趋势图（基于 `df`）。
   - 图表 2：API 定价历史走势折线图（基于 `df_price`）。如果是单模型，同时画 Input/Output 随着时间的曲线；如果是多模型，画 Input 价格随着时间的分别曲线。
   - 图表 3：权威底层基准跑分对比图（必须结合 `df_bench` 基准分数 和 `df_lmarena` 排名）。如果有具体测试分数（如 MMLU, MATH，从 df_bench 获取并融为长表绘制并列柱状图），就用分数；如果有 Arena Ranking（从 df_lmarena 获取），就用排名辅助标注。**拒绝空数据硬画图！提取真实的基准数据绘制交错对比！**
2. **多模型画图**：包含多个模型时，必须使用 Plotly 的 `color` 属性将它们重叠/并排渲染到同一张图中直观对比！
3. **数据预处理**：在对 `df`、`df_price` 等包含 `Date` 的表操作前，务必确保 `Date` 为 datetime 类型并进行排序。对宽表 `df_bench` 需要用 `pd.melt()` 处理成长表再画。
4. **输出格式**：
   - 第一部分：专业核心洞察结论。
   - 第二部分：**【核心硬性规定】这部分必须是一整块由 ```python 包裹的、可直接执行的代码区块！你绝对不能偷懒省略，必须完整输出包含 `import plotly` 的代码部分！**
   - 第三部分：趋势骤变与模型能力分析（必须结合联网搜索抓取的“价格战、大版本更新、新模型发布”等外部事实做归因）。

        - 🚨 **[代码格式要求]**：注意遵守 Python 基本语法缩进与换行，**切勿将多行代码（特别是 import 语句）合并在同一行，必须保持独立成行**。严禁遗漏最外层的 ```python 标记。

        **你的强制 Python 代码模板（请严格遵循并替换数据字段，图表顺序和形式不可更改）**：

```python
import plotly.express as px
# 引入基础库
import pandas as pd
# 引入前端组件
import streamlit as st

# 目标模型列表
targets = ['minimax-m2.5', 'qwen-max'] 

# 1. 用量趋势图
st.markdown("### 📊 全景扫描：用量热度趋势")
df['Date'] = pd.to_datetime(df['Date'])
plot_df = df[df['Display_Name'].isin(targets)].sort_values('Date')
if not plot_df.empty:
    st.plotly_chart(px.line(plot_df, x='Date', y='Total_Tokens', color='Display_Name', markers=True))

# 2. 定价走势图 (Price Trend)
st.markdown("### 💰 商业分析：API 定价趋势")
df_price['Date'] = pd.to_datetime(df_price['Date'])
price_df = df_price[df_price['Model'].str.contains('|'.join(targets), case=False, na=False)].sort_values('Date')
if not price_df.empty:
    if len(targets) == 1:
        st.dataframe(price_df.tail(1)[['Date', 'Model', 'Input_Price_1M', 'Output_Price_1M']], use_container_width=True)
        melted = pd.melt(price_df, id_vars=['Date', 'Model'], value_vars=['Input_Price_1M', 'Output_Price_1M'], var_name='Price_Type', value_name='Price ($/1M)')
        st.plotly_chart(px.line(melted, x='Date', y='Price ($/1M)', color='Price_Type', markers=True, title="单模型定价走势"))
    else:
        st.plotly_chart(px.line(price_df, x='Date', y='Input_Price_1M', color='Model', markers=True, title="多模型 Input 价格走势对比"))

# 3. 跑分水位与综合基准对比 (LMArena + Benchmarks)
st.markdown("### 🏆 技术底座：综合基准测试核心指标水位对比")
# (A) LMArena 全局表现评估
df_lmarena['Date'] = pd.to_datetime(df_lmarena['Date'])
latest_date = df_lmarena['Date'].max()
arena_df = df_lmarena[(df_lmarena['Date'] == latest_date) & (df_lmarena['Model'].str.contains('|'.join(targets), case=False, na=False))].copy()
if not arena_df.empty:
    st.dataframe(arena_df[['Model', 'Score_text', 'Rank_Overall', 'Rank_Coding', 'Rank_Hard_Prompts']], use_container_width=True)

# (B) 从权威机器测试集抽取精确指标 (df_bench 处理)
if df_bench is not None:
    df_bench['Date'] = pd.to_datetime(df_bench['Date'])
    b_latest = df_bench['Date'].max()
    # 提取三大核心权威指标：MMLU (常识/通用), MATH (数学推理), HumanEval (编程) 代码能力
    b_df = df_bench[(df_bench['Date'] == b_latest) & (df_bench['Metric'].isin(['MMLU', 'MATH', 'HumanEval']))].copy()
    if not b_df.empty:
        # 将宽图转换长表
        melted_b = b_df.melt(id_vars=['Date', 'Metric'], var_name='Model', value_name='Score')
        melted_b['Score'] = pd.to_numeric(melted_b['Score'], errors='coerce')
        # 筛选目标模型，并且清除没有分数的脏行
        melted_b = melted_b[melted_b['Model'].str.contains('|'.join(targets), case=False, na=False)].dropna(subset=['Score'])
        if not melted_b.empty:
            fig_bench = px.bar(melted_b, x='Metric', y='Score', color='Model', barmode='group', title="多维度基准独立能力横评得分对比")
            st.plotly_chart(fig_bench)
        else:
            st.info("⚠️ 该模型可能为闭源或小众，暂无官方公开核心基准分 (MMLU/MATH/HumanEval等)。")
    else:
        st.info("无底层维度数据")
```
"""

        # 初始化聊天历史
        if "ai_messages" not in st.session_state:
            st.session_state.ai_messages = []
        
        # 用于 exec 的命名空间
        import numpy as np
        try:
            import plotly.express as px
        except ImportError:
            px = None
            
        # 专给 AI 绘图使用去重的 df_price，优先提取 Weighted Average 供分析，避免不同 Cache Hit Rate 下的各类聚合混乱
        df_price_for_ai = df_price.copy() if df_price is not None else pd.DataFrame()
        if not df_price_for_ai.empty:
            def _get_daily_price(g):
                wa = g[g['Provider'] == 'Weighted Average']
                if not wa.empty:
                    return wa.iloc[0][['Input_Price_1M', 'Output_Price_1M']]
                return g[['Input_Price_1M', 'Output_Price_1M']].mean()
            df_price_for_ai = df_price_for_ai.groupby(['Date', 'Model']).apply(_get_daily_price).reset_index()
            
        exec_namespace = {
            "df": df, "df_price": df_price_for_ai, "df_bench": df_bench, "df_lmarena": df_lmarena,
            "st": st, "alt": alt, "pd": pd, "np": np, "os": os, "px": px,
        }
        
        # 极致增强版正则提取逻辑 (逻辑切换：先提取代码，避免被 XML 清洗误杀)
        def split_reply(reply):
            import re as _re
            
            # 1. 首先尝试提取所有代码块 (这是最宝贵的，必须先保护)
            # 支持 ```python, ```py, ```Python, ``` (没有语言标识), 以及不规范的空格
            blocks = _re.findall(r'```(?:python|py|Python)?\s*(.*?)\s*```', reply, _re.DOTALL | _re.IGNORECASE)
            combined_code = "\n".join(blocks).strip() if blocks else None
            
            # 2. 清洗回复文本（在此阶段可以清理 XML/Thinking）
            # 剔除代码块占位
            text_only = _re.sub(r'```(?:python|py|Python)?\s*.*?\s*```', '', reply, flags=_re.DOTALL | _re.IGNORECASE).strip()
            # 彻底剔除思考过程 (支持未严格闭合的情况)
            text_only = _re.sub(r'<think>.*?(</think>|$)', '', text_only, flags=_re.DOTALL | _re.IGNORECASE)
            # 贪婪匹配剔除所有形如 <tag>...</tag> 或 [tool_call] 的干扰
            text_only = _re.sub(r'<(?:parameter|tool_call|invoke|function|arg|name|call)[^>]*>.*?</(?:parameter|tool_call|invoke|function|arg|name|call)>', '', text_only, flags=_re.DOTALL)
            text_only = _re.sub(r'</?(?:parameter|tool_call|invoke|function|arg|name|call)[^>]*>', '', text_only) 
            text_only = _re.sub(r'\[tool_call\].*?$', '', text_only, flags=_re.MULTILINE) 
            # 深度二次清理：剔除任何残留的尖括号内容
            text_only = _re.sub(r'<[^>]+>', '', text_only).strip()
            
            return text_only, combined_code
        
        def safe_exec(code, ns):
            # 预处理数据类型
            for key in ['df', 'df_price', 'df_bench', 'df_lmarena']:
                frame = ns.get(key)
                if frame is not None and isinstance(frame, pd.DataFrame):
                    ns[key] = frame.copy()
            
            try:
                exec(code, ns)
            except Exception as e:
                st.error(f"代码执行失败: {str(e)}")
                with st.expander("查看错误详情"):
                    st.code(code)
                    import traceback
                    st.text(traceback.format_exc())
        
        # 显示历史对话
        for msg in st.session_state.ai_messages:
            with st.chat_message(msg["role"]):
                if msg["role"] == "assistant":
                    text_part, code = split_reply(msg["content"])
                    st.markdown(text_part)
                    if code:
                        try:
                            safe_exec(code, exec_namespace)
                        except Exception:
                            pass
                else:
                    st.markdown(msg["content"])
        
        # 用户输入
        user_query = st.chat_input("输入你的问题，例如: '分析 M2.5 本月的用量趋势，并查找它近期暴涨的原因'")
        
        if user_query:
            st.session_state.ai_messages.append({"role": "user", "content": user_query})
            with st.chat_message("user"):
                st.markdown(user_query)
            
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            for msg in st.session_state.ai_messages[-12:]:
                messages.append({"role": msg["role"], "content": msg["content"]})
            
            # 构建 API 请求 Payload (修改点 3)
            api_payload = {
                "model": AI_MODEL,
                "messages": messages,
                "max_tokens": 8000,
                "temperature": 0.2
            }
            if enable_web_search:
                if provider_name == "OpenRouter":
                    api_payload["plugins"] = [{"id": "web", "max_results": 4}]
                else:
                    import requests as _req
                    headers = {
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json"
                    }
                    
                    # 🌟 第 1 步：让 AI 作为“搜索专家”提炼关键词
                    with st.spinner("🧠 正在让 AI 提炼精准搜索关键词..."):
                        keyword_prompt = f"""
                        我需要你在搜索引擎上查阅最新资讯来辅助回答。
                        用户当前的原始问题是："{user_query}"
                        当前讨论的大模型是："{selected_model_label}"
                        
                        请你提取出 3-5 个最有利于在 DuckDuckGo 或 Google 上搜到近期新闻的关键词（以空格分隔）。
                        比如如果用户问“分析M2.5斜率放缓原因”，你应该输出：Minimax M2.5 降价 免费 动态。
                        
                        【最高指令】：你只需输出一行纯关键词，绝对不要输出任何其他解释、标点符号或前缀！
                        """
                        kw_payload = {
                            "model": AI_MODEL, 
                            "messages": [{"role": "user", "content": keyword_prompt}], 
                            "max_tokens": 150, 
                            "temperature": 0.1 
                        }
                        try:
                            kw_resp = _req.post(f"{provider_cfg['base_url']}/chat/completions", headers=headers, json=kw_payload, timeout=90)
                            kw_resp.raise_for_status()
                            raw_kw = kw_resp.json()['choices'][0]['message']['content']
                            import re as _re
                            clean_kw = _re.sub(r'<think>.*?</think>', '', raw_kw, flags=_re.DOTALL).strip()
                            search_query = clean_kw.replace('"', '').replace("'", "").replace("关键词：", "").replace("关键词:", "").split('\n')[0][:80] 
                            st.toast(f"🔑 AI 提取出搜索词: {search_query}")
                        except Exception as e:
                            search_query = f"{selected_model_label.split(' ')[0]} 大模型 近期动态" 
                            st.toast(f"⚠️ 关键词提取失败，使用备用词。")

                    # 🌟 第 2 步：鸭鸭拿着 AI 给的词去搜索 (修改点 1: max_results=15)
                    with st.spinner(f"🌐 鸭鸭正在搜索: '{search_query}'..."):
                        try:
                            import urllib.parse
                            import requests as _s_req
                            from bs4 import BeautifulSoup
                            
                            h={'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
                            r=_s_req.get('https://www.bing.com/search?q='+urllib.parse.quote(search_query), headers=h, timeout=8)
                            s=BeautifulSoup(r.text, 'html.parser')
                            search_results = [{'title':li.find('h2').text, 'body':li.find('p').text if li.find('p') else ''} for li in s.find_all('li', class_='b_algo')[:6] if li.find('h2')]
                            
                            if search_results:
                                context_str = f"【实时网络搜索参考资料 (搜索词: {search_query})】\n"
                                for r in search_results:
                                    context_str += f"- 标题: {r.get('title', '')}\n  摘要: {r.get('body', '')}\n"
                                
                                api_payload["messages"][-1]["content"] += f"\n\n请参考以下最新的网络搜索结果来辅助回答上述问题（如有帮助）：\n{context_str}\n\n【最高优先级指令】：无论你参考了什么外部资料，你的主要任务仍然是执行数据分析。如果你需要生成图表，请务必返回完全独立、无依赖报错的 Python st/alt 渲染代码，并使用 ```python ... ``` 包裹代码块！"
                                
                            else:
                                st.toast(f"⚠️ 搜索 '{search_query}' 未发现近期直接结果。")
                        except Exception as e:
                            st.toast(f"⚠️ 本地联网搜索受阻: {e}")
            
            with st.chat_message("assistant"):
                with st.spinner(f"AI ({provider_name}) 正在分析数据..." + (" (正在全网搜索线索 🌐)" if enable_web_search else "")):
                    try:
                        import requests as _req
                        headers = {
                            "Authorization": f"Bearer {api_key}",
                            "Content-Type": "application/json"
                        }
                        if provider_name == "OpenRouter":
                            headers.update({"HTTP-Referer": "http://localhost", "X-Title": "LLM-Dashboard"})
                        
                        resp = _req.post(
                            f"{provider_cfg['base_url']}/chat/completions",
                            headers=headers,
                            json=api_payload,
                            timeout=300 
                        )
                        if resp.status_code != 200:
                            raise Exception(f"API Error {resp.status_code}: {resp.text}")
                        result = resp.json()
                        ai_reply = result['choices'][0]['message']['content']
                    except Exception as e:
                        ai_reply = f"查询失败: {str(e)}"
                    
                    text_part, code = split_reply(ai_reply)
                    st.markdown(text_part)
                    if code:
                        with st.status("📊 正在执行数据可视化...", expanded=False) as status:
                            try:
                                safe_exec(code, exec_namespace)
                                status.update(label="✅ 可视化渲染完成", state="complete", expanded=False)
                            except Exception as code_e:
                                status.update(label="❌ 绘图执行失败", state="error", expanded=True)
                                st.error(f"代码运行报错: {str(code_e)}")
                                with st.expander("查看 AI 生成的代码"):
                                    st.code(code)
                
                st.session_state.ai_messages.append({
                    "role": "assistant", 
                    "content": ai_reply,
                })
        
        # 清空对话按钮
        if st.session_state.ai_messages:
            if st.button("清空对话历史"):
                st.session_state.ai_messages = []
                st.rerun()
                
# ========================================================
# 页面 1: T+N 横向对比 (每日消耗)
# ========================================================
elif page == NAV_TN_DAILY:
    st.subheader("模型增长曲线对比 (T+N 每日消耗)")
    st.info("横轴：上线天数 | 纵轴：当日 Token 消耗量")

    selected_names = st.multiselect(
        "选择要对比的模型:", 
        all_model_names, 
        default=all_model_names[:1] 
    )
    
    if selected_names:
        tn_data = []
        standard_ticks = [0, 1, 2, 3, 4, 5, 6, 7, 10, 14, 30, 60]
        final_tick_values = set(standard_ticks)
        
        max_days_global = 0

        for name in selected_names:
            m_df = df[df['Display_Name'] == name].sort_values('Date').copy()
            if m_df.empty: continue
            
            start_date = m_df.iloc[0]['Date']
            st.caption(f"📅 **{name}** 收录起始日: {start_date.strftime('%Y-%m-%d')}")
            
            if len(m_df) > 1: m_df = m_df.iloc[:-1]

            latest_date = m_df.iloc[-1]['Date']
            latest_day_diff = (latest_date - start_date).days
            final_tick_values.add(latest_day_diff)
            
            if latest_day_diff > max_days_global:
                max_days_global = latest_day_diff

            for _, row in m_df.iterrows():
                day_diff = (row['Date'] - start_date).days
                if day_diff in standard_ticks or day_diff == latest_day_diff:
                    tn_data.append({
                        'Model': name,
                        'Days_Since_Start': day_diff,
                        'Total_Tokens': row['Total_Tokens'],
                        'Label': f"T+{day_diff}" if day_diff != latest_day_diff else f"Latest (T+{day_diff})",
                        'Real_Date': row['Date'].strftime('%Y-%m-%d')
                    })
        
        if tn_data:
            df_tn = pd.DataFrame(tn_data)
            
            chart = alt.Chart(df_tn).mark_line(
                point=alt.OverlayMarkDef(size=100, filled=True, color="white", strokeWidth=2)
            ).encode(
                x=alt.X(
                    'Days_Since_Start', 
                    title='上线天数 (Days)',
                    axis=alt.Axis(values=list(final_tick_values), labelFontSize=20, titleFontSize=24, grid=True),
                    scale=alt.Scale(domain=[0, max_days_global + 1], clamp=True)
                ),
                y=alt.Y(
                    'Total_Tokens', 
                    title='Total Tokens (Billion)',
                    axis=alt.Axis(labelFontSize=20, titleFontSize=24)
                ),
                color=alt.Color('Model', 
                                scale=alt.Scale(scheme='tableau10'), 
                                legend=alt.Legend(title="模型名称", orient='bottom')),
                tooltip=['Model', 'Label', 'Total_Tokens', 'Real_Date']
            ).properties(height=500)
            
            st.altair_chart(chart, use_container_width=True)
            
            st.markdown("#### 📋 数据明细")
            df_pivot = df_tn.pivot_table(index='Model', columns='Days_Since_Start', values='Total_Tokens')
            df_pivot.columns = [f"T+{c}" for c in df_pivot.columns]
            st.dataframe(df_pivot.style.format("{:.4f} B"), use_container_width=True)
            
            data, name, mime, label = get_dataset_download(df_pivot.reset_index(), "tn_daily_comparison")
            st.download_button(label=label, data=data, file_name=name, mime=mime)

# ========================================================
# 页面 2: 多模型累计增长 (趋势对比)
# ========================================================
elif page == NAV_CUMULATIVE_COMPARE:
    st.subheader("多模型累计增长对比")
    
    selected_names = st.multiselect(
        "选择要对比的模型:", 
        all_model_names, 
        default=all_model_names[:3] if len(all_model_names) >=3 else all_model_names
    )

    if selected_names:
        plot_data = []
        max_day_plot = 0
        
        cols = st.columns(len(selected_names))
        for idx, name in enumerate(selected_names):
            m_df_temp = df[df['Display_Name'] == name].sort_values('Date')
            if not m_df_temp.empty:
                s_date = m_df_temp.iloc[0]['Date'].strftime('%Y-%m-%d')
                cols[idx].caption(f"📅 **{name}**: {s_date}")

        for name in selected_names:
            m_df = df[df['Display_Name'] == name].sort_values('Date').copy()
            m_df['Cum_Tokens'] = m_df['Total_Tokens'].cumsum()
            if len(m_df) > 1: m_df = m_df.iloc[:-1]
            if m_df.empty: continue

            start_date = m_df.iloc[0]['Date']
            current_max_day = (m_df.iloc[-1]['Date'] - start_date).days
            if current_max_day > max_day_plot:
                max_day_plot = current_max_day

            for _, row in m_df.iterrows():
                day_num = (row['Date'] - start_date).days
                plot_data.append({
                    'Model': name, 'Day': day_num,
                    'Date': row['Date'].strftime('%Y-%m-%d'),
                    'Cumulative_Tokens': row['Cum_Tokens']
                })

        if plot_data:
            df_plot = pd.DataFrame(plot_data)

            base = alt.Chart(df_plot).encode(
                x=alt.X('Day', title="上线天数 (Daily)", 
                        scale=alt.Scale(domain=[0, max_day_plot + 2], clamp=True),
                        axis=alt.Axis(labelFontSize=16, titleFontSize=18, grid=True)),
                y=alt.Y('Cumulative_Tokens', title='累计 Token (Billion)', 
                        axis=alt.Axis(labelFontSize=16, titleFontSize=18)),
                color=alt.Color('Model', 
                                title='模型名称', 
                                scale=alt.Scale(scheme='tableau10'),
                                legend=alt.Legend(orient='bottom')),
                tooltip=['Model', 'Day', 'Date', 'Cumulative_Tokens']
            )
            chart = (base.mark_line(strokeWidth=3) + base.mark_circle(size=60)).properties(height=600)
            
            st.altair_chart(chart, use_container_width=True)

            st.markdown("### 📅 累计数值明细")
            df_pivot = df_plot.pivot_table(index='Day', columns='Model', values='Cumulative_Tokens')
            st.dataframe(df_pivot.style.format("{:.4f} B"), use_container_width=True)

            data, name, mime, label = get_dataset_download(df_pivot.reset_index(), "cumulative_growth")
            st.download_button(label=label, data=data, file_name=name, mime=mime)

# ========================================================
# 页面 3: 单模型每日详情 (趋势分析 + 日期筛选)
# ========================================================
elif page == NAV_DETAIL_DAILY:
    st.subheader("单模型每日详情趋势")
    
    selected_name = st.selectbox("选择模型", all_model_names)
    m_df_full = df[df['Display_Name'] == selected_name].sort_values('Date').copy()
    
    if not m_df_full.empty:
        min_date = m_df_full['Date'].min().date()
        max_date = m_df_full['Date'].max().date()
        st.success(f"📅 **{selected_name}** 数据收录区间: {min_date} 至 {max_date}")

        col_filter1, col_filter2 = st.columns([1, 3])
        with col_filter1:
            date_range = st.date_input(
                "🔎 筛选时间段",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date
            )
        
        if len(date_range) == 2:
            start_filter, end_filter = date_range
            mask = (m_df_full['Date'].dt.date >= start_filter) & (m_df_full['Date'].dt.date <= end_filter)
            m_df = m_df_full.loc[mask]
        else:
            m_df = m_df_full

        if not m_df.empty:
            latest = m_df.iloc[-1]
            c1, c2, c3 = st.columns(3)
            c1.metric("选段最新日期", latest['Date'].strftime('%Y-%m-%d'))
            c2.metric("当日消耗", f"{latest['Total_Tokens']:.4f} B")
            if latest.get('Reasoning', 0) > 0 and latest.get('Completion', 0) > 0:
                ratio = (latest['Reasoning'] / latest['Completion']) * 100
                c3.metric("Reasoning 占比", f"{ratio:.1f}%")
            else:
                c3.metric("Prompt Tokens", f"{latest['Prompt']:.4f} B")

            chart = alt.Chart(m_df).mark_line(point=True).encode(
                x=alt.X('Date', title='日期', axis=alt.Axis(format='%m-%d')),
                y=alt.Y('Total_Tokens', title='Token (Billion)'),
                tooltip=['Date', 'Total_Tokens', 'Prompt', 'Completion']
            )
            
            st.altair_chart(chart, use_container_width=True)
            
            display_cols = ['Date', 'Total_Tokens', 'Prompt', 'Completion', 'Reasoning']
            valid_cols = [c for c in display_cols if c in m_df.columns]
            st.dataframe(m_df[valid_cols].sort_values('Date', ascending=False).style.format({'Total_Tokens':'{:.4f}'}), use_container_width=True)

            data, name, mime, label = get_dataset_download(m_df[valid_cols], f"{selected_name}_daily")
            st.download_button(label=label, data=data, file_name=name, mime=mime)
        else:
            st.warning("⚠️ 所选时间段内无数据。")

# ========================================================
# 页面 4: 原始数据检查
# ========================================================
elif page == NAV_RAW_DATA:
    st.subheader("数据库原始数据")
    
    st.markdown("#### 💾 全量数据下载")
    data, name, mime, label = get_dataset_download(df, "full_history_database")
    st.download_button(label=label, data=data, file_name=name, mime=mime)
    
    st.divider()
    
    check_name = st.selectbox("选择要检查的模型:", all_model_names)
    filtered_df = df[df['Display_Name'] == check_name].sort_values('Date', ascending=False)
    
    st.dataframe(
        filtered_df.style.format({
            'Prompt': '{:.6f} B', 'Completion': '{:.6f} B', 
            'Reasoning': '{:.6f} B', 'Total_Tokens': '{:.6f} B'
        }), use_container_width=True
    )

# ========================================================
# 页面 5: 每日速览与分析
# ========================================================
elif page == NAV_DAILY_BRIEF:
    st.subheader("模型表现速览与分析报告")
    st.caption("基于历史数据的多维度量化分析，所有指标均由数据自动计算生成。")

    latest_date = df['Date'].max()
    two_weeks_ago = latest_date - pd.Timedelta(days=14)
    seven_days_ago = latest_date - pd.Timedelta(days=7)

    metrics_list = []
    for name in all_model_names:
        m_df = df[df['Display_Name'] == name].sort_values('Date').copy()
        if m_df.empty:
            continue
        if len(m_df) > 1:
            m_df = m_df.iloc[:-1]
        if m_df.empty:
            continue

        first_date = m_df.iloc[0]['Date']
        last_date = m_df.iloc[-1]['Date']
        days_online = max((last_date - first_date).days, 1)
        cumulative = m_df['Total_Tokens'].sum()
        daily_avg = cumulative / days_online
        peak = m_df['Total_Tokens'].max()

        recent_df = m_df[m_df['Date'] >= seven_days_ago]
        recent_days = max(len(recent_df), 1)
        recent_avg = recent_df['Total_Tokens'].sum() / recent_days if not recent_df.empty else 0

        momentum = (recent_avg / daily_avg) if daily_avg > 0 else 0

        metrics_list.append({
            'Model': name,
            'First_Date': first_date,
            'Last_Date': last_date,
            'Days_Online': days_online,
            'Cumulative': round(cumulative, 4),
            'Daily_Avg': round(daily_avg, 4),
            'Recent_7d_Avg': round(recent_avg, 4),
            'Momentum': round(momentum, 2),
            'Peak': round(peak, 4),
        })

    df_metrics = pd.DataFrame(metrics_list)

    if df_metrics.empty:
        st.warning("暂无可分析的模型数据。")
        st.stop()

    df_metrics['Pct_Rank_DailyAvg'] = df_metrics['Daily_Avg'].rank(pct=True)

    new_models_df = df_metrics[df_metrics['First_Date'] >= two_weeks_ago].sort_values('First_Date', ascending=False)
    display_new = pd.DataFrame()
    if not new_models_df.empty:
        enhanced_new_models = []
        for row in new_models_df.itertuples():
            model_name = row.Model
            norm_name = normalize_model_name(model_name)
            
            input_price, output_price = None, None
            if df_price is not None and not df_price.empty:
                latest_price_date = df_price['Date'].max()
                price_rows = df_price[(df_price['Date'] == latest_price_date) & 
                                      (df_price['Provider'] == 'Weighted Average')]
                
                matched_price_model = fuzzy_match_model(norm_name, price_rows['Model'].unique().tolist(), threshold=0.55)
                if matched_price_model:
                    match_row = price_rows[price_rows['Model'] == matched_price_model[0]].iloc[0]
                    input_price = match_row.get('Input_Price_1M')
                    output_price = match_row.get('Output_Price_1M')

            arena_rank = None
            if df_lmarena is not None and not df_lmarena.empty:
                latest_lm_date = df_lmarena['Date'].max()
                lm_rows = df_lmarena[df_lmarena['Date'] == latest_lm_date]
                matched_lm_model = fuzzy_match_model(norm_name, lm_rows['Model'].unique().tolist(), threshold=0.55)
                if matched_lm_model:
                    match_row = lm_rows[lm_rows['Model'] == matched_lm_model[0]].iloc[0]
                    arena_rank = match_row.get('Rank_Overall')
            
            enhanced_new_models.append({
                '模型名称': model_name,
                '上线日期': row.First_Date.strftime('%Y-%m-%d'),
                '上线天数': row.Days_Online,
                '累计消耗 (B)': row.Cumulative,
                '日均消耗 (B)': row.Daily_Avg,
                '输入价格 ($/1M)': f"${input_price:.4f}" if pd.notna(input_price) else "-",
                '输出价格 ($/1M)': f"${output_price:.4f}" if pd.notna(output_price) else "-",
                'Arena 排名': f"{int(arena_rank)}" if pd.notna(arena_rank) else "-"
            })
        display_new = pd.DataFrame(enhanced_new_models)

    # ============================
    # 模块 A (置顶): AI 智能简报分析
    # ============================
    st.markdown("---")
    st.markdown("### 🤖 智能趋势简报")
    st.caption("基于今日数据的自动深度分析 (数据每日自动缓存，避免重复请求)")
    
    if not df_metrics.empty:
        top_momentum = df_metrics.nlargest(5, 'Momentum')
        low_momentum = df_metrics.nsmallest(5, 'Momentum')
        
        new_models_context = ""
        if not new_models_df.empty:
            new_models_context = display_new.to_string(index=False)
            
        # 【恢复原貌】完全恢复你原版的 ai_brief_prompt
        ai_brief_prompt = f"""
你是一位资深 TMT 行业投研分析师。请基于以下最新数据，直接撰写一份【大模型趋势追踪简报】。
当前日期: {latest_date.strftime('%Y-%m-%d')}

## 1. 近两周新上线模型（附带价格和外部能力排名）
{new_models_context if new_models_context else "近两周无新模型。"}

## 2. 增长动量 (Momentum) 极速上升的模型 TOP 5 (可能存在破圈或免费活动)
{top_momentum[['Model', 'Momentum', 'Daily_Avg', 'Recent_7d_Avg']].to_string(index=False)}

## 3. 增速放缓或萎缩的模型 TOP 5
{low_momentum[['Model', 'Momentum', 'Daily_Avg', 'Recent_7d_Avg']].to_string(index=False)}

## 分析要求（禁止输出任何 XML 或 <parameter> 标签，请直接基于上方文本资料进行分析,当你需要做原因分析时，可以结合联网信息寻找原因，但数据端请基于数据库和上方文本资料）：
请直接输出一段简练的、言之有物、有数据有逻辑的投研分析，不要任何寒暄、总结性套话或过度拔高的定性分析。
请务必参考以下风格和结构撰写（将其作为你的格式模板）：

“本周受 [某外部事件/节点] 影响，大模型推新 [加快/放缓]，共推出 [X] 款模型。

一、重点新模型追踪
近期新上线的一批模型中，只要是来自头部大厂（如 OpenAI、Anthropic、Google、Moonshot (Kimi)、DeepSeek、阿里、腾讯、百度等）的模型必须全部点名播报。其中最值得关注的是 [模型A] 和 [模型B]，其核心性能指标 [排名/基准] 达到 [X]，而价格为 [输入X/输出X]，对现有市场格局带来一定影响。

二、模型动量变动分析
在调用量增速方面，剔除长尾非旗舰模型后，[旗舰/新模型C] 上量猛增，近7日均值达到 [X] B，动量高达 [X]。经查证，这是因为其推出了 [具体的优惠政策/能力更新]；另外，[旗舰模型D] 增速停滞或萎缩，推测与 [具体退市/商业化策略] 有关。

三、多模态赛道观察
在图像、音频、视频等多模态领域，本周 [模型E/F] 表现亮眼 / [或暂无明显动向]。[基于数据的多模态能力与定价分析]”

任务规范：
1. 大厂追踪：新模型列表中，凡是属于头部主流厂商的（无论是国内外），必须全部提及，不遗漏。
2. 动量过滤：对于“极速上升的 TOP 5”，如果遇到不知名、非旗舰、非新上线的长尾模型，请直接忽略它，只挑其中具有行业代表性的旗舰或新模型进行分析。
3. 事实为准：必须使用准确的上方数据，并在陈述原因时必须通过网络搜索出确切的政策或版本事件（如某大厂在几号宣布了什么API免费计划，或者发布新版本）。坚决拒绝“引爆全球调用”等假大空的抒情主观词汇。
        """

        @st.cache_data(ttl=86400, show_spinner=False)
        def fetch_daily_ai_brief(prompt, provider="Google AI Studio", model_id=None):
            import requests as _req
            import re as _re
            cfg = AI_PROVIDERS.get(provider, AI_PROVIDERS["Google AI Studio"])
            key = cfg["key"]
            if not key: raise Exception(f"缺失 {provider} API Key")
            
            try:
                headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
                if provider == "OpenRouter":
                    headers.update({"HTTP-Referer": "http://localhost", "X-Title": "LLM-Dashboard"})
                
                if not model_id:
                    if cfg.get("models"):
                        model_id = list(cfg["models"].values())[0]
                    else:
                        model_id = "z-ai/glm-4.5-air:free"
                
                # 【恢复原貌】保留你原本未切割的 User Prompt 结构
                payload = {
                    "model": model_id,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 3000,
                }
                
                if provider == "OpenRouter":
                    payload["plugins"] = [{"id": "web", "max_results": 4}]
                else:
                    try:
                        import urllib.parse
                        import requests as _s_req
                        from bs4 import BeautifulSoup
                        
                        h={'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
                        r=_s_req.get('https://www.bing.com/search?q='+urllib.parse.quote("AI 大模型 近期动态"), headers=h, timeout=8)
                        s=BeautifulSoup(r.text, 'html.parser')
                        news_res = [{'title':li.find('h2').text, 'body':li.find('p').text if li.find('p') else ''} for li in s.find_all('li', class_='b_algo')[:5] if li.find('h2')]
                        if news_res:
                            context_str = "\n\n【补充资料：近期大模型相关新闻】：\n"
                            for r in news_res:
                                context_str += f"- {r.get('title', '')}: {r.get('body', '')}\n"
                            payload["messages"][0]["content"] += context_str
                    except Exception:
                        pass
                
                resp = _req.post(
                    f"{cfg['base_url']}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=300 
                )
                resp.raise_for_status()
                result = resp.json()
                raw_reply = result['choices'][0]['message']['content']
                
                
                clean_reply = _re.sub(r'<think>.*?</think>', '', raw_reply, flags=_re.DOTALL)
                clean_reply = _re.sub(r'</?[a-zA-Z0-9_:-]*tool_call[^>]*>', '', clean_reply)
                clean_reply = _re.sub(r'</?invoke[^>]*>', '', clean_reply)
                clean_reply = _re.sub(r'</?function[^>]*>', '', clean_reply)
                clean_reply = clean_reply.strip()
                
                return clean_reply if clean_reply else raw_reply
                
            except Exception as e:
                raise Exception(f"简报生成失败: {str(e)}")
                
        st.sidebar.divider()
        st.sidebar.markdown("### 📊 简报配置")
        brief_provider = st.sidebar.selectbox("简报服务商:", list(AI_PROVIDERS.keys()), index=2)
        
        available_models = AI_PROVIDERS[brief_provider]["models"]
        model_labels = list(available_models.keys())
        default_idx = 0
        if "GLM-5" in model_labels:
            default_idx = model_labels.index("GLM-5")
            
        brief_model_label = st.sidebar.selectbox("简报调用模型:", model_labels, index=default_idx)
        brief_model_id = available_models[brief_model_label]
        
        st.sidebar.caption(f"API 调用路径：`{brief_model_id}`")

        use_cache = st.sidebar.checkbox("优先读取每日凌晨自动生成的缓存简报 (推荐)", value=True)
        cache_path = "briefing_cache/daily_briefing_latest.md"
        
        if use_cache and os.path.exists(cache_path):
            st.info("💡 当前显示的是由 GitHub Actions 每日凌晨自动生成并缓存的行业简报。如需获取最新实时简报，请在左侧取消勾选“优先读取缓存”。")
            with open(cache_path, "r", encoding="utf-8") as f:
                st.markdown(f.read())
        else:
            with st.spinner(f"🤖 正在调用 {brief_provider} ({brief_model_label}) 生成实时简报 (耗时可能较长)..."):
                try:
                    brief_report = fetch_daily_ai_brief(ai_brief_prompt, provider=brief_provider, model_id=brief_model_id)
                    st.markdown(brief_report)
                except Exception as call_err:
                    st.error(f"🤖 分析报告生成失败: {call_err}")
                    if st.button("🔄 重试"):
                        fetch_daily_ai_brief.clear()
                        st.rerun()
    else:
        st.info("数据不足，无法生成总结报告。")

    # ============================
    # 模块 B: 近两周新增模型一览
    # ============================
    st.markdown("---")
    st.markdown("### 近两周新增模型一览")
    st.caption(f"统计区间: {two_weeks_ago.strftime('%Y-%m-%d')} ~ {latest_date.strftime('%Y-%m-%d')}")

    if new_models_df.empty:
        st.info("过去两周内没有新上线的模型。")
    else:
        st.markdown(f"过去两周共上线 **{len(new_models_df)}** 个新模型。")
        st.dataframe(
            display_new.style.format({'累计消耗 (B)': '{:.4f}', '日均消耗 (B)': '{:.4f}'}),
            use_container_width=True, hide_index=True
        )

    # ============================
    # 模块 C: 新模型累计增长对比
    # ============================
    if not new_models_df.empty:
        st.markdown("---")
        st.markdown("### 新模型累计增长对比")

        new_model_names = new_models_df['Model'].tolist()
        plot_new = []
        max_day_new = 0

        for name in new_model_names:
            m_df = df[df['Display_Name'] == name].sort_values('Date').copy()
            m_df['Cum_Tokens'] = m_df['Total_Tokens'].cumsum()
            if len(m_df) > 1:
                m_df = m_df.iloc[:-1]
            if m_df.empty:
                continue
            start_date = m_df.iloc[0]['Date']
            current_max = (m_df.iloc[-1]['Date'] - start_date).days
            if current_max > max_day_new:
                max_day_new = current_max
            for _, row in m_df.iterrows():
                day_n = (row['Date'] - start_date).days
                plot_new.append({
                    'Model': name, 'Day': day_n,
                    'Date': row['Date'].strftime('%Y-%m-%d'),
                    'Cumulative_Tokens': row['Cum_Tokens']
                })

        if plot_new:
            df_plot_new = pd.DataFrame(plot_new)
            base_new = alt.Chart(df_plot_new).encode(
                x=alt.X('Day', title='上线天数',
                        scale=alt.Scale(domain=[0, max_day_new + 2], clamp=True),
                        axis=alt.Axis(tickMinStep=1, format='d', labelFontSize=14, titleFontSize=16, grid=True)),
                y=alt.Y('Cumulative_Tokens', title='累计 Token (Billion)',
                        axis=alt.Axis(labelFontSize=14, titleFontSize=16)),
                color=alt.Color('Model', title='模型',
                                scale=alt.Scale(scheme='tableau10'),
                                legend=alt.Legend(orient='bottom')),
                tooltip=['Model', 'Day', 'Date', 'Cumulative_Tokens']
            )
            chart_new = (base_new.mark_line(strokeWidth=3) + base_new.mark_circle(size=60)).properties(height=500)
            st.altair_chart(chart_new, use_container_width=True)
        else:
            st.info("新模型暂无足够数据绘制趋势图。")

    # ============================
    # 模块 C: 分析摘要 (表格化)
    # ============================
    st.markdown("---")
    st.markdown("### 综合分析摘要")
    st.caption(f"分析基准日: {latest_date.strftime('%Y-%m-%d')}")

    with st.expander("累计消耗 Top 3", expanded=True):
        top3_cum = df_metrics.nlargest(3, 'Cumulative').copy()
        top3_cum['Rank'] = range(1, len(top3_cum) + 1)
        display_top3 = top3_cum[['Rank', 'Model', 'Cumulative', 'Days_Online', 'Daily_Avg']].copy()
        display_top3.columns = ['排名', '模型', '累计消耗 (B)', '上线天数', '日均消耗 (B)']
        st.dataframe(
            display_top3.style.format({'累计消耗 (B)': '{:.4f}', '日均消耗 (B)': '{:.4f}'}),
            use_container_width=True, hide_index=True
        )

    with st.expander("近7日增速领先 (Top 3)", expanded=True):
        top3_recent = df_metrics.nlargest(3, 'Recent_7d_Avg').copy()
        top3_recent['Rank'] = range(1, len(top3_recent) + 1)
        display_recent = top3_recent[['Rank', 'Model', 'Recent_7d_Avg']].copy()
        display_recent.columns = ['排名', '模型', '近7日日均消耗 (B)']
        st.dataframe(
            display_recent.style.format({'近7日日均消耗 (B)': '{:.4f}'}),
            use_container_width=True, hide_index=True
        )

    with st.expander("正在加速增长 (动量 > 1.2)", expanded=True):
        accel = df_metrics[df_metrics['Momentum'] >= 1.2].sort_values('Momentum', ascending=False)
        if not accel.empty:
            accel['Growth_Pct'] = (accel['Momentum'] - 1) * 100
            display_accel = accel[['Model', 'Momentum', 'Growth_Pct']].head(5).copy()
            display_accel.columns = ['模型', '动量值', '近期增速超出均值 (%)']
            st.dataframe(
                display_accel.style.format({'动量值': '{:.2f}', '近期增速超出均值 (%)': '+{:.0f}%'}),
                use_container_width=True, hide_index=True
            )
        else:
            st.info("暂无明显加速增长的模型。")

    with st.expander("增速放缓关注 (动量 < 0.8)", expanded=True):
        decel = df_metrics[(df_metrics['Momentum'] <= 0.8) & (df_metrics['Days_Online'] >= 7)].sort_values('Momentum')
        if not decel.empty:
            decel['Slowdown_Pct'] = (1 - decel['Momentum']) * 100
            display_decel = decel[['Model', 'Momentum', 'Slowdown_Pct']].head(5).copy()
            display_decel.columns = ['模型', '动量值', '近期增速低于均值 (%)']
            st.dataframe(
                display_decel.style.format({'动量值': '{:.2f}', '近期增速低于均值 (%)': '-{:.0f}%'}),
                use_container_width=True, hide_index=True
            )
        else:
            st.info("暂无明显增速放缓的模型。")

    if not new_models_df.empty:
        with st.expander("新模型初期表现评级", expanded=True):
            rating_data = []
            for row in new_models_df.itertuples():
                pct_rank = row.Pct_Rank_DailyAvg
                if pct_rank >= 0.90:
                    tier, desc = "S · 头部水平", f"超过 {pct_rank*100:.0f}% 模型"
                elif pct_rank >= 0.75:
                    tier, desc = "A · 表现优异", f"超过 {pct_rank*100:.0f}% 模型"
                elif pct_rank >= 0.50:
                    tier, desc = "B · 中等水平", "日均 > 中位数"
                elif pct_rank >= 0.25:
                    tier, desc = "C · 低于预期", f"仅超 {pct_rank*100:.0f}% 模型"
                else:
                    tier, desc = "D · 起步缓慢", f"后 {(1-pct_rank)*100:.0f}% 分位"
                
                rating_data.append({
                    '模型': row.Model,
                    '上线日期': row.First_Date.strftime('%m-%d'),
                    '日均消耗 (B)': row.Daily_Avg,
                    '评级': tier,
                    '说明': desc
                })
            
            df_rating = pd.DataFrame(rating_data)
            st.dataframe(
                df_rating.style.format({'日均消耗 (B)': '{:.4f}'}),
                use_container_width=True, hide_index=True
            )

    # ============================
    # 模块 D : 全模型表现排名 
    # ============================
    st.markdown("---")
    st.markdown("### 全模型表现排名 (Top 15)")

    RANK_OPTIONS = {
        '累计总量': 'Cumulative',
        '日均消耗': 'Daily_Avg',
        '近7日增速': 'Recent_7d_Avg',
        '增长动量': 'Momentum',
        '峰值消耗': 'Peak',
        '上线天数': 'Days_Online'
    }
    col_rank1, col_rank2 = st.columns([1, 3])
    with col_rank1:
        rank_label = st.selectbox("选择排名维度", list(RANK_OPTIONS.keys()))
    rank_col = RANK_OPTIONS[rank_label]

    df_ranked = df_metrics.sort_values(rank_col, ascending=False).head(15).reset_index(drop=True)
    df_ranked.index = df_ranked.index + 1

    chart_rank = alt.Chart(df_ranked).mark_bar(
        cornerRadiusTopLeft=4, cornerRadiusTopRight=4
    ).encode(
        x=alt.X('Model', sort='-y', title='模型',
                axis=alt.Axis(labelAngle=-45, labelFontSize=11)),
        y=alt.Y(rank_col, title=rank_label,
                axis=alt.Axis(labelFontSize=14, titleFontSize=16)),
        color=alt.Color('Model', legend=None, scale=alt.Scale(scheme='tableau10')),
        tooltip=['Model', alt.Tooltip(rank_col, title=rank_label, format='.4f')]
    ).properties(height=400)
    st.altair_chart(chart_rank, use_container_width=True)

    display_ranked = df_ranked[['Model', 'Days_Online', 'Cumulative', 'Daily_Avg', 'Recent_7d_Avg', 'Momentum', 'Peak']].copy()
    display_ranked.columns = ['模型', '上线天数', '累计 (B)', '日均 (B)', '近7日均 (B)', '动量', '峰值 (B)']

    def highlight_momentum(val):
        if isinstance(val, (int, float)):
            if val >= 1.2:
                return 'background-color: #d4edda; color: #155724'
            elif val <= 0.8:
                return 'background-color: #f8d7da; color: #721c24'
        return ''

    st.dataframe(
        display_ranked.style
            .format({'累计 (B)': '{:.4f}', '日均 (B)': '{:.4f}', '近7日均 (B)': '{:.4f}', '动量': '{:.2f}', '峰值 (B)': '{:.4f}'})
            .map(highlight_momentum, subset=['动量']),
        use_container_width=True, hide_index=False
    )
    st.caption("动量 > 1.2 (绿色背景) = 加速增长 · 动量 < 0.8 (红色背景) = 增速放缓")

    # ============================
    # 模块 E: 指标定义与公式说明
    # ============================
    st.markdown("---")
    st.markdown("### 附录: 指标定义与公式说明")
    with st.expander("查看完整指标说明", expanded=False):
        st.markdown("""
| 指标 | 定义 | 计算公式 |
|------|------|----------|
| **日均消耗** | 模型全生命周期内平均每天的 Token 消耗量 | `累计总量 ÷ 上线天数` |
| **近7日增速** | 最近 7 个自然日内的日平均 Token 消耗量 | `Σ(近7日 Total_Tokens) ÷ 近7日数据条数` |
| **增长动量** | 近期活跃度相对于全生命周期均值的比率 | `近7日增速 ÷ 日均消耗` |
| **峰值消耗** | 历史单日最高 Token 消耗量 | `max(每日 Total_Tokens)` |
| **累计总量** | 模型上线以来所有日期 Token 消耗之和 | `Σ(Total_Tokens)` |
| **上线天数** | 模型首次出现在数据库到最新数据的天数 | `最新数据日期 - 首次出现日期` |

**动量解读:**
- 动量 = 1.0 → 近期增速与全期均值持平
- 动量 > 1.2 → 近期处于加速增长阶段
- 动量 < 0.8 → 近期增速放缓，可能进入衰退期

**新模型评级说明:**

评级采用**百分位排名法 (Percentile Rank)**，将新模型的日均消耗放入全部模型的日均消耗分布中计算排名百分位:
- `百分位 = 日均消耗 < 该模型的模型数量 ÷ 总模型数`

| 评级 | 百分位区间 | 含义 |
|------|-----------|------|
| **S · 头部水平** | ≥ P90 | 日均消耗超过 90% 的模型，属于顶级表现 |
| **A · 表现优异** | P75 ~ P90 | 日均消耗处于前 25%，增长势头强劲 |
| **B · 中等水平** | P50 ~ P75 | 日均消耗高于中位数，表现中规中矩 |
| **C · 低于预期** | P25 ~ P50 | 日均消耗处于中位数以下，关注后续走势 |
| **D · 起步缓慢** | < P25     | 日均消耗处于倒数 25%，市场接受度较低 |
""")

# ========================================================
# 页面 6: 供应商价格与有效定价分析
# ========================================================
elif page == NAV_PRICING:
    st.subheader("模型定价")
    st.caption("基于 OpenRouter 前端 API 抓取的最新实际有效价格。")
    
    if df_price is None or df_price.empty:
        st.warning("暂未发现可用的定价数据。")
    else:
        all_models = sorted(df_price['Model'].unique())
        selected_price_model = st.selectbox("选择模型:", all_models, index=0)

        # 获取该模型所有历史时间点的数据
        m_price_df = df_price[df_price['Model'] == selected_price_model].copy()
        
        # 将最新的综合报价拆出来展示 KPI
        latest_pricing_date = m_price_df['Date'].max()
        df_latest_prices = m_price_df[m_price_df['Date'] == latest_pricing_date]
        
        weighted_avg = df_latest_prices[df_latest_prices['Provider'] == 'Weighted Average']
        provider_latest = df_latest_prices[df_latest_prices['Provider'] != 'Weighted Average'].sort_values('Input_Price_1M')
        
        if not weighted_avg.empty:
            wa_row = weighted_avg.iloc[0]
            st.markdown("### 最新有效价格 (Weighted Average)")
            col1, col2 = st.columns(2)
            col1.metric("Input Price ($/1M)", f"${wa_row['Input_Price_1M']:.4f}")
            col2.metric("Output Price ($/1M)", f"${wa_row['Output_Price_1M']:.4f}")
        
        st.markdown("---")
        
        # === 图1: 有效价格趋势 (Weighted Average 的 Input+Output 历史) ===
        st.markdown("### 有效价格趋势")
        wa_history = m_price_df[m_price_df['Provider'] == 'Weighted Average'].copy()
        if not wa_history.empty:
            wa_history['Date'] = pd.to_datetime(wa_history['Date'])
            wa_long = wa_history.melt(
                id_vars=['Date'],
                value_vars=['Input_Price_1M', 'Output_Price_1M'],
                var_name='Type', value_name='Price'
            ).dropna(subset=['Price'])
            wa_long['Type'] = wa_long['Type'].map({'Input_Price_1M': 'Input', 'Output_Price_1M': 'Output'})
            
            chart_wa = alt.Chart(wa_long).mark_line(point=True).encode(
                x=alt.X('Date:T', title='时间', axis=alt.Axis(format='%m/%d')),
                y=alt.Y('Price:Q', title='价格 ($/1M Tokens)'),
                color=alt.Color('Type:N', title='类型'),
                tooltip=['Date:T', 'Type', alt.Tooltip('Price:Q', format='$.4f')]
            ).properties(height=300)
            st.altair_chart(chart_wa, use_container_width=True)
        else:
            st.info("暂无有效价格历史数据。")
        
        st.markdown("---")
        
        # === 图2: 各供应商 Input 价格趋势折线图 ===
        st.markdown("### 各供应商 Input 价格趋势")
        provider_history = m_price_df[m_price_df['Provider'] != 'Weighted Average'].copy()
        if not provider_history.empty:
            provider_history['Date'] = pd.to_datetime(provider_history['Date'])
            
            chart_input = alt.Chart(provider_history).mark_line(point=True).encode(
                x=alt.X('Date:T', title='时间', axis=alt.Axis(format='%m/%d')),
                y=alt.Y('Input_Price_1M:Q', title='Input 价格 ($/1M Tokens)'),
                color=alt.Color('Provider:N', title='供应商', scale=alt.Scale(scheme='tableau20')),
                tooltip=['Date:T', 'Provider', alt.Tooltip('Input_Price_1M:Q', format='$.4f')]
            ).properties(height=350)
            st.altair_chart(chart_input, use_container_width=True)
        else:
            st.info("暂无供应商 Input 价格数据。")
        
        st.markdown("---")
        
        # === 图3: 各供应商 Output 价格趋势折线图 ===
        st.markdown("### 各供应商 Output 价格趋势")
        if not provider_history.empty:
            chart_output = alt.Chart(provider_history).mark_line(point=True).encode(
                x=alt.X('Date:T', title='时间', axis=alt.Axis(format='%m/%d')),
                y=alt.Y('Output_Price_1M:Q', title='Output 价格 ($/1M Tokens)'),
                color=alt.Color('Provider:N', title='供应商', scale=alt.Scale(scheme='tableau20')),
                tooltip=['Date:T', 'Provider', alt.Tooltip('Output_Price_1M:Q', format='$.4f')]
            ).properties(height=350)
            st.altair_chart(chart_output, use_container_width=True)
        else:
            st.info("暂无供应商 Output 价格数据。")
        
        st.markdown("---")
        
        # 供应商详情表格（最新一天）
        st.markdown("### 供应商详情 (最新)")
        if not provider_latest.empty:
            st.dataframe(
                provider_latest[['Provider', 'Input_Price_1M', 'Output_Price_1M', 'Cache_Hit_Rate']].style.format({
                    'Input_Price_1M': '${:.4f}',
                    'Output_Price_1M': '${:.4f}',
                    'Cache_Hit_Rate': '{:.1%}'
                }),
                use_container_width=True,
                hide_index=True
            )
            
        data, name, mime, label = get_dataset_download(df_price, "openrouter_pricing_full")
        st.download_button(label=label, data=data, file_name=name, mime=mime)

# ========================================================
# 页面 7: Benchmark 跑分数据矩阵
# ========================================================
elif page == NAV_BENCHMARK:
    st.subheader("基准测试与排行榜")
    st.caption("数据源：Artificial Analysis 基准跑分 + LMARENA (Chatbot Arena) 竞技排名。")
    
    tab1, tab2, tab3 = st.tabs([
        "单指标排行 (AA Benchmark)",
        "多指标矩阵 (AA Benchmark)",
        "LMARENA 竞技排名"
    ])
    
    # --- Tab 1 & 2: 原有 Artificial Analysis Benchmark ---
    if df_bench is None or df_bench.empty:
        with tab1:
            st.warning("暂未发现可用的 Benchmark 数据，请确认是否成功运行 `openrouter_benchmark_scraper.py`。")
        with tab2:
            st.warning("暂未发现可用的 Benchmark 数据。")
    else:
        latest_bench_date = df_bench['Date'].max()
        df_latest_bench = df_bench[df_bench['Date'] == latest_bench_date].drop(columns=['Date'])
        
        # 矩阵转置：让 Model 变成 index，Metrics 变成 columns
        bench_melted = df_latest_bench.melt(id_vars=['Metric'], var_name='Model', value_name='Score')
        bench_pivot = bench_melted.pivot_table(index='Model', columns='Metric', values='Score')
        
        metrics_available = bench_pivot.columns.tolist()
        
        with tab1:
            st.markdown("### 核心基准测试排行榜")
            st.info(f"数据更新于: **{latest_bench_date.strftime('%Y-%m-%d')}**")
            primary_metric = st.selectbox("选择排序指标:", metrics_available, index=0, key="tab1_metric")
            
            if primary_metric:
                bench_sorted = bench_pivot.sort_values(by=primary_metric, ascending=False).reset_index()
                bench_sorted = bench_sorted.dropna(subset=[primary_metric])
                
                top_10_models = bench_sorted['Model'].head(10).tolist()
                
                selected_b_models = st.multiselect(
                    "选择对比模型 (默认前10):", 
                    bench_sorted['Model'].tolist(), 
                    default=top_10_models,
                    key="tab1_models"
                )
                
                if selected_b_models:
                    plot_df = bench_sorted[bench_sorted['Model'].isin(selected_b_models)]
                    
                    chart_vertical = alt.Chart(plot_df).mark_bar(
                        cornerRadiusTopLeft=3, cornerRadiusTopRight=3
                    ).encode(
                        x=alt.X('Model:N', sort='-y', title='模型名称', axis=alt.Axis(labelAngle=-45, labelOverlap=False)),
                        y=alt.Y(f'{primary_metric}:Q', title='得分数值'),
                        color=alt.Color('Model:N', legend=None, scale=alt.Scale(scheme='tableau20')),
                        tooltip=['Model', alt.Tooltip(f'{primary_metric}:Q', format='.3f')]
                    ).properties(height=500)
                    
                    st.altair_chart(chart_vertical, use_container_width=True)
                else:
                    st.info("请至少选择一个模型进行对比绘制。")
                    
        with tab2:
            st.markdown("### 多维度指标交叉对比")
            st.info(f"数据更新于: **{latest_bench_date.strftime('%Y-%m-%d')}**")
            col_t1, col_t2 = st.columns([1, 2])
            with col_t1:
                t2_metric = st.selectbox("排序指标优先权:", metrics_available, index=0, key="tab2_main_metric")
            with col_t2:
                t2_metrics = st.multiselect("需要一并列出的其他指标:", metrics_available, default=metrics_available[:4] if len(metrics_available) >= 4 else metrics_available, key="tab2_metrics")
            
            t2_sorted = bench_pivot.sort_values(by=t2_metric, ascending=False).reset_index()
            t2_models_selected = st.multiselect(
                "需要放入表格对比的模型 (留空代表显示所有):",
                t2_sorted['Model'].tolist(),
                default=[]
            )
            
            display_cols = [t2_metric] + [m for m in t2_metrics if m != t2_metric]
            
            if t2_models_selected:
                display_df = bench_pivot.loc[t2_models_selected, display_cols].sort_values(by=t2_metric, ascending=False)
            else:
                display_df = bench_pivot.loc[:, display_cols].sort_values(by=t2_metric, ascending=False)
                
            st.dataframe(display_df.style.format("{:.3f}", na_rep='-'), use_container_width=True)
    
    # --- Tab 3: Arena 竞技排名 ---
    with tab3:
        st.markdown("### Arena 排行榜")
        st.caption("数据源: arena.ai · 由真人盲测对战的 ELO 分数")
        
        if df_lmarena is None or df_lmarena.empty:
            st.warning("暂未发现 Arena 排行榜数据。")
        else:
            latest_lm_date = df_lmarena['Date'].max()
            st.info(f"数据更新于: **{latest_lm_date.strftime('%Y-%m-%d')}**")
            
            df_latest_lm = df_lmarena[df_lmarena['Date'] == latest_lm_date].copy()
            
            # 两大类数据：ELO 排行榜 + Arena Overview
            arena_sub1, arena_sub2 = st.tabs(["🏆 ELO 排行榜", "📊 Arena Overview"])
            
            # ---- ELO 排行榜子标签 ----
            with arena_sub1:
                ELO_LABELS = {
                    'Score_text': '文本 (Text)',
                    'Score_code': '代码 (Code)',
                    'Score_vision': '视觉 (Vision)',
                    'Score_text_to_image': '文生图 (Text-to-Image)',
                    'Score_image_edit': '图像编辑 (Image Edit)',
                    'Score_search': '搜索 (Search)',
                    'Score_text_to_video': '文生视频 (Text-to-Video)',
                    'Score_image_to_video': '图生视频 (Image-to-Video)',
                }
                
                elo_options = {}
                for col_key, label in ELO_LABELS.items():
                    if col_key in df_latest_lm.columns and df_latest_lm[col_key].notna().sum() > 0:
                        elo_options[label] = col_key
                
                if not elo_options:
                    st.info("暂无 ELO 排行数据。")
                else:
                    selected_elo_label = st.selectbox("选择排行榜:", list(elo_options.keys()), index=0, key="arena_elo_cat")
                    selected_elo_col = elo_options[selected_elo_label]
                    
                    ranked_df = df_latest_lm.dropna(subset=[selected_elo_col]).copy()
                    ranked_df = ranked_df.sort_values(selected_elo_col, ascending=False).reset_index(drop=True)
                    
                    # Votes 列
                    votes_col = selected_elo_col.replace('Score_', 'Votes_')
                    
                    if not ranked_df.empty:
                        top_n = min(25, len(ranked_df))
                        top_df = ranked_df.head(top_n).copy()
                        top_df['ELO'] = top_df[selected_elo_col].astype(int)
                        
                        tooltip_fields = ['Model', alt.Tooltip('ELO:Q', title='ELO 分数')]
                        if votes_col in top_df.columns:
                            top_df['Votes'] = top_df[votes_col].fillna(0).astype(int)
                            tooltip_fields.append(alt.Tooltip('Votes:Q', title='投票数', format=','))
                        
                        chart_elo = alt.Chart(top_df).mark_bar(
                            cornerRadiusTopRight=3, cornerRadiusBottomRight=3
                        ).encode(
                            y=alt.Y('Model:N', 
                                    sort=alt.EncodingSortField(field='ELO', order='descending'),
                                    title=None, 
                                    axis=alt.Axis(labelOverlap=False)),
                            x=alt.X('ELO:Q', title='ELO 分数', scale=alt.Scale(zero=False)),
                            color=alt.value('#4C78A8'),
                            tooltip=tooltip_fields
                        ).properties(height=max(300, top_n * 28))
                        st.altair_chart(chart_elo, use_container_width=True)
                        
                        # 表格
                        disp_cols = ['Model', selected_elo_col]
                        disp_names = {'Model': '模型', selected_elo_col: 'ELO 分数'}
                        if votes_col in ranked_df.columns:
                            disp_cols.append(votes_col)
                            disp_names[votes_col] = '投票数'
                        display_lm = ranked_df[disp_cols].copy()
                        display_lm.rename(columns=disp_names, inplace=True)
                        st.dataframe(display_lm, use_container_width=True, hide_index=True, height=400)
                    else:
                        st.info("该排行榜暂无数据。")
            
            # ---- Arena Overview 子标签 ----
            with arena_sub2:
                RANK_LABELS = {
                    'Rank_Overall': '综合 (Overall)',
                    'Rank_Expert': '专家 (Expert)',
                    'Rank_Hard_Prompts': '困难提示词',
                    'Rank_Coding': '代码',
                    'Rank_Math': '数学',
                    'Rank_Creative_Writing': '创意写作',
                    'Rank_Instruction_Following': '指令遵循',
                    'Rank_Longer_Query': '长查询',
                }
                
                rank_options = {}
                for col_key, label in RANK_LABELS.items():
                    if col_key in df_latest_lm.columns and df_latest_lm[col_key].notna().sum() > 0:
                        rank_options[label] = col_key
                
                if not rank_options:
                    st.info("暂无 Arena Overview 数据。")
                else:
                    selected_rank_label = st.selectbox("排序维度:", list(rank_options.keys()), index=0, key="arena_ov_cat")
                    selected_rank_col = rank_options[selected_rank_label]
                    
                    ov_df = df_latest_lm.dropna(subset=[selected_rank_col]).copy()
                    ov_df = ov_df.sort_values(selected_rank_col, ascending=True).reset_index(drop=True)
                    
                    if not ov_df.empty:
                        top_n = min(30, len(ov_df))
                        top_df = ov_df.head(top_n).copy()
                        top_df['排名'] = top_df[selected_rank_col].astype(int)
                        
                        chart_ov = alt.Chart(top_df).mark_bar(
                            cornerRadiusTopRight=3, cornerRadiusBottomRight=3
                        ).encode(
                            y=alt.Y('Model:N',
                                    sort=alt.EncodingSortField(field='排名', order='ascending'),
                                    title=None,
                                    axis=alt.Axis(labelOverlap=False)),
                            x=alt.X('排名:Q', title='排名 (越小越好)', scale=alt.Scale(reverse=True)),
                            color=alt.value('#E45756'),
                            tooltip=['Model', alt.Tooltip('排名:Q', title='排名')]
                        ).properties(height=max(300, top_n * 25))
                        st.altair_chart(chart_ov, use_container_width=True)
                        
                        # 多维度排名表
                        st.markdown(f"#### Arena Overview 完整排名 (共 {len(ov_df)} 个模型)")
                        rank_cols_available = [c for c in RANK_LABELS.keys() if c in ov_df.columns]
                        display_ov = ov_df[['Model'] + rank_cols_available].copy()
                        rename_map = {'Model': '模型'}
                        rename_map.update({k: RANK_LABELS[k] for k in rank_cols_available})
                        display_ov.rename(columns=rename_map, inplace=True)
                        st.dataframe(display_ov, use_container_width=True, hide_index=True, height=500)
                    else:
                        st.info("该维度暂无数据。")
    
    st.markdown("---")
    col_dl1, col_dl2 = st.columns(2)
    if df_bench is not None:
        with col_dl1:
            data, name, mime, label = get_dataset_download(df_bench, "openrouter_benchmark_full")
            st.download_button(label="下载 AA Benchmark 数据", data=data, file_name=name, mime=mime)
    if df_lmarena is not None:
        with col_dl2:
            data, name, mime, label = get_dataset_download(df_lmarena, "lmarena_leaderboard_full")
            st.download_button(label="下载 LMARENA 数据", data=data, file_name=name, mime=mime)

# ========================================================
# 页面 8: 单模型深度探索
# ========================================================
elif page == NAV_SINGLE_MODEL:
    st.subheader("单模型分析")
    st.caption("综合用量、基准测试和定价数据，追踪单一模型。")

    # 获取包含过去现在所有记录下来的名字集合，统一消除重名干扰项
    raw_models = set(all_model_names) | set(all_pricing_models) | set(all_benchmark_models)
    normalized_map = {}
    for rm in raw_models:
        norm = normalize_model_name(rm)
        if norm not in normalized_map:
            normalized_map[norm] = []
        normalized_map[norm].append(rm)
        
    all_possible_models = sorted(list(normalized_map.keys()))
    
    if not all_possible_models:
        st.warning("暂未发现任何模型数据。")
    else:
        selected_model_norm = st.selectbox("选择模型:", all_possible_models)
        st.markdown("---")
        
        real_names = normalized_map[selected_model_norm]
        
        # 1. 累计用量趋势
        st.markdown("### 累计用量趋势")
        if df is not None and not df.empty:
            m_df = df[df['Model'].isin(real_names) | df['Display_Name'].isin(real_names)].sort_values('Date').copy()
                
            if not m_df.empty:
                m_df = m_df.groupby('Date', as_index=False)['Total_Tokens'].sum()
                m_df['Cumulative_Tokens'] = m_df['Total_Tokens'].cumsum()
                
                col_m1, col_m2 = st.columns(2)
                recent_7d = m_df.tail(7)['Total_Tokens'].sum()
                col_m1.metric("累计消耗", f"{m_df['Cumulative_Tokens'].iloc[-1]:.4f} Billion")
                col_m2.metric("近 7 天消耗", f"{recent_7d:.4f} Billion")
                    
                chart_cum = alt.Chart(m_df).mark_area(
                    opacity=0.6, 
                    color=alt.Gradient(
                        gradient='linear',
                        stops=[alt.GradientStop(color='orange', offset=0), alt.GradientStop(color='white', offset=1)],
                        x1=1, x2=1, y1=1, y2=0
                    )
                ).encode(
                    x=alt.X('Date:T', title='日期'),
                    y=alt.Y('Cumulative_Tokens:Q', title='累计 Tokens (Billion)'),
                    tooltip=['Date', 'Cumulative_Tokens', 'Total_Tokens']
                ).properties(height=350)
                st.altair_chart(chart_cum, use_container_width=True)
            else:
                st.info("该模型暂无 Token 消耗记录。")
        else:
            st.info("未连接到 Token 数据源。")

        st.markdown("---")
        
        # 2. 基准测试跑分
        st.markdown(f"### {selected_model_norm} 基准测试跑分")
        st.caption("同一模型可能有 Reasoning / Non-Reasoning 等变体。")
        if df_bench is not None and not df_bench.empty:
            latest_bench_date = df_bench['Date'].max()
            df_latest_bench = df_bench[(df_bench['Date'] == latest_bench_date) & (df_bench['Metric'].notna())].copy()
            
            bench_model_cols = [col for col in df_latest_bench.columns if col not in ['Date', 'Metric']]
            matched_b_cols = fuzzy_match_model(selected_model_norm, bench_model_cols, threshold=0.55)
            
            if matched_b_cols:
                tabs_b = st.tabs(matched_b_cols)
                
                for i, m_col in enumerate(matched_b_cols):
                    with tabs_b[i]:
                        model_scores = df_latest_bench[['Metric', m_col]].dropna()
                        if not model_scores.empty:
                            rank_data = []
                            for _, row in model_scores.iterrows():
                                metric = row['Metric']
                                score = row[m_col]
                                
                                # 增加 .empty 防护，防止触发 IndexError
                                metric_rows = df_latest_bench[df_latest_bench['Metric'] == metric]
                                if not metric_rows.empty:
                                    all_scores_flat = metric_rows.drop(columns=['Date', 'Metric']).iloc[0].dropna()
                                    all_scores_num = pd.to_numeric(all_scores_flat, errors='coerce').dropna()
                                    
                                    if score in all_scores_num.values:
                                        rank = all_scores_num.rank(method='min', ascending=False)[m_col]
                                        total = len(all_scores_num)
                                        percentile = (total - rank) / total * 100
                                        
                                        rank_data.append({
                                            '指标': metric,
                                            '得分': f"{score:.3f}",
                                            '排名': f"第 {int(rank)} / 共 {total}",
                                            '分位数': f"超越 {percentile:.1f}%"
                                        })
                            
                            if rank_data:
                                st.dataframe(pd.DataFrame(rank_data), use_container_width=True, hide_index=True)
                            else:
                                st.info("暂无可用测试数据。")
                        else:
                            st.info("暂无数据。")
            else:
                st.info("该模型未被收录于 Benchmark 数据中。")
        else:
            st.info("未连接到跑分数据源。")

        st.markdown("---")
        
        # 3. 定价分析（双图：有效价格趋势 + 各供应商价格柱状图）
        st.markdown("### 定价分析")
        if df_price is not None and not df_price.empty:
            m_price_df = df_price[df_price['Model'].isin(real_names)].copy()
            if not m_price_df.empty:
                latest_pricing_date = m_price_df['Date'].max()
                df_latest_prices = m_price_df[m_price_df['Date'] == latest_pricing_date]
                
                wa_row = df_latest_prices[df_latest_prices['Provider'] == 'Weighted Average']
                if not wa_row.empty:
                    wa = wa_row.iloc[0]
                    st.success(f"有效均价: Input **${wa['Input_Price_1M']:.4f}**/1M · Output **${wa['Output_Price_1M']:.4f}**/1M")
                
                # 图1: 有效价格趋势
                st.markdown("#### 有效价格趋势")
                wa_hist = m_price_df[m_price_df['Provider'] == 'Weighted Average'].copy()
                if not wa_hist.empty:
                    wa_hist['Date'] = pd.to_datetime(wa_hist['Date'])
                    wa_long = wa_hist.melt(
                        id_vars=['Date'],
                        value_vars=['Input_Price_1M', 'Output_Price_1M'],
                        var_name='Type', value_name='Price'
                    ).dropna(subset=['Price'])
                    wa_long['Type'] = wa_long['Type'].map({'Input_Price_1M': 'Input', 'Output_Price_1M': 'Output'})
                    
                    chart_wa = alt.Chart(wa_long).mark_line(point=True).encode(
                        x=alt.X('Date:T', title='时间', axis=alt.Axis(format='%m/%d')),
                        y=alt.Y('Price:Q', title='价格 ($/1M Tokens)'),
                        color=alt.Color('Type:N', title='类型'),
                        tooltip=['Date:T', 'Type', alt.Tooltip('Price:Q', format='$.4f')]
                    ).properties(height=250)
                    st.altair_chart(chart_wa, use_container_width=True)
                
                # 图2: 各供应商 Input 价格趋势
                st.markdown("#### 各供应商 Input 价格趋势")
                provider_history = m_price_df[m_price_df['Provider'] != 'Weighted Average'].copy()
                if not provider_history.empty:
                    provider_history['Date'] = pd.to_datetime(provider_history['Date'])
                    
                    chart_input = alt.Chart(provider_history).mark_line(point=True).encode(
                        x=alt.X('Date:T', title='时间', axis=alt.Axis(format='%m/%d')),
                        y=alt.Y('Input_Price_1M:Q', title='Input 价格 ($/1M Tokens)'),
                        color=alt.Color('Provider:N', title='供应商', scale=alt.Scale(scheme='tableau20')),
                        tooltip=['Date:T', 'Provider', alt.Tooltip('Input_Price_1M:Q', format='$.4f')]
                    ).properties(height=250)
                    st.altair_chart(chart_input, use_container_width=True)
                else:
                    st.info("暂无供应商 Input 价格趋势数据。")
                
                # 图3: 各供应商 Output 价格趋势
                st.markdown("#### 各供应商 Output 价格趋势")
                if not provider_history.empty:
                    chart_output = alt.Chart(provider_history).mark_line(point=True).encode(
                        x=alt.X('Date:T', title='时间', axis=alt.Axis(format='%m/%d')),
                        y=alt.Y('Output_Price_1M:Q', title='Output 价格 ($/1M Tokens)'),
                        color=alt.Color('Provider:N', title='供应商', scale=alt.Scale(scheme='tableau20')),
                        tooltip=['Date:T', 'Provider', alt.Tooltip('Output_Price_1M:Q', format='$.4f')]
                    ).properties(height=250)
                    st.altair_chart(chart_output, use_container_width=True)
                else:
                    st.info("暂无供应商 Output 价格趋势数据。")
                
                # 供应商详情表格
                st.markdown("#### 供应商详情 (最新)")
                provider_prices = df_latest_prices[df_latest_prices['Provider'] != 'Weighted Average'].sort_values('Input_Price_1M')
                if not provider_prices.empty:
                    st.dataframe(
                        provider_prices[['Provider', 'Input_Price_1M', 'Output_Price_1M', 'Cache_Hit_Rate']].style.format({
                            'Input_Price_1M': '${:.4f}',
                            'Output_Price_1M': '${:.4f}',
                            'Cache_Hit_Rate': '{:.1%}'
                        }),
                        use_container_width=True,
                        hide_index=True
                    )
            else:
                st.info("暂无该模型的定价数据。")
        else:
            st.info("未连接到定价数据源。")

        st.markdown("---")

        # 4. Arena (LMARENA) 排名
        st.markdown(f"### {selected_model_norm} 的 Arena 排名")
        if df_lmarena is not None and not df_lmarena.empty:
            latest_lm_date = df_lmarena['Date'].max()
            df_latest_lm = df_lmarena[df_lmarena['Date'] == latest_lm_date]
            
            lm_all_models = df_latest_lm['Model'].unique().tolist()
            matched_lm = fuzzy_match_model(selected_model_norm, lm_all_models, threshold=0.5)
            
            if matched_lm:
                lm_rows = df_latest_lm[df_latest_lm['Model'].isin(matched_lm)].copy()
                
                score_cols = [c for c in lm_rows.columns if c.startswith('Score_')]
                rank_cols = [c for c in lm_rows.columns if c.startswith('Rank_')]
                SCORE_LABELS = {
                    'Score_text': '文本', 'Score_code': '代码', 'Score_vision': '视觉',
                    'Score_text_to_image': '文生图', 'Score_image_edit': '图像编辑', 
                    'Score_search': '搜索', 'Score_text_to_video': '文生视频', 
                    'Score_image_to_video': '图生视频',
                }
                RANK_LABELS = {
                    'Rank_Overall': '综合', 'Rank_Expert': '专家', 'Rank_Hard_Prompts': '困难提示词',
                    'Rank_Coding': '代码', 'Rank_Math': '数学', 'Rank_Creative_Writing': '创意写作',
                    'Rank_Instruction_Following': '指令遵循', 'Rank_Longer_Query': '长查询',
                }
                
                rank_display = []
                for _, row in lm_rows.iterrows():
                    entry = {'模型': row['Model']}
                    for rc in rank_cols:
                        label = RANK_LABELS.get(rc, rc)
                        if pd.notna(row.get(rc)):
                            entry[f'{label}排名'] = int(row[rc])
                    for sc in score_cols:
                        label = SCORE_LABELS.get(sc, sc)
                        if pd.notna(row.get(sc)):
                            entry[f'{label} ELO'] = int(row[sc])
                    rank_display.append(entry)
                
                if rank_display:
                    st.dataframe(pd.DataFrame(rank_display), use_container_width=True, hide_index=True)
                else:
                    st.info("未找到该模型的排名数据。")
            else:
                st.info("该模型暂未被 Arena 收录。")
        else:
            st.info("未连接到 Arena 数据源。")