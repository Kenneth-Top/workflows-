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
            "Gemini 2.5 Pro": "gemini-2.5-pro",
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
    # 按 空格、横线、下划线 分割
    tokens = set(_re_global.split(r'[\s\-_]+', n.strip()))
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
    n = n.replace(' ', '-')
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
        selected_model_label = st.selectbox("选择模型:", list(provider_cfg["models"].keys()), index=0)
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
                display_names = sorted(_df['Display_Name'].dropna().unique().tolist())
                context_parts.append(f"""### Token 消耗数据 (变量名: df)
- 列: Date, Model, Prompt, Completion, Reasoning, Total_Tokens, Display_Name
- 记录数: {len(_df)}, 日期范围: {_df['Date'].min().strftime('%Y-%m-%d')} ~ {_df['Date'].max().strftime('%Y-%m-%d')}
- Token 单位: Billion (10亿)
- **全部可用模型列表 (Display_Name列)**: {', '.join(display_names)}""")

            if _df_price is not None and not _df_price.empty:
                price_models = sorted(_df_price['Model'].dropna().unique().tolist())
                context_parts.append(f"""### 定价数据 (变量名: df_price)
- 列: Date, Model, Provider, Input_Price_1M, Output_Price_1M, Cache_Hit_Rate
- 价格单位: $/1M Tokens
- **全部定价模型列表**: {', '.join(price_models)}""")

            if _df_bench is not None and not _df_bench.empty:
                context_parts.append(f"""### Benchmark 跑分 (变量名: df_bench)
- 结构: 宽表，每行是一个 Metric，每列是一个模型名
- Metric: {', '.join(_df_bench['Metric'].unique()[:8])}
- 模型数: {len([c for c in _df_bench.columns if c not in ['Date','Metric']])}""")

            if _df_lmarena is not None and not _df_lmarena.empty:
                score_cols = [c for c in _df_lmarena.columns if c.startswith('Score_')]
                context_parts.append(f"""### Arena 竞技排行 (变量名: df_lmarena)
- 数据源: arena.ai (原 LMARENA)
- 8 个 ELO 排行榜: {', '.join(c.replace('Score_','') for c in score_cols)}
- 模型示例: {', '.join(_df_lmarena['Model'].unique().tolist()[:30])}""")
            
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
        
        SYSTEM_PROMPT = f"""你是一位专业的 LLM 行业投研分析师，服务于机构投资者。你的核心任务是用**数据驱动的可视化图表**回答问题。绝对禁止在回复中输出任何 <tool_call>, <function> 等 XML 或内部工具调用标签。如果缺少信息，请直接回答“我不知道”或基于现有数据进行推测。

## 数据库
{db_context}

{web_search_rules}

## 数据过滤指南（最高优先级）
1. **杜绝盲目模糊匹配**：当用户询问特定模型（如 "M2.5", "minimax", "gemini"）时，**必须**仔细查阅我提供的“全部可用模型列表”，找出确切的目标名称。
2. **使用精准过滤**：严禁使用宽泛的 `str.contains('2.5')`！你必须使用 `isin` 或 `==` 进行精准匹配，例如：`df[df['Display_Name'].isin(['minimax-m2.5'])]`。
3. **区分同厂商新老模型**：如果用户查特定新模型，绝不要把该厂商几个月前的老模型数据一并算入，以免污染时间轴。

## 输出格式（严格遵守）
1. **文字部分**：用 3-5 句话给出核心结论，像投研报告摘要一样简洁。提供关键数据+洞察。
2. **代码部分**：必须生成一个 ```python``` 代码块，包含**至少 1 个图表**。

## 可视化指南
- 使用 `st.altair_chart(chart, use_container_width=True)` 展示图表。
- 用 `st.dataframe()` 展示辅助数据表格（可选）。
- 标题用中文，字号设为 16。
- 对所有数值列操作前**先确保类型正确** (如 `pd.to_numeric()`)。
- 只写一个代码块，包含所有图表和表格代码。
"""

        # 初始化聊天历史
        if "ai_messages" not in st.session_state:
            st.session_state.ai_messages = []
        
        # 用于 exec 的命名空间
        import numpy as np
        exec_namespace = {
            "df": df, "df_price": df_price, "df_bench": df_bench, "df_lmarena": df_lmarena,
            "st": st, "alt": alt, "pd": pd, "np": np, "os": os,
        }
        
        # 辅助函数：从 AI 回复中分离文字和代码
        def split_reply(reply):
            import re as _re
            code_blocks = _re.findall(r'```python\s*\n(.*?)```', reply, _re.DOTALL)
            text_only = _re.sub(r'```python\s*\n.*?```', '', reply, flags=_re.DOTALL).strip()
            return text_only, code_blocks[0] if code_blocks else None
        
        def safe_exec(code, ns):
            for key in ['df', 'df_price', 'df_lmarena']:
                frame = ns.get(key)
                if frame is not None and 'Model' in frame.columns:
                    frame = frame.copy()
                    frame['Model'] = frame['Model'].astype(str)
                    ns[key] = frame
            exec(code, ns)
        
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
            
            # 构建 API 请求 Payload (动态注入联网插件)
            api_payload = {
                "model": AI_MODEL,
                "messages": messages,
                "max_tokens": 4000,
                "temperature": 0.3
            }
            if enable_web_search:
                if provider_name == "OpenRouter":
                    api_payload["plugins"] = [{"id": "web", "max_results": 4}]
                else:
                    # 使用纯本地免费方案给非 OpenRouter 模型添加联网能力
                    # 提示词预处理：去除引号、多余空格，并只取前 60 个字符，防止搜索词过长导致 DDGS 返回空
                    clean_query = user_query.replace('"', '').replace("'", "").strip()[:60]
                    
                    with st.spinner(f"正在全网搜索线索: '{clean_query}'..."):
                        try:
                            from duckduckgo_search import DDGS
                            with DDGS() as ddgs:
                                search_results = list(ddgs.text(clean_query, max_results=5))
                            
                            if search_results:
                                context_str = "【实时网络搜索参考资料】\n"
                                for r in search_results:
                                    context_str += f"- 标题: {r.get('title', '')}\n  摘要: {r.get('body', '')}\n"
                                
                                # 注入上下文和强制输出格式要求
                                api_payload["messages"][-1]["content"] += f"\n\n请参考以下最新的网络搜索结果来辅助回答上述问题：\n{context_str}\n\n【最高优先级指令】：无论你参考了什么外部资料，你的主要任务仍然是执行数据分析。如果你需要生成图表，请务必返回完全独立、无依赖报错的 Python st/alt 渲染代码，并使用 ```python ... ``` 包裹代码块！"
                                st.toast(f"✅ 已抓取与 '{clean_query}' 相关的实时数据！")
                            else:
                                st.toast(f"⚠️ 搜索 '{clean_query}' 未发现直接结果，将凭模型知识库回答。")
                        except Exception as e:
                            st.toast(f"⚠️ 本地联网搜索受阻: {e}，将正常发送文本。")
            
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
                            timeout=75
                        )
                        if resp.status_code != 200:
                            raise Exception(f"API Error {resp.status_code}: {resp.text}")
                        result = resp.json()
                        ai_reply = result['choices'][0]['message']['content']
                    except Exception as e:
                        ai_reply = f"查询失败: {str(e)}"
                    
                    # 渲染文字和代码
                    text_part, code = split_reply(ai_reply)
                    st.markdown(text_part)
                    if code:
                        try:
                            safe_exec(code, exec_namespace)
                        except Exception as code_e:
                            st.warning("⚠️ 图表渲染出错，大模型生成的代码可能存在语法错误或引用了不存在的列名。")
                            with st.expander("查看错误详情"):
                                st.code(str(code_e) + "\n\n" + code)
                
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
            m_df = df[df['Display_Name'] == name].sort_values('Date')
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
            
            # 【配色优化】使用 tableau10 高对比配色
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
                # 关键修改：scale=alt.Scale(scheme='tableau10')
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
            m_df = df[df['Display_Name'] == name].sort_values('Date')
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

            # 【配色优化】使用 tableau10 高对比配色
            base = alt.Chart(df_plot).encode(
                x=alt.X('Day', title="上线天数 (Daily)", 
                        scale=alt.Scale(domain=[0, max_day_plot + 2], clamp=True),
                        axis=alt.Axis(labelFontSize=16, titleFontSize=18, grid=True)),
                y=alt.Y('Cumulative_Tokens', title='累计 Token (Billion)', 
                        axis=alt.Axis(labelFontSize=16, titleFontSize=18)),
                # 关键修改：scale=alt.Scale(scheme='tableau10')
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
    m_df_full = df[df['Display_Name'] == selected_name].sort_values('Date')
    
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

    # --- 预计算所有模型的指标 ---
    latest_date = df['Date'].max()
    two_weeks_ago = latest_date - pd.Timedelta(days=14)
    seven_days_ago = latest_date - pd.Timedelta(days=7)

    metrics_list = []
    for name in all_model_names:
        m_df = df[df['Display_Name'] == name].sort_values('Date')
        if m_df.empty:
            continue
        # 去掉最后一天（当天未结算数据，和其他页面逻辑保持一致）
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

        # 近 7 日增速
        recent_df = m_df[m_df['Date'] >= seven_days_ago]
        recent_days = max(len(recent_df), 1)
        recent_avg = recent_df['Total_Tokens'].sum() / recent_days if not recent_df.empty else 0

        # 增长动量
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

    # 计算百分位排名（供后续模块使用）
    df_metrics['Pct_Rank_DailyAvg'] = df_metrics['Daily_Avg'].rank(pct=True)

    # 提前计算新模型数据供 AI 简报使用
    new_models_df = df_metrics[df_metrics['First_Date'] >= two_weeks_ago].sort_values('First_Date', ascending=False)
    display_new = pd.DataFrame()
    if not new_models_df.empty:
        enhanced_new_models = []
        for row in new_models_df.itertuples():
            model_name = row.Model
            norm_name = normalize_model_name(model_name)
            
            # --- 查价格 ---
            input_price, output_price = None, None
            if df_price is not None and not df_price.empty:
                latest_price_date = df_price['Date'].max()
                price_rows = df_price[(df_price['Date'] == latest_price_date) & 
                                      (df_price['Provider'] == 'Weighted Average')]
                
                # 精确匹配（防止部分名称被误杀）
                matched_price_model = fuzzy_match_model(norm_name, price_rows['Model'].unique().tolist(), threshold=0.55)
                if matched_price_model:
                    match_row = price_rows[price_rows['Model'] == matched_price_model[0]].iloc[0]
                    input_price = match_row.get('Input_Price_1M')
                    output_price = match_row.get('Output_Price_1M')

            # --- 查 LMARENA 排名 ---
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
    
    # 构造给 AI 的当日关键数据
    if not df_metrics.empty:
        # 获取表现最优和最差的模型
        top_momentum = df_metrics.nlargest(5, 'Momentum')
        low_momentum = df_metrics.nsmallest(5, 'Momentum')
        
        # 提取新模型简报数据
        new_models_context = ""
        if not new_models_df.empty:
            new_models_context = display_new.to_string(index=False)
            
        ai_brief_prompt = f"""
你是一位资深 TMT 行业投研分析师。请基于以下最新数据，直接撰写一份【大模型趋势追踪简报】。
当前日期: {latest_date.strftime('%Y-%m-%d')}

## 1. 近两周新上线模型（附带价格和外部能力排名）
{new_models_context if new_models_context else "近两周无新模型。"}

## 2. 增长动量 (Momentum) 极速上升的模型 TOP 5 (可能存在破圈或免费活动)
{top_momentum[['Model', 'Momentum', 'Daily_Avg', 'Recent_7d_Avg']].to_string(index=False)}

## 3. 增速放缓或萎缩的模型 TOP 5
{low_momentum[['Model', 'Momentum', 'Daily_Avg', 'Recent_7d_Avg']].to_string(index=False)}

## 分析要求（务必使用网络插件检索政策/活动原因）：
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
        def fetch_daily_ai_brief(prompt, provider="Google AI Studio"):
            import requests as _req
            cfg = AI_PROVIDERS.get(provider, AI_PROVIDERS["Google AI Studio"])
            key = cfg["key"]
            if not key: raise Exception(f"缺失 {provider} API Key")
            
            try:
                headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
                
                # 动态获取该提供商配置的第一个模型，防止硬编码导致模型不存在 (400) 问题
                if cfg.get("models"):
                    model = list(cfg["models"].values())[0]
                else:
                    model = "z-ai/glm-4.5-air:free" # Fallback
                
                payload = {
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 2500,
                }
                
                # 只有 OpenRouter 支持这里的 web 插件语法
                if provider == "OpenRouter":
                    payload["plugins"] = [{"id": "web", "max_results": 4}]
                else:
                    # 对于生简报的非 OpenRouter 模型，也执行一轮预搜索帮助它寻找最新的资讯
                    try:
                        from duckduckgo_search import DDGS
                        with DDGS() as ddgs:
                            news_res = list(ddgs.text("AI 大模型 近期动态", max_results=5, timelimit='w'))
                        if news_res:
                            context_str = "\n\n【补充资料：近期大模型相关新闻】：\n"
                            for r in news_res:
                                context_str += f"- {r.get('title', '')}: {r.get('body', '')}\n"
                            payload["messages"][0]["content"] += context_str
                    except Exception as e:
                        pass
                
                resp = _req.post(
                    f"{cfg['base_url']}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=300
                )
                resp.raise_for_status()
                result = resp.json()
                return result['choices'][0]['message']['content']
            except Exception as e:
                raise Exception(f"简报生成失败: {str(e)}")
                
        # 允许切换简报提供商并展示当前具体模型
        st.sidebar.divider()
        brief_provider = st.sidebar.selectbox("简报生成商:", list(AI_PROVIDERS.keys()), index=1, help="默认使用 Google (Gemini 2.5 Flash)")
        
        # 实时显示该商调用的具体模型（取配置中第一个）
        current_brief_model = list(AI_PROVIDERS[brief_provider]["models"].values())[0] if AI_PROVIDERS[brief_provider].get("models") else "Unknown"
        st.sidebar.caption(f"当前简报通道模型：`{current_brief_model}`")

        with st.spinner(f"🤖 正在调用 {brief_provider} ({current_brief_model}) 生成当日简报..."):
            try:
                brief_report = fetch_daily_ai_brief(ai_brief_prompt, provider=brief_provider)
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
            m_df = df[df['Display_Name'] == name].sort_values('Date')
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
                        # 强制 tickMinStep=1 避免显示小数刻度
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

    # Top 3 累计消耗
    with st.expander("累计消耗 Top 3", expanded=True):
        top3_cum = df_metrics.nlargest(3, 'Cumulative').copy()
        top3_cum['Rank'] = range(1, len(top3_cum) + 1)
        display_top3 = top3_cum[['Rank', 'Model', 'Cumulative', 'Days_Online', 'Daily_Avg']].copy()
        display_top3.columns = ['排名', '模型', '累计消耗 (B)', '上线天数', '日均消耗 (B)']
        st.dataframe(
            display_top3.style.format({'累计消耗 (B)': '{:.4f}', '日均消耗 (B)': '{:.4f}'}),
            use_container_width=True, hide_index=True
        )

    # 近 7 日增速最快
    with st.expander("近7日增速领先 (Top 3)", expanded=True):
        top3_recent = df_metrics.nlargest(3, 'Recent_7d_Avg').copy()
        top3_recent['Rank'] = range(1, len(top3_recent) + 1)
        display_recent = top3_recent[['Rank', 'Model', 'Recent_7d_Avg']].copy()
        display_recent.columns = ['排名', '模型', '近7日日均消耗 (B)']
        st.dataframe(
            display_recent.style.format({'近7日日均消耗 (B)': '{:.4f}'}),
            use_container_width=True, hide_index=True
        )

    # 加速增长中的模型
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

    # 增速放缓的模型
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

    # 新模型速评（使用百分位排名五级制）
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
                                
                                all_scores_flat = df_latest_bench[df_latest_bench['Metric'] == metric].drop(columns=['Date', 'Metric']).iloc[0].dropna()
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
