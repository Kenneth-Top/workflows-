import streamlit as st
import pandas as pd
import altair as alt
import os

st.set_page_config(page_title="OpenRouter 模型追踪", layout="wide")

DATA_FILE = "history_database.csv"

st.title("🚀 OpenRouter 模型追踪看板")
st.caption("单位: Billion Tokens (十亿)")  # <--- 新增全局单位说明

# --- 🛠️ 诊断模块 ---
if not os.path.exists(DATA_FILE):
    st.error(f"❌ 找不到文件 `{DATA_FILE}`，请等待爬虫运行。")
    st.stop()

# --- 数据加载 ---
@st.cache_data(ttl=600)
def load_data():
    try:
        df = pd.read_csv(DATA_FILE)
        if df.empty: return None, "CSV 文件为空"
        df['Date'] = pd.to_datetime(df['Date'])
        return df, None
    except Exception as e:
        return None, str(e)

df, error = load_data()
if error:
    st.error(f"数据加载失败: {error}")
    st.stop()

st.sidebar.title("导航")
page = st.sidebar.radio("选择视图", [
    "📊 T+N 横向对比 (每日消耗)", 
    "📈 单模型累积增长 (历史总量)",   
    "📉 单模型每日详情 (趋势分析)",   
    "🔍 原始数据检查"
])

# === 页面 1: T+N 对比 ===
if page == "📊 T+N 横向对比":
    st.subheader("🏆 模型增长曲线对比 (T+N)")
    st.info("💡 横轴：上线天数 | 纵轴：Token 总量 (Billion)")

    # 1. 筛选器
    all_models = df['Model'].unique()
    selected_models = st.multiselect(
        "选择要对比的模型:", 
        all_models, 
        default=all_models[:1] 
    )
    
    if selected_models:
        # 2. 数据准备 & 刻度计算
        tn_data = []
        # 定义标准刻度列表
        standard_ticks = [0, 1, 2, 3, 4, 5, 6, 7, 10, 14, 30, 60]
        # 用集合来存储最终要显示的 X 轴刻度 (会自动去重)
        final_tick_values = set(standard_ticks)

        for m in selected_models:
            m_df = df[df['Model'] == m].sort_values('Date')
            if m_df.empty: continue
            
            # === 逻辑：切掉今天，只取到昨天 ===
            if len(m_df) > 1:
                m_df = m_df.iloc[:-1]

            start_date = m_df.iloc[0]['Date']
            latest_date = m_df.iloc[-1]['Date']
            
            # 计算该模型的 Latest 是第几天，并加入到刻度列表里
            latest_day_diff = (latest_date - start_date).days
            final_tick_values.add(latest_day_diff)

            for _, row in m_df.iterrows():
                day_diff = (row['Date'] - start_date).days
                
                # 只保留标准节点 OR 最新节点
                if day_diff in standard_ticks or day_diff == latest_day_diff:
                    tn_data.append({
                        'Model': m,
                        'Days_Since_Start': day_diff,
                        'Total_Tokens': row['Total_Tokens'],
                        'Label': f"T+{day_diff}" if day_diff != latest_day_diff else f"Latest (T+{day_diff})",
                        'Real_Date': row['Date'].strftime('%Y-%m-%d')
                    })
        
        if tn_data:
            df_tn = pd.DataFrame(tn_data)
            
            # 3. 强力绘图 (大字体、粗线条)
            chart = alt.Chart(df_tn).mark_line(
                point=alt.OverlayMarkDef(size=100, filled=True, color="white", strokeWidth=2) # 点放大，白芯
            ).encode(
                x=alt.X(
                    'Days_Since_Start', 
                    title='上线天数 (Days)',
                    # 核心修改：指定只显示这些刻度，并加大加粗
                    axis=alt.Axis(
                        values=list(final_tick_values), # 强制只显示 T+N 和 Latest 的数字
                        labelFontSize=30,    # 刻度数字大小
                        labelFontWeight='bold', # 刻度加粗
                        titleFontSize=36,    # 标题大小
                        titleFontWeight='bold', # 标题加粗
                        grid=True            # 显示网格辅助看线
                    ),
                    scale=alt.Scale(type='linear') # 保持线性间距 (0-30短, 30-60长)
                ),
                y=alt.Y(
                    'Total_Tokens', 
                    title='Total Tokens (Billion)',
                    # 核心修改：Y轴也加大加粗
                    axis=alt.Axis(
                        labelFontSize=30,
                        labelFontWeight='bold',
                        titleFontSize=36,
                        titleFontWeight='bold'
                    )
                ),
                color=alt.Color('Model', legend=alt.Legend(
                    title="模型名称",
                    titleFontSize=30,
                    labelFontSize=28,
                    labelFontWeight='bold',
                    orient='bottom' # 图例放到底部，给图表留更宽的空间
                )),
                tooltip=['Model', 'Label', 'Total_Tokens', 'Real_Date']
            ).properties(
                height=500 # 增加图表高度，让纵轴更舒展
            ).interactive()
            
            st.altair_chart(chart, use_container_width=True)
            
            # 4. 表格
            st.markdown("#### 📋 数据明细")
            df_pivot = df_tn.pivot_table(index='Model', columns='Days_Since_Start', values='Total_Tokens')
            # 列名重命名，让表格也显示 T+
            df_pivot.columns = [f"T+{c}" for c in df_pivot.columns]
            st.dataframe(df_pivot.style.format("{:.4f} B"), use_container_width=True)

# ========================================================
# 📋 页面 2: 单模型累积增长 (增量堆积版)
# ========================================================

elif page == "📈 单模型累积增长 (历史总量)":
    st.subheader("🏔️ 单模型全生命周期堆叠图 (Daily Full Stack)")
    st.info("💡 **高能预警**：这里展示了每一天的增量如何一层层堆叠成今天的总量。每一层颜色代表那一天的贡献。")

    # 1. 选择模型
    all_models = df['Model'].unique()
    target_model = st.selectbox("选择模型:", all_models)

    if target_model:
        # 2. 数据处理
        m_df = df[df['Model'] == target_model].sort_values('Date')
        m_df['Cum_Tokens'] = m_df['Total_Tokens'].cumsum()

        # 切掉今天
        if len(m_df) > 1:
            m_df = m_df.iloc[:-1]
        
        # 数据量检查
        total_days = len(m_df)
        if total_days > 150:
            st.warning(f"⚠️ 当前模型历史数据长达 {total_days} 天，生成全堆叠图可能需要几秒钟，请耐心等待渲染...")

        if not m_df.empty:
            start_date = m_df.iloc[0]['Date']
            
            # --- 第一步：计算每一天的纯增量 ---
            # 这次我们不再筛选 T+N，而是保留每一天！
            daily_increments = []
            previous_cum = 0
            
            for _, row in m_df.iterrows():
                day_num = (row['Date'] - start_date).days
                current_cum = row['Cum_Tokens']
                inc = current_cum - previous_cum
                
                daily_increments.append({
                    'Day': day_num,
                    'Date': row['Date'].strftime('%Y-%m-%d'),
                    'Increment': inc,
                    'Total_Cum': current_cum,
                    'Label': f"Day {day_num}"
                })
                previous_cum = current_cum

            # --- 第二步：构建 N*N 的全量堆叠数据 ---
            # 这是一个计算密集型操作
            stack_plot_data = []
            
            # 限制一下：如果天数太多(比如超过200天)，这种图可能会卡死浏览器
            # 但既然您要求“试试”，我们就全跑
            for i, target_day in enumerate(daily_increments):
                # 针对每一天 (作为X轴的一根柱子)
                # 我们需要把它拆解成 0...i 所有之前的增量 (作为堆叠块)
                
                for j in range(i + 1):
                    source_day = daily_increments[j]
                    
                    stack_plot_data.append({
                        'X_Day': target_day['Day'],      # X轴位置
                        'X_Date': target_day['Date'],    # X轴日期
                        'Total_Height': target_day['Total_Cum'], # 总高度(用于画线)
                        
                        'Comp_Day': source_day['Day'],   # 增量来源哪一天(用于上色)
                        'Comp_Inc': source_day['Increment'], # 增量大小(用于堆叠高度)
                        'Comp_Date': source_day['Date']  # 增量日期
                    })

            df_stack = pd.DataFrame(stack_plot_data)

            # === 3. 绘图 (动态调整版) ===
            
            # 动态计算柱子宽度：天数越少柱子越粗，天数越多柱子越细
            # 最小 2px，最大 50px
            bar_width = max(2, min(50, 800 // total_days))
            
            base = alt.Chart(df_stack).encode(
                x=alt.X('X_Day:Q', title="上线天数 (Daily)",
                        axis=alt.Axis(labelFontSize=16, titleFontSize=18, grid=False))
            )

            # A层: 全彩堆叠柱
            bars = base.mark_bar(size=bar_width).encode(
                y=alt.Y('Comp_Inc', stack='zero', title='累计 Token (Billion)',
                        axis=alt.Axis(labelFontSize=16, titleFontSize=18)),
                # 颜色：使用 continuous 渐变色，因为 discrete 颜色不够用
                color=alt.Color('Comp_Day:Q', 
                                title="增量来源(天)",
                                scale=alt.Scale(scheme='turbo'), # 使用 turbo 这种高对比度彩虹色
                                legend=None), # 隐藏图例，否则会遮住图表
                order=alt.Order('Comp_Day', sort='ascending'), # 保证早期的在下面
                tooltip=[
                    alt.Tooltip('X_Date', title='当前日期'),
                    alt.Tooltip('Total_Height', title='当前总累计(B)', format='.4f'),
                    alt.Tooltip('Comp_Date', title='增量来源日期'),
                    alt.Tooltip('Comp_Inc', title='该层贡献量(B)', format='.4f')
                ]
            )

            # B层: 顶部轮廓线
            # 去重取总高度
            line_data = df_stack[['X_Day', 'Total_Height']].drop_duplicates()
            line = alt.Chart(line_data).mark_line(color="black", strokeWidth=2).encode(
                x='X_Day:Q',
                y='Total_Height'
            )

            final_chart = (bars + line).properties(
                height=600,
                title=alt.TitleParams(text=f"{target_model} 每日全量沉积图", fontSize=24)
            ).interactive()

            st.altair_chart(final_chart, use_container_width=True)

            # 4. 下方显示简单的每日数据表
            st.markdown("### 📅 每日增长明细")
            st.dataframe(
                pd.DataFrame(daily_increments)[['Date', 'Day', 'Total_Cum', 'Increment']].style.format({
                    'Total_Cum': '{:.4f} B',
                    'Increment': '{:.4f} B'
                }),
                use_container_width=True
            )
            
# === 页面 3: 单模型详情 ===
elif page == "📈 单模型历史详情":
    selected_model = st.selectbox("选择模型", df['Model'].unique())
    m_df = df[df['Model'] == selected_model].sort_values('Date')
    
    latest = m_df.iloc[-1]
    col1, col2, col3 = st.columns(3)
    col1.metric("最新日期", latest['Date'].strftime('%Y-%m-%d'))
    col2.metric("Total Tokens", f"{latest['Total_Tokens']:.4f} B")
    
    # 智能显示：如果有推理数据，显示推理占比
    if latest['Reasoning'] > 0:
        ratio = (latest['Reasoning'] / latest['Completion']) * 100
        col3.metric("Reasoning", f"{latest['Reasoning']:.4f} B ({ratio:.1f}%)")
    else:
        col3.metric("Prompt Tokens", f"{latest['Prompt']:.4f} B")

    chart = alt.Chart(m_df).mark_line(point=True).encode(
        x=alt.X('Date', title='日期', axis=alt.Axis(format='%m-%d')),
        y=alt.Y('Total_Tokens', title='Tokens (Billion)'), # <--- 修改单位
        tooltip=['Date', 'Total_Tokens', 'Prompt', 'Completion', 'Reasoning']
    ).interactive()
    
    st.altair_chart(chart, use_container_width=True)

# === 页面 4: 原始数据 ===
else:
    st.subheader("🔍 数据库原始数据")
    check_model = st.selectbox("选择要检查的模型:", df['Model'].unique())
    filtered_df = df[df['Model'] == check_model].sort_values('Date', ascending=False)
    
    # 格式化显示
    st.dataframe(
        filtered_df.style.format({
            'Prompt': '{:.6f} B',
            'Completion': '{:.6f} B',
            'Reasoning': '{:.6f} B',
            'Total_Tokens': '{:.6f} B'
        }), 
        use_container_width=True
    )











