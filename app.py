import streamlit as st
import pandas as pd
import altair as alt
import os

# === 1. 基础配置 ===
st.set_page_config(page_title="OpenRouter 模型追踪", layout="wide")
DATA_FILE = "history_database.csv"

# 页面标题
st.title("🚀 OpenRouter 模型追踪看板")
st.caption("单位: Billion Tokens (十亿)")

# 定义页面名称常量 (防止字符串不匹配导致的 Bug)
NAV_TN_DAILY = "📊 T+N 横向对比 (每日消耗)"
NAV_STACK_FULL = "📈 单模型累积增长 (历史总量)"
NAV_DETAIL_DAILY = "📉 单模型每日详情 (趋势分析)"
NAV_RAW_DATA = "🔍 原始数据检查"

# === 2. 数据加载函数 ===
@st.cache_data(ttl=600)
def load_data():
    if not os.path.exists(DATA_FILE):
        return None, f"❌ 找不到文件 `{DATA_FILE}`，请等待爬虫运行。"
    try:
        df = pd.read_csv(DATA_FILE)
        if df.empty: return None, "CSV 文件为空"
        df['Date'] = pd.to_datetime(df['Date'])
        return df, None
    except Exception as e:
        return None, str(e)

df, error = load_data()
if error:
    st.error(error)
    st.stop()

# === 3. 侧边栏导航 ===
st.sidebar.title("导航")
page = st.sidebar.radio("选择视图", [
    NAV_TN_DAILY,
    NAV_STACK_FULL,
    NAV_DETAIL_DAILY,
    NAV_RAW_DATA
])

# 获取所有模型列表 (全局复用)
all_models = df['Model'].unique()

# ========================================================
# 页面 1: T+N 横向对比 (每日消耗) - 修复了逻辑不执行的问题
# ========================================================
if page == NAV_TN_DAILY:
    st.subheader("🏆 模型增长曲线对比 (T+N 每日消耗)")
    st.info("💡 横轴：上线天数 (仅显示关键节点) | 纵轴：当日 Token 消耗量")

    # 1. 筛选器
    selected_models = st.multiselect(
        "选择要对比的模型:", 
        all_models, 
        default=all_models[:1] 
    )
    
    if selected_models:
        tn_data = []
        standard_ticks = [0, 1, 2, 3, 4, 5, 6, 7, 10, 14, 30, 60]
        final_tick_values = set(standard_ticks)

        for m in selected_models:
            m_df = df[df['Model'] == m].sort_values('Date')
            if m_df.empty: continue
            
            # 切掉今天，只取到昨天
            if len(m_df) > 1:
                m_df = m_df.iloc[:-1]

            start_date = m_df.iloc[0]['Date']
            latest_date = m_df.iloc[-1]['Date']
            latest_day_diff = (latest_date - start_date).days
            final_tick_values.add(latest_day_diff)

            for _, row in m_df.iterrows():
                day_diff = (row['Date'] - start_date).days
                
                # 核心过滤：只保留 T+N 列表里的天数，或者最新的一天
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
            
            # 绘图：强制显示 T+N 刻度
            chart = alt.Chart(df_tn).mark_line(
                point=alt.OverlayMarkDef(size=100, filled=True, color="white", strokeWidth=2)
            ).encode(
                x=alt.X(
                    'Days_Since_Start', 
                    title='上线天数 (Days)',
                    axis=alt.Axis(
                        values=list(final_tick_values), # <--- 关键：强制只显示 T+N
                        labelFontSize=20, labelFontWeight='bold',
                        titleFontSize=24, titleFontWeight='bold',
                        grid=True
                    ),
                    scale=alt.Scale(type='linear')
                ),
                y=alt.Y(
                    'Total_Tokens', 
                    title='Total Tokens (Billion)',
                    axis=alt.Axis(labelFontSize=20, labelFontWeight='bold', titleFontSize=24, titleFontWeight='bold')
                ),
                color=alt.Color('Model', legend=alt.Legend(
                    title="模型名称", titleFontSize=20, labelFontSize=16, orient='bottom'
                )),
                tooltip=['Model', 'Label', 'Total_Tokens', 'Real_Date']
            ).properties(height=500).interactive()
            
            st.altair_chart(chart, use_container_width=True)
            
            # 表格透视
            st.markdown("#### 📋 数据明细")
            df_pivot = df_tn.pivot_table(index='Model', columns='Days_Since_Start', values='Total_Tokens')
            df_pivot.columns = [f"T+{c}" for c in df_pivot.columns]
            st.dataframe(df_pivot.style.format("{:.4f} B"), use_container_width=True)

# ========================================================
# 页面 2: 单模型累积增长 (全堆叠版)
# ========================================================
elif page == NAV_STACK_FULL:
    st.subheader("🏔️ 单模型全生命周期堆叠图 (每日全量)")
    st.info("💡 展示每一天的增量如何一层层堆叠成今天的总量。")

    target_model = st.selectbox("选择模型:", all_models)

    if target_model:
        m_df = df[df['Model'] == target_model].sort_values('Date')
        m_df['Cum_Tokens'] = m_df['Total_Tokens'].cumsum()

        if len(m_df) > 1: m_df = m_df.iloc[:-1] # 切掉今天
        
        total_days = len(m_df)
        if total_days > 150:
            st.warning(f"⚠️ 数据长达 {total_days} 天，渲染可能需要几秒钟...")

        if not m_df.empty:
            start_date = m_df.iloc[0]['Date']
            
            # 1. 计算增量
            daily_increments = []
            previous_cum = 0
            for _, row in m_df.iterrows():
                day_num = (row['Date'] - start_date).days
                current_cum = row['Cum_Tokens']
                inc = current_cum - previous_cum
                daily_increments.append({
                    'Day': day_num, 'Date': row['Date'].strftime('%Y-%m-%d'),
                    'Increment': inc, 'Total_Cum': current_cum, 'Label': f"Day {day_num}"
                })
                previous_cum = current_cum

            # 2. 构建堆叠数据
            stack_plot_data = []
            for i, target_day in enumerate(daily_increments):
                for j in range(i + 1):
                    source_day = daily_increments[j]
                    stack_plot_data.append({
                        'X_Day': target_day['Day'], 'X_Date': target_day['Date'],
                        'Total_Height': target_day['Total_Cum'],
                        'Comp_Day': source_day['Day'], 'Comp_Inc': source_day['Increment'],
                        'Comp_Date': source_day['Date']
                    })

            df_stack = pd.DataFrame(stack_plot_data)

            # 3. 绘图
            bar_width = max(2, min(50, 800 // (total_days or 1)))
            base = alt.Chart(df_stack).encode(x=alt.X('X_Day:Q', title="上线天数", axis=alt.Axis(labelFontSize=16, titleFontSize=18, grid=False)))
            
            bars = base.mark_bar(size=bar_width).encode(
                y=alt.Y('Comp_Inc', stack='zero', title='累计 Token (Billion)', axis=alt.Axis(labelFontSize=16, titleFontSize=18)),
                color=alt.Color('Comp_Day:Q', scale=alt.Scale(scheme='turbo'), legend=None),
                order=alt.Order('Comp_Day', sort='ascending'),
                tooltip=['X_Date', 'Total_Height', 'Comp_Date', 'Comp_Inc']
            )
            
            line_data = df_stack[['X_Day', 'Total_Height']].drop_duplicates()
            line = alt.Chart(line_data).mark_line(color="black", strokeWidth=2).encode(x='X_Day:Q', y='Total_Height')

            st.altair_chart((bars + line).properties(height=600).interactive(), use_container_width=True)
            st.dataframe(pd.DataFrame(daily_increments)[['Date', 'Day', 'Total_Cum', 'Increment']].style.format({'Total_Cum':'{:.4f} B','Increment':'{:.4f} B'}), use_container_width=True)

# ========================================================
# 页面 3: 单模型每日详情 (趋势分析)
# ========================================================
elif page == NAV_DETAIL_DAILY:
    st.subheader("📉 单模型每日详情趋势")
    selected_model = st.selectbox("选择模型", all_models)
    m_df = df[df['Model'] == selected_model].sort_values('Date')
    
    if not m_df.empty:
        latest = m_df.iloc[-1]
        col1, col2, col3 = st.columns(3)
        col1.metric("最新日期", latest['Date'].strftime('%Y-%m-%d'))
        col2.metric("当日消耗", f"{latest['Total_Tokens']:.4f} B")
        
        if latest.get('Reasoning', 0) > 0 and latest.get('Completion', 0) > 0:
            ratio = (latest['Reasoning'] / latest['Completion']) * 100
            col3.metric("Reasoning 占比", f"{ratio:.1f}%")
        else:
            col3.metric("Prompt Tokens", f"{latest['Prompt']:.4f} B")

        chart = alt.Chart(m_df).mark_line(point=True).encode(
            x=alt.X('Date', title='日期', axis=alt.Axis(format='%m-%d')),
            y=alt.Y('Total_Tokens', title='Token (Billion)'),
            tooltip=['Date', 'Total_Tokens', 'Prompt', 'Completion']
        ).interactive()
        
        st.altair_chart(chart, use_container_width=True)
        st.dataframe(m_df.sort_values('Date', ascending=False).style.format({'Total_Tokens':'{:.4f}'}), use_container_width=True)

# ========================================================
# 页面 4: 原始数据检查
# ========================================================
elif page == NAV_RAW_DATA:
    st.subheader("🔍 数据库原始数据")
    check_model = st.selectbox("选择要检查的模型:", all_models)
    filtered_df = df[df['Model'] == check_model].sort_values('Date', ascending=False)
    
    st.dataframe(
        filtered_df.style.format({
            'Prompt': '{:.6f} B', 'Completion': '{:.6f} B', 
            'Reasoning': '{:.6f} B', 'Total_Tokens': '{:.6f} B'
        }), use_container_width=True
    )
