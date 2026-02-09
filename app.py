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
    st.subheader("🏔️ 单模型历史累计增长 & 阶段增量")
    st.info("💡 **彩色块**: 本阶段新增消耗 | **灰色底座**: 之前积累的历史存量")

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
        
        if not m_df.empty:
            start_date = m_df.iloc[0]['Date']
            latest_date = m_df.iloc[-1]['Date']
            latest_day = (latest_date - start_date).days

            # 筛选关键节点
            standard_ticks = [0, 1, 2, 3, 4, 5, 6, 7, 10, 14, 30, 60]
            
            plot_data = []
            previous_cum = 0 
            
            for _, row in m_df.iterrows():
                day = (row['Date'] - start_date).days
                
                if day in standard_ticks or day == latest_day:
                    current_cum = row['Cum_Tokens']
                    increment = current_cum - previous_cum
                    
                    plot_data.append({
                        'Label': f"T+{day}" if day != latest_day else f"Latest (T+{day})",
                        'Day_Num': day, 
                        'Date': row['Date'].strftime('%Y-%m-%d'),
                        'Total_Cum': current_cum,    
                        'Base_Cum': previous_cum,    
                        'Increment': increment       
                    })
                    previous_cum = current_cum

            df_plot = pd.DataFrame(plot_data)

            # === 3. 绘图 (修复颜色 + 修复字体) ===
            
            # 基础 X 轴
            base = alt.Chart(df_plot).encode(
                x=alt.X('Label', sort=alt.EncodingSortField(field="Day_Num", order='ascending'), 
                        title="时间节点 (T+N)",
                        axis=alt.Axis(labelFontSize=20, labelFontWeight='bold', labelAngle=0, titleFontSize=24))
            )

            # A层: 灰色底座 (存量)
            bar_base = base.mark_bar(color="#E0E0E0", size=50).encode(
                y=alt.Y('Base_Cum', title='累计 Token (Billion)', 
                        axis=alt.Axis(labelFontSize=20, titleFontSize=24)),
                tooltip=['Label', 'Base_Cum']
            )

            # B层: 彩色增量块 (修复点：让每个柱子颜色不一样！)
            bar_inc = base.mark_bar(size=50).encode(
                y=alt.Y('Base_Cum'), 
                y2=alt.Y2('Total_Cum'), 
                # 【关键修改】颜色根据 Label 变化，不再是单一蓝色
                color=alt.Color('Label', legend=None, scale=alt.Scale(scheme='tableau20')), 
                tooltip=[
                    alt.Tooltip('Label', title='节点'),
                    alt.Tooltip('Increment', title='本阶段新增(B)', format='.4f'),
                    alt.Tooltip('Date', title='日期')
                ]
            )

            # C层: 折线
            line = base.mark_line(color="#333333", strokeWidth=4).encode(
                y='Total_Cum'
            )

            # D层: 红色数据点
            points = base.mark_point(
                filled=True, fill="#FF4B4B", color="#FFFFFF", strokeWidth=2, size=200
            ).encode(
                y='Total_Cum',
                tooltip=['Label', 'Total_Cum', 'Increment']
            )

            final_chart = (bar_base + bar_inc + line + points).properties(
                height=600,
                title=alt.TitleParams(text=f"{target_model} 阶梯增长图", fontSize=24)
            ).interactive()

            st.altair_chart(final_chart, use_container_width=True)

            # 4. 下方表格 (修复点：移除了报错的 gradient 样式)
            st.markdown("### 📊 阶段增量详情表")
            
            table_data = df_plot[['Label', 'Date', 'Total_Cum', 'Increment']].copy()
            table_data.columns = ['节点', '统计日期', '历史累计总量 (B)', '本阶段新增量 (B)']
            
            # 直接显示，不加 fancy 的背景色，防止报错
            st.dataframe(
                table_data.style.format({
                    '历史累计总量 (B)': '{:.4f}', 
                    '本阶段新增量 (B)': '{:.4f}'
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







