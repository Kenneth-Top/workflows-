import streamlit as st
import pandas as pd
import altair as alt

st.set_page_config(page_title="OpenRouter 模型追踪", layout="wide")

DATA_FILE = "history_database.csv"

# --- 数据加载函数 ---
@st.cache_data(ttl=600) # 缓存10分钟
def load_data():
    try:
        df = pd.read_csv(DATA_FILE)
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    except FileNotFoundError:
        return None

df = load_data()

if df is None:
    st.error("数据库文件尚未生成，请等待 GitHub Action 运行完成。")
    st.stop()

# --- 侧边栏导航 ---
st.sidebar.title("🚀 模型追踪看板")
page = st.sidebar.radio("选择视图", ["📊 T+N 横向对比 (主页)", "📈 单模型详情页"])

# === 页面 1: T+N 横向对比 ===
if page == "📊 T+N 横向对比 (主页)":
    st.title("🏆 模型增长曲线对比 (T+N)")
    st.markdown("横轴为**天数**，严格按照时间间隔排列 (T+30 和 T+60 之间会有真实距离)。")

    # 1. 计算 T+N 数据
    # 对每个模型，找到最早的 Date 作为 T+0
    models = df['Model'].unique()
    tn_data = []

    for m in models:
        m_df = df[df['Model'] == m].sort_values('Date')
        if m_df.empty: continue
        
        start_date = m_df.iloc[0]['Date']
        
        # 筛选关键节点
        target_days = [0, 1, 2, 3, 4, 5, 6, 7, 10, 14, 30, 60]
        
        # 还要加上最新的一天 (Latest)
        latest_row = m_df.iloc[-1]
        latest_day_diff = (latest_row['Date'] - start_date).days
        
        # 构建绘图数据
        # 我们遍历每一天的数据，如果它正好是目标天数，或者是最后一天，就加入
        for _, row in m_df.iterrows():
            day_diff = (row['Date'] - start_date).days
            
            # 只有当它是关键节点 OR 它是最新一天时，才显示在图上
            is_target = day_diff in target_days
            is_latest = (day_diff == latest_day_diff)
            
            if is_target or is_latest:
                tn_data.append({
                    'Model': m,
                    'Days_Since_Start': day_diff, # X轴：真实天数
                    'Total_Tokens': row['Total_Tokens'],
                    'Label': f"T+{day_diff}" if not is_latest else f"Latest (T+{day_diff})",
                    'Real_Date': row['Date'].strftime('%Y-%m-%d')
                })

    df_tn = pd.DataFrame(tn_data)

    # 2. 绘制 T+N 走势图 (使用 Altair 保证 X 轴是线性时间刻度)
    chart = alt.Chart(df_tn).mark_line(point=True).encode(
        x=alt.X('Days_Since_Start', title='上线天数 (T+N)', scale=alt.Scale(type='linear')), # 线性轴
        y=alt.Y('Total_Tokens', title='Total Tokens (Billion)'),
        color='Model',
        tooltip=['Model', 'Label', 'Total_Tokens', 'Real_Date']
    ).interactive()

    st.altair_chart(chart, use_container_width=True)

    # 3. 下方表格 (透视表)
    st.subheader("📋 T+N 详细数据表")
    # 为了表格好看，我们只展示标准节点
    df_pivot = df_tn[df_tn['Label'].str.contains("Latest") == False].pivot_table(
        index='Model', 
        columns='Days_Since_Start', 
        values='Total_Tokens'
    )
    # 重命名列名为 T+N
    df_pivot.columns = [f"T+{c}" for c in df_pivot.columns]
    
    # 把 Latest 数据拼接到最后一列
    df_latest = df_tn[df_tn['Label'].str.contains("Latest")].set_index('Model')[['Total_Tokens']]
    df_latest.columns = ['Latest_Tokens']
    
    final_table = df_pivot.join(df_latest)
    st.dataframe(final_table.style.format("{:.3f}"), use_container_width=True)

# === 页面 2: 单模型详情页 ===
else:
    st.title("📈 单模型历史走势详情")
    
    selected_model = st.selectbox("选择模型", df['Model'].unique())
    
    # 筛选数据
    m_df = df[df['Model'] == selected_model].sort_values('Date')
    
    # 获取最早记录日期
    start_date = m_df.iloc[0]['Date'].strftime('%Y-%m-%d')
    latest_tokens = m_df.iloc[-1]['Total_Tokens']
    
    col1, col2 = st.columns(2)
    col1.metric("最早记录日期 (T+0)", start_date)
    col2.metric("最新 Token 总量", f"{latest_tokens:.3f} B")
    
    # 1. 历史走势图
    chart_detail = alt.Chart(m_df).mark_line(area=True).encode(
        x=alt.X('Date', title='日期', axis=alt.Axis(format='%Y-%m-%d')),
        y=alt.Y('Total_Tokens', title='Token (B)'),
        tooltip=['Date', 'Total_Tokens', 'Prompt', 'Completion']
    ).properties(height=400)
    
    st.altair_chart(chart_detail, use_container_width=True)
    
    # 2. 详细数据表
    st.subheader(f"{selected_model} - 每日明细")
    st.dataframe(
        m_df[['Date', 'Total_Tokens', 'Prompt', 'Completion', 'Reasoning']].sort_values('Date', ascending=False).style.format({
            'Total_Tokens': '{:.4f}',
            'Prompt': '{:.4f}',
            'Completion': '{:.4f}',
            'Reasoning': '{:.4f}'
        }),
        use_container_width=True
    )
