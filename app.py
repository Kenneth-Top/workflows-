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
page = st.sidebar.radio("选择视图", ["📊 T+N 横向对比", "📈 单模型历史详情", "🔍 原始数据检查"])

# === 页面 1: T+N 对比 ===
if page == "📊 T+N 横向对比":
    st.subheader("🏆 模型增长曲线对比 (T+N)")
    st.info("💡 横轴：上线天数 | 纵轴：Token 总量 (Billion)") # <--- 补充单位提示
    
    # 1. 筛选器
    all_models = df['Model'].unique()
    # 默认只选中列表里的第一个模型
    selected_models = st.multiselect("选择要对比的模型:", all_models, default=all_models[:1])
    
    if selected_models:
        # 2. 计算逻辑
        tn_data = []
        for m in selected_models:
            m_df = df[df['Model'] == m].sort_values('Date')
            if m_df.empty: continue
            
            start_date = m_df.iloc[0]['Date']
            latest_date = m_df.iloc[-1]['Date']
            
            for _, row in m_df.iterrows():
                day_diff = (row['Date'] - start_date).days
                is_latest = (row['Date'] == latest_date)
                
                # 只保留关键节点 (0,1...60) + 最新数据
                target_days = [0, 1, 2, 3, 4, 5, 6, 7, 10, 14, 30, 60]
                if day_diff in target_days or is_latest:
                    tn_data.append({
                        'Model': m,
                        'Days_Since_Start': day_diff,
                        'Total_Tokens': row['Total_Tokens'],
                        'Label': f"T+{day_diff}" if not is_latest else "Latest"
                    })
        
        if tn_data:
            df_tn = pd.DataFrame(tn_data)
            
            # 3. 绘图
            chart = alt.Chart(df_tn).mark_line(point=True).encode(
                x=alt.X('Days_Since_Start', title='上线天数 (Days)', scale=alt.Scale(type='linear')),
                y=alt.Y('Total_Tokens', title='Total Tokens (Billion)'), # <--- 修改单位
                color='Model',
                tooltip=['Model', 'Days_Since_Start', 'Total_Tokens', 'Label']
            ).interactive()
            
            st.altair_chart(chart, use_container_width=True)
            
            # 4. 表格
            df_pivot = df_tn.pivot_table(index='Model', columns='Days_Since_Start', values='Total_Tokens')
            st.dataframe(df_pivot.style.format("{:.4f} B"), use_container_width=True) # <--- 表格格式化加单位

# === 页面 2: 单模型详情 ===
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

# === 页面 3: 原始数据 ===
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

