import streamlit as st
import pandas as pd
import altair as alt
import os

# 设置页面
st.set_page_config(page_title="OpenRouter 模型追踪", layout="wide")

DATA_FILE = "history_database.csv"

st.title("🚀 OpenRouter 模型追踪看板")

# --- 🛠️ 诊断模块 ---
if not os.path.exists(DATA_FILE):
    st.error(f"❌ 严重错误：找不到文件 `{DATA_FILE}`")
    st.info("💡 请检查 GitHub Action 是否成功运行并生成了 CSV 文件。")
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

# === 页面 1: T+N 对比 (已增加筛选功能) ===
if page == "📊 T+N 横向对比":
    st.subheader("🏆 模型增长曲线对比 (T+N)")
    
    # 1. 筛选器
    all_models = df['Model'].unique()
    selected_models = st.multiselect("选择要对比的模型:", all_models, default=all_models)
    
    if not selected_models:
        st.warning("请至少选择一个模型。")
    else:
        # 2. 计算逻辑
        tn_data = []
        for m in selected_models:
            m_df = df[df['Model'] == m].sort_values('Date')
            if m_df.empty: continue
            
            start_date = m_df.iloc[0]['Date']
            latest_date = m_df.iloc[-1]['Date']
            
            for _, row in m_df.iterrows():
                day_diff = (row['Date'] - start_date).days
                
                # 只保留关键节点 + 最新数据 (减少图表杂乱)
                # 如果你想看每天的连线，可以去掉 day_diff <= 60 的限制
                is_latest = (row['Date'] == latest_date)
                if day_diff <= 60 or is_latest:
                    tn_data.append({
                        'Model': m,
                        'Days_Since_Start': day_diff,
                        'Total_Tokens': row['Total_Tokens'],
                        'Label': f"T+{day_diff}" if not is_latest else "Latest"
                    })
        
        if tn_data:
            df_tn = pd.DataFrame(tn_data)
            
            # 3. 绘图 (修复了 mark_line 报错)
            chart = alt.Chart(df_tn).mark_line(point=True).encode(
                x=alt.X('Days_Since_Start', title='上线天数 (Days)', scale=alt.Scale(type='linear')),
                y=alt.Y('Total_Tokens', title='Token (Billion)'),
                color='Model',
                tooltip=['Model', 'Days_Since_Start', 'Total_Tokens', 'Label']
            ).interactive()
            
            st.altair_chart(chart, use_container_width=True)
            
            # 4. 下方表格 (透视表)
            st.markdown("#### 📋 数据明细")
            # 仅展示选中的模型
            df_pivot = df_tn.pivot_table(index='Model', columns='Days_Since_Start', values='Total_Tokens')
            st.dataframe(df_pivot.style.format("{:.4f}"), use_container_width=True)

# === 页面 2: 单模型详情 (修复了 mark_line 报错) ===
elif page == "📈 单模型历史详情":
    selected_model = st.selectbox("选择模型", df['Model'].unique())
    m_df = df[df['Model'] == selected_model].sort_values('Date')
    
    # 顶部指标
    latest = m_df.iloc[-1]
    col1, col2, col3 = st.columns(3)
    col1.metric("最新日期", latest['Date'].strftime('%Y-%m-%d'))
    col2.metric("Total Tokens", f"{latest['Total_Tokens']:.4f} B")
    col3.metric("Prompt Tokens", f"{latest['Prompt']:.4f} B")

    # 绘图 (修复报错：去掉了 area=True，改用 mark_area 或 mark_line)
    # 这里用 mark_line + point 看起来更清晰
    chart = alt.Chart(m_df).mark_line(point=True).encode(
        x=alt.X('Date', title='日期', axis=alt.Axis(format='%m-%d')),
        y=alt.Y('Total_Tokens', title='Token (Billion)'),
        tooltip=['Date', 'Total_Tokens', 'Prompt', 'Completion', 'Reasoning']
    ).interactive()
    
    st.altair_chart(chart, use_container_width=True)

# === 页面 3: 原始数据检查 (已改为单模型筛选) ===
else:
    st.subheader("🔍 数据库原始数据")
    
    check_model = st.selectbox("选择要检查的模型:", df['Model'].unique())
    
    # 只显示该模型的数据，并按日期倒序
    filtered_df = df[df['Model'] == check_model].sort_values('Date', ascending=False)
    
    st.dataframe(
        filtered_df.style.format({
            'Prompt': '{:.6f}',
            'Completion': '{:.6f}',
            'Reasoning': '{:.6f}',
            'Total_Tokens': '{:.6f}'
        }), 
        use_container_width=True
    )
