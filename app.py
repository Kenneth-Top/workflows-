import streamlit as st
import pandas as pd
import altair as alt
import os

# 设置页面
st.set_page_config(page_title="OpenRouter 模型追踪", layout="wide")

DATA_FILE = "history_database.csv"

st.title("🚀 OpenRouter 模型追踪看板")

# --- 🛠️ 诊断模块：检查文件是否存在 ---
if not os.path.exists(DATA_FILE):
    st.error(f"❌ 严重错误：找不到文件 `{DATA_FILE}`")
    st.warning("当前目录下只有这些文件：")
    st.write(os.listdir("."))
    st.info("💡 建议：请检查 GitHub 仓库里是否真的生成了 CSV 文件？GitHub Action 是否 Push 成功？")
    st.stop() # 停止运行后续代码

# --- 数据加载模块 ---
@st.cache_data(ttl=600)
def load_data():
    try:
        # 尝试读取
        df = pd.read_csv(DATA_FILE)
        
        # 检查是否为空
        if df.empty:
            return None, "CSV 文件存在，但里面是空的（没有数据）。"
            
        # 检查必要的列是否存在
        required_cols = ['Date', 'Model', 'Total_Tokens']
        if not all(col in df.columns for col in required_cols):
            return None, f"列名不匹配！当前列名: {list(df.columns)}"

        # 转换日期格式
        df['Date'] = pd.to_datetime(df['Date'])
        return df, None
        
    except Exception as e:
        return None, f"读取 CSV 失败: {str(e)}"

# 加载数据
df, error_msg = load_data()

# 如果加载失败，打印错误信息
if error_msg:
    st.error(f"❌ 数据加载错误: {error_msg}")
    st.stop()

# --- 如果数据正常，显示内容 ---
st.success(f"✅ 成功加载数据！共 {len(df)} 条记录，更新至 {df['Date'].max().date()}")

# 侧边栏
st.sidebar.title("导航")
page = st.sidebar.radio("选择视图", ["📊 全模型 T+N 对比", "📈 单模型历史详情", "🔍 原始数据检查"])

# === 页面 1: T+N 对比 ===
if page == "📊 全模型 T+N 对比":
    st.subheader("🏆 模型增长曲线对比 (T+N)")
    
    # 计算逻辑
    models = df['Model'].unique()
    tn_data = []

    for m in models:
        # 对每个模型单独处理
        m_df = df[df['Model'] == m].sort_values('Date')
        if m_df.empty: continue
        
        start_date = m_df.iloc[0]['Date']
        
        for _, row in m_df.iterrows():
            day_diff = (row['Date'] - start_date).days
            
            # 我们只保留 T+0 到 T+60 以及最新的数据，减少图表噪点
            # 如果你想看全部，可以把这个 if 去掉
            if day_diff <= 60 or row['Date'] == m_df.iloc[-1]['Date']:
                tn_data.append({
                    'Model': m,
                    'Days_Since_Start': day_diff,
                    'Total_Tokens': row['Total_Tokens'],
                    'Date_Str': row['Date'].strftime('%Y-%m-%d')
                })
    
    if tn_data:
        df_tn = pd.DataFrame(tn_data)
        
        # 绘图
        chart = alt.Chart(df_tn).mark_line(point=True).encode(
            x=alt.X('Days_Since_Start', title='上线天数 (Days)', scale=alt.Scale(type='linear')),
            y=alt.Y('Total_Tokens', title='Token (Billion)'),
            color='Model',
            tooltip=['Model', 'Days_Since_Start', 'Total_Tokens', 'Date_Str']
        ).interactive()
        
        st.altair_chart(chart, use_container_width=True)
    else:
        st.warning("数据不足，无法生成 T+N 图表。")

# === 页面 2: 单模型详情 ===
elif page == "📈 单模型历史详情":
    selected_model = st.selectbox("选择模型", df['Model'].unique())
    m_df = df[df['Model'] == selected_model].sort_values('Date')
    
    # 绘图
    chart = alt.Chart(m_df).mark_line(area=True).encode(
        x=alt.X('Date', title='日期', axis=alt.Axis(format='%m-%d')),
        y=alt.Y('Total_Tokens', title='Token (B)'),
        tooltip=['Date', 'Total_Tokens', 'Prompt', 'Completion']
    ).interactive()
    
    st.altair_chart(chart, use_container_width=True)

# === 页面 3: 原始数据检查 (调试用) ===
else:
    st.subheader("🔍 数据库原始数据")
    st.write("如果图表不对，请检查这里的数据是否正确：")
    st.dataframe(df.sort_values(['Model', 'Date'], ascending=[True, False]), use_container_width=True)
