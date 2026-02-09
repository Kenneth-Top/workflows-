import streamlit as st
import pandas as pd
import altair as alt

st.title("大模型 Token 增长趋势看板")

# 1. 读取数据
df = pd.read_csv("streamlit_source.csv")
df['Date'] = pd.to_datetime(df['Date']) # 关键：转为时间格式

# 2. 侧边栏筛选
models = st.multiselect("选择模型", df['Model'].unique(), default=df['Model'].unique())
filtered_df = df[df['Model'].isin(models)]

# 3. 绘制折线图 (X轴使用真实日期，解决了刻度问题)
chart = alt.Chart(filtered_df).mark_line(point=True).encode(
    x=alt.X('Date', title='日期', axis=alt.Axis(format='%m-%d')), # 真实时间轴
    y=alt.Y('Total_Tokens', title='Total Tokens (B)'),
    color='Model',
    tooltip=['Model', 'Date', 'Total_Tokens', 'Day_Index']
).interactive()

st.altair_chart(chart, use_container_width=True)

# 4. 展示最新数据 (Latest)
st.subheader("📊 最新数据 (Yesterday/Today)")
latest_df = df[df['Is_Latest'] == True]
st.dataframe(latest_df[['Model', 'Date', 'Total_Tokens', 'Prompt', 'Completion']])