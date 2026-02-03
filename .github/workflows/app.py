import streamlit as st
import pandas as pd
import plotly.express as px

# 设置网页标题和布局
st.set_page_config(page_title="LLM Token Growth Monitor", layout="wide")

st.title("🤖 OpenRouter 模型数据增长监控")
st.markdown("数据来源：自动爬取 OpenRouter | 更新频率：每日")

# --- 1. 读取数据 ---
@st.cache_data # 缓存数据，避免每次刷新都重读
def load_data():
    # 这里的路径对应你 GitHub 仓库里的文件名
    try:
        df = pd.read_csv("latest_summary.csv")
        return df
    except FileNotFoundError:
        st.error("找不到数据文件，请先运行爬虫脚本。")
        return pd.DataFrame()

df = load_data()

if not df.empty:
    # --- 2. 主仪表盘 (T+0 ~ T+60 表格) ---
    st.subheader("📊 全模型数据汇总表")
    
    # 可以在这里加个简单的搜索框
    search_term = st.text_input("🔍 搜索模型名称:", "")
    if search_term:
        df_display = df[df['Model'].str.contains(search_term, case=False)]
    else:
        df_display = df
        
    # 展示交互式表格
    st.dataframe(
        df_display,
        use_container_width=True,
        height=500,
        hide_index=True
    )

    st.divider()

    # --- 3. 单模型可视化 (详情页) ---
    st.subheader("📈 模型增长曲线详情")
    
    # 下拉选择框
    model_list = df['Model'].unique().tolist()
    selected_model = st.selectbox("选择要查看的模型:", model_list)

    if selected_model:
        # 提取该模型的数据
        model_data = df[df['Model'] == selected_model].iloc[0]
        
        # 把 T+0, T+1... 转成绘图格式
        # 假设列名都是 "T+0", "T+1" 等
        time_cols = [c for c in df.columns if c.startswith("T+")]
        
        # 准备绘图数据
        plot_data = []
        for col in time_cols:
            val = model_data.get(col)
            if pd.notnull(val): # 过滤掉空值
                day_num = int(col.replace("T+", ""))
                plot_data.append({"Day": day_num, "Tokens": val})
        
        df_plot = pd.DataFrame(plot_data).sort_values("Day")

        if not df_plot.empty:
            # 使用 Plotly 画漂亮的折线图
            fig = px.line(df_plot, x="Day", y="Tokens", title=f"{selected_model} Token Growth", markers=True)
            fig.update_layout(xaxis_title="天数 (T+N)", yaxis_title="Total Tokens (百万/十亿)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("该模型暂无有效的时间序列数据。")