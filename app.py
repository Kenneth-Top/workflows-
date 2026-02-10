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

# 定义页面名称常量
NAV_TN_DAILY = "📊 T+N 横向对比 (每日消耗)"
NAV_CUMULATIVE_COMPARE = "📈 多模型累计增长 (趋势对比)" # <--- 改名了
NAV_DETAIL_DAILY = "📉 单模型每日详情 (趋势分析)"
NAV_RAW_DATA = "🔍 原始数据检查"

# === 2. 工具函数 ===

@st.cache_data(ttl=600)
def load_data():
    if not os.path.exists(DATA_FILE):
        return None, f"❌ 找不到文件 `{DATA_FILE}`，请等待爬虫运行。"
    try:
        df = pd.read_csv(DATA_FILE)
        if df.empty: return None, "CSV 文件为空"
        df['Date'] = pd.to_datetime(df['Date'])
        
        # 【优化2】名称清洗：去掉 '/' 前面的厂商名
        # 例如 'deepseek/deepseek-v3' -> 'deepseek-v3'
        df['Display_Name'] = df['Model'].apply(lambda x: x.split('/')[-1] if '/' in x else x)
        
        return df, None
    except Exception as e:
        return None, str(e)

# 【优化3】Excel/CSV 下载转换函数
@st.cache_data
def convert_df(df):
    # 使用 utf-8-sig 编码，防止 Excel 打开中文乱码
    return df.to_csv(index=False).encode('utf-8-sig')

df, error = load_data()
if error:
    st.error(error)
    st.stop()

# === 3. 侧边栏导航 ===
st.sidebar.title("导航")
page = st.sidebar.radio("选择视图", [
    NAV_TN_DAILY,
    NAV_CUMULATIVE_COMPARE,
    NAV_DETAIL_DAILY,
    NAV_RAW_DATA
])

# 获取所有模型列表 (用于筛选)
# 使用 Display_Name 做展示，但逻辑处理时可能还需要 Model 字段来确保唯一性
# 为了方便，我们这里主要用 Display_Name，如果重名风险低的话
all_model_names = df['Display_Name'].unique()
all_models_full = df['Model'].unique()

# ========================================================
# 页面 1: T+N 横向对比 (每日消耗)
# ========================================================
if page == NAV_TN_DAILY:
    st.subheader("🏆 模型增长曲线对比 (T+N 每日消耗)")
    st.info("💡 横轴：上线天数 (仅显示关键节点) | 纵轴：当日 Token 消耗量")

    # 1. 筛选器 (使用短名字)
    selected_names = st.multiselect(
        "选择要对比的模型:", 
        all_model_names, 
        default=all_model_names[:1] 
    )
    
    if selected_names:
        tn_data = []
        standard_ticks = [0, 1, 2, 3, 4, 5, 6, 7, 10, 14, 30, 60]
        final_tick_values = set(standard_ticks)

        for name in selected_names:
            # 通过 Display_Name 找到原始数据
            m_df = df[df['Display_Name'] == name].sort_values('Date')
            if m_df.empty: continue
            
            # 切掉今天
            if len(m_df) > 1:
                m_df = m_df.iloc[:-1]

            start_date = m_df.iloc[0]['Date']
            latest_date = m_df.iloc[-1]['Date']
            latest_day_diff = (latest_date - start_date).days
            final_tick_values.add(latest_day_diff)

            for _, row in m_df.iterrows():
                day_diff = (row['Date'] - start_date).days
                
                if day_diff in standard_ticks or day_diff == latest_day_diff:
                    tn_data.append({
                        'Model': name, # 使用短名字
                        'Days_Since_Start': day_diff,
                        'Total_Tokens': row['Total_Tokens'],
                        'Label': f"T+{day_diff}" if day_diff != latest_day_diff else f"Latest (T+{day_diff})",
                        'Real_Date': row['Date'].strftime('%Y-%m-%d')
                    })
        
        if tn_data:
            df_tn = pd.DataFrame(tn_data)
            
            # 绘图
            chart = alt.Chart(df_tn).mark_line(
                point=alt.OverlayMarkDef(size=100, filled=True, color="white", strokeWidth=2)
            ).encode(
                x=alt.X(
                    'Days_Since_Start', 
                    title='上线天数 (Days)',
                    axis=alt.Axis(
                        values=list(final_tick_values),
                        labelFontSize=20, labelFontWeight='bold',
                        titleFontSize=24, titleFontWeight='bold',
                        grid=True
                    ),
                    # 【优化4】锁定横轴，不显示负数
                    scale=alt.Scale(domainMin=0, nice=False) 
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
            
            # 表格与下载
            st.markdown("#### 📋 数据明细")
            df_pivot = df_tn.pivot_table(index='Model', columns='Days_Since_Start', values='Total_Tokens')
            df_pivot.columns = [f"T+{c}" for c in df_pivot.columns]
            st.dataframe(df_pivot.style.format("{:.4f} B"), use_container_width=True)
            
            # 【优化3】下载按钮
            csv = convert_df(df_pivot.reset_index())
            st.download_button(
                label="📥 下载表格数据 (CSV/Excel)",
                data=csv,
                file_name='tn_daily_comparison.csv',
                mime='text/csv',
            )

# ========================================================
# 页面 2: 多模型累计增长 (重构：趋势对比)
# ========================================================
elif page == NAV_CUMULATIVE_COMPARE:
    st.subheader("📈 多模型累计增长对比")
    st.info("💡 比较不同模型的历史总量增长轨迹。斜率越陡峭，代表该阶段增长越快。")

    # 【优化1】改为多选，支持对比
    selected_names = st.multiselect(
        "选择要对比的模型:", 
        all_model_names, 
        default=all_model_names[:3] if len(all_model_names) >=3 else all_model_names
    )

    if selected_names:
        plot_data = []
        
        for name in selected_names:
            # 数据处理：计算每个模型的累积值
            m_df = df[df['Display_Name'] == name].sort_values('Date')
            
            # 计算累积
            m_df['Cum_Tokens'] = m_df['Total_Tokens'].cumsum()

            # 切掉今天
            if len(m_df) > 1: m_df = m_df.iloc[:-1]
            
            if m_df.empty: continue

            start_date = m_df.iloc[0]['Date']
            
            for _, row in m_df.iterrows():
                day_num = (row['Date'] - start_date).days
                plot_data.append({
                    'Model': name, # 使用短名字
                    'Day': day_num,
                    'Date': row['Date'].strftime('%Y-%m-%d'),
                    'Cumulative_Tokens': row['Cum_Tokens']
                })

        if plot_data:
            df_plot = pd.DataFrame(plot_data)

            # 【优化1 & 4】绘图：多模型对比线图 + 区域填充 (透明度) + 锁定坐标轴
            # 基础图表
            base = alt.Chart(df_plot).encode(
                x=alt.X('Day', title="上线天数 (Daily)", 
                        # 锁定 X 轴不显示负数
                        scale=alt.Scale(domainMin=0, nice=False),
                        axis=alt.Axis(labelFontSize=16, titleFontSize=18, grid=True)),
                y=alt.Y('Cumulative_Tokens', title='累计 Token (Billion)', 
                        axis=alt.Axis(labelFontSize=16, titleFontSize=18)),
                color=alt.Color('Model', title='模型名称', legend=alt.Legend(orient='bottom')),
                tooltip=['Model', 'Day', 'Date', 'Cumulative_Tokens']
            )

            # 线条
            lines = base.mark_line(strokeWidth=3)
            
            # 数据点 (方便看具体位置)
            points = base.mark_circle(size=60)

            # 组合
            chart = (lines + points).properties(
                height=600,
                title=alt.TitleParams(text="累计增长趋势对比", fontSize=24)
            ).interactive()

            st.altair_chart(chart, use_container_width=True)

            # 下方表格
            st.markdown("### 📅 累计数值明细")
            # 透视表：行是天数，列是模型
            df_pivot = df_plot.pivot_table(index='Day', columns='Model', values='Cumulative_Tokens')
            st.dataframe(df_pivot.style.format("{:.4f} B"), use_container_width=True)

            # 【优化3】下载按钮
            csv = convert_df(df_pivot.reset_index())
            st.download_button(
                label="📥 下载累计增长数据 (CSV/Excel)",
                data=csv,
                file_name='cumulative_growth_comparison.csv',
                mime='text/csv',
            )

# ========================================================
# 页面 3: 单模型每日详情 (趋势分析)
# ========================================================
elif page == NAV_DETAIL_DAILY:
    st.subheader("📉 单模型每日详情趋势")
    
    # 使用短名字选择
    selected_name = st.selectbox("选择模型", all_model_names)
    
    # 过滤数据
    m_df = df[df['Display_Name'] == selected_name].sort_values('Date')
    
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
        
        # 数据表
        display_cols = ['Date', 'Total_Tokens', 'Prompt', 'Completion', 'Reasoning']
        # 确保列存在
        valid_cols = [c for c in display_cols if c in m_df.columns]
        st.dataframe(m_df[valid_cols].sort_values('Date', ascending=False).style.format({'Total_Tokens':'{:.4f}'}), use_container_width=True)

        # 【优化3】下载按钮
        csv = convert_df(m_df[valid_cols])
        st.download_button(
            label=f"📥 下载 {selected_name} 每日详情 (CSV/Excel)",
            data=csv,
            file_name=f'{selected_name}_daily_details.csv',
            mime='text/csv',
        )

# ========================================================
# 页面 4: 原始数据检查
# ========================================================
elif page == NAV_RAW_DATA:
    st.subheader("🔍 数据库原始数据")
    
    # 全局下载
    st.markdown("#### 💾 全量数据下载")
    csv_all = convert_df(df)
    st.download_button(
        label="📥 下载完整数据库 (CSV/Excel)",
        data=csv_all,
        file_name='full_history_database.csv',
        mime='text/csv',
    )
    
    st.divider()
    
    check_name = st.selectbox("选择要检查的模型:", all_model_names)
    filtered_df = df[df['Display_Name'] == check_name].sort_values('Date', ascending=False)
    
    st.dataframe(
        filtered_df.style.format({
            'Prompt': '{:.6f} B', 'Completion': '{:.6f} B', 
            'Reasoning': '{:.6f} B', 'Total_Tokens': '{:.6f} B'
        }), use_container_width=True
    )
