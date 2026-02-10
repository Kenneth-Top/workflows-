import streamlit as st
import pandas as pd
import altair as alt
import os
import io

# === 1. 基础配置 ===
st.set_page_config(page_title="OpenRouter 模型追踪", layout="wide")
DATA_FILE = "history_database.csv"

# 页面标题
st.title("🚀 OpenRouter 模型追踪看板")
st.caption("单位: Billion Tokens (十亿)")

# 定义页面名称常量
NAV_TN_DAILY = "📊 T+N 横向对比 (每日消耗)"
NAV_CUMULATIVE_COMPARE = "📈 多模型累计增长 (趋势对比)"
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
        
        # 名称清洗：去掉 '/' 前面的厂商名
        df['Display_Name'] = df['Model'].apply(lambda x: x.split('/')[-1] if '/' in x else x)
        
        return df, None
    except Exception as e:
        return None, str(e)

# Excel/CSV 智能导出函数
def get_dataset_download(df, filename_prefix):
    try:
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Sheet1')
        data = output.getvalue()
        file_name = f"{filename_prefix}.xlsx"
        mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        label = "📥 下载 Excel 文件 (.xlsx)"
    except ImportError:
        data = df.to_csv(index=False).encode('utf-8-sig')
        file_name = f"{filename_prefix}.csv"
        mime = "text/csv"
        label = "📥 下载 CSV 文件 (Excel兼容)"
    
    return data, file_name, mime, label

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

all_model_names = df['Display_Name'].unique()

# ========================================================
# 页面 1: T+N 横向对比 (每日消耗)
# ========================================================
if page == NAV_TN_DAILY:
    st.subheader("🏆 模型增长曲线对比 (T+N 每日消耗)")
    st.info("💡 横轴：上线天数 | 纵轴：当日 Token 消耗量")

    selected_names = st.multiselect(
        "选择要对比的模型:", 
        all_model_names, 
        default=all_model_names[:1] 
    )
    
    if selected_names:
        tn_data = []
        standard_ticks = [0, 1, 2, 3, 4, 5, 6, 7, 10, 14, 30, 60]
        final_tick_values = set(standard_ticks)
        
        max_days_global = 0

        for name in selected_names:
            m_df = df[df['Display_Name'] == name].sort_values('Date')
            if m_df.empty: continue
            
            start_date = m_df.iloc[0]['Date']
            st.caption(f"📅 **{name}** 收录起始日: {start_date.strftime('%Y-%m-%d')}")
            
            if len(m_df) > 1: m_df = m_df.iloc[:-1]

            latest_date = m_df.iloc[-1]['Date']
            latest_day_diff = (latest_date - start_date).days
            final_tick_values.add(latest_day_diff)
            
            if latest_day_diff > max_days_global:
                max_days_global = latest_day_diff

            for _, row in m_df.iterrows():
                day_diff = (row['Date'] - start_date).days
                if day_diff in standard_ticks or day_diff == latest_day_diff:
                    tn_data.append({
                        'Model': name,
                        'Days_Since_Start': day_diff,
                        'Total_Tokens': row['Total_Tokens'],
                        'Label': f"T+{day_diff}" if day_diff != latest_day_diff else f"Latest (T+{day_diff})",
                        'Real_Date': row['Date'].strftime('%Y-%m-%d')
                    })
        
        if tn_data:
            df_tn = pd.DataFrame(tn_data)
            
            # 【配色优化】使用 tableau10 高对比配色
            chart = alt.Chart(df_tn).mark_line(
                point=alt.OverlayMarkDef(size=100, filled=True, color="white", strokeWidth=2)
            ).encode(
                x=alt.X(
                    'Days_Since_Start', 
                    title='上线天数 (Days)',
                    axis=alt.Axis(values=list(final_tick_values), labelFontSize=20, titleFontSize=24, grid=True),
                    scale=alt.Scale(domain=[0, max_days_global + 1], clamp=True)
                ),
                y=alt.Y(
                    'Total_Tokens', 
                    title='Total Tokens (Billion)',
                    axis=alt.Axis(labelFontSize=20, titleFontSize=24)
                ),
                # 关键修改：scale=alt.Scale(scheme='tableau10')
                color=alt.Color('Model', 
                                scale=alt.Scale(scheme='tableau10'), 
                                legend=alt.Legend(title="模型名称", orient='bottom')),
                tooltip=['Model', 'Label', 'Total_Tokens', 'Real_Date']
            ).properties(height=500)
            
            st.altair_chart(chart, use_container_width=True)
            
            st.markdown("#### 📋 数据明细")
            df_pivot = df_tn.pivot_table(index='Model', columns='Days_Since_Start', values='Total_Tokens')
            df_pivot.columns = [f"T+{c}" for c in df_pivot.columns]
            st.dataframe(df_pivot.style.format("{:.4f} B"), use_container_width=True)
            
            data, name, mime, label = get_dataset_download(df_pivot.reset_index(), "tn_daily_comparison")
            st.download_button(label=label, data=data, file_name=name, mime=mime)

# ========================================================
# 页面 2: 多模型累计增长 (趋势对比)
# ========================================================
elif page == NAV_CUMULATIVE_COMPARE:
    st.subheader("📈 多模型累计增长对比")
    
    selected_names = st.multiselect(
        "选择要对比的模型:", 
        all_model_names, 
        default=all_model_names[:3] if len(all_model_names) >=3 else all_model_names
    )

    if selected_names:
        plot_data = []
        max_day_plot = 0
        
        cols = st.columns(len(selected_names))
        for idx, name in enumerate(selected_names):
            m_df_temp = df[df['Display_Name'] == name].sort_values('Date')
            if not m_df_temp.empty:
                s_date = m_df_temp.iloc[0]['Date'].strftime('%Y-%m-%d')
                cols[idx].caption(f"📅 **{name}**: {s_date}")

        for name in selected_names:
            m_df = df[df['Display_Name'] == name].sort_values('Date')
            m_df['Cum_Tokens'] = m_df['Total_Tokens'].cumsum()
            if len(m_df) > 1: m_df = m_df.iloc[:-1]
            if m_df.empty: continue

            start_date = m_df.iloc[0]['Date']
            current_max_day = (m_df.iloc[-1]['Date'] - start_date).days
            if current_max_day > max_day_plot:
                max_day_plot = current_max_day

            for _, row in m_df.iterrows():
                day_num = (row['Date'] - start_date).days
                plot_data.append({
                    'Model': name, 'Day': day_num,
                    'Date': row['Date'].strftime('%Y-%m-%d'),
                    'Cumulative_Tokens': row['Cum_Tokens']
                })

        if plot_data:
            df_plot = pd.DataFrame(plot_data)

            # 【配色优化】使用 tableau10 高对比配色
            base = alt.Chart(df_plot).encode(
                x=alt.X('Day', title="上线天数 (Daily)", 
                        scale=alt.Scale(domain=[0, max_day_plot + 2], clamp=True),
                        axis=alt.Axis(labelFontSize=16, titleFontSize=18, grid=True)),
                y=alt.Y('Cumulative_Tokens', title='累计 Token (Billion)', 
                        axis=alt.Axis(labelFontSize=16, titleFontSize=18)),
                # 关键修改：scale=alt.Scale(scheme='tableau10')
                color=alt.Color('Model', 
                                title='模型名称', 
                                scale=alt.Scale(scheme='tableau10'),
                                legend=alt.Legend(orient='bottom')),
                tooltip=['Model', 'Day', 'Date', 'Cumulative_Tokens']
            )
            chart = (base.mark_line(strokeWidth=3) + base.mark_circle(size=60)).properties(height=600)
            
            st.altair_chart(chart, use_container_width=True)

            st.markdown("### 📅 累计数值明细")
            df_pivot = df_plot.pivot_table(index='Day', columns='Model', values='Cumulative_Tokens')
            st.dataframe(df_pivot.style.format("{:.4f} B"), use_container_width=True)

            data, name, mime, label = get_dataset_download(df_pivot.reset_index(), "cumulative_growth")
            st.download_button(label=label, data=data, file_name=name, mime=mime)

# ========================================================
# 页面 3: 单模型每日详情 (趋势分析 + 日期筛选)
# ========================================================
elif page == NAV_DETAIL_DAILY:
    st.subheader("📉 单模型每日详情趋势")
    
    selected_name = st.selectbox("选择模型", all_model_names)
    m_df_full = df[df['Display_Name'] == selected_name].sort_values('Date')
    
    if not m_df_full.empty:
        min_date = m_df_full['Date'].min().date()
        max_date = m_df_full['Date'].max().date()
        st.success(f"📅 **{selected_name}** 数据收录区间: {min_date} 至 {max_date}")

        col_filter1, col_filter2 = st.columns([1, 3])
        with col_filter1:
            date_range = st.date_input(
                "🔎 筛选时间段",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date
            )
        
        if len(date_range) == 2:
            start_filter, end_filter = date_range
            mask = (m_df_full['Date'].dt.date >= start_filter) & (m_df_full['Date'].dt.date <= end_filter)
            m_df = m_df_full.loc[mask]
        else:
            m_df = m_df_full

        if not m_df.empty:
            latest = m_df.iloc[-1]
            c1, c2, c3 = st.columns(3)
            c1.metric("选段最新日期", latest['Date'].strftime('%Y-%m-%d'))
            c2.metric("当日消耗", f"{latest['Total_Tokens']:.4f} B")
            if latest.get('Reasoning', 0) > 0 and latest.get('Completion', 0) > 0:
                ratio = (latest['Reasoning'] / latest['Completion']) * 100
                c3.metric("Reasoning 占比", f"{ratio:.1f}%")
            else:
                c3.metric("Prompt Tokens", f"{latest['Prompt']:.4f} B")

            chart = alt.Chart(m_df).mark_line(point=True).encode(
                x=alt.X('Date', title='日期', axis=alt.Axis(format='%m-%d')),
                y=alt.Y('Total_Tokens', title='Token (Billion)'),
                tooltip=['Date', 'Total_Tokens', 'Prompt', 'Completion']
            )
            
            st.altair_chart(chart, use_container_width=True)
            
            display_cols = ['Date', 'Total_Tokens', 'Prompt', 'Completion', 'Reasoning']
            valid_cols = [c for c in display_cols if c in m_df.columns]
            st.dataframe(m_df[valid_cols].sort_values('Date', ascending=False).style.format({'Total_Tokens':'{:.4f}'}), use_container_width=True)

            data, name, mime, label = get_dataset_download(m_df[valid_cols], f"{selected_name}_daily")
            st.download_button(label=label, data=data, file_name=name, mime=mime)
        else:
            st.warning("⚠️ 所选时间段内无数据。")

# ========================================================
# 页面 4: 原始数据检查
# ========================================================
elif page == NAV_RAW_DATA:
    st.subheader("🔍 数据库原始数据")
    
    st.markdown("#### 💾 全量数据下载")
    data, name, mime, label = get_dataset_download(df, "full_history_database")
    st.download_button(label=label, data=data, file_name=name, mime=mime)
    
    st.divider()
    
    check_name = st.selectbox("选择要检查的模型:", all_model_names)
    filtered_df = df[df['Display_Name'] == check_name].sort_values('Date', ascending=False)
    
    st.dataframe(
        filtered_df.style.format({
            'Prompt': '{:.6f} B', 'Completion': '{:.6f} B', 
            'Reasoning': '{:.6f} B', 'Total_Tokens': '{:.6f} B'
        }), use_container_width=True
    )
