import streamlit as st
import pandas as pd
import altair as alt
import os
import io

# === 1. 基础配置 ===
st.set_page_config(page_title="OpenRouter 模型追踪", layout="wide")
DATA_FILE = "history_database.csv"

# 页面标题
st.title("OpenRouter 模型追踪看板")
st.caption("单位: Billion Tokens (十亿)")

# 定义页面名称常量
NAV_TN_DAILY = "T+N 横向对比 (每日消耗)"
NAV_CUMULATIVE_COMPARE = "多模型累计增长 (趋势对比)"
NAV_DETAIL_DAILY = "单模型每日详情 (趋势分析)"
NAV_RAW_DATA = "原始数据检查"
NAV_DAILY_BRIEF = "每日速览"

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
    NAV_DAILY_BRIEF,
    NAV_TN_DAILY,
    NAV_CUMULATIVE_COMPARE,
    NAV_DETAIL_DAILY,
    NAV_RAW_DATA
])

all_model_names = df['Display_Name'].unique()

# 数据概览面板
st.sidebar.divider()
st.sidebar.markdown("#### 📊 数据概览")
st.sidebar.metric("追踪模型数", len(all_model_names))
st.sidebar.caption(
    f"📅 数据区间: {df['Date'].min().strftime('%Y-%m-%d')} ~ {df['Date'].max().strftime('%Y-%m-%d')}"
)

# ========================================================
# 页面 1: T+N 横向对比 (每日消耗)
# ========================================================
if page == NAV_TN_DAILY:
    st.subheader("模型增长曲线对比 (T+N 每日消耗)")
    st.info("横轴：上线天数 | 纵轴：当日 Token 消耗量")

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
    st.subheader("多模型累计增长对比")
    
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
    st.subheader("单模型每日详情趋势")
    
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
    st.subheader("数据库原始数据")
    
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

# ========================================================
# 页面 5: 每日速览与分析
# ========================================================
elif page == NAV_DAILY_BRIEF:
    st.subheader("模型表现速览与分析报告")
    st.caption("基于历史数据的多维度量化分析，所有指标均由数据自动计算生成。")

    # --- 预计算所有模型的指标 ---
    latest_date = df['Date'].max()
    two_weeks_ago = latest_date - pd.Timedelta(days=14)
    seven_days_ago = latest_date - pd.Timedelta(days=7)

    metrics_list = []
    for name in all_model_names:
        m_df = df[df['Display_Name'] == name].sort_values('Date')
        if m_df.empty:
            continue
        # 去掉最后一天（当天未结算数据，和其他页面逻辑保持一致）
        if len(m_df) > 1:
            m_df = m_df.iloc[:-1]
        if m_df.empty:
            continue

        first_date = m_df.iloc[0]['Date']
        last_date = m_df.iloc[-1]['Date']
        days_online = max((last_date - first_date).days, 1)
        cumulative = m_df['Total_Tokens'].sum()
        daily_avg = cumulative / days_online
        peak = m_df['Total_Tokens'].max()

        # 近 7 日增速
        recent_df = m_df[m_df['Date'] >= seven_days_ago]
        recent_days = max(len(recent_df), 1)
        recent_avg = recent_df['Total_Tokens'].sum() / recent_days if not recent_df.empty else 0

        # 增长动量
        momentum = (recent_avg / daily_avg) if daily_avg > 0 else 0

        metrics_list.append({
            'Model': name,
            'First_Date': first_date,
            'Last_Date': last_date,
            'Days_Online': days_online,
            'Cumulative': round(cumulative, 4),
            'Daily_Avg': round(daily_avg, 4),
            'Recent_7d_Avg': round(recent_avg, 4),
            'Momentum': round(momentum, 2),
            'Peak': round(peak, 4),
        })

    df_metrics = pd.DataFrame(metrics_list)

    if df_metrics.empty:
        st.warning("暂无可分析的模型数据。")
        st.stop()

    # 计算百分位排名（供后续模块使用）
    df_metrics['Pct_Rank_DailyAvg'] = df_metrics['Daily_Avg'].rank(pct=True)

    # ============================
    # 模块 A: 近两周新增模型一览
    # ============================
    st.markdown("---")
    st.markdown("### 近两周新增模型一览")
    st.caption(f"统计区间: {two_weeks_ago.strftime('%Y-%m-%d')} ~ {latest_date.strftime('%Y-%m-%d')}")

    new_models_df = df_metrics[df_metrics['First_Date'] >= two_weeks_ago].sort_values('First_Date', ascending=False)

    if new_models_df.empty:
        st.info("过去两周内没有新上线的模型。")
    else:
        st.markdown(f"过去两周共上线 **{len(new_models_df)}** 个新模型。")
        display_new = new_models_df[['Model', 'First_Date', 'Days_Online', 'Cumulative', 'Daily_Avg']].copy()
        display_new.columns = ['模型名称', '上线日期', '上线天数', '累计消耗 (B)', '日均消耗 (B)']
        display_new['上线日期'] = display_new['上线日期'].dt.strftime('%Y-%m-%d')
        st.dataframe(
            display_new.style.format({'累计消耗 (B)': '{:.4f}', '日均消耗 (B)': '{:.4f}'}),
            use_container_width=True, hide_index=True
        )

    # ============================
    # 模块 B (原 D): 新模型累计增长对比
    # ============================
    if not new_models_df.empty:
        st.markdown("---")
        st.markdown("### 新模型累计增长对比")

        new_model_names = new_models_df['Model'].tolist()
        plot_new = []
        max_day_new = 0

        for name in new_model_names:
            m_df = df[df['Display_Name'] == name].sort_values('Date')
            m_df['Cum_Tokens'] = m_df['Total_Tokens'].cumsum()
            if len(m_df) > 1:
                m_df = m_df.iloc[:-1]
            if m_df.empty:
                continue
            start_date = m_df.iloc[0]['Date']
            current_max = (m_df.iloc[-1]['Date'] - start_date).days
            if current_max > max_day_new:
                max_day_new = current_max
            for _, row in m_df.iterrows():
                day_n = (row['Date'] - start_date).days
                plot_new.append({
                    'Model': name, 'Day': day_n,
                    'Date': row['Date'].strftime('%Y-%m-%d'),
                    'Cumulative_Tokens': row['Cum_Tokens']
                })

        if plot_new:
            df_plot_new = pd.DataFrame(plot_new)
            base_new = alt.Chart(df_plot_new).encode(
                x=alt.X('Day', title='上线天数',
                        scale=alt.Scale(domain=[0, max_day_new + 2], clamp=True),
                        axis=alt.Axis(labelFontSize=14, titleFontSize=16, grid=True)),
                y=alt.Y('Cumulative_Tokens', title='累计 Token (Billion)',
                        axis=alt.Axis(labelFontSize=14, titleFontSize=16)),
                color=alt.Color('Model', title='模型',
                                scale=alt.Scale(scheme='tableau10'),
                                legend=alt.Legend(orient='bottom')),
                tooltip=['Model', 'Day', 'Date', 'Cumulative_Tokens']
            )
            chart_new = (base_new.mark_line(strokeWidth=3) + base_new.mark_circle(size=60)).properties(height=500)
            st.altair_chart(chart_new, use_container_width=True)
        else:
            st.info("新模型暂无足够数据绘制趋势图。")

    # ============================
    # 模块 C: 分析摘要 (表格化)
    # ============================
    st.markdown("---")
    st.markdown("### 综合分析摘要")
    st.caption(f"分析基准日: {latest_date.strftime('%Y-%m-%d')}")

    # Top 3 累计消耗
    with st.expander("累计消耗 Top 3", expanded=True):
        top3_cum = df_metrics.nlargest(3, 'Cumulative').copy()
        top3_cum['Rank'] = range(1, len(top3_cum) + 1)
        display_top3 = top3_cum[['Rank', 'Model', 'Cumulative', 'Days_Online', 'Daily_Avg']].copy()
        display_top3.columns = ['排名', '模型', '累计消耗 (B)', '上线天数', '日均消耗 (B)']
        st.dataframe(
            display_top3.style.format({'累计消耗 (B)': '{:.4f}', '日均消耗 (B)': '{:.4f}'}),
            use_container_width=True, hide_index=True
        )

    # 近 7 日增速最快
    with st.expander("近7日增速领先 (Top 3)", expanded=True):
        top3_recent = df_metrics.nlargest(3, 'Recent_7d_Avg').copy()
        top3_recent['Rank'] = range(1, len(top3_recent) + 1)
        display_recent = top3_recent[['Rank', 'Model', 'Recent_7d_Avg']].copy()
        display_recent.columns = ['排名', '模型', '近7日日均消耗 (B)']
        st.dataframe(
            display_recent.style.format({'近7日日均消耗 (B)': '{:.4f}'}),
            use_container_width=True, hide_index=True
        )

    # 加速增长中的模型
    with st.expander("正在加速增长 (动量 > 1.2)", expanded=True):
        accel = df_metrics[df_metrics['Momentum'] >= 1.2].sort_values('Momentum', ascending=False)
        if not accel.empty:
            accel['Growth_Pct'] = (accel['Momentum'] - 1) * 100
            display_accel = accel[['Model', 'Momentum', 'Growth_Pct']].head(5).copy()
            display_accel.columns = ['模型', '动量值', '近期增速超出均值 (%)']
            st.dataframe(
                display_accel.style.format({'动量值': '{:.2f}', '近期增速超出均值 (%)': '+{:.0f}%'}),
                use_container_width=True, hide_index=True
            )
        else:
            st.info("暂无明显加速增长的模型。")

    # 增速放缓的模型
    with st.expander("增速放缓关注 (动量 < 0.8)", expanded=True):
        decel = df_metrics[(df_metrics['Momentum'] <= 0.8) & (df_metrics['Days_Online'] >= 7)].sort_values('Momentum')
        if not decel.empty:
            decel['Slowdown_Pct'] = (1 - decel['Momentum']) * 100
            display_decel = decel[['Model', 'Momentum', 'Slowdown_Pct']].head(5).copy()
            display_decel.columns = ['模型', '动量值', '近期增速低于均值 (%)']
            st.dataframe(
                display_decel.style.format({'动量值': '{:.2f}', '近期增速低于均值 (%)': '-{:.0f}%'}),
                use_container_width=True, hide_index=True
            )
        else:
            st.info("暂无明显增速放缓的模型。")

    # 新模型速评（使用百分位排名五级制）
    if not new_models_df.empty:
        with st.expander("新模型初期表现评级", expanded=True):
            rating_data = []
            for row in new_models_df.itertuples():
                pct_rank = row.Pct_Rank_DailyAvg
                if pct_rank >= 0.90:
                    tier, desc = "S · 头部水平", f"超过 {pct_rank*100:.0f}% 模型"
                elif pct_rank >= 0.75:
                    tier, desc = "A · 表现优异", f"超过 {pct_rank*100:.0f}% 模型"
                elif pct_rank >= 0.50:
                    tier, desc = "B · 中等水平", "日均 > 中位数"
                elif pct_rank >= 0.25:
                    tier, desc = "C · 低于预期", f"仅超 {pct_rank*100:.0f}% 模型"
                else:
                    tier, desc = "D · 起步缓慢", f"后 {(1-pct_rank)*100:.0f}% 分位"
                
                rating_data.append({
                    '模型': row.Model,
                    '上线日期': row.First_Date.strftime('%m-%d'),
                    '日均消耗 (B)': row.Daily_Avg,
                    '评级': tier,
                    '说明': desc
                })
            
            df_rating = pd.DataFrame(rating_data)
            st.dataframe(
                df_rating.style.format({'日均消耗 (B)': '{:.4f}'}),
                use_container_width=True, hide_index=True
            )

    # ============================
    # 模块 D (原 B): 全模型表现排名 (移至最后)
    # ============================
    st.markdown("---")
    st.markdown("### 全模型表现排名 (Top 15)")

    RANK_OPTIONS = {
        '累计总量': 'Cumulative',
        '日均消耗': 'Daily_Avg',
        '近7日增速': 'Recent_7d_Avg',
        '增长动量': 'Momentum',
        '峰值消耗': 'Peak',
        '上线天数': 'Days_Online'
    }
    col_rank1, col_rank2 = st.columns([1, 3])
    with col_rank1:
        rank_label = st.selectbox("选择排名维度", list(RANK_OPTIONS.keys()))
    rank_col = RANK_OPTIONS[rank_label]

    df_ranked = df_metrics.sort_values(rank_col, ascending=False).head(15).reset_index(drop=True)
    df_ranked.index = df_ranked.index + 1

    chart_rank = alt.Chart(df_ranked).mark_bar(
        cornerRadiusTopLeft=4, cornerRadiusTopRight=4
    ).encode(
        x=alt.X('Model', sort='-y', title='模型',
                axis=alt.Axis(labelAngle=-45, labelFontSize=11)),
        y=alt.Y(rank_col, title=rank_label,
                axis=alt.Axis(labelFontSize=14, titleFontSize=16)),
        color=alt.Color('Model', legend=None, scale=alt.Scale(scheme='tableau10')),
        tooltip=['Model', alt.Tooltip(rank_col, title=rank_label, format='.4f')]
    ).properties(height=400)
    st.altair_chart(chart_rank, use_container_width=True)

    display_ranked = df_ranked[['Model', 'Days_Online', 'Cumulative', 'Daily_Avg', 'Recent_7d_Avg', 'Momentum', 'Peak']].copy()
    display_ranked.columns = ['模型', '上线天数', '累计 (B)', '日均 (B)', '近7日均 (B)', '动量', '峰值 (B)']

    def highlight_momentum(val):
        if isinstance(val, (int, float)):
            if val >= 1.2:
                return 'background-color: #d4edda; color: #155724'
            elif val <= 0.8:
                return 'background-color: #f8d7da; color: #721c24'
        return ''

    st.dataframe(
        display_ranked.style
            .format({'累计 (B)': '{:.4f}', '日均 (B)': '{:.4f}', '近7日均 (B)': '{:.4f}', '动量': '{:.2f}', '峰值 (B)': '{:.4f}'})
            .map(highlight_momentum, subset=['动量']),
        use_container_width=True, hide_index=False
    )
    st.caption("动量 > 1.2 (绿色背景) = 加速增长 · 动量 < 0.8 (红色背景) = 增速放缓")

    # ============================
    # 模块 E: 近两周新模型动态 (RSS)
    # ============================
    st.markdown("---")
    st.markdown("### 近两周新模型动态")

    if new_models_df.empty:
        st.info("近两周内无新上线模型，暂无相关新闻可检索。")
    else:
        import re as _re
        import requests as _requests


        # ── AI 专业媒体 RSS 源 ──
        RSS_FEEDS = [
            ("TechCrunch AI",   "https://techcrunch.com/category/artificial-intelligence/feed/"),
            ("VentureBeat AI",  "https://venturebeat.com/category/ai/feed/"),
            ("The Verge AI",    "https://www.theverge.com/rss/ai-artificial-intelligence/index.xml"),
            ("Ars Technica",    "https://feeds.arstechnica.com/arstechnica/technology-lab"),
        ]

        # ── 从模型全名提取品牌名 ──
        def extract_brand(full_name):
            base = full_name.split('/')[-1]
            return base.split('-')[0].lower()

        # ── 自动打标签：返回匹配的品牌名，无匹配返回 None ──
        def detect_tag(text, brand_label_map):
            text_lower = text.lower()
            for brand, label in brand_label_map.items():
                if brand in text_lower:
                    return label
            return None

        # ── 构建品牌名标签映射（RSS 来源本身是 AI 媒体，不需要过滤短词）──
        model_names_raw = new_models_df['Model'].tolist()
        brand_label_map = {}
        for full_name in model_names_raw:
            brand = extract_brand(full_name)
            if brand and len(brand) >= 3:
                brand_label_map[brand] = brand
        # 补充常见厂商别名，提高召回率
        ALIAS_MAP = {"gpt": "openai", "o1": "openai", "o3": "openai", "step": "stepfun"}
        for short, full in ALIAS_MAP.items():
            if short in brand_label_map:
                brand_label_map[full] = full

        cutoff = latest_date - pd.Timedelta(days=14)
        cutoff_str = cutoff.strftime('%Y-%m-%d')

        # ── 翻译函数（缓存 24 小时）──
        @st.cache_data(ttl=86400)
        def translate_zh(text):
            if not text or not text.strip():
                return text
            try:
                from deep_translator import GoogleTranslator
                return GoogleTranslator(source='en', target='zh-CN').translate(text[:500])
            except Exception:
                return text

        # ── 抓取并解析 RSS（缓存 3 小时，使用 feedparser）──
        @st.cache_data(ttl=10800)
        def fetch_rss_articles(cutoff_str):
            import feedparser
            cutoff_dt = pd.Timestamp(cutoff_str, tz='UTC')
            results = []
            for feed_name, feed_url in RSS_FEEDS:
                try:
                    feed = feedparser.parse(feed_url)
                    for entry in feed.entries:
                        title = entry.get('title', '').strip()
                        link  = entry.get('link', '#')
                        # 摘要：优先 summary，其次 content
                        desc_raw = entry.get('summary', '') or ''
                        if not desc_raw and entry.get('content'):
                            desc_raw = entry['content'][0].get('value', '')
                        import re as _re2
                        desc = _re2.sub(r'<[^>]+>', '', desc_raw).strip()[:300]
                        # 发布时间：feedparser 统一解析为 time.struct_time
                        pub_parsed = entry.get('published_parsed') or entry.get('updated_parsed')
                        if pub_parsed:
                            pub_dt = pd.Timestamp(*pub_parsed[:6], tz='UTC')
                        else:
                            pub_dt = pd.Timestamp.now(tz='UTC')
                        if pub_dt < cutoff_dt:
                            continue
                        results.append({
                            'title': title, 'desc': desc, 'link': link,
                            'source': feed_name, 'date': pub_dt.strftime('%Y-%m-%d'),
                        })
                except Exception:
                    continue
            results.sort(key=lambda x: x['date'], reverse=True)
            return results


        brand_display = ', '.join(list(brand_label_map.keys())[:8])
        st.caption(f"数据来源: TechCrunch / VentureBeat / The Verge / Ars Technica · 每3小时更新 · 匹配品牌: {brand_display}")

        all_articles = fetch_rss_articles(cutoff_str)

        # ── 过滤出与新模型相关的文章 ──
        matched = []
        for art in all_articles:
            tag = detect_tag(f"{art['title']} {art['desc']}", brand_label_map)
            if tag is not None:
                art['tag'] = tag
                matched.append(art)

        if not matched:
            st.info("近两周内 AI 媒体中未找到这些模型的相关报道。")
        else:
            st.markdown(f"共找到 **{len(matched)}** 条相关报道（标题和摘要已翻译为中文）")
            for art in matched:
                title_zh = translate_zh(art['title']) if art['title'] else "无标题"
                desc_zh = translate_zh(art['desc']) if art['desc'] else ""
                with st.expander(
                    f"[{art['tag']}]  {title_zh}  ·  {art['source']}  ·  {art['date']}",
                    expanded=False
                ):
                    if desc_zh:
                        st.markdown(desc_zh)
                    st.caption(f"原文: {art['title']}")
                    st.markdown(f"[阅读原文 →]({art['link']})")



    # ============================
    # 模块 F: 指标定义与公式说明
    # ============================
    st.markdown("---")
    st.markdown("### 附录: 指标定义与计算公式")
    with st.expander("查看完整指标说明", expanded=False):
        st.markdown("""
| 指标 | 定义 | 计算公式 |
|------|------|----------|
| **日均消耗** | 模型全生命周期内平均每天的 Token 消耗量 | `累计总量 ÷ 上线天数` |
| **近7日增速** | 最近 7 个自然日内的日平均 Token 消耗量 | `Σ(近7日 Total_Tokens) ÷ 近7日数据条数` |
| **增长动量** | 近期活跃度相对于全生命周期均值的比率 | `近7日增速 ÷ 日均消耗` |
| **峰值消耗** | 历史单日最高 Token 消耗量 | `max(每日 Total_Tokens)` |
| **累计总量** | 模型上线以来所有日期 Token 消耗之和 | `Σ(Total_Tokens)` |
| **上线天数** | 模型首次出现在数据库到最新数据的天数 | `最新数据日期 - 首次出现日期` |

**动量解读:**
- 动量 = 1.0 → 近期增速与全期均值持平
- 动量 > 1.2 → 近期处于加速增长阶段
- 动量 < 0.8 → 近期增速放缓，可能进入衰退期

**新模型评级说明:**

评级采用**百分位排名法 (Percentile Rank)**，将新模型的日均消耗放入全部模型的日均消耗分布中计算排名百分位:
- `百分位 = 日均消耗 < 该模型的模型数量 ÷ 总模型数`

| 评级 | 百分位区间 | 含义 |
|------|-----------|------|
| **S · 头部水平** | ≥ P90 | 日均消耗超过 90% 的模型，属于顶级表现 |
| **A · 表现优异** | P75 ~ P90 | 日均消耗处于前 25%，增长势头强劲 |
| **B · 中等水平** | P50 ~ P75 | 日均消耗高于中位数，表现中规中矩 |
| **C · 低于预期** | P25 ~ P50 | 日均消耗处于中位数以下，关注后续走势 |
| **D · 起步缓慢** | < P25 | 日均消耗处于后 25%，可能尚未被广泛采用 |
""")
