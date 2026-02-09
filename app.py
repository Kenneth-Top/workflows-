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
    st.subheader("🏔️ 单模型历史累计增长 (全堆叠透视)")
    st.info("💡 每一根柱子都展示了从上线第一天到当天的**全部分层结构**。您可以清晰看到早期的增量是如何垫高后期的总量的。")

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
            
            # --- 第一步：计算每个节点的“纯增量” (Atomic Increment) ---
            # 我们先算好 T+0, T+1, T+2... 每个阶段分别长了多少
            increments_map = {} # 存储 {Day: Increment}
            previous_cum = 0
            
            # 这里我们需要遍历每一行来获取精确的每日/每阶段增量
            # 为了对应 T+N 逻辑，我们将时间段归类给最近的 T+N 节点
            # 但为了简化且严谨，我们直接用您定义的节点来切分
            
            node_data = [] # 存储关键节点的信息
            
            # 先收集所有关键节点的数据
            for _, row in m_df.iterrows():
                day = (row['Date'] - start_date).days
                if day in standard_ticks or day == latest_day:
                    current_cum = row['Cum_Tokens']
                    inc = current_cum - previous_cum
                    
                    # 记录这个节点本身的增量信息
                    node_info = {
                        'Day': day,
                        'Label': f"T+{day}" if day != latest_day else f"Latest (T+{day})",
                        'Increment': inc,
                        'Total_Cum': current_cum,
                        'Date': row['Date'].strftime('%Y-%m-%d')
                    }
                    node_data.append(node_info)
                    previous_cum = current_cum

            # --- 第二步：构建“全堆叠”数据 (Full Stack Construction) ---
            # 这是一个 N*N 的嵌套循环
            # 对于第 i 个柱子 (Target_Day)，它由 0 到 i 所有阶段的增量块堆叠而成
            
            stack_plot_data = []
            
            for i, target_node in enumerate(node_data):
                # 这一层循环决定 X 轴有几根柱子
                target_label = target_node['Label']
                target_day_num = target_node['Day']
                
                # 这一层循环决定这根柱子里有几个色块 (从 T+0 一直堆到 T+i)
                current_stack_height = 0
                for j in range(i + 1):
                    source_node = node_data[j]
                    
                    stack_plot_data.append({
                        'X_Label': target_label,       # X轴：属于哪根柱子
                        'X_Order': target_day_num,     # X轴排序用
                        'Component_Label': source_node['Label'], # 色块名称：来自哪个阶段的增量
                        'Component_Order': source_node['Day'],   # 色块排序：保证早期的在下面
                        'Increment_Value': source_node['Increment'], # 色块高度
                        'Total_Height_At_Node': target_node['Total_Cum'], # 这根柱子的总高度（用于画线）
                        'Date': target_node['Date']
                    })

            df_stack = pd.DataFrame(stack_plot_data)

            # === 3. 绘图 (全彩堆叠) ===
            
            # 基础图表
            base = alt.Chart(df_stack).encode(
                x=alt.X('X_Label', sort=alt.EncodingSortField(field="X_Order", order='ascending'), 
                        title="时间节点 (T+N)",
                        axis=alt.Axis(labelFontSize=20, labelFontWeight='bold', labelAngle=0, titleFontSize=24))
            )

            # 图层 A: 彩色堆叠柱 (Stacked Bar)
            # Altair 会自动根据 color 分组进行堆叠
            bars = base.mark_bar(size=55).encode(
                y=alt.Y('Increment_Value', title='累计 Token (Billion)', stack='zero', # stack='zero' 表示从0开始堆
                        axis=alt.Axis(labelFontSize=20, titleFontSize=24)),
                # 颜色：根据“增量来源”上色
                color=alt.Color('Component_Label', 
                                sort=alt.EncodingSortField(field="Component_Order", order='ascending'),
                                legend=alt.Legend(title="增量来源阶段", orient='bottom', columns=6), # 图例放下面
                                scale=alt.Scale(scheme='tableau20')),
                # 排序：必须指定 order，保证 T+0 在最下面
                order=alt.Order('Component_Order', sort='ascending'),
                tooltip=[
                    alt.Tooltip('X_Label', title='当前时刻'),
                    alt.Tooltip('Component_Label', title='增量来源'),
                    alt.Tooltip('Increment_Value', title='该层增量(B)', format='.4f')
                ]
            )

            # 图层 B: 折线 (连接柱子顶部)
            # 为了只画一条线，我们需要对数据去重，每个 X 只留一个总高度点
            line_data = df_stack[['X_Label', 'X_Order', 'Total_Height_At_Node']].drop_duplicates()
            
            line = alt.Chart(line_data).mark_line(color="#333333", strokeWidth=4).encode(
                x=alt.X('X_Label', sort=alt.EncodingSortField(field="X_Order", order='ascending')),
                y='Total_Height_At_Node'
            )

            # 图层 C: 红色总点
            points = alt.Chart(line_data).mark_point(
                filled=True, fill="#FF4B4B", color="#FFFFFF", strokeWidth=2, size=200
            ).encode(
                x=alt.X('X_Label', sort=alt.EncodingSortField(field="X_Order", order='ascending')),
                y='Total_Height_At_Node',
                tooltip=[
                    alt.Tooltip('X_Label', title='时间节点'),
                    alt.Tooltip('Total_Height_At_Node', title='累计总量(B)', format='.4f')
                ]
            )

            final_chart = (bars + line + points).properties(
                height=700, # 高度稍微加大一点，容纳底部图例
                title=alt.TitleParams(text=f"{target_model} 全周期堆叠增长图", fontSize=24)
            ).interactive()

            st.altair_chart(final_chart, use_container_width=True)

            # 4. 下方表格 (保持简洁)
            st.markdown("### 📊 阶段增量详情表")
            # 这里的表格还是展示每个阶段增加了多少，不用改
            # 我们从 line_data 里提取总量，再从 increments 里提取增量，稍微处理一下
            
            # 重新生成一个简单的表格数据
            table_rows = []
            prev = 0
            for item in node_data:
                table_rows.append({
                    '节点': item['Label'],
                    '日期': item['Date'],
                    '累计总量 (B)': item['Total_Cum'],
                    '本阶段新增 (B)': item['Increment']
                })
            
            st.dataframe(
                pd.DataFrame(table_rows).style.format({
                    '累计总量 (B)': '{:.4f}', 
                    '本阶段新增 (B)': '{:.4f}'
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








