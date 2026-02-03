from playwright.sync_api import sync_playwright
import pandas as pd
import time
from datetime import datetime
import re

# ================= 批量配置区域 =================
# 格式: ("模型ID", 蓝色柱子最高峰真实值, "可选:手动起始日期YYYY-MM-DD")

MODELS_CONFIG = [
    # 示例:
    # ("openai/gpt-5.1", 125.5, "2025-12-22"),
    ("moonshotai/kimi-k2-thinking",13.5,None),
    ("moonshotai/kimi-k2.5", 76.5, None),
    ("deepseek/deepseek-v3.2",83.5,None),
    ("deepseek/deepseek-v3.2-speciale",0.663,None),
    ("z-ai/glm-4.6v",1.11,None),
    ("z-ai/glm-4.7",35,None),
    ("z-ai/glm-4.7-flash",2.48,None),
    ("minimax/minimax-m2-her", 0.322, None),
    ("anthropic/claude-opus-4.5",156.5,None),
    ("openai/gpt-5.1-codex",3.99,None),
    ("openai/gpt-5.1",9,None),
    ("openai/gpt-5.2",43.4,None),
    ("openai/gpt-5.2-codex",15,None),
    ("google/gemini-3-pro-preview",75.1,None),
    ("google/gemini-3-pro-image-preview", 0.485,None),
    ("google/gemini-3-flash-preview",118,None),
    ("x-ai/grok-4.1-fast", 86, None),
    
]


OUTPUT_FILENAME = f"All_Models_Analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx"
# ======================================================




def scrape_single_model(playwright_instance, model_id):
    """ 负责单个模型的爬取工作 (保持不变) """
    url = f"https://openrouter.ai/{model_id}"
    print(f"\n🚀 [开始处理] 模型: {model_id}")
    
    browser = playwright_instance.chromium.launch(headless=True)
    page = browser.new_page()
    
    print(f"🔗 正在访问: {url} ...")
    page.goto(url)
    
    created_date_str = None
    try:
        page.wait_for_selector('body', timeout=10000)
        created_date_str = page.evaluate("""() => {
            const elements = document.querySelectorAll('div, span, p');
            for (const el of elements) {
                const match = el.innerText.match(/Created [A-Za-z]+ \d{1,2}, \d{4}/);
                if (match) return match[0];
            }
            return null;
        }""")
        if created_date_str:
            print(f"📅 自动抓取到日期文本: {created_date_str}")
    except:
        pass

    try:
        page.wait_for_selector('path.recharts-rectangle', timeout=20000)
        print("✅ 图表元素已加载，执行滚动...")
        for i in range(6): 
            page.evaluate(f"window.scrollBy(0, 500)")
            time.sleep(0.5)
        time.sleep(2.5) 
    except:
        print(f"❌ [{model_id}] 图表加载超时，跳过。")
        browser.close()
        return None, None

    bars_data = page.evaluate("""() => {
        const paths = document.querySelectorAll('path.recharts-rectangle');
        const result = [];
        const colorMap = {'#0088FE': 'Prompt', '#00C49F': 'Completion', '#FFBB28': 'Reasoning'};

        paths.forEach(p => {
            const height = parseFloat(p.getAttribute('height') || 0);
            const x = parseFloat(p.getAttribute('x') || 0);
            const y = parseFloat(p.getAttribute('y') || 0); 
            const fill = p.getAttribute('fill') || "";
            const colorHex = fill.toUpperCase();
            
            if (height > 0 && colorMap[colorHex]) {
                result.push({type: colorMap[colorHex], height_px: height, x_pos: x, y_pos: y});
            }
        });
        return result;
    }""")
    
    browser.close()
    
    if not bars_data:
        print(f"❌ [{model_id}] 未提取到柱状图数据。")
        return None, None

    df_all = pd.DataFrame(bars_data)
    if len(df_all) > 0:
        max_y = df_all['y_pos'].max()
        df_recent = df_all[df_all['y_pos'] > (max_y - 400)].copy()
    else:
        df_recent = df_all
        
    df_recent['x_pos'] = df_recent['x_pos'].round(0)
    
    return df_recent, created_date_str

def process_single_df(df, model_id, calibration_val, auto_date_str, manual_date_input):
    """ 负责清洗、校准、T+N计算 (已更新严格天数逻辑) """
    if df is None or df.empty:
        return None, None

    # === 新增：物理去重，防止完全重叠的元素干扰 ===
    df = df.drop_duplicates(subset=['x_pos', 'y_pos', 'type', 'height_px'])
    # ============================================

    # 1. 透视与清洗
    
    df_pivot = df.pivot_table(index='x_pos', columns='type', values='height_px', aggfunc='max').fillna(0)
    df_pivot = df_pivot.sort_index()
    
    for col in ['Prompt', 'Completion', 'Reasoning']:
        if col not in df_pivot.columns: df_pivot[col] = 0.0
            
    df_pivot = df_pivot[(df_pivot['Completion'] > 0.000000000001) | (df_pivot['Reasoning'] > 0.00000000001)].reset_index(drop=True)

    # 2. 校准
    if 'Prompt' in df_pivot.columns:
        max_px = df_pivot['Prompt'].max()
    else:
        max_px = df_pivot.iloc[:, 0].max()
        
    if max_px == 0:
        scale_factor = 1.0
    else:
        scale_factor = float(calibration_val) / max_px
        print(f"⚖️ [{model_id}] 校准系数: {scale_factor:.4f}")

    df_final = df_pivot[['Prompt', 'Completion', 'Reasoning']] * scale_factor
    df_final['Total_Tokens'] = df_final.sum(axis=1)
    df_final = df_final.round(3)

    # 3. 日期处理
    start_date = None
    if manual_date_input:
        try:
            start_date = pd.to_datetime(manual_date_input)
        except: pass
    if not start_date and auto_date_str:
        try:
            clean_date_str = auto_date_str.replace("Created ", "").strip()
            start_date = datetime.strptime(clean_date_str, "%b %d, %Y")
        except: pass
    if not start_date:
        start_date = datetime.now()

    df_final['Date'] = [start_date + pd.Timedelta(days=i) for i in range(len(df_final))]
    cols = ['Date', 'Total_Tokens', 'Prompt', 'Completion', 'Reasoning']
    df_final = df_final[cols]

    # 4. 生成 T+N 汇总 (修改：统一使用严格天数，不使用 DateOffset)
    summary_data = []
    
    # 包含所有需要的天数，包括30和60
    target_days = [0, 1, 2, 3, 4, 5, 6, 7, 10, 14, 30, 60]

    for d in target_days:
        # 【核心修改】严格使用 Timedelta (物理天数)，而非 DateOffset (月份)
        t_date = start_date + pd.Timedelta(days=d)
        
        row = df_final[df_final['Date'] == t_date]
        
        # 记录 Start_Date 方便后续透视表展示
        entry = {
            'Model': model_id, 
            'Start_Date': start_date.strftime('%Y-%m-%d'),
            'Time_Label': f"T+{d}", 
            'Date': t_date
        }
        
        if not row.empty:
            # 只要总数
            entry['Total_Tokens'] = row.iloc[0]['Total_Tokens']
        else:
            entry['Total_Tokens'] = None # 用 None 方便后续 Excel 显示空白
            
        summary_data.append(entry)

    return df_final, summary_data

def run_batch_process():
    all_summaries = []
    all_full_data = {} 

    with sync_playwright() as p:
        for model_id, cal_val, man_date in MODELS_CONFIG:
            try:
                df_raw, auto_date = scrape_single_model(p, model_id)
                if df_raw is None: continue
                
                df_full, summary_list = process_single_df(df_raw, model_id, cal_val, auto_date, man_date)
                
                if df_full is not None:
                    safe_sheet_name = model_id.split('/')[-1][:30] 
                    all_full_data[safe_sheet_name] = df_full
                    all_summaries.extend(summary_list)
                    
            except Exception as e:
                print(f"❌ 处理模型 {model_id} 错误: {e}")
                continue

    if not all_summaries:
        print("\n⚠️ 没有数据。")
        return

    # 在 run_batch_process 函数的最后部分
    print(f"\n💾 正在保存数据...")
    
    # 1. 生成宽表 (Master Summary) 用于网页展示
    df_long = pd.DataFrame(all_summaries)
    df_wide = df_long.pivot(index=['Model', 'Start_Date'], columns='Time_Label', values='Total_Tokens')
    desired_order = [f"T+{d}" for d in [0, 1, 2, 3, 4, 5, 6, 7, 10, 14, 30, 60]]
    existing_cols = [c for c in desired_order if c in df_wide.columns]
    df_wide = df_wide[existing_cols].reset_index() # reset_index 让 Model 变回普通列
    
    # 【关键修改】保存为一个固定的 CSV 文件，方便网页读取
    # 每次运行覆盖旧文件，保证网页看到的是最新的
    df_wide.to_csv("latest_summary.csv", index=False) 
    
    # 如果你想保留历史记录，也可以把 all_full_data 保存成 pickle 或者其他格式
    # 但为了网页展示，一个 latest_summary.csv 足够了
    print("✅ latest_summary.csv 已更新")

if __name__ == "__main__":
    run_batch_process()

    
