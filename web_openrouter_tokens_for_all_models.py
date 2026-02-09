from playwright.sync_api import sync_playwright
import pandas as pd
import time
from datetime import datetime, timedelta
import re

# ================= 批量配置区域 =================
# 格式: ("模型ID", "可选:手动起始日期YYYY-MM-DD")
# 注意：现在不需要填最高值的数字了！脚本会自动去读！
MODELS_CONFIG = [
    ("moonshotai/kimi-k2-thinking", None),
    ("moonshotai/kimi-k2.5", None),
    ("deepseek/deepseek-v3.2", None),
    ("minimax/minimax-m2.1", None),
    ("x-ai/grok-4.1-fast", None),
    ("openai/gpt-5.1", None), 
    # 你可以继续添加更多...
]

OUTPUT_SUMMARY = f"summary_matrix.csv"
OUTPUT_STREAMLIT = f"streamlit_source.csv"
# ======================================================

def parse_tooltip_value(text):
    """
    将 Tooltip 里的 "97.5B", "800M", "10k" 转化为浮点数
    """
    text = text.upper().replace(',', '')
    match = re.search(r'(\d+\.?\d*)\s*([BKM]?)', text)
    if not match:
        return 0.0
    
    val = float(match.group(1))
    unit = match.group(2)
    
    if unit == 'B': return val * 1_000_000_000 # 统一转换为基础单位或保持 B 为单位
    # 这里为了方便 Excel 阅读，我们统一转换为 Billion (B) 为单位
    if unit == 'B': return val
    if unit == 'M': return val / 1000
    if unit == 'K': return val / 1_000_000
    return val / 1_000_000_000 # 无单位默认为个位，转为B

def scrape_and_calibrate(playwright_instance, model_id):
    url = f"https://openrouter.ai/{model_id}"
    print(f"\n🚀 [开始处理] 模型: {model_id}")
    
    browser = playwright_instance.chromium.launch(headless=True) # 调试时改 False
    page = browser.new_page()
    page.goto(url)
    
    # 1. 抓取创建日期
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
            print(f"📅 自动日期: {created_date_str}")
    except: pass

    # 2. 滚动加载图表
    try:
        page.wait_for_selector('path.recharts-rectangle', timeout=20000)
        print("✅ 图表加载，滚动激活...")
        for i in range(6): 
            page.evaluate(f"window.scrollBy(0, 500)")
            time.sleep(0.5)
        time.sleep(2.0) 
    except:
        print(f"❌ 图表加载失败")
        browser.close()
        return None, None, None

    # 3. 提取像素数据 (带去重)
    bars_data = page.evaluate("""() => {
        const paths = document.querySelectorAll('path.recharts-rectangle');
        const uniqueBars = new Map();
        const colorMap = {'#0088FE': 'Prompt', '#00C49F': 'Completion', '#FFBB28': 'Reasoning'};

        paths.forEach(p => {
            const height = parseFloat(p.getAttribute('height') || 0);
            const x = parseFloat(p.getAttribute('x') || 0);
            const fill = p.getAttribute('fill') || "";
            const colorHex = fill.toUpperCase();
            
            const style = window.getComputedStyle(p);
            if (style.opacity === '0' || style.visibility === 'hidden') return;
            if (height <= 1 || !colorMap[colorHex]) return;

            const key = `${Math.round(x)}_${colorHex}`;
            if (uniqueBars.has(key)) {
                if (height > uniqueBars.get(key).height_px) {
                    uniqueBars.set(key, {type: colorMap[colorHex], height_px: height, x_pos: x, dom_index: -1}); # index稍后在py处理
                }
            } else {
                uniqueBars.set(key, {type: colorMap[colorHex], height_px: height, x_pos: x, dom_index: -1});
            }
        });
        return Array.from(uniqueBars.values());
    }""")

    if not bars_data:
        browser.close()
        return None, None, None

    df = pd.DataFrame(bars_data)
    
    # 筛选底部图表
    max_y = df['x_pos'].max() # 这里没取y，假设数据比较干净，或者沿用之前的逻辑
    df['x_pos'] = df['x_pos'].round(0)
    
    # ================= 核心升级：自动校准 (Tooltip) =================
    # 1. 找到 Prompt (蓝色) 最高的那根柱子的 x_pos
    if 'Prompt' in df['type'].values:
        max_bar = df[df['type'] == 'Prompt'].sort_values('height_px', ascending=False).iloc[0]
    else:
        max_bar = df.sort_values('height_px', ascending=False).iloc[0]
    
    target_x = max_bar['x_pos']
    max_px_height = max_bar['height_px']
    print(f"🔍 正在执行自动校准... 目标 X坐标: {target_x}")

    # 2. Playwright 查找对应的 DOM 元素并悬停
    # 我们需要重新在页面找这个元素，因为之前的 evaluate 传回的是纯数据
    try:
        # 使用 CSS 选择器定位：找到 x 属性接近 target_x 的 path
        # 注意：SVG 的 x 可能是小数，我们用模糊匹配逻辑不太好写 CSS，
        # 最好的办法是遍历所有 path，找到 x 匹配的那个，然后 hover
        
        box = page.evaluate_handle(f"""(targetX) => {{
            const paths = document.querySelectorAll('path.recharts-rectangle');
            let target = null;
            let minDiff = 1.0;
            
            paths.forEach(p => {{
                const x = parseFloat(p.getAttribute('x') || 0);
                const fill = p.getAttribute('fill');
                // 确保是蓝色 Prompt 柱子
                if (Math.abs(x - targetX) < minDiff && fill.toUpperCase() === '#0088FE') {{
                    target = p;
                    minDiff = Math.abs(x - targetX);
                }}
            }});
            return target;
        }}""", target_x)
        
        if box:
            box.hover()
            time.sleep(1) # 等待 Tooltip 弹出
            
            # 3. 读取 Tooltip 内容
            # OpenRouter Tooltip 通常在一个 class 为 'recharts-tooltip-wrapper' 的 div 里
            tooltip_text = page.locator('.recharts-tooltip-wrapper').inner_text()
            print(f"💬 捕获 Tooltip: {tooltip_text.replace(chr(10), ' | ')}")
            
            # 解析 Prompt 的数值
            # Tooltip 格式通常是: "Date \n Prompt 80B \n Completion..."
            # 我们用正则提取 Prompt 后面的数字
            match = re.search(r'Prompt\s*([\d\.,]+[KMB]?)', tooltip_text, re.IGNORECASE)
            if match:
                raw_val = match.group(1)
                real_val = parse_tooltip_value(raw_val)
                scale_factor = real_val / max_px_height
                print(f"⚖️ 自动校准成功! 读数: {raw_val} -> 系数: {scale_factor:.6f}")
            else:
                print("⚠️ Tooltip 解析失败，使用默认系数 1.0")
                scale_factor = 1.0
        else:
            print("⚠️ 未找到目标柱子 DOM，无法悬停。")
            scale_factor = 1.0
            
    except Exception as e:
        print(f"⚠️ 自动校准过程报错: {e}")
        scale_factor = 1.0

    browser.close()
    return df, created_date_str, scale_factor

def process_data(df, model_id, scale_factor, auto_date_str, manual_date):
    # 1. 透视 & 清洗
    df_pivot = df.pivot_table(index='x_pos', columns='type', values='height_px', aggfunc='max').fillna(0)
    df_pivot = df_pivot.sort_index()
    for col in ['Prompt', 'Completion', 'Reasoning']:
        if col not in df_pivot.columns: df_pivot[col] = 0.0
    
    # 2. 应用比例尺
    df_final = df_pivot * scale_factor
    df_final['Total_Tokens'] = df_final.sum(axis=1)
    df_final = df_final.round(3)

    # 3. 确定日期
    start_date = None
    if manual_date:
        try: start_date = pd.to_datetime(manual_date)
        except: pass
    if not start_date and auto_date_str:
        try:
            clean = auto_date_str.replace("Created ", "").strip()
            start_date = datetime.strptime(clean, "%b %d, %Y")
        except: pass
    if not start_date: start_date = datetime.now()

    # 4. 生成完整日期列
    df_final['Date'] = [start_date + timedelta(days=i) for i in range(len(df_final))]
    df_final['Day_Index'] = range(len(df_final)) # T+0, T+1...
    
    # 5. 生成两个版本的输出
    
    # --- A. 宽表数据 (T+N 汇总) ---
    summary_row = {'Model': model_id, 'Start_Date': start_date.strftime('%Y-%m-%d')}
    targets = [0, 1, 2, 3, 4, 5, 6, 7, 10, 14, 30, 60]
    
    for t in targets:
        # 严格使用iloc查找，如果 index 超出范围则为空
        if t < len(df_final):
            val = df_final.iloc[t]['Total_Tokens']
            summary_row[f"T+{t}"] = val
        else:
            summary_row[f"T+{t}"] = None
            
    # --- B. 长表数据 (Streamlit 用) ---
    # 包含每一天的数据，加上 Latest 标记
    streamlit_df = df_final.copy()
    streamlit_df['Model'] = model_id
    streamlit_df['Is_Latest'] = False
    streamlit_df.iloc[-1, streamlit_df.columns.get_loc('Is_Latest')] = True # 标记最后一行
    
    # 重命名列以符合 Streamlit 习惯
    streamlit_df = streamlit_df[['Date', 'Model', 'Total_Tokens', 'Day_Index', 'Is_Latest', 'Prompt', 'Completion', 'Reasoning']]

    return summary_row, streamlit_df

def run():
    all_summaries = []
    all_streamlit_data = []

    with sync_playwright() as p:
        for model_id, manual_date in MODELS_CONFIG:
            try:
                # 爬取 + 自动获取系数
                df_raw, auto_date, scale = scrape_and_calibrate(p, model_id)
                
                if df_raw is not None:
                    # 处理数据
                    summ_row, st_df = process_data(df_raw, model_id, scale, auto_date, manual_date)
                    all_summaries.append(summ_row)
                    all_streamlit_data.append(st_df)
            except Exception as e:
                print(f"❌ 处理 {model_id} 异常: {e}")

    # 保存宽表 (Excel/CSV)
    if all_summaries:
        df_sum = pd.DataFrame(all_summaries)
        # 排序每一列
        cols = ['Model', 'Start_Date'] + [f"T+{d}" for d in [0, 1, 2, 3, 4, 5, 6, 7, 10, 14, 30, 60]]
        cols = [c for c in cols if c in df_sum.columns]
        df_sum = df_sum[cols]
        df_sum.to_csv(OUTPUT_SUMMARY, index=False)
        print(f"\n✅ 宽表已保存: {OUTPUT_SUMMARY}")

    # 保存长表 (Streamlit)
    if all_streamlit_data:
        df_st = pd.concat(all_streamlit_data)
        df_st.to_csv(OUTPUT_STREAMLIT, index=False)
        print(f"✅ Streamlit源数据已保存: {OUTPUT_STREAMLIT}")

if __name__ == "__main__":
    run()