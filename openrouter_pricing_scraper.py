import requests
import pandas as pd
from datetime import datetime
import os
import concurrent.futures

def fetch_models():
    print("Fetching all models to get their canonical slugs...")
    url = "https://openrouter.ai/api/v1/models"
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.json().get('data', [])
    except Exception as e:
        print(f"Error fetching models: {e}")
        return []

def fetch_effective_pricing(model, session):
    model_id = model.get('id')
    canonical_slug = model.get('canonical_slug')
    
    if not canonical_slug:
        return []
        
    url = f"https://openrouter.ai/api/frontend/stats/effective-pricing?permaslug={canonical_slug}&variant=standard"
    try:
        response = session.get(url, timeout=15)
        response.raise_for_status()
        data = response.json().get('data', {})
        
        records = []
        current_date = datetime.now().strftime("%Y-%m-%d")
        
        # 1. Overall Weighted Average Price (当日快照)
        weighted_input = data.get('weightedInputPrice')
        weighted_output = data.get('weightedOutputPrice')
        
        if weighted_input is not None and weighted_output is not None:
            records.append({
                'Date': current_date,
                'Model': model_id,
                'Provider': 'Weighted Average',
                'Input_Price_1M': float(weighted_input),
                'Output_Price_1M': float(weighted_output),
                'Cache_Hit_Rate': None
            })
            
        # 2. Provider Specific Effective Prices (当日快照，含 CacheHitRate)
        provider_summaries = data.get('providerSummaries', [])
        for p in provider_summaries:
            records.append({
                'Date': current_date,
                'Model': model_id,
                'Provider': p.get('providerName', 'Unknown'),
                'Input_Price_1M': float(p.get('effectiveInputPrice', -1)),
                'Output_Price_1M': float(p.get('effectiveOutputPrice', -1)),
                'Cache_Hit_Rate': float(p.get('cacheHitRate', -1)) if p.get('cacheHitRate') is not None else None
            })

        # 3. 7-Day Chart Data：逐日逐供应商的历史定价 (不含 Weighted Average / Cache Hit)
        input_chart = data.get('inputChartData', [])
        output_chart = data.get('outputChartData', [])

        # 先把 output chart 按日期索引化，方便合并
        output_by_date = {}
        for point in output_chart:
            date_str = point.get('x', '')[:10]
            output_by_date[date_str] = point.get('y', {})

        for point in input_chart:
            date_str = point.get('x', '')[:10]
            if not date_str or date_str == current_date:
                # 当天数据已由 providerSummaries 覆盖（含 cacheHitRate），跳过
                continue
            input_prices = point.get('y', {})
            output_prices = output_by_date.get(date_str, {})

            for provider_name, inp_price in input_prices.items():
                out_price = output_prices.get(provider_name)
                records.append({
                    'Date': date_str,
                    'Model': model_id,
                    'Provider': provider_name,
                    'Input_Price_1M': float(inp_price),
                    'Output_Price_1M': float(out_price) if out_price is not None else None,
                    'Cache_Hit_Rate': None  # 历史数据无 cache hit 信息
                })

        return records
    except Exception as e:
        # Some models might not have this stats routing available yet
        return []

def update_pricing_database(new_df, file_name="openrouter_pricing_provider_records.csv"):
    if new_df.empty:
        print("No pricing records found today.")
        return
        
    if os.path.exists(file_name):
        existing_df = pd.read_csv(file_name)
        
        # 增量合并：以 Date + Model + Provider 为唯一键进行 upsert
        combined = pd.concat([existing_df, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=['Date', 'Model', 'Provider'], keep='last')
    else:
        combined = new_df
        
    combined.sort_values(by=['Date', 'Model', 'Provider'], inplace=True)
    combined.to_csv(file_name, index=False)
    
    unique_dates = combined['Date'].nunique()
    print(f"✅ Provider pricing database updated successfully. Saved to {file_name}")
    print(f"   Total records: {len(combined)}, Date range: {unique_dates} days")

def main():
    models = fetch_models()
    if not models:
        print("No models extracted.")
        return
        
    print(f"Found {len(models)} models. Fetching effective pricing (including 7-day chart data) concurrently...")
    
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    })
    
    all_records = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(fetch_effective_pricing, model, session): model for model in models}
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            records = future.result()
            all_records.extend(records)
            if (i+1) % 50 == 0:
                print(f"Progress: {i+1} / {len(models)}...")

    print("Completed API calls. Building dataframe...")
    new_df = pd.DataFrame(all_records)
    
    if not new_df.empty:
        new_df = new_df[['Date', 'Model', 'Provider', 'Input_Price_1M', 'Output_Price_1M', 'Cache_Hit_Rate']]
        
    print(f"Extracted {len(new_df)} effective pricing records.")
    update_pricing_database(new_df)

if __name__ == "__main__":
    main()
