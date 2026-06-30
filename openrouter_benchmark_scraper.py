import requests
import pandas as pd
import json
from datetime import datetime
import os
import time
import concurrent.futures

FRONTEND_BENCHMARKS_API = "https://openrouter.ai/api/frontend/v1/rankings/benchmarks"

def fetch_model_slugs():
    print("Fetching all models to get their slugs...")
    url = "https://openrouter.ai/api/v1/models"
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json().get('data', [])
        return [model.get('id') for model in data if model.get('id')]
    except Exception as e:
        print(f"Error fetching models: {e}")
        return []

def fetch_frontend_benchmarks(session):
    try:
        response = session.get(FRONTEND_BENCHMARKS_API, timeout=60)
        response.raise_for_status()
        return response.json().get("data", {}) or {}
    except Exception as e:
        print(f"Error fetching frontend benchmark API: {e}")
        return {}

def build_frontend_benchmark_dataframe(payload):
    current_date = datetime.now().strftime("%Y-%m-%d")
    rows = []

    for category, items in (payload.get("aaData") or {}).items():
        row = {"Date": current_date, "Metric": f"AA {category}"}
        for item in items or []:
            name = item.get("aa_name") or item.get("permaslug") or item.get("uid")
            score = item.get("score")
            if name and score is not None:
                row[name] = score
        if len(row) > 2:
            rows.append(row)

    for metric, items in (payload.get("daData") or {}).items():
        row = {"Date": current_date, "Metric": f"DA {metric}"}
        for item in items or []:
            name = item.get("aa_name") or item.get("permaslug") or item.get("uid")
            score = item.get("score")
            if name and score is not None:
                row[name] = score
        if len(row) > 2:
            rows.append(row)

    return pd.DataFrame(rows)

def fetch_benchmark_for_model(slug, session):
    url = f"https://openopen_router_internal.ai/api/internal/v1/artificial-analysis-benchmarks?slug={slug}"
    # The actual functional URL from the test script was:
    url = f"https://openrouter.ai/api/internal/v1/artificial-analysis-benchmarks?slug={slug}"
    try:
        response = session.get(url, timeout=15)
        response.raise_for_status()
        data = response.json()
        benchmarks = data.get('data', [])
        
        extracted_evals = []
        if benchmarks:
            for b in benchmarks:
                b_name = b.get('aa_name', slug)
                # Fallback to slug if aa_name is weirdly null
                if not b_name: b_name = slug
                
                b_data = b.get('benchmark_data', {})
                evals = b_data.get('evaluations', {})
                
                if evals:
                    extracted_evals.append((b_name, evals))
                
            return extracted_evals
            
        return []
    except Exception as e:
        # Many models may not have benchmarks, don't spam print
        return []

def build_benchmark_dataframe(scraper_results):
    current_date = datetime.now().strftime("%Y-%m-%d")
    
    # Restructure data: metrics_dict[metric][slug] = score
    metrics_dict = {}
    
    for evaluations_list in scraper_results:
        for b_name, evals in evaluations_list:
            if not evals: continue
            for metric, score in evals.items():
                if metric not in metrics_dict:
                    metrics_dict[metric] = {}
                metrics_dict[metric][b_name] = score
            
    # Now convert metrics_dict to a list of rows
    rows = []
    for metric, slug_scores in metrics_dict.items():
        row = {
            'Date': current_date,
            'Metric': metric
        }
        row.update(slug_scores)
        rows.append(row)
        
    df = pd.DataFrame(rows)
    return df

def update_benchmark_database(new_df, file_name="openrouter_benchmark_records.csv"):
    if new_df.empty:
        print("No benchmark records found today.")
        return
        
    if os.path.exists(file_name):
        existing_df = pd.read_csv(file_name)
        current_date = datetime.now().strftime("%Y-%m-%d")
        
        # 存储机制： 同样采用增量逻辑，确保数据表能够随着新指标的加入和新模型的上线自动横向或纵向自动扩展
        # Remove today's rows to overwrite
        existing_df = existing_df[existing_df['Date'] != current_date]
        
        # Concat will automatically handle new columns (metrics) and rows
        final_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        final_df = new_df
        
    # Sort first by Date, then by Metric
    final_df.sort_values(by=['Date', 'Metric'], inplace=True)
    
    # Place Date and Metric at the very beginning of the columns
    cols = final_df.columns.tolist()
    fixed_cols = ['Date', 'Metric']
    remaining_cols = sorted([c for c in cols if c not in fixed_cols])
    final_df = final_df[fixed_cols + remaining_cols]
    
    final_df.to_csv(file_name, index=False)
    print(f"✅ Benchmark matrix updated successfully. Saved to {file_name}")
    print(f"Metrics count: {len(new_df['Metric'].unique())}, Models count: {len(remaining_cols)}")

def main():
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    })

    payload = fetch_frontend_benchmarks(session)
    frontend_df = build_frontend_benchmark_dataframe(payload)
    if not frontend_df.empty:
        print(f"Fetched frontend benchmark API rows: {len(frontend_df)}")
        update_benchmark_database(frontend_df)
        return

    slugs = fetch_model_slugs()
    if not slugs:
        print("No slugs extracted.")
        return
        
    print(f"Found {len(slugs)} models. Fetching benchmarks concurrently...")
    
    results = []
    # Use ThreadPool to fetch quickly
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(fetch_benchmark_for_model, slug, session): slug for slug in slugs}
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            evals_list = future.result()
            if evals_list:
                results.append(evals_list)
            if (i+1) % 50 == 0:
                print(f"Progress: {i+1} / {len(slugs)}...")

    print("Completed API calls. Building matrix...")
    new_df = build_benchmark_dataframe(results)
    update_benchmark_database(new_df)

if __name__ == "__main__":
    main()
