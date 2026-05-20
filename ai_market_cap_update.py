import csv
import json
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import quote

import requests


ROOT = Path(__file__).resolve().parent
COMPANY_FILE = ROOT / "ai_market_companies.csv"
HISTORY_FILE = ROOT / "ai_market_cap_history.csv"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
USD_HKD_RATE = float(os.getenv("USD_HKD_RATE", "7.8"))
FMP_API_KEY = os.getenv("FMP_API_KEY")


def read_csv(path):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fieldnames):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def yahoo_chart(ticker, start_date):
    period1 = int(datetime.fromisoformat(start_date).replace(tzinfo=timezone.utc).timestamp())
    period2 = int((datetime.now(timezone.utc) + timedelta(days=1)).timestamp())
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{quote(ticker)}"
        f"?period1={period1}&period2={period2}&interval=1d&events=history"
    )
    response = requests.get(url, headers={"User-Agent": UA}, timeout=30)
    response.raise_for_status()
    payload = response.json()
    result = payload.get("chart", {}).get("result", [])
    if not result:
        return []
    timestamps = result[0].get("timestamp") or []
    quote_data = (result[0].get("indicators", {}).get("quote") or [{}])[0]
    closes = quote_data.get("close") or []
    rows = []
    for ts, close in zip(timestamps, closes):
        if close is None:
            continue
        rows.append({
            "date": datetime.fromtimestamp(ts, tz=timezone.utc).date().isoformat(),
            "close": float(close),
        })
    return rows


def fmp_market_cap_history(ticker, start_date):
    if not FMP_API_KEY:
        return []
    url = f"https://financialmodelingprep.com/stable/historical-market-capitalization"
    params = {"symbol": ticker, "from": start_date, "apikey": FMP_API_KEY}
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    payload = response.json()
    if isinstance(payload, dict):
        payload = payload.get("historical") or payload.get("data") or []
    rows = []
    for item in payload or []:
        date = item.get("date") or item.get("Date")
        market_cap = item.get("marketCap") or item.get("market_cap") or item.get("marketCapitalization")
        if not date or market_cap in (None, ""):
            continue
        rows.append({"date": str(date)[:10], "market_cap": float(market_cap)})
    return rows


def alpha_vantage_market_cap(ticker):
    api_key = os.getenv("ALPHAVANTAGE_API_KEY")
    if not api_key:
        return None
    symbol = ticker.replace(".HK", ".HKG")
    url = "https://www.alphavantage.co/query"
    params = {"function": "OVERVIEW", "symbol": symbol, "apikey": api_key}
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        payload = response.json()
        value = float(payload.get("MarketCapitalization") or 0)
        if value > 0:
            return value
    except Exception:
        return None
    return None


def native_to_hkd(value, currency):
    if currency == "HKD":
        return value
    if currency == "USD":
        return value * USD_HKD_RATE
    return value


def upsert_market_cap(existing, company, date, native_cap, close, source, updated_at):
    hkd_cap = native_to_hkd(native_cap, company.get("currency", "HKD"))
    existing[(date, company["company_id"])] = {
        "Date": date,
        "Company_ID": company["company_id"],
        "Ticker": company.get("ticker", "").strip(),
        "Market_Cap_Billion_HKD": f"{hkd_cap / 1_000_000_000:.6f}",
        "Market_Cap_Native_Billion": f"{native_cap / 1_000_000_000:.6f}",
        "Currency": company.get("currency", ""),
        "Close": "" if close is None else f"{float(close):.6f}",
        "Source": source,
        "Updated_At": updated_at,
    }


def main():
    companies = read_csv(COMPANY_FILE)
    history = read_csv(HISTORY_FILE)
    fieldnames = [
        "Date",
        "Company_ID",
        "Ticker",
        "Market_Cap_Billion_HKD",
        "Market_Cap_Native_Billion",
        "Currency",
        "Close",
        "Source",
        "Updated_At",
    ]
    existing = {(row["Date"], row["Company_ID"]): row for row in history}
    today = datetime.now(timezone.utc).date()
    start_date = os.getenv("AI_MARKET_START_DATE", "2026-01-01")
    updated_at = datetime.now(timezone.utc).isoformat()

    for company in companies:
        if company.get("listed_status") != "listed":
            continue
        ticker = company.get("ticker", "").strip()
        shares = float(company.get("shares_outstanding") or 0)
        if not ticker or shares <= 0:
            continue
        fmp_rows = []
        try:
            fmp_rows = fmp_market_cap_history(ticker, start_date)
        except Exception as exc:
            print(f"[warn] failed to fetch FMP market cap for {ticker}: {exc}")
        for item in fmp_rows:
            date = item["date"]
            if datetime.fromisoformat(date).date() > today:
                continue
            upsert_market_cap(existing, company, date, item["market_cap"], None, "fmp_historical_market_cap", updated_at)

        try:
            chart_rows = yahoo_chart(ticker, start_date)
        except Exception as exc:
            print(f"[warn] failed to fetch {ticker}: {exc}")
            chart_rows = []

        for item in chart_rows:
            date = item["date"]
            if fmp_rows and (date, company["company_id"]) in existing:
                continue
            if datetime.fromisoformat(date).date() > today:
                continue
            close = item["close"]
            native_cap = close * shares
            upsert_market_cap(existing, company, date, native_cap, close, "yahoo_chart", updated_at)

        current_cap = alpha_vantage_market_cap(ticker)
        if current_cap:
            latest = max(chart_rows, key=lambda row: row["date"], default=None)
            if latest:
                native_cap = current_cap
                hkd_cap = native_to_hkd(native_cap, company.get("currency", "HKD"))
                existing[(latest["date"], company["company_id"])]["Market_Cap_Billion_HKD"] = f"{hkd_cap / 1_000_000_000:.6f}"
                existing[(latest["date"], company["company_id"])]["Market_Cap_Native_Billion"] = f"{native_cap / 1_000_000_000:.6f}"
                existing[(latest["date"], company["company_id"])]["Source"] = "alpha_vantage_overview"
        time.sleep(0.8)

    rows = sorted(existing.values(), key=lambda row: (row["Date"], row["Company_ID"]))
    write_csv(HISTORY_FILE, rows, fieldnames)
    print(json.dumps({"rows": len(rows), "updated_at": updated_at}, ensure_ascii=False))


if __name__ == "__main__":
    main()
