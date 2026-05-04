"""Fetch OpenRouter market-share and app usage data for the static dashboard."""

from __future__ import annotations

import csv
import json
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional
from urllib.parse import quote

import requests

BASE_URL = "https://openrouter.ai"
RANKINGS_URL = f"{BASE_URL}/rankings"
APPS_URL = f"{BASE_URL}/apps"
PROVIDER_URL = f"{BASE_URL}/provider"
PROVIDERS_URL = f"{BASE_URL}/providers"
TOP_APP_LIMIT = 60
CATEGORY_DEFINITIONS = [
    ("programming", "Programming"),
    ("roleplay", "Roleplay"),
    ("marketing", "Marketing"),
    ("marketing/seo", "SEO"),
    ("technology", "Technology"),
    ("science", "Science"),
    ("translation", "Translation"),
    ("legal", "Legal"),
    ("finance", "Finance"),
    ("health", "Health"),
    ("trivia", "Trivia"),
    ("academia", "Academia"),
]

MARKET_CSV = Path("openrouter_market_share_records.csv")
APPS_CSV = Path("openrouter_apps_records.csv")
APP_USAGE_CSV = Path("openrouter_app_model_usage_records.csv")
PROVIDER_USAGE_CSV = Path("openrouter_provider_usage_records.csv")
CATEGORY_USAGE_CSV = Path("openrouter_category_usage_records.csv")
MARKET_APPS_JSON = Path("openrouter_market_apps.json")

SESSION = requests.Session()
SESSION.headers.update(
    {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
        )
    }
)


def fetch_text(url: str) -> str:
    response = SESSION.get(url, timeout=60)
    response.raise_for_status()
    return response.text


def decode_next_rsc(html: str) -> str:
    chunks: list[str] = []
    pattern = re.compile(r"self\.__next_f\.push\((\[.*?\])\)</script>", re.S)
    for match in pattern.finditer(html):
        try:
            payload = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        if len(payload) > 1 and isinstance(payload[1], str):
            chunks.append(payload[1])
    if not chunks:
        raise RuntimeError("No Next.js RSC payload found")
    return "".join(chunks)


def extract_balanced(text: str, start_index: int, opener: str, closer: str) -> str:
    depth = 0
    in_string = False
    escaped = False
    for index in range(start_index, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == opener:
            depth += 1
        elif char == closer:
            depth -= 1
            if depth == 0:
                return text[start_index : index + 1]
    raise RuntimeError("Unclosed JSON value in RSC payload")


def arrays_for_key(text: str, key: str) -> list[list[Any]]:
    arrays: list[list[Any]] = []
    key_pattern = re.compile(rf'"{re.escape(key)}"\s*:')
    for match in key_pattern.finditer(text):
        start = text.find("[", match.end())
        if start < 0:
            continue
        try:
            parsed = json.loads(extract_balanced(text, start, "[", "]"))
        except (json.JSONDecodeError, RuntimeError):
            continue
        arrays.append(parsed)
    return arrays


def objects_starting_with(text: str, prefix: str) -> list[dict[str, Any]]:
    objects: list[dict[str, Any]] = []
    for match in re.finditer(re.escape(prefix), text):
        try:
            parsed = json.loads(extract_balanced(text, match.start(), "{", "}"))
        except (json.JSONDecodeError, RuntimeError):
            continue
        if isinstance(parsed, dict):
            objects.append(parsed)
    return objects


def is_market_share_series(series: list[Any]) -> bool:
    if len(series) < 20 or not isinstance(series[0], dict):
        return False
    ys = series[0].get("ys")
    if not isinstance(ys, dict) or not ys:
        return False
    keys = list(ys.keys())
    return all("/" not in key for key in keys) and any(key in ys for key in ["openai", "anthropic", "google"])


def is_weekly_model_series(series: list[Any]) -> bool:
    if len(series) < 20 or not isinstance(series[0], dict):
        return False
    dated = [str(point.get("x", ""))[:10] for point in series if point.get("x")]
    if len(dated) < 20:
        return False
    try:
        parsed_dates = [datetime.fromisoformat(date).date() for date in dated]
    except ValueError:
        return False
    if not all((right - left).days == 7 for left, right in zip(parsed_dates, parsed_dates[1:])):
        return False
    ys = series[-1].get("ys")
    return isinstance(ys, dict) and any("/" in str(key) for key in ys)


def fetch_market_share() -> list[dict[str, Any]]:
    text = decode_next_rsc(fetch_text(RANKINGS_URL))
    candidates = [series for series in arrays_for_key(text, "data") if is_market_share_series(series)]
    if not candidates:
        raise RuntimeError("Could not find OpenRouter market-share series")
    series = max(candidates, key=len)

    rows: list[dict[str, Any]] = []
    for point in series:
        date = str(point.get("x", ""))[:10]
        values = point.get("ys") or {}
        total_tokens = sum(float(value or 0) for value in values.values())
        if not date or total_tokens <= 0:
            continue
        for author, tokens in values.items():
            token_value = float(tokens or 0)
            rows.append(
                {
                    "Date": date,
                    "Author": author,
                    "Tokens": round(token_value / 1e9, 6),
                    "Share": round(token_value / total_tokens, 8),
                }
            )
    return rows


def usage_series_from_category_page(category: str) -> list[dict[str, Any]]:
    url = f"{RANKINGS_URL}?category={quote(category, safe='')}"
    text = decode_next_rsc(fetch_text(url))
    complete_week_cutoff = (datetime.now(timezone.utc).date() - timedelta(days=7)).isoformat()
    candidates = [
        series
        for series in arrays_for_key(text, "data")
        if is_weekly_model_series(series) and str(series[-1].get("x", ""))[:10] <= complete_week_cutoff
    ]
    if not candidates:
        raise RuntimeError(f"Could not find weekly category series for {category}")
    return max(candidates, key=len)


def fetch_category_usage() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, (category, label) in enumerate(CATEGORY_DEFINITIONS, start=1):
        print(f"Fetching category usage {index}/{len(CATEGORY_DEFINITIONS)}: {category}")
        series = usage_series_from_category_page(category)
        for point in series:
            date = str(point.get("x", ""))[:10]
            values = point.get("ys") or {}
            author_totals: dict[str, float] = {}
            for model, tokens in values.items():
                model_key = str(model)
                author = model_key.split("/", 1)[0] if "/" in model_key else model_key.lower()
                author_totals[author] = author_totals.get(author, 0) + float(tokens or 0)
            category_total = sum(author_totals.values())
            if not date or category_total <= 0:
                continue
            for author, tokens in author_totals.items():
                rows.append(
                    {
                        "Date": date,
                        "Category": category,
                        "Category_Label": label,
                        "Author": author,
                        "Tokens": round(tokens / 1e9, 6),
                        "Share_In_Category": round(tokens / category_total, 8),
                    }
                )
        time.sleep(0.25)
    return rows


def fetch_top_apps(limit: int = TOP_APP_LIMIT) -> list[dict[str, Any]]:
    text = decode_next_rsc(fetch_text(APPS_URL))
    unique: dict[int, dict[str, Any]] = {}
    for item in objects_starting_with(text, '{"app_id":'):
        if "rank" not in item or "app" not in item:
            continue
        app_id = item.get("app_id")
        if not isinstance(app_id, int):
            continue
        existing = unique.get(app_id)
        if existing is None or int(item.get("total_tokens") or 0) > int(existing.get("total_tokens") or 0):
            unique[app_id] = item

    apps = sorted(unique.values(), key=lambda item: int(item.get("total_tokens") or 0), reverse=True)[:limit]
    rows: list[dict[str, Any]] = []
    snapshot_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    for item in apps:
        app = item.get("app") or {}
        rows.append(
            {
                "Date": snapshot_date,
                "Rank": int(item.get("rank") or 0),
                "App_ID": int(item.get("app_id")),
                "App_Title": app.get("title") or app.get("origin_url") or f"App {item.get('app_id')}",
                "App_Slug": app.get("slug"),
                "Origin_URL": app.get("origin_url"),
                "Categories": ";".join(app.get("categories") or []),
                "Total_Tokens": int(item.get("total_tokens") or 0),
                "Total_Requests": int(item.get("total_requests") or 0),
                "Created_At": app.get("created_at"),
            }
        )
    return rows


def usage_series_from_app_page(slug: str) -> list[dict[str, Any]]:
    text = decode_next_rsc(fetch_text(f"{APPS_URL}/{slug}"))
    candidates = []
    for series in arrays_for_key(text, "data"):
        if not series or not isinstance(series[0], dict):
            continue
        ys = series[0].get("ys")
        if isinstance(ys, dict) and any("/" in key or key == "Others" for key in ys):
            candidates.append(series)
    if not candidates:
        return []
    return max(candidates, key=len)


def fetch_app_model_usage(apps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    daily_rows: list[dict[str, Any]] = []
    slugged_apps = [app for app in apps if app.get("App_Slug")]
    for index, app in enumerate(slugged_apps, start=1):
        slug = str(app["App_Slug"])
        print(f"Fetching app usage {index}/{len(slugged_apps)}: {slug}")
        try:
            series = usage_series_from_app_page(slug)
        except Exception as exc:
            print(f"  skipped {slug}: {exc}")
            continue
        if not series:
            continue

        dates = sorted({str(point.get("x", ""))[:10] for point in series if point.get("x")})
        latest_date = dates[-1] if dates else None
        for point in series:
            date = str(point.get("x", ""))[:10]
            if not date:
                continue
            if date == latest_date:
                continue
            for model, tokens in (point.get("ys") or {}).items():
                daily_rows.append(
                    {
                        "Date": date,
                        "App_ID": app["App_ID"],
                        "App_Title": app["App_Title"],
                        "App_Slug": slug,
                        "Model": model,
                        "Tokens": round(float(tokens or 0) / 1e9, 6),
                    }
                )
        time.sleep(0.25)
    return daily_rows


def fetch_provider_catalog() -> list[dict[str, Any]]:
    text = decode_next_rsc(fetch_text(PROVIDERS_URL))
    candidates = arrays_for_key(text, "providers")
    if not candidates:
        return []
    providers = [item for item in candidates[0] if isinstance(item, dict) and item.get("slug")]
    providers.sort(key=lambda item: int(item.get("dailyTokens") or 0), reverse=True)
    return providers


def usage_series_from_provider_page(provider_slug: str) -> list[dict[str, Any]]:
    text = decode_next_rsc(fetch_text(f"{PROVIDER_URL}/{provider_slug}"))
    candidates = []
    for series in arrays_for_key(text, "data"):
        if not series or not isinstance(series[0], dict):
            continue
        ys = series[0].get("ys")
        if isinstance(ys, dict) and any("/" in str(key) for key in ys):
            candidates.append(series)
    if not candidates:
        return []
    return max(candidates, key=len)


def fetch_provider_usage(providers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    daily_rows: list[dict[str, Any]] = []
    for index, provider in enumerate(providers, start=1):
        provider_slug = str(provider.get("slug"))
        provider_display = str(provider.get("displayName") or provider.get("name") or provider_slug)
        print(f"Fetching provider usage {index}/{len(providers)}: {provider_slug}")
        try:
            series = usage_series_from_provider_page(provider_slug)
        except Exception as exc:
            print(f"  skipped {provider_slug}: {exc}")
            continue
        if not series:
            continue

        dates = sorted({str(point.get("x", ""))[:10] for point in series if point.get("x")})
        latest_date = dates[-1] if dates else None
        for point in series:
            date = str(point.get("x", ""))[:10]
            if not date or date == latest_date:
                continue
            values = point.get("ys") or {}
            tokens = sum(float(value or 0) for value in values.values())
            daily_rows.append(
                {
                    "Date": date,
                    "Provider": provider_slug,
                    "Provider_Display": provider_display,
                    "Tokens": round(tokens / 1e9, 6),
                    "Model_Count": sum(1 for value in values.values() if float(value or 0) > 0),
                }
            )
        time.sleep(0.25)
    return daily_rows


def read_csv(path: Path, key_fields: list[str]) -> dict[tuple[str, ...], dict[str, Any]]:
    if not path.exists():
        return {}
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        return {tuple(str(row.get(field, "")) for field in key_fields): row for row in reader}


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def safe_fetch(label: str, fetcher, fallback_path: Optional[Path] = None) -> list[dict[str, Any]]:
    try:
        return fetcher()
    except Exception as exc:
        print(f"{label} failed: {exc}")
        if fallback_path is not None:
            fallback_rows = read_csv_rows(fallback_path)
            if fallback_rows:
                print(f"Using existing {fallback_path} rows: {len(fallback_rows)}")
                return fallback_rows
        return []


def write_csv(path: Path, rows: list[dict[str, Any]], headers: list[str], key_fields: list[str]) -> list[dict[str, Any]]:
    combined = read_csv(path, key_fields)
    for row in rows:
        combined[tuple(str(row.get(field, "")) for field in key_fields)] = row
    final_rows = sorted(combined.values(), key=lambda row: tuple(str(row.get(field, "")) for field in key_fields))
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=headers, lineterminator="\n")
        writer.writeheader()
        writer.writerows(final_rows)
    return final_rows


def write_current_csv(path: Path, rows: list[dict[str, Any]], headers: list[str]) -> list[dict[str, Any]]:
    final_rows = sorted(rows, key=lambda row: int(row.get("Rank") or 10**9))
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=headers, lineterminator="\n")
        writer.writeheader()
        writer.writerows(final_rows)
    return final_rows


def write_replace_csv(path: Path, rows: list[dict[str, Any]], headers: list[str], key_fields: list[str]) -> list[dict[str, Any]]:
    final_rows = sorted(rows, key=lambda row: tuple(str(row.get(field, "")) for field in key_fields))
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=headers, lineterminator="\n")
        writer.writeheader()
        writer.writerows(final_rows)
    return final_rows


def latest_app_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = str(row.get("App_ID") or row.get("App_Slug") or row.get("App_Title") or "")
        if not key:
            continue
        existing = latest.get(key)
        row_date = str(row.get("Date") or "")
        existing_date = str(existing.get("Date") or "") if existing else ""
        if existing is None or (row_date, int(row.get("Total_Tokens") or 0)) >= (
            existing_date,
            int(existing.get("Total_Tokens") or 0),
        ):
            latest[key] = row
    return sorted(latest.values(), key=lambda row: int(row.get("Rank") or 10**9))


def write_json(
    market_rows: list[dict[str, Any]],
    app_rows: list[dict[str, Any]],
    app_usage_rows: list[dict[str, Any]],
    provider_usage_rows: list[dict[str, Any]],
    category_usage_rows: list[dict[str, Any]],
) -> None:
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sources": {
            "rankings": RANKINGS_URL,
            "apps": APPS_URL,
            "providers": PROVIDER_URL,
            "provider_catalog": PROVIDERS_URL,
        },
        "market_share": market_rows,
        "apps": latest_app_rows(app_rows),
        "app_model_usage": app_usage_rows,
        "provider_usage": provider_usage_rows,
        "category_usage": category_usage_rows,
    }
    MARKET_APPS_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    market_rows = safe_fetch("Market share", fetch_market_share, MARKET_CSV)
    app_rows = safe_fetch("Top apps", fetch_top_apps, APPS_CSV)
    app_usage_rows = safe_fetch("App model usage", lambda: fetch_app_model_usage(app_rows), APP_USAGE_CSV)
    provider_rows = safe_fetch("Provider catalog", fetch_provider_catalog)
    provider_usage_rows = safe_fetch(
        "Provider usage",
        lambda: fetch_provider_usage(provider_rows),
        PROVIDER_USAGE_CSV,
    )
    category_usage_rows = safe_fetch("Category usage", fetch_category_usage, CATEGORY_USAGE_CSV)

    market_all = write_csv(MARKET_CSV, market_rows, ["Date", "Author", "Tokens", "Share"], ["Date", "Author"])
    apps_all = write_current_csv(
        APPS_CSV,
        app_rows,
        ["Date", "Rank", "App_ID", "App_Title", "App_Slug", "Origin_URL", "Categories", "Total_Tokens", "Total_Requests", "Created_At"],
    )
    usage_all = write_csv(
        APP_USAGE_CSV,
        app_usage_rows,
        ["Date", "App_ID", "App_Title", "App_Slug", "Model", "Tokens"],
        ["Date", "App_ID", "Model"],
    )
    provider_usage_all = write_csv(
        PROVIDER_USAGE_CSV,
        provider_usage_rows,
        ["Date", "Provider", "Provider_Display", "Tokens", "Model_Count"],
        ["Date", "Provider"],
    )
    category_usage_all = write_replace_csv(
        CATEGORY_USAGE_CSV,
        category_usage_rows,
        ["Date", "Category", "Category_Label", "Author", "Tokens", "Share_In_Category"],
        ["Date", "Category", "Author"],
    )
    write_json(market_all, apps_all, usage_all, provider_usage_all, category_usage_all)
    print(f"Saved market rows: {len(market_all)}")
    print(f"Saved app rows: {len(apps_all)}")
    print(f"Saved app model usage rows: {len(usage_all)}")
    print(f"Saved provider usage rows: {len(provider_usage_all)}")
    print(f"Saved category usage rows: {len(category_usage_all)}")


if __name__ == "__main__":
    main()
