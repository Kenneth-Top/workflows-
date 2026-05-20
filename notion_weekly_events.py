import json
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlparse

import requests


ROOT = Path(__file__).resolve().parent
LOCAL_NEWS_FILE = ROOT / "notion_weekly_news.json"
DRAFT_FILE = ROOT / "ai_market_events_draft.json"
NOTION_VERSION = "2022-06-28"

VENDOR_ALIASES = {
    "minimax": ["minimax", "海螺", "hailuo", "m2", "m1"],
    "zhipu": ["智谱", "zhipu", "glm", "z.ai"],
    "kimi": ["kimi", "moonshot", "月之暗面"],
    "stepfun": ["阶跃", "stepfun"],
    "alibaba": ["阿里", "通义", "qwen", "alibaba"],
    "bytedance": ["字节", "豆包", "seed", "bytedance"],
    "tencent": ["腾讯", "混元", "hunyuan", "tencent"],
    "openai": ["openai", "chatgpt", "gpt"],
    "anthropic": ["anthropic", "claude"],
    "google": ["google", "gemini"],
    "meta": ["meta", "llama"],
    "spacex": ["spacex", "xai", "grok"],
}


def notion_headers():
    token = os.getenv("NOTION_API_KEY")
    if not token:
        return None
    return {
        "Authorization": f"Bearer {token}",
        "Notion-Version": NOTION_VERSION,
        "Content-Type": "application/json",
    }


def normalize_notion_id(value):
    text = (value or "").strip()
    if not text:
        return ""
    if text.startswith("collection://"):
        text = text.replace("collection://", "", 1)
    if "notion.so" in text:
        path = urlparse(text).path.rstrip("/")
        text = path.split("/")[-1]
        if "-" in text:
            text = text.split("-")[-1]
    match = re.search(r"([0-9a-fA-F]{32})", text.replace("-", ""))
    return match.group(1) if match else text


def rich_text_text(items):
    return "".join((item.get("plain_text") or "") for item in (items or []))


def block_text(block):
    block_type = block.get("type")
    data = block.get(block_type, {})
    return rich_text_text(data.get("rich_text"))


def page_title(page):
    for prop in page.get("properties", {}).values():
        if prop.get("type") == "title":
            return rich_text_text(prop.get("title"))
    return page.get("id", "")


def fetch_block_children(block_id, headers):
    url = f"https://api.notion.com/v1/blocks/{block_id}/children?page_size=100"
    blocks = []
    while url:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        payload = response.json()
        blocks.extend(payload.get("results", []))
        url = payload.get("next_cursor")
        if url:
            url = f"https://api.notion.com/v1/blocks/{block_id}/children?page_size=100&start_cursor={url}"
    return blocks


def fetch_database_pages(source_id, headers):
    url = f"https://api.notion.com/v1/databases/{source_id}/query"
    response = requests.post(url, headers=headers, json={"page_size": 30}, timeout=30)
    response.raise_for_status()
    return response.json().get("results", [])


def read_notion_items():
    headers = notion_headers()
    source_id = normalize_notion_id(os.getenv("NOTION_WEEKLY_SOURCE_ID"))
    if not headers or not source_id:
        return []
    try:
        pages = fetch_database_pages(source_id, headers)
        items = []
        for page in pages:
            title = page_title(page)
            content = "\n".join(filter(None, (block_text(block) for block in fetch_block_children(page["id"], headers))))
            items.append({
                "date": page.get("last_edited_time", "")[:10],
                "title": title,
                "summary": content[:1200],
                "source_url": page.get("url", ""),
            })
        return items
    except Exception:
        try:
            blocks = fetch_block_children(source_id, headers)
        except Exception as exc:
            print(f"[warn] cannot read Notion source: {exc}")
            return []
        return [{
            "date": datetime.now(timezone.utc).date().isoformat(),
            "title": block_text(block)[:80],
            "summary": block_text(block),
            "source_url": "",
        } for block in blocks if block_text(block)]


def read_local_items():
    if not LOCAL_NEWS_FILE.exists():
        return []
    with LOCAL_NEWS_FILE.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload.get("items", [])


def detect_vendor(text):
    lowered = text.lower()
    matches = []
    for vendor, aliases in VENDOR_ALIASES.items():
        if any(alias.lower() in lowered for alias in aliases):
            matches.append(vendor)
    return matches or ["macro"]


def detect_event_type(text):
    lowered = text.lower()
    if re.search(r"发布|release|launch|上线", lowered):
        return "model_release"
    if re.search(r"价格|price|token plan|订阅|降价|涨价", lowered):
        return "pricing"
    if re.search(r"上市|融资|ipo|估值|财报|业绩", lowered):
        return "financing"
    if re.search(r"爆发|潮|行业|生态|openrouter|openclaw", lowered):
        return "macro"
    return "company"


def short_title(title, summary):
    text = (title or summary or "").strip().replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    if len(text) <= 24:
        return text
    return text[:24]


def within_last_week(date_text):
    try:
        date = datetime.fromisoformat(date_text[:10]).date()
    except Exception:
        return True
    today = datetime.now(timezone.utc).date()
    return today - timedelta(days=10) <= date <= today


def explicit_event_date(text, fallback_date):
    match = re.search(r"(20\d{2})[-/.年](\d{1,2})[-/.月](\d{1,2})", text)
    if match:
        year, month, day = match.groups()
        return f"{int(year):04d}-{int(month):02d}-{int(day):02d}", "explicit_text_date", False
    match = re.search(r"(?<!\d)(\d{1,2})月(\d{1,2})日", text)
    if match:
        month, day = match.groups()
        try:
            year = datetime.fromisoformat(fallback_date[:10]).year
        except Exception:
            year = datetime.now(timezone.utc).year
        return f"{year:04d}-{int(month):02d}-{int(day):02d}", "explicit_text_month_day", False
    return fallback_date[:10], "notion_page_date_needs_web_check", True


def build_events(items):
    events = []
    for index, item in enumerate(items):
        fallback_date = (item.get("date") or datetime.now(timezone.utc).date().isoformat())[:10]
        text = f"{item.get('title', '')}\n{item.get('summary', '')}"
        date, date_basis, needs_date_verification = explicit_event_date(text, fallback_date)
        if not text.strip() or not within_last_week(fallback_date):
            continue
        event_type = detect_event_type(text)
        for vendor in detect_vendor(text):
            title = short_title(item.get("title", ""), item.get("summary", ""))
            events.append({
                "id": f"draft-{date}-{vendor}-{index}",
                "date": date,
                "vendor": vendor,
                "company": vendor,
                "event_type": event_type,
                "title": title,
                "summary": (item.get("summary") or item.get("title") or "")[:240],
                "source": "notion",
                "source_url": item.get("source_url", ""),
                "confidence": "medium" if vendor != "macro" else "low",
                "date_basis": date_basis,
                "needs_date_verification": needs_date_verification,
                "status": "draft",
            })
    return events


def main():
    items = read_notion_items() or read_local_items()
    events = build_events(items)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "notion_api" if os.getenv("NOTION_API_KEY") and os.getenv("NOTION_WEEKLY_SOURCE_ID") else "local_export",
        "events": events,
    }
    with DRAFT_FILE.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"wrote {len(events)} draft events")


if __name__ == "__main__":
    main()
