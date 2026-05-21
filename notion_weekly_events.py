import argparse
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import requests


ROOT = Path(__file__).resolve().parent
LOCAL_NEWS_FILE = ROOT / "notion_weekly_news.json"
DRAFT_FILE = ROOT / "ai_market_events_draft.json"
NOTION_VERSION = "2022-06-28"

VENDOR_ALIASES = {
    "minimax": ["minimax", "mini max", "海螺", "hailuo", "m2", "m1", "mimo", "maxclaw", "mmx"],
    "zhipu": ["智谱", "zhipu", "glm", "z.ai", "autoclaw"],
    "kimi": ["kimi", "moonshot", "月之暗面"],
    "stepfun": ["阶跃", "stepfun"],
    "alibaba": ["阿里", "通义", "qwen", "alibaba", "淘天"],
    "bytedance": ["字节", "豆包", "seed", "seedance", "bytedance"],
    "tencent": ["腾讯", "混元", "hunyuan", "yuanbao", "元宝", "wechat", "微信"],
    "openai": ["openai", "chatgpt", "gpt", "sora", "codex", "oai"],
    "anthropic": ["anthropic", "claude", "opus", "cowork", "mythos"],
    "google": ["google", "gemini", "deepmind", "antigravity", "synthid", "flow"],
    "meta": ["meta", "llama", "muse"],
    "deepseek": ["deepseek", "深度求索"],
    "spacex": ["spacex", "xai", "grok"],
}

IMPORTANT_MACRO_TERMS = [
    "openrouter",
    "openclaw",
    "agent",
    "智能体",
    "算力网",
    "六张网",
    "token调用",
    "ai融资",
    "ai搜索",
    "ai购物",
    "世界模型",
    "agentic",
]

DATE_OVERRIDES = [
    (re.compile(r"Gemini\s*3\.5|3\.5\s*Flash|Gemini Omni|Omni Flash|Antigravity\s*2\.0|Gemini Spark|Google Flow|Information agents|Search Agents", re.I), "2026-05-19", "web_verified_google_io"),
    (re.compile(r"Universal Cart|UCP|AP2|Agent Payments Protocol|Agent 电商", re.I), "2026-05-19", "web_verified_google_shopping"),
    (re.compile(r"SynthID|OpenAI.*SynthID|Kakao.*SynthID|ElevenLabs.*SynthID", re.I), "2026-05-19", "web_verified_synthid"),
    (re.compile(r"六张网|算力网.*7万亿|7\s*万亿.*算力网|全球算力网络", re.I), "2026-03-06", "web_verified_ndrc"),
]

SOURCE_OVERRIDES = [
    (re.compile(r"Gemini\s*3\.5", re.I), "https://blog.google/innovation-and-ai/models-and-research/gemini-models/gemini-3-5/"),
    (re.compile(r"Gemini Omni|Omni Flash|Google Flow", re.I), "https://blog.google/innovation-and-ai/technology/developers-tools/google-io-2026-collection/"),
    (re.compile(r"Antigravity", re.I), "https://blog.google/innovation-and-ai/technology/developers-tools/google-io-2026-developer-highlights/"),
    (re.compile(r"Gemini Spark|Information agents|Search Agents", re.I), "https://blog.google/innovation-and-ai/products/gemini-app/next-evolution-gemini-app/"),
    (re.compile(r"Universal Cart|UCP|AP2|Agent Payments Protocol|Agent 电商", re.I), "https://blog.google/products-and-platforms/products/shopping/google-shopping-cart/"),
    (re.compile(r"SynthID|OpenAI.*SynthID|Kakao.*SynthID|ElevenLabs.*SynthID", re.I), "https://openai.com/index/advancing-content-provenance/"),
    (re.compile(r"六张网|算力网.*7万亿|7\s*万亿.*算力网|全球算力网络", re.I), "https://zgeo.net/news/ndrc-7-trillion-investment-ai-infrastructure-digital-transformation"),
]


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
    if block_type == "child_page":
        return data.get("title", "")
    return rich_text_text(data.get("rich_text"))


def page_title(page):
    for prop in page.get("properties", {}).values():
        if prop.get("type") == "title":
            return rich_text_text(prop.get("title"))
    return page.get("id", "")


def page_week_date(title, fallback_date):
    match = re.search(r"(\d{2})(\d{2})(\d{2})", title or "")
    if not match:
        return fallback_date[:10]
    year, month, day = match.groups()
    return f"20{year}-{month}-{day}"


def notion_request(method, url, headers, **kwargs):
    for attempt in range(4):
        try:
            response = requests.request(method, url, headers=headers, timeout=45, **kwargs)
            response.raise_for_status()
            return response
        except requests.RequestException:
            if attempt == 3:
                raise
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError("unreachable")


def fetch_block_children(block_id, headers):
    url = f"https://api.notion.com/v1/blocks/{block_id}/children?page_size=100"
    blocks = []
    while url:
        response = notion_request("GET", url, headers)
        payload = response.json()
        blocks.extend(payload.get("results", []))
        cursor = payload.get("next_cursor")
        url = f"https://api.notion.com/v1/blocks/{block_id}/children?page_size=100&start_cursor={cursor}" if cursor else ""
    return blocks


def fetch_database_pages(source_id, headers, since_date):
    url = f"https://api.notion.com/v1/databases/{source_id}/query"
    pages = []
    cursor = None
    while True:
        body = {"page_size": 100, "sorts": [{"timestamp": "created_time", "direction": "ascending"}]}
        if cursor:
            body["start_cursor"] = cursor
        response = notion_request("POST", url, headers, json=body)
        payload = response.json()
        for page in payload.get("results", []):
            title = page_title(page)
            week_date = page_week_date(title, page.get("created_time", ""))
            if week_date >= since_date:
                pages.append(page)
        cursor = payload.get("next_cursor")
        if not payload.get("has_more"):
            break
    return pages


def source_url_for(text, fallback):
    for pattern, url in SOURCE_OVERRIDES:
        if pattern.search(text):
            return url
    return fallback


def row_from_block(block):
    return {
        "id": block.get("id", ""),
        "type": block.get("type", ""),
        "text": block_text(block).strip(),
    }


def page_rows(page_id, headers):
    rows = []
    for block in fetch_block_children(page_id, headers):
        top = row_from_block(block)
        rows.append(top)
        if block.get("has_children") and block.get("type") == "heading_1":
            for child in fetch_block_children(block["id"], headers):
                rows.append(row_from_block(child))
    return rows


def read_notion_items(since_date):
    headers = notion_headers()
    source_id = normalize_notion_id(os.getenv("NOTION_WEEKLY_SOURCE_ID"))
    if not headers or not source_id:
        return []
    try:
        pages = fetch_database_pages(source_id, headers, since_date)
    except Exception:
        pages = []

    items = []
    if pages:
        for page in pages:
            title = page_title(page)
            week_date = page_week_date(title, page.get("created_time", ""))
            rows = page_rows(page["id"], headers)
            current_section = ""
            for idx, row in enumerate(rows):
                text = row["text"]
                if not text:
                    continue
                if row["type"] == "heading_1":
                    current_section = text
                    continue
                if row["type"] not in {"heading_2", "heading_3"}:
                    continue
                if len(text) < 6:
                    continue
                items.append({
                    "date": week_date,
                    "title": text,
                    "summary": text,
                    "section": current_section,
                    "source_url": page.get("url", ""),
                    "source_page_title": title,
                })
        return items

    try:
        blocks = fetch_block_children(source_id, headers)
    except Exception as exc:
        print(f"[warn] cannot read Notion source: {exc}")
        return []
    for block in blocks:
        text = block_text(block)
        if text:
            items.append({
                "date": datetime.now(timezone.utc).date().isoformat(),
                "title": text[:80],
                "summary": text,
                "section": "",
                "source_url": "",
                "source_page_title": "",
            })
    return items


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
    if matches:
        return matches
    if any(term.lower() in lowered for term in IMPORTANT_MACRO_TERMS):
        return ["macro"]
    return []


def detect_event_type(text):
    lowered = text.lower()
    if re.search(r"发布|release|launch|上线|推出|开源|灰度|测试|上榜|进入", lowered):
        return "model_release"
    if re.search(r"价格|price|token plan|订阅|降价|涨价|计费|收费", lowered):
        return "pricing"
    if re.search(r"上市|融资|ipo|估值|财报|业绩|收入|arr|投资|收购|债券", lowered):
        return "financing"
    if re.search(r"爆发|潮|行业|生态|openrouter|openclaw|算力网|六张网|token调用", lowered):
        return "macro"
    return "company"


def short_title(title, summary):
    text = (title or summary or "").strip().replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"^[#\-\d、.\s]+", "", text)
    if len(text) <= 32:
        return text
    return text[:32]


def explicit_event_date(text, fallback_date):
    for pattern, date, basis in DATE_OVERRIDES:
        if pattern.search(text):
            return date, basis, False
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
    return fallback_date[:10], "notion_week_date_needs_web_check", True


def useful_item(item):
    text = f"{item.get('section', '')}\n{item.get('title', '')}\n{item.get('summary', '')}"
    if detect_vendor(text):
        return True
    return False


def event_id(date, vendor, title):
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower())[:42].strip("-") or "event"
    return f"draft-{date}-{vendor}-{slug}"


def build_events(items):
    events = {}
    for item in items:
        fallback_date = (item.get("date") or datetime.now(timezone.utc).date().isoformat())[:10]
        text = f"{item.get('section', '')}\n{item.get('title', '')}\n{item.get('summary', '')}"
        if not text.strip() or not useful_item(item):
            continue
        date, date_basis, needs_date_verification = explicit_event_date(text, fallback_date)
        event_type = detect_event_type(text)
        for vendor in detect_vendor(text):
            title = short_title(item.get("title", ""), item.get("summary", ""))
            if not title:
                continue
            key = event_id(date, vendor, title)
            events[key] = {
                "id": key,
                "date": date,
                "vendor": vendor,
                "company": vendor,
                "event_type": event_type,
                "title": title,
                "summary": (item.get("summary") or item.get("title") or "")[:300],
                "source": "notion",
                "source_url": source_url_for(text, item.get("source_url", "")),
                "source_page_title": item.get("source_page_title", ""),
                "confidence": "medium" if vendor != "macro" else "low",
                "date_basis": date_basis,
                "needs_date_verification": needs_date_verification,
                "status": "draft",
            }
    return sorted(events.values(), key=lambda event: (event["date"], event["vendor"], event["title"]))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--since", default=os.getenv("NOTION_SINCE_DATE") or f"{datetime.now(timezone.utc).year}-01-01")
    return parser.parse_args()


def main():
    args = parse_args()
    items = read_notion_items(args.since) or read_local_items()
    events = build_events(items)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "notion_api" if os.getenv("NOTION_API_KEY") and os.getenv("NOTION_WEEKLY_SOURCE_ID") else "local_export",
        "since": args.since,
        "events": events,
    }
    with DRAFT_FILE.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"wrote {len(events)} draft events from {len(items)} notion items since {args.since}")


if __name__ == "__main__":
    main()
