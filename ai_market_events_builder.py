import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MANUAL_FILE = ROOT / "ai_market_events_manual.json"
PRICING_FILE = ROOT / "artificial_analysis_pricing.json"
OUTPUT_FILE = ROOT / "ai_market_events.json"

PROVIDER_VENDOR_MAP = {
    "MiniMax": "minimax",
    "Z.ai": "zhipu",
    "Kimi": "kimi",
    "Moonshot AI": "kimi",
    "Alibaba Cloud": "alibaba",
    "Tencent Cloud": "tencent",
    "OpenAI": "openai",
    "Anthropic": "anthropic",
    "Google": "google",
    "Google Vertex AI": "google",
    "Meta": "meta",
}


def read_json(path, fallback):
    if not path.exists():
        return fallback
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def compact_model_name(name):
    if not name:
        return ""
    text = str(name).strip()
    for suffix in (" Preview", " Instruct"):
        text = text.replace(suffix, "")
    return text


def pricing_events():
    payload = read_json(PRICING_FILE, {})
    events = []
    seen = set()
    for row in payload.get("records", []):
        provider = row.get("Provider") or ""
        vendor = PROVIDER_VENDOR_MAP.get(provider)
        release_date = (row.get("Release_Date") or "").strip()
        model = compact_model_name(row.get("Model") or row.get("Host_Model"))
        if not vendor or not release_date or not model:
            continue
        key = (release_date, vendor, model)
        if key in seen:
            continue
        seen.add(key)
        events.append({
            "id": f"{release_date}-{vendor}-{model}".lower().replace(" ", "-").replace("/", "-"),
            "date": release_date,
            "vendor": vendor,
            "company": vendor,
            "event_type": "model_release",
            "title": f"{model} 发布",
            "summary": f"Artificial Analysis 记录 {model} 的发布时间为 {release_date}。",
            "source": "artificial_analysis",
            "source_url": row.get("Source_URL") or payload.get("source_url") or "",
            "status": "approved",
        })
    return events


def main():
    manual_payload = read_json(MANUAL_FILE, {"events": []})
    events_by_id = {}
    for event in manual_payload.get("events", []):
        events_by_id[event["id"]] = event
    for event in pricing_events():
        events_by_id.setdefault(event["id"], event)
    events = sorted(events_by_id.values(), key=lambda item: (item.get("date", ""), item.get("vendor", ""), item.get("title", "")))
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "manual_and_artificial_analysis",
        "events": events,
    }
    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"wrote {len(events)} events to {OUTPUT_FILE.name}")


if __name__ == "__main__":
    main()
