import json
import re
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MANUAL_FILE = ROOT / "ai_market_events_manual.json"
PRICING_FILE = ROOT / "artificial_analysis_pricing.json"
OUTPUT_FILE = ROOT / "ai_market_events.json"

PROVIDER_VENDOR_MAP = {
    "MiniMax": "minimax",
    "DeepSeek": "deepseek",
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

MODEL_VENDOR_RULES = [
    ("deepseek", ("deepseek", "deepseek-ai")),
    ("minimax", ("minimax", "minimaxai")),
    ("kimi", ("kimi", "moonshot")),
    ("zhipu", ("glm", "zai-org", "z.ai")),
    ("alibaba", ("qwen", "alibaba")),
    ("bytedance", ("doubao", "bytedance", "seed-")),
    ("tencent", ("hunyuan", "tencent")),
    ("openai", ("gpt", "openai/", "o1", "o3", "o4")),
    ("anthropic", ("claude", "anthropic")),
    ("google", ("gemini", "gemma", "google/")),
    ("meta", ("llama", "meta-llama")),
    ("spacex", ("grok", "xai")),
    ("stepfun", ("stepfun", "step-")),
]


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
    text = re.sub(r"\s+\((Non-reasoning|Reasoning|Reasoning,\s*[^)]+|high|low|Max Effort|High Effort)\)", "", text, flags=re.I)
    return text


def infer_model_vendor(row):
    haystack = " ".join(str(row.get(key) or "") for key in (
        "Model",
        "Host_Model",
        "Host_API_ID",
        "Model_Slug",
        "Host_Model_Slug",
    )).lower()
    for vendor, needles in MODEL_VENDOR_RULES:
        if any(needle in haystack for needle in needles):
            return vendor
    return PROVIDER_VENDOR_MAP.get(row.get("Provider") or "")


def event_dedupe_key(event):
    title = str(event.get("title") or "").lower()
    for token in ("发布", "release", "minimax", "mini max", "kimi", "模型"):
        title = title.replace(token, "")
    title = re.sub(r"[-_\s]+", "", title)
    title = re.sub(r"[^a-z0-9.]+", "", title)
    return (
        event.get("date", ""),
        event.get("vendor", ""),
        event.get("event_type", ""),
        title,
    )


def merge_events(events):
    priority = {"artificial_analysis": 0, "manual": 1, "notion": 2}
    merged = {}
    for event in events:
        key = event_dedupe_key(event)
        current = merged.get(key)
        if not current:
            merged[key] = event
            continue
        current_priority = priority.get(current.get("source", ""), 9)
        event_priority = priority.get(event.get("source", ""), 9)
        if event_priority < current_priority:
            event.setdefault("source_url", current.get("source_url", ""))
            event.setdefault("summary", current.get("summary", ""))
            merged[key] = event
        else:
            if event.get("source_url") and event["source_url"] not in current.get("source_url", ""):
                current["source_url"] = "; ".join(filter(None, [current.get("source_url", ""), event["source_url"]]))
            if event.get("source") and event["source"] not in current.get("source", ""):
                current["source"] = f"{current.get('source', '')}+{event['source']}"
    return list(merged.values())


def pricing_events():
    payload = read_json(PRICING_FILE, {})
    events = []
    seen = set()
    for row in payload.get("records", []):
        vendor = infer_model_vendor(row)
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
    events_by_id = {
        event["id"]: event
        for event in manual_payload.get("events", [])
        if event.get("event_type") != "model_release"
    }
    for event in pricing_events():
        events_by_id.setdefault(event["id"], event)
    events = sorted(merge_events(events_by_id.values()), key=lambda item: (item.get("date", ""), item.get("vendor", ""), item.get("title", "")))
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
