import argparse
import json
import os
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlparse

import requests


ROOT = Path(__file__).resolve().parent
LOCAL_NEWS_FILE = ROOT / "notion_weekly_news.json"
DRAFT_FILE = ROOT / "ai_market_events_draft.json"
AA_PRICING_FILE = ROOT / "artificial_analysis_pricing.json"
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

AI_NATIVE_VENDORS = {"openai", "anthropic", "google", "meta", "deepseek", "kimi", "minimax", "zhipu", "bytedance", "tencent", "alibaba", "spacex"}

AI_RELEVANCE_TERMS = [
    "ai", "aigc", "agi", "llm", "大模型", "模型", "多模态", "推理", "智能体", "agent",
    "openclaw", "chatgpt", "claude", "gemini", "gpt", "sora", "codex",
    "deepseek", "kimi", "minimax", "glm", "qwen", "通义", "豆包", "混元", "grok",
    "llama", "mimo", "hailuo", "海螺", "seedance", "veo", "flow", "antigravity",
    "算力", "gpu", "tpu", "芯片", "token", "api", "推理成本", "训练", "开源",
]

HIGH_VALUE_BUSINESS_TERMS = [
    "融资", "估值", "上市", "ipo", "募资", "投资", "收购", "债券", "arr", "收入",
    "付费", "订阅", "价格", "降价", "涨价", "token plan", "商业化", "算力协议",
]

LOW_VALUE_TERMS = [
    "财季预览", "核心电商", "广告变现", "平台成瘾", "判罚", "游戏行业",
    "效率应用排行", "访问量前20", "iphone maus", "市场情绪", "值得一看",
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

PREFIX_VENDOR_MAP = {
    "OpenAI": "openai",
    "OAI": "openai",
    "Anthropic": "anthropic",
    "Claude": "anthropic",
    "Google": "google",
    "Gemini": "google",
    "GCP": "google",
    "Meta": "meta",
    "Deepseek": "deepseek",
    "DeepSeek": "deepseek",
    "Kimi": "kimi",
    "月之暗面": "kimi",
    "Minimax": "minimax",
    "MiniMax": "minimax",
    "通义千问": "alibaba",
    "阿里": "alibaba",
    "Alibaba": "alibaba",
    "Qwen": "alibaba",
    "字节": "bytedance",
    "Bytedance": "bytedance",
    "ByteDance": "bytedance",
    "豆包": "bytedance",
    "腾讯": "tencent",
    "混元": "tencent",
    "xAI": "spacex",
    "Grok": "spacex",
    "SpaceX": "spacex",
    "智谱": "zhipu",
    "GLM": "zhipu",
    "Similarweb数据": "macro",
    "Similarweb": "macro",
}

NON_TARGET_PREFIXES = {
    "Amazon", "AWS", "Oracle", "微软", "Microsoft", "Apple", "Cursor", "MongoDB",
    "NVIDIA", "MSFT", "CRWV", "CoreWeave", "Cloudflare", "Reddit", "Notion",
    "SAP", "CRM", "Uber", "PLTR", "NET", "RDDT", "Unity", "Shopify", "Tesla",
    "CloudX", "ADBE", "TTD", "NOW", "Snowflake", "Cerebras", "Ant", "蚂蚁灵波",
}

DROP_PATTERNS = [
    re.compile(r"专题\*?$"),
    re.compile(r"模型一月情况|模型发布专题|CapEx专题|AWS：?$"),
    re.compile(r"值得一看的研究|研究：|TMT市场情绪"),
    re.compile(r"^\s*(应用|广告|软件行业|模型与CSP)\s*$"),
    re.compile(r"财季预览|核心电商业务|广告变现|平台成瘾|Unity CEO谈世界模型"),
    re.compile(r"美国Appstore 效率应用排行|Gemini首次进入全球访问量前20|iPhone MAUs"),
    re.compile(r"Seedance\*?$"),
    re.compile(r"ChatGPT Health|Codex推出宠物功能|Google于安卓推出Gemini Intelligence"),
    re.compile(r"OpenRouter|Openrouter|OpenRouter Token|Tokens?消耗|Token数据"),
]

TITLE_OVERRIDES = [
    (re.compile(r"Similarweb.*GPT.*Gemini.*Grok.*Deepseek", re.I), "Similarweb：Gemini/Grok 扩张", "Similarweb 数据显示 GPT 份额止跌、Gemini 继续扩张，Grok 首次超过 DeepSeek。"),
    (re.compile(r"LLM公司对OpenClaw态度分歧", re.I), "OpenClaw 生态分化", "OpenAI 主动拥抱 OpenClaw，Anthropic 与 Google 因经济原因限制第三方工具使用。"),
    (re.compile(r"Google.*I/O|I/O：Agent", re.I), "Google I/O Agent 生态发布", "Google I/O 发布 Gemini 3.5 Flash、Omni/Flow、Antigravity 2.0、Search Agents 与 Agent 电商能力。"),
    (re.compile(r"OpenAI.*122B|OpenAI融资\$110B|OpenAI完成\$122B", re.I), "OpenAI 完成 122B 融资", "OpenAI 完成约 1220 亿美元融资，投后估值约 8520 亿美元。"),
    (re.compile(r"OpenAI.*\$110B", re.I), "OpenAI 新融资估值上行", "周报记录 OpenAI 大额融资与估值继续上行。"),
    (re.compile(r"Anthropic.*Opus 4\.7|Opus4\.7", re.I), "Claude Opus 4.7 发布", "Anthropic 发布 Claude Opus 4.7，并同步推出 Claude Design 等能力。"),
    (re.compile(r"Deepseek发布V4|DeepSeek发布V4|DeepSeek V4", re.I), "DeepSeek V4 发布", "DeepSeek 发布 V4 Flash/Pro，进一步强化国产旗舰模型叙事。"),
    (re.compile(r"腾讯、阿里.*Deepseek|Tencent.*Alibaba.*DeepSeek", re.I), "腾讯阿里洽投 DeepSeek", "腾讯、阿里被报道洽谈投资 DeepSeek，估值超过 200 亿美元。"),
    (re.compile(r"Kimi：发布新模型Kimi K2\.5|Kimi K2\.5上榜", re.I), "Kimi K2.5 发布", "Kimi K2.5 发布，推动月之暗面模型升级叙事。"),
    (re.compile(r"Kimi.*K2\.6", re.I), "Kimi K2.6 发布", "Kimi K2.6 发布，延续月之暗面高频模型更新节奏。"),
    (re.compile(r"Minimax.*M2\.7|MiniMax.*M2\.7|M2\.7", re.I), "MiniMax M2.7 发布", "MiniMax M2.7 发布，强化 MiniMax 模型能力升级叙事。"),
    (re.compile(r"GLM-5\.1|智谱.*GLM 5\.1", re.I), "GLM-5.1 发布", "智谱 GLM-5.1 发布，强化国产模型升级节奏。"),
    (re.compile(r"六张网|算力网", re.I), "中国推进算力网建设", "中国推进六张网与算力网建设，AI 基础设施投资预期升温。"),
    (re.compile(r"SpaceX.*IPO|SpaceX最早.*IPO", re.I), "SpaceX IPO 预期升温", "周报记录 SpaceX 最早可能提交 IPO 并寻求大额募资。"),
    (re.compile(r"字节：估值", re.I), "字节估值上调", "周报记录字节估值较前期继续上调。"),
    (re.compile(r"Bytedance：进军云计算市场|字节.*云计算", re.I), "字节进军云计算", "字节被记录为加速进入云计算与 AI 基础设施市场。"),
    (re.compile(r"腾讯.*HY 3\.0|混元3", re.I), "腾讯混元 3 发布/预热", "腾讯混元 3/HY 3.0 进入发布窗口，强化腾讯模型厂商叙事。"),
]

VENDOR_TITLE_OVERRIDES = [
    (re.compile(r"MiniMax M2\.7|M2\.7|Minimax.*M2\.7", re.I), "minimax"),
    (re.compile(r"MiniMax M2\.5|M2\.5|MiMo|Mimo", re.I), "minimax"),
    (re.compile(r"Kimi K2\.5|Kimi K2\.6|月之暗面", re.I), "kimi"),
    (re.compile(r"GLM-5|GLM-5\.1|智谱", re.I), "zhipu"),
    (re.compile(r"DeepSeek V4|Deepseek", re.I), "deepseek"),
    (re.compile(r"Grok 4\.2|Grok 4\.20|xAI", re.I), "spacex"),
    (re.compile(r"Claude Opus|Anthropic|Claude", re.I), "anthropic"),
    (re.compile(r"Gemini|Google", re.I), "google"),
    (re.compile(r"Qwen|阿里|Alibaba|通义", re.I), "alibaba"),
    (re.compile(r"Seedance|豆包|字节|ByteDance", re.I), "bytedance"),
    (re.compile(r"混元|腾讯|Hunyuan", re.I), "tencent"),
    (re.compile(r"OpenAI|ChatGPT|GPT-|Codex|Sora", re.I), "openai"),
]

DATE_TITLE_OVERRIDES = [
    (re.compile(r"OpenAI 完成 122B 融资|OpenAI.*122B|OpenAI融资\$110B", re.I), "2026-03-31", "web_verified_openai_funding"),
    (re.compile(r"GPT-5\.3-Codex|GPT-5\.3 Codex", re.I), "2026-02-05", "artificial_analysis_release_date"),
    (re.compile(r"GPT-5\.5|Image 2\.0", re.I), "2026-04-23", "artificial_analysis_release_date"),
    (re.compile(r"Claude Opus 4\.6|Opus 4\.6", re.I), "2026-02-05", "artificial_analysis_release_date"),
    (re.compile(r"Claude Opus 4\.7|Opus4\.7", re.I), "2026-04-16", "artificial_analysis_release_date"),
    (re.compile(r"DeepSeek V4|Deepseek发布V4|DeepSeek发布V4", re.I), "2026-04-24", "artificial_analysis_release_date"),
    (re.compile(r"Kimi.*K2\.5", re.I), "2026-01-27", "artificial_analysis_release_date"),
    (re.compile(r"Kimi.*K2\.6", re.I), "2026-04-20", "artificial_analysis_release_date"),
    (re.compile(r"MiniMax.*M2\.7|Minimax.*M2\.7|M2\.7", re.I), "2026-03-18", "artificial_analysis_release_date"),
    (re.compile(r"GLM[-\s]?5\.1", re.I), "2026-04-07", "artificial_analysis_release_date"),
    (re.compile(r"GLM[-\s]?5(?!\.)|GLM-5V", re.I), "2026-02-11", "artificial_analysis_release_date"),
    (re.compile(r"Gemini 3\.1 Flash Lite|Gemini 3\.1 Flash-Lite", re.I), "2026-03-03", "artificial_analysis_release_date"),
    (re.compile(r"Grok 4\.2|Grok 4\.20", re.I), "2026-03-10", "artificial_analysis_release_date"),
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


def clean_text(text):
    text = (text or "").strip().replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"^[#\-\d、.\s]+", "", text)
    return text.strip()


def split_event_fragments(text):
    text = clean_text(text)
    fragments = [part.strip() for part in re.split(r"[；;]\s*", text) if part.strip()]
    expanded = []
    for fragment in fragments:
        if re.search(r"、", fragment) and re.search(r"发布|推出|开源|上线|release|launch", fragment, re.I):
            expanded.extend(part.strip() for part in re.split(r"、", fragment) if part.strip())
        else:
            expanded.append(fragment)
    return expanded


def text_prefix(text):
    match = re.match(r"^([A-Za-z][A-Za-z0-9 .+\-/]{1,28}|[\u4e00-\u9fa5A-Za-z0-9 .+\-/]{1,18})[:：]", text)
    return match.group(1).strip() if match else ""


def vendor_from_prefix(text):
    prefix = text_prefix(text)
    if not prefix:
        return ""
    normalized = prefix.strip()
    for name, vendor in PREFIX_VENDOR_MAP.items():
        if normalized.lower() == name.lower():
            return vendor
    if normalized in NON_TARGET_PREFIXES:
        return "drop"
    return ""


def is_market_data(text):
    return bool(re.search(r"^(Similarweb|美国Appstore|Appstore|AI 留存率|AI流量出口)", text, re.I))


def should_drop_text(text):
    cleaned = clean_text(text)
    if len(cleaned) < 8:
        return True
    if any(pattern.search(cleaned) for pattern in DROP_PATTERNS):
        return True
    lowered = cleaned.lower()
    if any(term.lower() in lowered for term in LOW_VALUE_TERMS) and not any(term.lower() in lowered for term in {"openai", "anthropic", "deepseek", "minimax", "kimi", "glm", "qwen"}):
        return True
    prefix = text_prefix(cleaned)
    return prefix in NON_TARGET_PREFIXES


def has_ai_relevance(text):
    lowered = text.lower()
    return any(term.lower() in lowered for term in AI_RELEVANCE_TERMS)


def has_business_relevance(text):
    lowered = text.lower()
    return any(term.lower() in lowered for term in HIGH_VALUE_BUSINESS_TERMS)


def is_high_value_event(text, vendor, event_type):
    cleaned = clean_text(text)
    lowered = cleaned.lower()
    if should_drop_text(cleaned):
        return False
    if event_type == "market_data":
        return bool(re.search(r"similarweb.*(gpt|gemini|grok|deepseek|claude|ai)|ai 留存率|ai流量出口", lowered, re.I))
    if event_type == "macro":
        return has_ai_relevance(lowered) and bool(re.search(r"openclaw|算力网|六张网|ai融资|生态|爆发|agent", lowered, re.I))
    if event_type == "model_release":
        return has_ai_relevance(lowered) and bool(re.search(r"发布|release|launch|上线|推出|开源|preview|beta|灰度", lowered, re.I))
    if event_type in {"financing", "pricing"}:
        return vendor in AI_NATIVE_VENDORS and (has_ai_relevance(lowered) or has_business_relevance(lowered))
    if vendor in AI_NATIVE_VENDORS:
        return has_ai_relevance(lowered) and bool(re.search(r"合作|协议|收入|arr|招聘|团队|算力|芯片|数据中心|agent|智能体|api|企业|政府|广告|设备|搜索|购物|生态|使用量|调用", lowered, re.I))
    return False


def subject_hint(text):
    if is_market_data(text):
        return "macro"
    prefix_vendor = vendor_from_prefix(text)
    if prefix_vendor and prefix_vendor not in {"drop", "macro"}:
        return prefix_vendor
    lowered = text.lower()
    first_positions = []
    for vendor, aliases in VENDOR_ALIASES.items():
        if vendor == "spacex":
            aliases = ["spacex", "xai", "grok"]
        for alias in aliases:
            if alias in {"m1", "m2", "seed", "flow", "agent", "oai", "gpt"}:
                continue
            pos = lowered.find(alias.lower())
            if pos >= 0:
                first_positions.append((pos, vendor))
                break
    if not first_positions:
        for pattern, vendor in VENDOR_TITLE_OVERRIDES:
            if pattern.search(text):
                return vendor
        return ""
    first_positions.sort()
    return first_positions[0][1]


def rewrite_title_summary(text, vendor, event_type):
    cleaned = clean_text(text)
    for pattern, title, summary in TITLE_OVERRIDES:
        if pattern.search(cleaned):
            if event_type == "market_data" and re.search(r"发布$", title):
                continue
            return title, summary
    if "：" in cleaned or ":" in cleaned:
        subject, rest = re.split(r"[:：]", cleaned, maxsplit=1)
        rest = rest.strip()
        if subject in PREFIX_VENDOR_MAP or subject in {"OAI", "GCP"}:
            cleaned = f"{subject}：{rest}"
    title = cleaned
    title = re.sub(r"^OpenAI：", "OpenAI：", title)
    title = re.sub(r"^Anthropic：", "Anthropic：", title)
    title = re.sub(r"^Google：", "Google：", title)
    title = re.sub(r"^Alibaba：", "阿里：", title)
    title = re.sub(r"^Bytedance：", "字节：", title)
    if len(title) > 30:
        parts = re.split(r"，|,|；|;", title)
        title = parts[0].strip()
        if len(title) > 30:
            title = title[:28].rstrip() + "..."
    summary = cleaned
    if event_type == "market_data":
        summary = f"周报市场数据记录：{cleaned}"
    elif event_type == "model_release":
        summary = f"周报记录模型/产品更新：{cleaned}"
    elif event_type == "financing":
        summary = f"周报记录投融资或估值事件：{cleaned}"
    elif event_type == "pricing":
        summary = f"周报记录定价或商业化变化：{cleaned}"
    else:
        summary = f"周报记录公司事件：{cleaned}"
    return title, summary[:320]


def aa_release_lookup():
    if not AA_PRICING_FILE.exists():
        return []
    try:
        with AA_PRICING_FILE.open("r", encoding="utf-8") as f:
            records = json.load(f).get("records", [])
    except Exception:
        return []
    models = {}
    for row in records:
        model = row.get("Model") or ""
        date = row.get("Release_Date") or ""
        if not model or not date:
            continue
        simple = re.sub(r"\s*\([^)]*\)", "", model).strip()
        models.setdefault(simple.lower(), (simple, date))
    return sorted(models.values(), key=lambda item: len(item[0]), reverse=True)


AA_RELEASES = aa_release_lookup()


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
    cleaned = clean_text(text)
    if should_drop_text(cleaned):
        return []
    if is_market_data(cleaned):
        return ["macro"]
    prefix_vendor = vendor_from_prefix(cleaned)
    if prefix_vendor == "drop":
        return []
    if prefix_vendor:
        return [prefix_vendor]
    hinted = subject_hint(cleaned)
    if hinted:
        return [hinted]
    lowered = text.lower()
    matches = []
    for vendor, aliases in VENDOR_ALIASES.items():
        for alias in aliases:
            if alias in {"m1", "m2", "seed", "flow", "agent", "oai", "gpt"}:
                continue
            if re.search(rf"(?<![A-Za-z0-9]){re.escape(alias.lower())}(?![A-Za-z0-9])", lowered):
                matches.append(vendor)
                break
    if matches:
        return matches
    if any(term.lower() in lowered for term in IMPORTANT_MACRO_TERMS):
        return ["macro"]
    return []


def detect_event_type(text):
    lowered = text.lower()
    if is_market_data(text):
        return "market_data"
    if re.search(r"价格|price|token plan|订阅|降价|涨价|计费|收费", lowered):
        return "pricing"
    if re.search(r"上市|融资|ipo|估值|募资|领投|投后估值|收购|债券", lowered):
        return "financing"
    if re.search(r"爆发|潮|行业|生态|openclaw|算力网|六张网", lowered):
        return "macro"
    if re.search(r"发布|release|launch|上线|推出|开源|灰度|preview|beta", lowered):
        if not re.search(r"广告|购物搜索|访问量|份额|上榜|进入.*前\d+", lowered):
            return "model_release"
    return "company"


def short_title(title, summary):
    text = clean_text(title or summary or "")
    if len(text) <= 32:
        return text
    return text[:30].rstrip() + "..."


def explicit_event_date(text, fallback_date, event_type):
    if event_type == "model_release":
        for pattern, date, basis in DATE_TITLE_OVERRIDES:
            if pattern.search(text):
                return date, basis, False
    for pattern, date, basis in DATE_OVERRIDES:
        if pattern.search(text):
            return date, basis, False
    lowered = text.lower()
    if event_type == "model_release":
        for model, release_date in AA_RELEASES:
            if model.lower() in lowered and re.search(r"发布|release|launch|上线|推出|开源|preview|beta", lowered):
                return release_date, "artificial_analysis_release_date", False
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
    vendors = detect_vendor(text)
    if not vendors:
        return False
    event_type = detect_event_type(text)
    return any(is_high_value_event(text, vendor, event_type) for vendor in vendors)


def event_id(date, vendor, title):
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower())[:42].strip("-") or "event"
    return f"draft-{date}-{vendor}-{slug}"


def build_events(items):
    events = {}
    for item in items:
        fallback_date = (item.get("date") or datetime.now(timezone.utc).date().isoformat())[:10]
        raw_title = item.get("title", "")
        fragments = split_event_fragments(raw_title)
        for fragment in fragments:
            text = fragment
            if not text.strip() or should_drop_text(text) or not useful_item({"title": text, "summary": text, "section": ""}):
                continue
            event_type = detect_event_type(text)
            has_manual_date = any(pattern.search(text) for pattern, _, _ in DATE_OVERRIDES)
            if event_type == "model_release" or has_manual_date:
                date, date_basis, needs_date_verification = explicit_event_date(text, fallback_date, event_type)
            else:
                date, date_basis, needs_date_verification = fallback_date, "notion_week_observation_date", True
            for vendor in detect_vendor(text):
                if not is_high_value_event(text, vendor, event_type):
                    continue
                title, summary = rewrite_title_summary(text, vendor, event_type)
                if not title:
                    continue
                if subject_hint(text) and vendor != subject_hint(text) and event_type not in {"market_data", "macro"}:
                    continue
                key = event_id(date, vendor, title)
                events[key] = {
                    "id": key,
                    "date": date,
                    "vendor": vendor,
                    "company": vendor,
                    "event_type": event_type,
                    "title": title,
                    "summary": summary,
                    "source": "notion",
                    "source_url": source_url_for(text, item.get("source_url", "")),
                    "source_page_title": item.get("source_page_title", ""),
                    "source_section": item.get("section", ""),
                    "raw_title": raw_title,
                    "confidence": "medium" if vendor != "macro" else "low",
                    "date_basis": date_basis,
                    "needs_date_verification": needs_date_verification,
                    "status": "draft",
                }
    return sorted(events.values(), key=lambda event: (event["date"], event["vendor"], event["title"]))


def default_since_date():
    lookback_days = int(os.getenv("NOTION_LOOKBACK_DAYS") or "10")
    return (datetime.now(timezone.utc).date() - timedelta(days=lookback_days)).isoformat()


def merge_existing_events(events, replace=False):
    if replace or not DRAFT_FILE.exists():
        return events
    try:
        with DRAFT_FILE.open("r", encoding="utf-8") as f:
            existing = json.load(f).get("events", [])
    except Exception:
        existing = []
    merged = {event.get("id"): event for event in existing if event.get("id")}
    for event in events:
        merged[event["id"]] = event
    return sorted(merged.values(), key=lambda event: (event.get("date", ""), event.get("vendor", ""), event.get("title", "")))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--since", default=os.getenv("NOTION_SINCE_DATE") or default_since_date())
    parser.add_argument("--replace", action="store_true", help="Replace draft file instead of merging into existing draft pool.")
    return parser.parse_args()


def main():
    args = parse_args()
    items = read_notion_items(args.since) or read_local_items()
    new_events = build_events(items)
    events = merge_existing_events(new_events, replace=args.replace)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "notion_api" if os.getenv("NOTION_API_KEY") and os.getenv("NOTION_WEEKLY_SOURCE_ID") else "local_export",
        "since": args.since,
        "mode": "replace" if args.replace else "merge_existing",
        "new_event_count": len(new_events),
        "events": events,
    }
    with DRAFT_FILE.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"wrote {len(events)} draft events ({len(new_events)} new/updated) from {len(items)} notion items since {args.since}")


if __name__ == "__main__":
    main()
