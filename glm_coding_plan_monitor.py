"""Monitor GLM Coding Plan daily sell-out times.

The public BigModel page is a Vue app, so the monitor uses Playwright to
observe the rendered pricing cards without clicking any purchase button.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo


SOURCE_URL = "https://bigmodel.cn/glm-coding"
API_BASE_URL = "https://bigmodel.cn/api"
API_BATCH_PREVIEW_URL = f"{API_BASE_URL}/biz/pay/batch-preview"
TZ = ZoneInfo("Asia/Shanghai")
SALE_START = time(10, 0, 0)
DEFAULT_MONITOR_START = time(9, 59, 50)
DEFAULT_MONITOR_END = time(10, 5, 0)
HISTORY_PATH = Path("glm_coding_plan_history.csv")
SNAPSHOT_PATH = Path("glm_coding_plan_snapshots.json")

TIERS = [
    ("lite", "Lite"),
    ("pro", "Pro"),
    ("max", "Max"),
]

DURATIONS = [
    ("month", "连续包月", "Month"),
    ("quarter", "连续包季", "Quarter"),
    ("year", "连续包年", "Year"),
]

PLAN_COLUMNS = [
    f"{tier_label}_{duration_label}"
    for _, tier_label in TIERS
    for _, _, duration_label in DURATIONS
]

PRODUCT_IDS = {
    "Lite_Month": "product-02434c",
    "Pro_Month": "product-1df3e1",
    "Max_Month": "product-2fc421",
    "Lite_Quarter": "product-b8ea38",
    "Pro_Quarter": "product-fef82f",
    "Max_Quarter": "product-5d3a03",
    "Lite_Year": "product-70a804",
    "Pro_Year": "product-5643e6",
    "Max_Year": "product-d46f8b",
}

STATUS_COLUMNS = [f"{column}_Status" for column in PLAN_COLUMNS]

CSV_HEADERS = [
    "Date",
    "Run_Start",
    "Run_End",
    "Source",
    *PLAN_COLUMNS,
    *STATUS_COLUMNS,
    "Notes",
]


@dataclass
class Observation:
    observed_at: str
    tier: str
    tier_label: str
    duration: str
    duration_label: str
    plan_key: str
    status: str
    button_text: str
    card_text: str
    source: str = "public_page"


def now_local() -> datetime:
    return datetime.now(TZ)


def combine_today(target: time, base: datetime | None = None) -> datetime:
    base = base or now_local()
    return datetime.combine(base.date(), target, tzinfo=TZ)


def clean_text(value: str | None) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def infer_status(card_text: str, button_text: str, disabled: bool) -> str:
    text = f"{card_text} {button_text}"
    if re.search(r"售罄|已抢光|抢光|无库存|今日已售|已售完", text):
        return "sold_out"
    if re.search(r"未开始|即将开售|10[:：]00|明日", text) and not re.search(r"订阅|抢购", button_text):
        return "pre_sale"
    if disabled and re.search(r"订阅|抢购|购买", text):
        return "disabled"
    # The public marketing page keeps showing "特惠订阅" even after the daily quota
    # is gone. Treat it as non-authoritative unless the rendered text explicitly
    # says sold out.
    return "unknown"


def plan_column(tier_label: str, duration_column_label: str) -> str:
    return f"{tier_label}_{duration_column_label}"


def status_to_cn(status: str) -> str:
    return {
        "sold_out": "已售罄",
        "available": "可订阅",
        "available_at_end": "窗口结束仍可订阅",
        "missed_window": "错过监控窗口",
        "auth_required": "需要登录态",
        "pre_sale": "未开售",
        "disabled": "按钮不可用",
        "unknown": "未知",
        "not_observed": "未观测",
        "error": "监控失败",
    }.get(status, status)


def bigmodel_api_headers() -> dict[str, str] | None:
    authorization = clean_text(os.environ.get("BIGMODEL_AUTHORIZATION"))
    cookie = clean_text(os.environ.get("BIGMODEL_COOKIE"))
    if not authorization and not cookie:
        return None

    headers = {
        "Accept": "application/json, text/plain, */*",
        "Content-Type": "application/json;charset=UTF-8",
        "Origin": "https://bigmodel.cn",
        "Referer": SOURCE_URL,
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
    }
    if authorization:
        headers["Authorization"] = authorization
    if cookie:
        headers["Cookie"] = cookie
    return headers


def product_column_parts(plan_key: str) -> tuple[str, str, str, str]:
    tier_label, duration_column_label = plan_key.split("_", 1)
    tier_key = tier_label.lower()
    duration = next((key for key, _, label in DURATIONS if label == duration_column_label), duration_column_label.lower())
    duration_label = next((label for key, label, column_label in DURATIONS if column_label == duration_column_label), duration_column_label)
    return tier_key, tier_label, duration, duration_label


def infer_api_status(product: dict[str, Any]) -> str:
    if product.get("soldOut"):
        return "sold_out"
    if product.get("forbidden") or product.get("disabled"):
        return "disabled"
    if product.get("isLimitBuy"):
        return "disabled"
    return "available"


def api_product_plan_key(product: dict[str, Any]) -> str | None:
    tier_label = clean_text(product.get("productName"))
    unit = clean_text(product.get("unit"))
    duration_column_label = {
        "month": "Month",
        "quarter": "Quarter",
        "year": "Year",
    }.get(unit)
    if tier_label in {"Lite", "Pro", "Max"} and duration_column_label:
        return plan_column(tier_label, duration_column_label)
    return None


def fetch_api_observations(headers: dict[str, str]) -> tuple[list[Observation], str | None]:
    payload = json.dumps({"invitationCode": ""}).encode("utf-8")
    request = Request(API_BATCH_PREVIEW_URL, data=payload, headers=headers, method="POST")
    try:
        with urlopen(request, timeout=30) as response:
            raw = response.read().decode("utf-8", errors="replace")
    except (HTTPError, URLError, TimeoutError) as exc:
        return [], f"api_error:{clean_text(str(exc))[:160]}"

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return [], "api_error:invalid_json"

    code = parsed.get("code")
    if code != 200:
        if code in {401, 1001}:
            return [], "api_auth_required"
        if code == 555:
            return [], "api_busy"
        return [], f"api_code_{code}:{clean_text(parsed.get('msg'))[:120]}"

    products = parsed.get("data", {}).get("productList") or []
    product_by_id = {item.get("productId"): item for item in products if item.get("productId")}
    product_by_plan_key = {
        plan_key: item
        for item in products
        for plan_key in [api_product_plan_key(item)]
        if plan_key
    }
    observed_at = now_local().isoformat(timespec="seconds")
    observations: list[Observation] = []
    for plan_key, product_id in PRODUCT_IDS.items():
        product = product_by_id.get(product_id) or product_by_plan_key.get(plan_key)
        if not product:
            continue
        tier_key, tier_label, duration, duration_label = product_column_parts(plan_key)
        status = infer_api_status(product)
        observations.append(
            Observation(
                observed_at=observed_at,
                tier=tier_key,
                tier_label=tier_label,
                duration=duration,
                duration_label=duration_label,
                plan_key=plan_key,
                status=status,
                button_text="batch-preview",
                card_text=json.dumps(
                    {
                        "product_id": product_id,
                        "product_name": product.get("productName"),
                        "sold_out": product.get("soldOut"),
                        "forbidden": product.get("forbidden"),
                        "is_limit_buy": product.get("isLimitBuy"),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                source="api_batch_preview",
            )
        )
    return observations, None


async def safe_goto(page: Any, url: str) -> None:
    try:
        await page.goto(url, wait_until="networkidle", timeout=60_000)
    except Exception:
        await page.goto(url, wait_until="domcontentloaded", timeout=60_000)
        await page.wait_for_timeout(2_000)


async def safe_reload(page: Any) -> None:
    try:
        await page.reload(wait_until="networkidle", timeout=60_000)
    except Exception:
        await page.reload(wait_until="domcontentloaded", timeout=60_000)
        await page.wait_for_timeout(1_500)


async def select_duration(page: Any, duration_label: str) -> None:
    tab = page.locator(".switch-tab-item", has_text=duration_label).first
    await tab.click(timeout=10_000)
    await page.wait_for_timeout(600)


async def read_cards(page: Any, duration_key: str, duration_label: str, duration_column_label: str) -> list[Observation]:
    cards = await page.evaluate(
        """
        () => {
          const candidates = Array.from(document.querySelectorAll('.package-card-box, .package-card'));
          const cards = candidates.filter((el) => /\\b(Lite|Pro|Max)\\b/.test(el.innerText || ''));
          return cards.map((card) => {
            const text = (card.innerText || '').replace(/\\s+/g, ' ').trim();
            const title = (text.match(/\\b(Lite|Pro|Max)\\b/) || [null, ''])[1];
            const button = card.querySelector('button');
            return {
              title,
              text,
              buttonText: button ? (button.innerText || '').replace(/\\s+/g, ' ').trim() : '',
              disabled: Boolean(button && (button.disabled || button.className.includes('disabled') || button.getAttribute('aria-disabled') === 'true'))
            };
          }).filter((item, index, arr) => item.title && arr.findIndex((other) => other.title === item.title) === index);
        }
        """
    )
    observed_at = now_local().isoformat(timespec="seconds")
    observations: list[Observation] = []
    for item in cards:
        tier_label = item["title"]
        tier_key = tier_label.lower()
        button_text = clean_text(item.get("buttonText"))
        card_text = clean_text(item.get("text"))
        status = infer_status(card_text, button_text, bool(item.get("disabled")))
        observations.append(
            Observation(
                observed_at=observed_at,
                tier=tier_key,
                tier_label=tier_label,
                duration=duration_key,
                duration_label=duration_label,
                plan_key=plan_column(tier_label, duration_column_label),
                status=status,
                button_text=button_text,
                card_text=card_text[:500],
            )
        )
    return observations


async def collect_once(page: Any) -> list[Observation]:
    observations: list[Observation] = []
    for duration_key, duration_label, duration_column_label in DURATIONS:
        await select_duration(page, duration_label)
        observations.extend(await read_cards(page, duration_key, duration_label, duration_column_label))
    return observations


async def monitor(args: argparse.Namespace) -> tuple[dict[str, str], list[Observation], list[str]]:
    try:
        from playwright.async_api import async_playwright
    except ImportError as exc:
        raise SystemExit("Playwright is required. Install with: pip install playwright && python -m playwright install chromium") from exc

    notes: list[str] = []
    run_date = now_local().date().isoformat()
    target_start = combine_today(DEFAULT_MONITOR_START)
    target_end = combine_today(DEFAULT_MONITOR_END)
    sale_start = combine_today(SALE_START)
    current = now_local()
    started_after_window = False

    if args.once:
        target_start = current
        target_end = current
        notes.append("once_mode")
    elif current < target_start:
        await asyncio.sleep((target_start - current).total_seconds())
    elif current > target_end:
        started_after_window = True
        notes.append("started_after_window")
        target_end = current
    elif current > sale_start:
        notes.append("started_after_10")

    first_sold_out: dict[str, str] = {}
    latest_status: dict[str, str] = {}
    latest_source: dict[str, str] = {}
    observations: list[Observation] = []
    api_headers = bigmodel_api_headers()
    if not api_headers:
        notes.append("no_bigmodel_auth")

    def record_batch(batch: list[Observation], observed_now: datetime) -> None:
        observations.extend(batch)
        for item in batch:
            latest_status[item.plan_key] = item.status
            latest_source[item.plan_key] = item.source
            if observed_now >= sale_start and item.status == "sold_out" and item.plan_key not in first_sold_out:
                first_sold_out[item.plan_key] = observed_now.strftime("%H:%M:%S")

    used_api = False
    if api_headers:
        while True:
            observed_now = now_local()
            batch, api_note = await asyncio.to_thread(fetch_api_observations, api_headers)
            if api_note and api_note not in notes:
                notes.append(api_note)
            if not batch:
                break
            used_api = True
            record_batch(batch, observed_now)
            if args.once or now_local() >= target_end:
                break
            await asyncio.sleep(max(1, args.poll_interval))

    if not used_api:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True, args=["--no-sandbox"])
            context = await browser.new_context(
                locale="zh-CN",
                timezone_id="Asia/Shanghai",
                viewport={"width": 1440, "height": 1200},
            )
            page = await context.new_page()

            async def route_handler(route: Any) -> None:
                if route.request.resource_type in {"image", "font", "media"}:
                    await route.abort()
                else:
                    await route.continue_()

            await page.route("**/*", route_handler)
            await safe_goto(page, SOURCE_URL)

            while True:
                observed_now = now_local()
                if args.reload_each_poll:
                    await safe_reload(page)
                batch = await collect_once(page)
                if "public_page_status_only" not in notes:
                    notes.append("public_page_status_only")
                record_batch(batch, observed_now)

                if args.once or now_local() >= target_end:
                    break
                await asyncio.sleep(max(1, args.poll_interval))

            await browser.close()

    row = {
        "Date": run_date,
        "Run_Start": target_start.strftime("%H:%M:%S"),
        "Run_End": now_local().strftime("%H:%M:%S"),
        "Source": SOURCE_URL,
        "Notes": ";".join(notes),
    }
    auth_required = "api_auth_required" in notes
    for column in PLAN_COLUMNS:
        row[column] = first_sold_out.get(column, "")
        raw_status = latest_status.get(column, "not_observed")
        reliable_inventory = latest_source.get(column) == "api_batch_preview"
        if column in first_sold_out:
            raw_status = "sold_out"
        elif raw_status == "sold_out":
            raw_status = "sold_out"
        elif auth_required and not reliable_inventory:
            raw_status = "auth_required"
        elif started_after_window and not reliable_inventory:
            raw_status = "missed_window"
        elif raw_status == "available" and reliable_inventory:
            raw_status = "available_at_end"
        elif raw_status == "available":
            raw_status = "unknown"
        row[f"{column}_Status"] = raw_status
    return row, observations, notes


def read_history(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def write_history(path: Path, rows: list[dict[str, str]]) -> None:
    normalized = []
    for row in rows:
        normalized.append({header: clean_text(row.get(header, "")).replace(",", "，") for header in CSV_HEADERS})
    normalized.sort(key=lambda item: item["Date"])
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_HEADERS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(normalized)


def upsert_history(row: dict[str, str]) -> None:
    rows = [item for item in read_history(HISTORY_PATH) if item.get("Date") != row["Date"]]
    rows.append(row)
    write_history(HISTORY_PATH, rows)


def write_snapshot(row: dict[str, str], observations: list[Observation], notes: list[str]) -> None:
    payload = {
        "generated_at": now_local().isoformat(timespec="seconds"),
        "source_url": SOURCE_URL,
        "monitor_window": {
            "start": DEFAULT_MONITOR_START.isoformat(),
            "sale_start": SALE_START.isoformat(),
            "end": DEFAULT_MONITOR_END.isoformat(),
            "timezone": "Asia/Shanghai",
        },
        "latest_run": row,
        "notes": notes,
        "observations": [
            {
                "observed_at": item.observed_at,
                "tier": item.tier,
                "tier_label": item.tier_label,
                "duration": item.duration,
                "duration_label": item.duration_label,
                "plan_key": item.plan_key,
                "status": item.status,
                "status_label": status_to_cn(item.status),
                "button_text": item.button_text,
                "card_text": item.card_text,
                "source": item.source,
            }
            for item in observations
        ],
    }
    SNAPSHOT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="Run one observation immediately.")
    parser.add_argument("--poll-interval", type=float, default=5.0, help="Seconds between polling rounds.")
    parser.add_argument("--reload-each-poll", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    try:
        row, observations, notes = await monitor(args)
    except Exception as exc:
        notes = [f"error:{clean_text(str(exc))[:180]}"]
        row = {
            "Date": now_local().date().isoformat(),
            "Run_Start": combine_today(DEFAULT_MONITOR_START).strftime("%H:%M:%S"),
            "Run_End": now_local().strftime("%H:%M:%S"),
            "Source": SOURCE_URL,
            "Notes": ";".join(notes).replace(",", "，"),
        }
        for column in PLAN_COLUMNS:
            row[column] = ""
            row[f"{column}_Status"] = "error"
        observations = []
    if not observations:
        notes.append("no_cards_observed")
        row["Notes"] = ";".join(notes)
    upsert_history(row)
    write_snapshot(row, observations, notes)
    print(json.dumps({"row": row, "observations": len(observations)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
