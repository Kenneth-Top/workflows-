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
DEFAULT_MONITOR_END = time(11, 0, 0)
DEFAULT_LATE_RETRY_SECONDS = 180
DEFAULT_API_POLL_INTERVAL = 0.5
DEFAULT_API_TIMEOUT = 2.0
DEFAULT_PAGE_POLL_INTERVAL = 0.5
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


def format_local_time(value: datetime) -> str:
    return value.strftime("%H:%M:%S.%f")[:-3]


def combine_today(target: time, base: datetime | None = None) -> datetime:
    base = base or now_local()
    return datetime.combine(base.date(), target, tzinfo=TZ)


def clean_text(value: str | None) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def infer_status(card_text: str, button_text: str, disabled: bool) -> str:
    text = f"{card_text} {button_text}"
    if re.search(r"售罄|已抢光|抢光|无库存|今日已售|已售完", text):
        return "sold_out"
    if re.search(r"抢购人数过多|请刷新再试|系统繁忙|稍后再试", text):
        return "purchase_busy"
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
        "purchase_busy": "抢购拥堵",
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


def bigmodel_cookie_header() -> str:
    return clean_text(os.environ.get("BIGMODEL_COOKIE"))


def bigmodel_authorization_header() -> str:
    return clean_text(os.environ.get("BIGMODEL_AUTHORIZATION"))


def playwright_cookies_from_header(cookie_header: str) -> list[dict[str, Any]]:
    cookies = []
    for part in cookie_header.split(";"):
        if "=" not in part:
            continue
        name, value = part.split("=", 1)
        name = clean_text(name)
        value = value.strip()
        if not name:
            continue
        cookies.append(
            {
                "name": name,
                "value": value,
                "domain": ".bigmodel.cn",
                "path": "/",
                "secure": True,
                "sameSite": "Lax",
            }
        )
    return cookies


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


def fetch_api_observations(headers: dict[str, str], timeout: float = DEFAULT_API_TIMEOUT) -> tuple[list[Observation], str | None]:
    payload = json.dumps({"invitationCode": ""}).encode("utf-8")
    request = Request(API_BATCH_PREVIEW_URL, data=payload, headers=headers, method="POST")
    try:
        with urlopen(request, timeout=timeout) as response:
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
    observed_at = now_local().isoformat(timespec="milliseconds")
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
    if not products:
        return [], "api_empty_product_list"
    if not observations:
        return [], f"api_unmapped_products:{len(products)}"
    return observations, None


async def safe_goto(page: Any, url: str) -> None:
    try:
        await page.goto(url, wait_until="domcontentloaded", timeout=20_000)
        await page.wait_for_timeout(500)
    except Exception:
        await page.goto(url, wait_until="commit", timeout=20_000)
        await page.wait_for_timeout(500)


async def safe_reload(page: Any) -> None:
    try:
        await page.reload(wait_until="domcontentloaded", timeout=15_000)
        await page.wait_for_timeout(350)
    except Exception:
        await page.reload(wait_until="commit", timeout=15_000)
        await page.wait_for_timeout(350)


async def select_duration(page: Any, duration_label: str) -> None:
    tab = page.locator(".switch-tab-item", has_text=duration_label).first
    await tab.click(timeout=10_000)
    await page.wait_for_timeout(150)


async def read_cards(page: Any, duration_key: str, duration_label: str, duration_column_label: str, source: str) -> list[Observation]:
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
    observed_at = now_local().isoformat(timespec="milliseconds")
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
                source=source,
            )
        )
    return observations


async def collect_duration_page(page: Any, duration: tuple[str, str, str], source: str, reload_each_poll: bool) -> list[Observation]:
    duration_key, duration_label, duration_column_label = duration
    if reload_each_poll:
        await safe_reload(page)
        await select_duration(page, duration_label)
    return await read_cards(page, duration_key, duration_label, duration_column_label, source)


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
        target_end = current + timedelta(seconds=max(0, args.late_retry_seconds))
    elif current > sale_start:
        notes.append("started_after_10")

    first_sold_out: dict[str, str] = {}
    latest_status: dict[str, str] = {}
    latest_source: dict[str, str] = {}
    last_not_sold_at: dict[str, str] = {}
    sold_out_windows: dict[str, str] = {}
    observations: list[Observation] = []
    api_attempts = 0
    api_busy_count = 0
    api_error_count = 0
    first_api_busy_at = ""
    first_api_success_at = ""
    page_attempts = 0
    page_observation_batches = 0
    first_page_success_at = ""
    api_headers = bigmodel_api_headers()
    if not api_headers:
        notes.append("no_bigmodel_auth")
    cookie_header = bigmodel_cookie_header()
    authorization_header = bigmodel_authorization_header()
    page_source = "authenticated_page" if cookie_header or authorization_header else "public_page"
    if args.page_monitor and not cookie_header:
        notes.append("no_bigmodel_cookie_for_page")
    playwright = None
    browser = None
    context = None
    duration_pages: dict[str, Any] = {}

    def record_batch(batch: list[Observation], observed_now: datetime) -> None:
        observations.extend(batch)
        for item in batch:
            current_status = latest_status.get(item.plan_key)
            current_source = latest_source.get(item.plan_key)
            reliable_inventory = item.source == "api_batch_preview"
            current_reliable_inventory = current_source == "api_batch_preview"
            if not (
                current_reliable_inventory
                and current_status == "sold_out"
                and not reliable_inventory
                and item.status != "sold_out"
            ):
                latest_status[item.plan_key] = item.status
                latest_source[item.plan_key] = item.source
            observed_label = format_local_time(observed_now)
            if observed_now >= sale_start and item.status != "sold_out":
                last_not_sold_at[item.plan_key] = observed_label
            if (
                not started_after_window
                and observed_now >= sale_start
                and item.status == "sold_out"
                and item.plan_key not in first_sold_out
            ):
                first_sold_out[item.plan_key] = observed_label
                previous = last_not_sold_at.get(item.plan_key)
                if previous:
                    sold_out_windows[item.plan_key] = f"{previous}~{observed_label}"

    async def ensure_context() -> Any:
        nonlocal playwright, browser, context
        if context is not None:
            return context
        playwright = await async_playwright().start()
        browser = await playwright.chromium.launch(headless=True, args=["--no-sandbox"])
        extra_http_headers = {"Authorization": authorization_header} if authorization_header else None
        context = await browser.new_context(
            locale="zh-CN",
            timezone_id="Asia/Shanghai",
            viewport={"width": 1440, "height": 1200},
            extra_http_headers=extra_http_headers,
        )
        cookies = playwright_cookies_from_header(cookie_header)
        if cookies:
            await context.add_cookies(cookies)

        async def route_handler(route: Any) -> None:
            if route.request.resource_type in {"image", "font", "media"}:
                await route.abort()
            else:
                await route.continue_()

        await context.route("**/*", route_handler)
        return context

    async def open_duration_page(duration: tuple[str, str, str]) -> tuple[str, Any]:
        duration_key, duration_label, _ = duration
        current_context = await ensure_context()
        duration_page = await current_context.new_page()
        await safe_goto(duration_page, SOURCE_URL)
        await select_duration(duration_page, duration_label)
        return duration_key, duration_page

    async def ensure_duration_pages() -> dict[str, Any]:
        missing = [
            duration
            for duration in DURATIONS
            if duration[0] not in duration_pages or duration_pages[duration[0]].is_closed()
        ]
        if missing:
            await ensure_context()
            opened = await asyncio.gather(*(open_duration_page(duration) for duration in missing))
            duration_pages.update(opened)
        return duration_pages

    async def collect_page_observations() -> list[Observation]:
        nonlocal page_attempts, page_observation_batches, first_page_success_at
        page_attempts += 1
        try:
            current_pages = await ensure_duration_pages()
            duration_tasks = [
                (
                    duration_key,
                    collect_duration_page(
                        current_pages[duration_key],
                        (duration_key, duration_label, duration_column_label),
                        page_source,
                        args.reload_each_poll,
                    ),
                )
                for duration_key, duration_label, duration_column_label in DURATIONS
            ]
            results = await asyncio.gather(*(task for _, task in duration_tasks), return_exceptions=True)
            batch = []
            for (duration_key, _), result in zip(duration_tasks, results):
                if isinstance(result, Exception):
                    note = f"page_{duration_key}_error:{clean_text(str(result))[:100]}"
                    if note not in notes:
                        notes.append(note)
                    continue
                batch.extend(result)
        except Exception as exc:
            note = f"page_error:{clean_text(str(exc))[:120]}"
            if note not in notes:
                notes.append(note)
            return []
        if batch:
            page_observation_batches += 1
            if not first_page_success_at:
                first_page_success_at = format_local_time(now_local())
            note = f"{page_source}_status"
            if note not in notes:
                notes.append(note)
        return batch

    async def poll_page_until_end() -> None:
        while True:
            observed_now = now_local()
            if observed_now >= target_end:
                break
            batch = await collect_page_observations()
            record_batch(batch, observed_now)
            if now_local() >= target_end:
                break
            await asyncio.sleep(max(0.1, args.page_poll_interval))

    if args.page_monitor and not args.once and now_local() < sale_start:
        try:
            await ensure_duration_pages()
            notes.append("page_prewarmed")
            notes.append("parallel_duration_pages")
        except Exception as exc:
            notes.append(f"page_prewarm_error:{clean_text(str(exc))[:120]}")

    page_task: asyncio.Task[None] | None = None
    if args.page_monitor and not args.once:
        if "parallel_duration_pages" not in notes:
            notes.append("parallel_duration_pages")
        page_task = asyncio.create_task(poll_page_until_end())
        notes.append("page_async_poller")

    used_api = False
    if api_headers:
        while True:
            observed_now = now_local()
            api_attempts += 1
            batch, api_note = await asyncio.to_thread(fetch_api_observations, api_headers, args.api_timeout)
            if api_note and api_note not in notes:
                notes.append(api_note)
            if api_note == "api_busy":
                api_busy_count += 1
                if not first_api_busy_at:
                    first_api_busy_at = format_local_time(observed_now)
            elif api_note and api_note.startswith("api_error:"):
                api_error_count += 1
            retryable_api_failure = bool(
                api_note
                and (
                    api_note == "api_busy"
                    or api_note.startswith("api_error:")
                    or api_note == "api_empty_product_list"
                    or api_note.startswith("api_unmapped_products:")
                )
            )
            if not batch and retryable_api_failure and not args.once and now_local() < target_end:
                await asyncio.sleep(max(0.1, args.poll_interval))
                continue
            if not batch:
                break
            used_api = True
            if not first_api_success_at:
                first_api_success_at = format_local_time(observed_now)
            record_batch(batch, observed_now)
            if args.once or now_local() >= target_end:
                break
            await asyncio.sleep(max(0.1, args.poll_interval))

    if api_attempts:
        notes.append(f"api_attempts={api_attempts}")
        notes.append(f"api_busy_count={api_busy_count}")
        if api_error_count:
            notes.append(f"api_error_count={api_error_count}")
        if first_api_busy_at:
            notes.append(f"first_api_busy={first_api_busy_at}")
        if first_api_success_at:
            notes.append(f"first_api_success={first_api_success_at}")
        notes.append(f"api_poll_interval={args.poll_interval:g}s")

    if page_task:
        await page_task
    elif (not used_api or args.once) and args.page_monitor:
        while True:
            observed_now = now_local()
            batch = await collect_page_observations()
            record_batch(batch, observed_now)
            if args.once or now_local() >= target_end:
                break
            await asyncio.sleep(max(0.5, args.page_poll_interval))

    for plan_key, window in sorted(sold_out_windows.items()):
        note = f"{plan_key}_sold_between={window}"
        if note not in notes:
            notes.append(note)

    if page_attempts:
        notes.append(f"page_attempts={page_attempts}")
        notes.append(f"page_batches={page_observation_batches}")
        notes.append(f"page_source={page_source}")
        if first_page_success_at:
            notes.append(f"first_page_success={first_page_success_at}")

    if browser:
        await browser.close()
    if playwright:
        await playwright.stop()

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


def merge_history_row(existing: dict[str, str] | None, row: dict[str, str]) -> dict[str, str]:
    if not existing:
        return row
    merged = dict(row)
    for column in PLAN_COLUMNS:
        if not clean_text(merged.get(column)) and clean_text(existing.get(column)):
            merged[column] = existing[column]
    previous_notes = clean_text(existing.get("Notes"))
    current_notes = clean_text(merged.get("Notes"))
    if previous_notes and current_notes and previous_notes != current_notes:
        merged["Notes"] = f"{previous_notes};{current_notes}"
    elif previous_notes and not current_notes:
        merged["Notes"] = previous_notes
    return merged


def upsert_history(row: dict[str, str]) -> dict[str, str]:
    existing_rows = read_history(HISTORY_PATH)
    existing = next((item for item in existing_rows if item.get("Date") == row["Date"]), None)
    rows = [item for item in existing_rows if item.get("Date") != row["Date"]]
    merged = merge_history_row(existing, row)
    rows.append(merged)
    write_history(HISTORY_PATH, rows)
    return merged


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
    parser.add_argument("--poll-interval", type=float, default=DEFAULT_API_POLL_INTERVAL, help="Seconds between API polling rounds.")
    parser.add_argument("--api-timeout", type=float, default=DEFAULT_API_TIMEOUT, help="Seconds before one API request is treated as timed out.")
    parser.add_argument("--page-monitor", action=argparse.BooleanOptionalAction, default=True, help="Also inspect the rendered BigModel pricing page with the configured cookie.")
    parser.add_argument("--page-poll-interval", type=float, default=DEFAULT_PAGE_POLL_INTERVAL, help="Seconds between rendered-page polling rounds.")
    parser.add_argument(
        "--late-retry-seconds",
        type=float,
        default=DEFAULT_LATE_RETRY_SECONDS,
        help="Seconds to keep retrying API inventory when the job starts after the monitor window.",
    )
    parser.add_argument("--reload-each-poll", action=argparse.BooleanOptionalAction, default=False)
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
    row = upsert_history(row)
    write_snapshot(row, observations, notes)
    print(json.dumps({"row": row, "observations": len(observations)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
