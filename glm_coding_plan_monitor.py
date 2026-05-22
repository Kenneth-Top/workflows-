"""Monitor GLM Coding Plan daily sell-out times.

The public BigModel page is a Vue app, so the monitor uses Playwright to
observe the rendered pricing cards without clicking any purchase button.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import re
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


SOURCE_URL = "https://bigmodel.cn/glm-coding"
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
    if re.search(r"订阅|抢购|购买", button_text):
        return "available"
    if disabled and re.search(r"订阅|抢购|购买", text):
        return "disabled"
    return "unknown"


def plan_column(tier_label: str, duration_column_label: str) -> str:
    return f"{tier_label}_{duration_column_label}"


def status_to_cn(status: str) -> str:
    return {
        "sold_out": "已售罄",
        "available": "可订阅",
        "available_at_end": "窗口结束仍可订阅",
        "pre_sale": "未开售",
        "disabled": "按钮不可用",
        "unknown": "未知",
        "not_observed": "未观测",
        "error": "监控失败",
    }.get(status, status)


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

    if args.once:
        target_start = current
        target_end = current
        notes.append("once_mode")
    elif current < target_start:
        await asyncio.sleep((target_start - current).total_seconds())
    elif current > target_end:
        notes.append("started_after_window")
        target_end = current
    elif current > sale_start:
        notes.append("started_after_10")

    first_sold_out: dict[str, str] = {}
    latest_status: dict[str, str] = {}
    observations: list[Observation] = []

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
            observations.extend(batch)
            for item in batch:
                latest_status[item.plan_key] = item.status
                if observed_now >= sale_start and item.status == "sold_out" and item.plan_key not in first_sold_out:
                    first_sold_out[item.plan_key] = observed_now.strftime("%H:%M:%S")

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
    for column in PLAN_COLUMNS:
        row[column] = first_sold_out.get(column, "")
        raw_status = latest_status.get(column, "not_observed")
        if column not in first_sold_out and raw_status == "available":
            raw_status = "available_at_end"
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
