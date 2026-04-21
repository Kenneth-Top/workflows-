"""Fetch model pricing from Artificial Analysis and publish static CSV/JSON files.

The Cloudflare Pages dashboard is static, so this scraper reads the public
Next.js/RSC payload embedded in the Artificial Analysis models page and converts
provider/model pricing into flat files that the frontend can fetch directly.
"""

from __future__ import annotations

import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

SOURCE_URL = "https://artificialanalysis.ai/models/?pricing=input-and-output-pricing"
CSV_FILE = Path("artificial_analysis_pricing.csv")
JSON_FILE = Path("artificial_analysis_pricing.json")

HEADERS = [
    "Date",
    "Provider",
    "Provider_Slug",
    "Model",
    "Model_Slug",
    "Host_Model",
    "Host_Model_Slug",
    "Host_API_ID",
    "Input_Price_1M",
    "Output_Price_1M",
    "Cache_Write_Price_1M",
    "Cache_Hit_Price_1M",
    "Cache_Storage_Price_Per_Hour_Per_1M",
    "Blended_3_1_Price_1M",
    "Context_Window",
    "Reasoning_Model",
    "JSON_Mode",
    "Function_Calling",
    "Source_URL",
]


def fetch_html() -> str:
    response = requests.get(
        SOURCE_URL,
        timeout=90,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
            )
        },
    )
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
        raise RuntimeError("No Next.js RSC payload found in Artificial Analysis page")
    return "".join(chunks)


def extract_array(text: str, start_index: int) -> str:
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
        elif char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                return text[start_index : index + 1]
    raise RuntimeError("Unclosed JSON array in RSC payload")


def largest_object_array(text: str, key: str) -> list[dict[str, Any]]:
    best: list[dict[str, Any]] = []
    key_pattern = re.compile(rf'"{re.escape(key)}"\s*:')
    for match in key_pattern.finditer(text):
        start = text.find("[", match.end())
        if start < 0:
            continue
        try:
            parsed = json.loads(extract_array(text, start))
        except (json.JSONDecodeError, RuntimeError):
            continue
        if parsed and isinstance(parsed[0], dict) and len(parsed) > len(best):
            best = parsed
    if not best:
        raise RuntimeError(f"No object array found for key: {key}")
    return best


def normalize_number(value: Any) -> float | int | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return value
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def fallback_provider_name(host_model: dict[str, Any]) -> str:
    label = str(host_model.get("host_model_string") or host_model.get("slug") or "Unknown")
    return label.split("_", 1)[0].replace("-", " ").strip() or "Unknown"


def build_records(models: list[dict[str, Any]], hosts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    generated_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    host_by_id = {host.get("id"): host for host in hosts}
    records: list[dict[str, Any]] = []

    for model in models:
        for host_model in model.get("host_models") or []:
            if host_model.get("deleted"):
                continue

            prices = [
                host_model.get("price_1m_input_tokens"),
                host_model.get("price_1m_output_tokens"),
                host_model.get("cache_write_price"),
                host_model.get("cache_hit_price"),
                host_model.get("cache_storage_price_per_hour_per_1m_tokens"),
            ]
            if all(price is None for price in prices):
                continue

            host = host_by_id.get(host_model.get("host_id"), {})
            provider = host.get("name") or fallback_provider_name(host_model)
            provider_slug = host.get("slug") or str(provider).lower().replace(" ", "_")

            records.append(
                {
                    "Date": generated_date,
                    "Provider": provider,
                    "Provider_Slug": provider_slug,
                    "Model": model.get("name") or model.get("short_name") or model.get("slug"),
                    "Model_Slug": model.get("slug"),
                    "Host_Model": host_model.get("host_model_string"),
                    "Host_Model_Slug": host_model.get("slug"),
                    "Host_API_ID": host_model.get("host_api_id"),
                    "Input_Price_1M": normalize_number(host_model.get("price_1m_input_tokens")),
                    "Output_Price_1M": normalize_number(host_model.get("price_1m_output_tokens")),
                    "Cache_Write_Price_1M": normalize_number(host_model.get("cache_write_price")),
                    "Cache_Hit_Price_1M": normalize_number(host_model.get("cache_hit_price")),
                    "Cache_Storage_Price_Per_Hour_Per_1M": normalize_number(
                        host_model.get("cache_storage_price_per_hour_per_1m_tokens")
                    ),
                    "Blended_3_1_Price_1M": normalize_number(host_model.get("price_1m_blended_3_to_1")),
                    "Context_Window": normalize_number(
                        host_model.get("context_window_if_different_to_model")
                        or model.get("context_window_tokens")
                    ),
                    "Reasoning_Model": bool(model.get("reasoning_model")),
                    "JSON_Mode": bool(host_model.get("json_mode")),
                    "Function_Calling": bool(host_model.get("function_calling")),
                    "Source_URL": SOURCE_URL,
                }
            )

    records.sort(key=lambda row: (str(row["Provider"]), str(row["Model"]), str(row["Host_Model"])))
    return records


def write_outputs(records: list[dict[str, Any]]) -> None:
    with CSV_FILE.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=HEADERS)
        writer.writeheader()
        writer.writerows(records)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_url": SOURCE_URL,
        "record_count": len(records),
        "providers": sorted({record["Provider"] for record in records if record.get("Provider")}),
        "records": records,
    }
    JSON_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    html = fetch_html()
    text = decode_next_rsc(html)
    models = largest_object_array(text, "models")
    hosts = largest_object_array(text, "hosts")
    records = build_records(models, hosts)
    if not records:
        raise RuntimeError("No Artificial Analysis pricing records extracted")
    write_outputs(records)
    print(f"Saved {len(records)} records across {len(set(r['Provider'] for r in records))} providers")


if __name__ == "__main__":
    main()
