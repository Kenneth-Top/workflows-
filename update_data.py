import requests
import re
import json
import pandas as pd
from datetime import datetime, timedelta, timezone
import os
import time

# === 配置 ===
DATA_FILE = "history_database.csv"
MODELS_API = "https://openrouter.ai/api/v1/models"
RANKINGS_URL = "https://openrouter.ai/rankings"
RANKINGS_MODELS_API = "https://openrouter.ai/api/frontend/v1/rankings/models"
MODEL_ACTIVITY_API = "https://openrouter.ai/api/frontend/v1/stats/model-activity"
ACTIVITY_LOOKBACK_DAYS = int(os.environ.get("OPENROUTER_ACTIVITY_LOOKBACK_DAYS", "45"))
RECENT_REFRESH_DAYS = int(os.environ.get("OPENROUTER_RECENT_REFRESH_DAYS", "3"))
CANONICAL_TO_ID = {}
MODEL_TO_PERMASLUG = {}

# 公共 Session（复用连接池，避免每次请求都创建新 Session）
SESSION = requests.Session()
SESSION.trust_env = False
SESSION.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
})


def register_canonical_mapping(canonical_slug, model_id):
    existing = CANONICAL_TO_ID.get(canonical_slug)
    if existing is None or (existing.endswith(":free") and not model_id.endswith(":free")):
        CANONICAL_TO_ID[canonical_slug] = model_id


def fetch_all_model_ids():
    """从 OpenRouter API 自动获取所有可用模型的 id 列表"""
    global CANONICAL_TO_ID, MODEL_TO_PERMASLUG
    print("🔍 正在从 OpenRouter API 获取模型列表...")
    try:
        resp = SESSION.get(MODELS_API, timeout=30)
        resp.raise_for_status()
        models = resp.json().get("data", [])
        CANONICAL_TO_ID = {}
        MODEL_TO_PERMASLUG = {}
        for model in models:
            model_id = model.get("id")
            if not model_id:
                continue
            CANONICAL_TO_ID[model_id] = model_id
            canonical_slug = model.get("canonical_slug")
            if canonical_slug:
                register_canonical_mapping(canonical_slug, model_id)
                MODEL_TO_PERMASLUG[model_id] = canonical_slug
            else:
                MODEL_TO_PERMASLUG[model_id] = model_id
        # 按 created 倒序排列（最新的在前）
        models.sort(key=lambda m: m.get("created", 0), reverse=True)
        ids = [m["id"] for m in models]
        print(f"✅ 发现 {len(ids)} 个模型")
        return ids
    except Exception as e:
        print(f"❌ 获取模型列表失败: {e}")
        return []


def variant_model_id(model_id, variant):
    if not variant or variant == "standard":
        return model_id
    if model_id.endswith(f":{variant}"):
        return model_id
    return f"{model_id}:{variant}"


def variants_for_model(model_id, observed_variants, all_model_set):
    if ":" in model_id:
        return [model_id.rsplit(":", 1)[1]]

    variants = {None, "free"}
    variants.update(observed_variants.get(model_id, set()))
    variants = {
        variant
        for variant in variants
        if not variant or f"{model_id}:{variant}" not in all_model_set
    }
    return [variant for variant in variants if variant != "standard"]


def fetch_model_activity(model_id, variant=None):
    """从 OpenRouter 官方前端 activity API 获取单模型每日用量。"""
    permaslug = MODEL_TO_PERMASLUG.get(model_id, model_id)
    params = {"permaslug": permaslug}
    if variant:
        params["variant"] = variant
    try:
        resp = SESSION.get(MODEL_ACTIVITY_API, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json().get("data", {}).get("analytics", []) or []
    except Exception as e:
        suffix = f":{variant}" if variant else ""
        print(f"  ❌ activity API 错误 {model_id}{suffix}: {e}")
    return []


def current_utc_date():
    return datetime.now(timezone.utc).date()


def latest_complete_utc_date_str():
    return (current_utc_date() - timedelta(days=1)).isoformat()


def fetch_rankings_day_models(max_existing_date, current_utc_date_str, latest_complete_date, refresh_dates=None):
    """从 OpenRouter 当前公开 rankings API 获取最新完整日的模型 token 用量。"""
    print("📊 正在从 OpenRouter rankings API 获取最新日榜...")
    resp = SESSION.get(RANKINGS_MODELS_API, params={"view": "day"}, timeout=60)
    resp.raise_for_status()
    rows = resp.json().get("data", []) or []

    refresh_dates = refresh_dates or set()
    records = []
    available_days = []
    for row in rows:
        record_date_str = parse_day(row.get("date"))
        if not record_date_str:
            continue
        if record_date_str == current_utc_date_str or record_date_str > latest_complete_date:
            continue
        available_days.append(record_date_str)
        if max_existing_date and record_date_str <= max_existing_date and record_date_str not in refresh_dates:
            continue

        prompt_tokens = int(row.get("total_prompt_tokens") or 0)
        completion_tokens = int(row.get("total_completion_tokens") or 0)
        reasoning_tokens = int(row.get("total_native_tokens_reasoning") or 0)
        total_tokens = prompt_tokens + completion_tokens
        if total_tokens <= 0:
            continue

        raw_model = row.get("variant_permaslug") or row.get("model_permaslug")
        model = normalize_model_id(raw_model)
        records.append({
            'Date': datetime.strptime(record_date_str, "%Y-%m-%d"),
            'Model': model,
            'Prompt': round(prompt_tokens / 1e9, 6),
            'Completion': round(completion_tokens / 1e9, 6),
            'Reasoning': round(reasoning_tokens / 1e9, 6),
            'Total_Tokens': round(total_tokens / 1e9, 6)
        })

    latest_available_date = max(available_days) if available_days else None
    print(f"✅ rankings API 新增记录: {len(records)}")
    if latest_available_date:
        print(f"📅 rankings API 最新可用日期: {latest_available_date}")
    return records, latest_available_date


def date_key(record):
    value = record.get("Date")
    if hasattr(value, "strftime"):
        return value.strftime("%Y-%m-%d")
    return str(value)[:10]


def build_activity_records(model_id, variant, analytics, target_dates=None):
    records = []
    latest_available_date = None
    for record in analytics:
        record_date_str = str(record.get("date", ""))[:10]
        if not record_date_str:
            continue
        latest_available_date = max(latest_available_date or record_date_str, record_date_str)
        if target_dates is not None and record_date_str not in target_dates:
            continue

        p = (record.get('total_prompt_tokens') or 0) / 1e9
        c = (record.get('total_completion_tokens') or 0) / 1e9
        r = (record.get('total_native_tokens_reasoning') or 0) / 1e9
        t = p + c
        if t <= 0:
            continue

        raw_model = record.get("variant_permaslug") or record.get("model_permaslug")
        output_model_id = normalize_model_id(raw_model) if raw_model else variant_model_id(model_id, variant)
        records.append({
            'Date': datetime.strptime(record_date_str, "%Y-%m-%d"),
            'Model': output_model_id,
            'Prompt': round(p, 6),
            'Completion': round(c, 6),
            'Reasoning': round(r, 6),
            'Total_Tokens': round(t, 6)
        })
    return records, latest_available_date


def recent_activity_target_dates(df_old, latest_complete_date):
    """Fill holes in the recent activity window and refresh the latest few days."""
    complete_date = datetime.strptime(latest_complete_date, "%Y-%m-%d").date()
    start = complete_date - timedelta(days=ACTIVITY_LOOKBACK_DAYS)
    refresh_start = complete_date - timedelta(days=max(RECENT_REFRESH_DAYS - 1, 0))
    existing_days = set()
    if not df_old.empty and "Date" in df_old:
        existing_days = set(df_old["Date"].dt.strftime("%Y-%m-%d"))

    target_dates = set()
    current = start
    while current <= complete_date:
        day = current.isoformat()
        if day not in existing_days or current >= refresh_start:
            target_dates.add(day)
        current += timedelta(days=1)
    return target_dates


def activity_model_candidates(all_models, existing_models):
    """Use current catalog models plus historical/special models still served by activity API."""
    candidates = []
    seen = set()
    for model in list(all_models) + sorted(str(model) for model in existing_models if str(model)):
        if model and model not in seen:
            candidates.append(model)
            seen.add(model)
    return candidates


def decode_next_rsc(html):
    """提取 Next.js RSC 文本块，OpenRouter rankings 的图表数据在这里。"""
    chunks = []
    for match in re.finditer(r"self\.__next_f\.push\((\[.*?\])\)</script>", html, re.S):
        try:
            payload = json.loads(match.group(1))
        except Exception:
            continue
        if len(payload) >= 2 and isinstance(payload[1], str):
            chunks.append(payload[1])
    return "\n".join(chunks)


def extract_balanced(text, start_index, opener, closer):
    """从 start_index 开始提取配平的 JSON 数组/对象字符串。"""
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
                return text[start_index:index + 1]
    return None


def arrays_after_key(text, key):
    """从 RSC 文本中找到形如 "data":[...] 的数组。"""
    arrays = []
    marker = f'"{key}":'
    search_from = 0
    while True:
        marker_index = text.find(marker, search_from)
        if marker_index == -1:
            break
        array_start = text.find("[", marker_index + len(marker))
        if array_start == -1:
            break
        raw_array = extract_balanced(text, array_start, "[", "]")
        if raw_array:
            try:
                arrays.append(json.loads(raw_array))
            except Exception:
                pass
            search_from = array_start + len(raw_array)
        else:
            search_from = marker_index + len(marker)
    return arrays


def parse_day(value):
    value = str(value or "")[:10]
    try:
        datetime.strptime(value, "%Y-%m-%d")
        return value
    except Exception:
        return None


def normalize_model_id(raw_model):
    """把 rankings 的 canonical slug 转成前端历史库一直使用的 OpenRouter model id。"""
    raw_model = str(raw_model)
    if raw_model in CANONICAL_TO_ID:
        mapped_model = CANONICAL_TO_ID[raw_model]
        if raw_model.endswith(":free") or not mapped_model.endswith(":free"):
            return mapped_model

    free_suffix = ":free" if raw_model.endswith(":free") else ""
    base_model = raw_model[:-5] if free_suffix else raw_model
    undated = re.sub(r"-\d{8}$", "", base_model) + free_suffix
    if undated in CANONICAL_TO_ID:
        mapped_model = CANONICAL_TO_ID[undated]
        if free_suffix or not mapped_model.endswith(":free"):
            return mapped_model
    return undated


def daily_spacing_score(days):
    parsed = [datetime.strptime(day, "%Y-%m-%d") for day in days if day]
    if len(parsed) < 2:
        return 0
    diffs = [(b - a).days for a, b in zip(parsed, parsed[1:])]
    return sum(1 for diff in diffs if diff == 1) / len(diffs)


def looks_like_model_token_series(series):
    """识别 rankings 中的每日模型 token 图表，排除请求数和分类占比图。"""
    if not isinstance(series, list) or len(series) < 20:
        return False

    days = [parse_day(point.get("x")) for point in series if isinstance(point, dict)]
    days = [day for day in days if day]
    if daily_spacing_score(days[-30:]) < 0.75:
        return False

    latest_ys = {}
    for point in reversed(series):
        if isinstance(point, dict) and isinstance(point.get("ys"), dict):
            latest_ys = point["ys"]
            if latest_ys:
                break

    slash_keys = [key for key in latest_ys if "/" in str(key)]
    if len(slash_keys) < 5:
        return False

    latest_total = sum(float(value or 0) for value in latest_ys.values())
    return latest_total > 1e9


def fetch_rankings_daily_tokens(max_existing_date, current_utc_date_str, latest_complete_date):
    """从 rankings 页抓取每日模型 token 总量，避免逐模型页面 analytics 失效。"""
    print("📈 正在从 OpenRouter rankings 提取每日模型 token 序列...")
    resp = SESSION.get(RANKINGS_URL, timeout=60)
    resp.raise_for_status()

    rsc_text = decode_next_rsc(resp.text)
    candidates = [
        series
        for series in arrays_after_key(rsc_text, "data")
        if looks_like_model_token_series(series)
    ]
    if not candidates:
        raise RuntimeError("OpenRouter rankings 页面未找到每日模型 token 序列")

    series = max(
        candidates,
        key=lambda item: sum(float(value or 0) for value in (item[-1].get("ys") or {}).values()),
    )

    records = []
    available_days = []
    for point in series:
        if not isinstance(point, dict):
            continue
        day = parse_day(point.get("x"))
        if not day:
            continue
        if day == current_utc_date_str or day > latest_complete_date:
            continue
        available_days.append(day)
        if max_existing_date and day <= max_existing_date:
            continue

        for raw_model, tokens in (point.get("ys") or {}).items():
            if str(raw_model).lower() == "others":
                continue
            total_tokens = float(tokens or 0) / 1e9
            if total_tokens <= 0:
                continue
            model = normalize_model_id(raw_model)
            records.append({
                'Date': datetime.strptime(day, "%Y-%m-%d"),
                'Model': model,
                'Prompt': 0,
                'Completion': 0,
                'Reasoning': 0,
                'Total_Tokens': round(total_tokens, 6)
            })

    print(f"✅ rankings 新增记录: {len(records)}")
    latest_available_date = max(available_days) if available_days else None
    if latest_available_date:
        print(f"📅 rankings 最新可用日期: {latest_available_date}")
    return records, latest_available_date


def fetch_analytics(model_id):
    """从模型页面 HTML 中提取 analytics 数据"""
    url = f"https://openrouter.ai/{model_id}"
    try:
        resp = SESSION.get(url, timeout=20)
        if resp.status_code != 200:
            return None

        match = re.search(r'\\?"analytics\\?":\s*(\[\{.*?\}\])', resp.text)
        if match:
            raw = match.group(1).replace('\\"', '"').replace('\\\\', '\\')
            return json.loads(raw)
    except Exception as e:
        print(f"  ❌ 错误 {model_id}: {e}")
    return None


def update_database():
    # 1. 读取旧数据
    columns = ['Date', 'Model', 'Prompt', 'Completion', 'Reasoning', 'Total_Tokens']
    if os.path.exists(DATA_FILE):
        try:
            df_old = pd.read_csv(DATA_FILE)
            df_old['Date'] = pd.to_datetime(df_old['Date'])
        except Exception:
            df_old = pd.DataFrame(columns=columns)
    else:
        df_old = pd.DataFrame(columns=columns)
    max_existing_date = None
    if not df_old.empty:
        max_existing_date = df_old['Date'].max().strftime("%Y-%m-%d")
        print(f"📅 当前历史库最新日期: {max_existing_date}")

    # 2. 自动获取模型列表
    all_models = fetch_all_model_ids()
    if not all_models:
        print("⚠️ 无法获取模型列表，终止")
        return
    all_model_set = set(all_models)

    # 识别新模型（CSV 中尚未出现的模型）
    existing_models = set(df_old['Model'].unique()) if not df_old.empty else set()
    new_models = [m for m in all_models if m not in existing_models]
    if new_models:
        print(f"🆕 发现 {len(new_models)} 个新模型:")
        for m in new_models[:10]:  # 最多显示10个
            print(f"   + {m}")
        if len(new_models) > 10:
            print(f"   ... 及另外 {len(new_models) - 10} 个")

    observed_variants = {}
    for existing_model in existing_models:
        if ":" not in str(existing_model):
            continue
        base_model, variant = str(existing_model).rsplit(":", 1)
        observed_variants.setdefault(base_model, set()).add(variant)

    # 3. 计算最近窗口内需要回填或刷新的完整日。
    current_utc_date_str = current_utc_date().isoformat()
    latest_complete_date = latest_complete_utc_date_str()
    print(f"📅 本次只抓取 UTC 完整日，截止到: {latest_complete_date}")
    target_dates = recent_activity_target_dates(df_old, latest_complete_date)

    # 4. rankings day 是最新完整日的权威来源，稍后覆盖 activity 同日同模型记录。
    rankings_records = []
    try:
        rankings_records, latest_available_date = fetch_rankings_day_models(
            max_existing_date,
            current_utc_date_str,
            latest_complete_date,
            target_dates,
        )
    except Exception as e:
        print(f"⚠️ rankings API 抓取失败，准备回退到旧 activity API: {e}")
        latest_available_date = None

    if (
        not rankings_records
        and latest_available_date
        and max_existing_date
        and max_existing_date >= latest_available_date
    ):
        print("✅ 历史库已覆盖 rankings API 最新可用日期，继续检查 activity 缺口")

    ranking_refresh_dates = {date_key(record) for record in rankings_records}
    activity_target_dates = target_dates - ranking_refresh_dates

    # 5. 逐模型 activity API 负责最近窗口内 rankings day 没覆盖的历史缺口和明细字段。
    activity_records = []
    latest_available_dates = []
    if activity_target_dates:
        print(
            f"🧩 准备从 activity API 回填/刷新 {len(activity_target_dates)} 天: "
            f"{min(activity_target_dates)} ~ {max(activity_target_dates)}"
        )
    candidates = activity_model_candidates(all_models, existing_models)
    all_model_set.update(existing_models)
    for i, model in enumerate(candidates):
        if not activity_target_dates:
            break
        print(f"🚀 [{i+1}/{len(candidates)}] 正在抓取 activity: {model}")
        for variant in variants_for_model(model, observed_variants, all_model_set):
            data = fetch_model_activity(model, variant)
            if not data:
                continue
            records, latest_available_date = build_activity_records(
                model, variant, data, activity_target_dates
            )
            activity_records.extend(records)
            if latest_available_date:
                latest_available_dates.append(latest_available_date)
        time.sleep(0.15)

    latest_available_date = max(latest_available_dates) if latest_available_dates else None
    if latest_available_date:
        print(f"📅 activity API 最新可用日期: {latest_available_date}")

    if (
        not activity_records
        and not rankings_records
        and latest_available_date
        and max_existing_date
        and max_existing_date >= latest_available_date
    ):
        print("✅ 历史库已覆盖 activity API 最新可用日期，无需更新")
        return

    if rankings_records and activity_records:
        activity_reasoning = {
            (date_key(record), record["Model"]): record.get("Reasoning", 0)
            for record in activity_records
        }
        for record in rankings_records:
            key = (date_key(record), record["Model"])
            if not record.get("Reasoning") and activity_reasoning.get(key):
                record["Reasoning"] = activity_reasoning[key]

    new_records = activity_records + rankings_records
    if not new_records:
        raise RuntimeError("本次未抓取到任何 token 数据，停止工作流，避免静默卡住")

    df_new = pd.DataFrame(new_records)

    # 5. 增量合并 (Upsert)：Date + Model 为唯一键
    df_combined = pd.concat([df_old, df_new])
    df_combined = df_combined.drop_duplicates(
        subset=['Date', 'Model'], keep='last'
    )

    # 6. 保存
    df_combined.to_csv(DATA_FILE, index=False)
    print(f"✅ 数据库更新完成！当前总记录数: {len(df_combined)}")


if __name__ == "__main__":
    if "HTTP_PROXY" in os.environ:
        del os.environ["HTTP_PROXY"]
    update_database()
