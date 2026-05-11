import requests
import re
import json
import pandas as pd
from datetime import datetime
import os
import time

# === 配置 ===
DATA_FILE = "history_database.csv"
MODELS_API = "https://openrouter.ai/api/v1/models"
RANKINGS_URL = "https://openrouter.ai/rankings"
CANONICAL_TO_ID = {}

# 公共 Session（复用连接池，避免每次请求都创建新 Session）
SESSION = requests.Session()
SESSION.trust_env = False
SESSION.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
})


def fetch_all_model_ids():
    """从 OpenRouter API 自动获取所有可用模型的 id 列表"""
    global CANONICAL_TO_ID
    print("🔍 正在从 OpenRouter API 获取模型列表...")
    try:
        resp = SESSION.get(MODELS_API, timeout=30)
        resp.raise_for_status()
        models = resp.json().get("data", [])
        CANONICAL_TO_ID = {}
        for model in models:
            model_id = model.get("id")
            if not model_id:
                continue
            CANONICAL_TO_ID[model_id] = model_id
            canonical_slug = model.get("canonical_slug")
            if canonical_slug:
                CANONICAL_TO_ID[canonical_slug] = model_id
        # 按 created 倒序排列（最新的在前）
        models.sort(key=lambda m: m.get("created", 0), reverse=True)
        ids = [m["id"] for m in models]
        print(f"✅ 发现 {len(ids)} 个模型")
        return ids
    except Exception as e:
        print(f"❌ 获取模型列表失败: {e}")
        return []


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
        return CANONICAL_TO_ID[raw_model]

    free_suffix = ":free" if raw_model.endswith(":free") else ""
    base_model = raw_model[:-5] if free_suffix else raw_model
    undated = re.sub(r"-\d{8}$", "", base_model) + free_suffix
    if undated in CANONICAL_TO_ID:
        return CANONICAL_TO_ID[undated]
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


def fetch_rankings_daily_tokens(max_existing_date, today_str):
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
        if not day or day == today_str:
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

    # 识别新模型（CSV 中尚未出现的模型）
    existing_models = set(df_old['Model'].unique()) if not df_old.empty else set()
    new_models = [m for m in all_models if m not in existing_models]
    if new_models:
        print(f"🆕 发现 {len(new_models)} 个新模型:")
        for m in new_models[:10]:  # 最多显示10个
            print(f"   + {m}")
        if len(new_models) > 10:
            print(f"   ... 及另外 {len(new_models) - 10} 个")

    # 3. 优先从 rankings 页批量提取每日模型 token 数据
    new_records = []
    rankings_latest_date = None
    today_str = datetime.utcnow().strftime("%Y-%m-%d")  # OpenRouter 使用 UTC 日期
    try:
        new_records, rankings_latest_date = fetch_rankings_daily_tokens(max_existing_date, today_str)
    except Exception as e:
        print(f"⚠️ rankings 批量提取失败，回退到逐模型页面: {e}")

    if (
        not new_records
        and rankings_latest_date
        and max_existing_date
        and max_existing_date >= rankings_latest_date
    ):
        print("✅ 历史库已覆盖 rankings 最新可用日期，无需更新")
        return

    if not new_records:
        for i, model in enumerate(all_models):
            print(f"🚀 [{i+1}/{len(all_models)}] 正在抓取: {model}")
            data = fetch_analytics(model)
            if not data:
                continue

            for record in data:
                # 过滤当天未结算数据（当天统计不完整，会导致数值偏低）
                record_date_str = record['date'][:10]  # "2026-02-13 00:00:00" -> "2026-02-13"
                if record_date_str == today_str:
                    continue
                if max_existing_date and record_date_str <= max_existing_date:
                    continue

                p = (record.get('total_prompt_tokens') or 0) / 1e9
                c = (record.get('total_completion_tokens') or 0) / 1e9
                r = (record.get('total_native_tokens_reasoning') or 0) / 1e9
                t = p + c  # Total = Prompt + Completion (OpenAI 标准)

                new_records.append({
                    'Date': datetime.strptime(record['date'], "%Y-%m-%d %H:%M:%S"),
                    'Model': model,
                    'Prompt': round(p, 6),
                    'Completion': round(c, 6),
                    'Reasoning': round(r, 6),
                    'Total_Tokens': round(t, 6)
                })
            time.sleep(1)

    if not new_records:
        raise RuntimeError("本次未抓取到任何 token 数据，停止工作流，避免静默卡住")

    df_new = pd.DataFrame(new_records)

    # 4. 增量合并 (Upsert)：Date + Model 为唯一键
    df_combined = pd.concat([df_old, df_new])
    df_combined = df_combined.drop_duplicates(
        subset=['Date', 'Model'], keep='last'
    )

    # 5. 保存
    df_combined.to_csv(DATA_FILE, index=False)
    print(f"✅ 数据库更新完成！当前总记录数: {len(df_combined)}")


if __name__ == "__main__":
    if "HTTP_PROXY" in os.environ:
        del os.environ["HTTP_PROXY"]
    update_database()
