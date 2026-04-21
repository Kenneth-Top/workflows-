const state = {
  tokens: [],
  filteredTokens: [],
  cumulativeRows: [],
  sampleRows: [],
  models: [],
  charts: {
    cumulative: null,
    token: null,
  },
  reports: [],
  currentSamples: [],
};

const palette = ["#0f8b8d", "#d1495b", "#edae49", "#2e6f95", "#6c5ce7", "#247ba0", "#9a6324", "#008080"];

const sampleFilters = [
  { id: "filter-user", label: "用户类型", columns: ["user_group_4"] },
  { id: "filter-region", label: "地区", columns: ["region_bucket"] },
  { id: "filter-sentiment", label: "情绪", columns: ["sentiment"] },
  { id: "filter-scene", label: "场景", columns: ["scene_tags", "scenario_tags"] },
  { id: "filter-praise", label: "夸赞指标", columns: ["praise_tags"] },
  { id: "filter-competitor", label: "竞品", columns: ["competitor_tags"] },
];

const alertConfig = {
  newModelLookbackDays: 30,
  rampObservationWindow: 7,
  rampPercentileThreshold: 95,
  slopeDropThreshold: 0.3,
  slopeMinDays: 2,
};

function $(selector) {
  return document.querySelector(selector);
}

function showToast(message, persistent = false) {
  const toast = $("#toast");
  toast.textContent = message;
  toast.classList.remove("hidden");
  if (!persistent) {
    window.setTimeout(() => toast.classList.add("hidden"), 2600);
  }
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function parseCsv(text) {
  const lines = text.trim().split(/\r?\n/);
  const headers = lines.shift().split(",");
  return lines.filter(Boolean).map((line) => {
    const cells = line.split(",");
    const row = {};
    headers.forEach((header, index) => {
      row[header] = cells[index] ?? "";
    });
    return row;
  });
}

async function loadCsv(path) {
  const response = await fetch(path, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`无法读取 ${path}`);
  }
  return parseCsv(await response.text());
}

async function loadJson(path) {
  const response = await fetch(path, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`无法读取 ${path}`);
  }
  return response.json();
}

function numberValue(value) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : 0;
}

function shortNumber(value) {
  if (value >= 1000) return `${(value / 1000).toFixed(2)} T`;
  if (value >= 1) return `${value.toFixed(2)} B`;
  return `${value.toFixed(4)} B`;
}

function groupBy(items, keyFn) {
  return items.reduce((acc, item) => {
    const key = keyFn(item);
    if (!acc.has(key)) acc.set(key, []);
    acc.get(key).push(item);
    return acc;
  }, new Map());
}

function selectedValues(selector) {
  return Array.from($(selector).selectedOptions).map((item) => item.value);
}

function daysBetween(startDate, endDate) {
  const start = new Date(`${startDate}T00:00:00Z`).getTime();
  const end = new Date(`${endDate}T00:00:00Z`).getTime();
  return Math.round((end - start) / 86400000);
}

function percentile(values, pct) {
  const sorted = values.slice().sort((a, b) => a - b);
  if (!sorted.length) return Infinity;
  if (sorted.length === 1) return sorted[0];
  const position = (pct / 100) * (sorted.length - 1);
  const lower = Math.floor(position);
  const upper = Math.ceil(position);
  if (lower === upper) return sorted[lower];
  return sorted[lower] + (sorted[upper] - sorted[lower]) * (position - lower);
}

function downloadCsv(filename, headers, rows) {
  const csv = [
    headers.join(","),
    ...rows.map((row) => headers.map((header) => {
      const value = row[header] ?? "";
      return `"${String(value).replaceAll('"', '""')}"`;
    }).join(",")),
  ].join("\n");
  const blob = new Blob([`\ufeff${csv}`], { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}

function setupTabs() {
  document.querySelectorAll(".tab").forEach((button) => {
    button.addEventListener("click", () => {
      document.querySelectorAll(".tab").forEach((item) => item.classList.remove("active"));
      document.querySelectorAll(".view").forEach((item) => item.classList.remove("active"));
      button.classList.add("active");
      $(`#${button.dataset.view}-view`).classList.add("active");
    });
  });
}

function setupCumulativeControls() {
  $("#cumulative-model-search").addEventListener("input", renderCumulativeModelOptions);
  $("#cumulative-model-select").addEventListener("change", renderCumulative);
  $("#cumulative-day-window").addEventListener("change", renderCumulative);
  $("#download-cumulative").addEventListener("click", () => {
    if (!state.cumulativeRows.length) return;
    const headers = Object.keys(state.cumulativeRows[0]);
    downloadCsv("cumulative_compare.csv", headers, state.cumulativeRows);
  });
}

function setupTokenControls() {
  $("#model-search").addEventListener("input", renderSingleModelOptions);
  $("#model-select").addEventListener("change", renderSingleModel);
  $("#range-select").addEventListener("change", renderSingleModel);
  $("#download-filtered").addEventListener("click", () => {
    if (!state.filteredTokens.length) return;
    downloadCsv(
      "single_model_tokens.csv",
      ["Date", "Model", "Total_Tokens"],
      state.filteredTokens,
    );
  });
}

function setupProductControls() {
  $("#vendor-select").addEventListener("change", renderProductOptions);
  $("#product-select").addEventListener("change", renderProductReport);
  $("#sample-search").addEventListener("input", renderSamples);
  $("#reset-sample-filters").addEventListener("click", resetSampleFilters);
  $("#download-samples").addEventListener("click", () => {
    if (!state.sampleRows.length) return;
    downloadCsv("ai_product_samples.csv", ["t", "handle", "sentiment", "region_bucket", "narrative_bucket", "translated", "u"], state.sampleRows);
  });
  sampleFilters.forEach((filter) => {
    $(`#${filter.id}`).addEventListener("change", renderSamples);
  });
}

function renderCumulativeModelOptions() {
  const search = $("#cumulative-model-search").value.trim().toLowerCase();
  const select = $("#cumulative-model-select");
  const previous = new Set(selectedValues("#cumulative-model-select"));
  const visibleModels = state.models.filter((model) => model.toLowerCase().includes(search));

  select.innerHTML = "";
  visibleModels.forEach((model, index) => {
    const option = document.createElement("option");
    option.value = model;
    option.textContent = model;
    option.selected = previous.has(model) || (!previous.size && index < 3);
    select.append(option);
  });
  renderCumulative();
}

function selectedCumulativeModels() {
  const selected = selectedValues("#cumulative-model-select");
  return selected.length ? selected : state.models.slice(0, 3);
}

function buildCumulativeSeries(models) {
  const grouped = groupBy(state.tokens.filter((row) => models.includes(row.Display_Name)), (row) => row.Display_Name);
  const windowValue = $("#cumulative-day-window").value;
  const dayLimit = windowValue === "all" ? Infinity : Number(windowValue);
  let maxDay = 0;

  const series = models.map((model) => {
    const rows = (grouped.get(model) || []).slice().sort((a, b) => a.Date.localeCompare(b.Date));
    if (!rows.length) return { model, points: [], startDate: "" };
    const startDate = rows[0].Date;
    let running = 0;
    const points = [];

    rows.forEach((row) => {
      const day = daysBetween(startDate, row.Date);
      if (day > dayLimit) return;
      running += row.Total_Tokens;
      maxDay = Math.max(maxDay, day);
      points.push({
        day,
        date: row.Date,
        value: Number(running.toFixed(6)),
      });
    });
    return { model, points, startDate };
  });

  return { series, maxDay };
}

function renderCumulative() {
  if (!state.tokens.length) return;
  const models = selectedCumulativeModels();
  const { series, maxDay } = buildCumulativeSeries(models);
  const days = Array.from({ length: maxDay + 1 }, (_, index) => index);
  const datasets = series.map((item, index) => {
    const byDay = new Map(item.points.map((point) => [point.day, point.value]));
    return {
      label: `${item.model}${item.startDate ? ` (${item.startDate})` : ""}`,
      data: days.map((day) => byDay.get(day) ?? null),
      borderColor: palette[index % palette.length],
      backgroundColor: palette[index % palette.length],
      tension: 0.25,
      pointRadius: 0,
      borderWidth: 2,
      spanGaps: true,
    };
  });

  const leaders = series
    .map((item) => ({ model: item.model, value: item.points.at(-1)?.value || 0 }))
    .sort((a, b) => b.value - a.value);

  $("#cum-model-count").textContent = models.length.toLocaleString();
  $("#cum-max-day").textContent = `${maxDay} 天`;
  $("#cum-leader").textContent = leaders[0]?.model || "-";
  $("#cum-leader-total").textContent = shortNumber(leaders[0]?.value || 0);

  if (state.charts.cumulative) state.charts.cumulative.destroy();
  state.charts.cumulative = new Chart($("#cumulative-chart"), {
    type: "line",
    data: { labels: days.map((day) => `Day ${day}`), datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      plugins: {
        legend: { position: "bottom" },
        tooltip: { callbacks: { label: (ctx) => `${ctx.dataset.label}: ${shortNumber(ctx.parsed.y || 0)}` } },
      },
      scales: {
        x: { title: { display: true, text: "模型收录后的天数" }, ticks: { maxTicksLimit: 12 } },
        y: { title: { display: true, text: "累计 Tokens (Billion)" } },
      },
    },
  });
  renderCumulativeTable(series, days);
}

function renderCumulativeTable(series, days) {
  const table = $("#cumulative-table");
  const models = series.map((item) => item.model);
  table.querySelector("thead").innerHTML = `<tr><th>Day</th>${models.map((model) => `<th>${escapeHtml(model)}</th>`).join("")}</tr>`;

  const byModel = new Map(series.map((item) => [item.model, new Map(item.points.map((point) => [point.day, point.value]))]));
  const rows = days.map((day) => {
    const row = { Day: day };
    models.forEach((model) => {
      row[model] = byModel.get(model).get(day) ?? "";
    });
    return row;
  });
  state.cumulativeRows = rows;

  table.querySelector("tbody").innerHTML = rows.map((row) => `
    <tr>
      <td>${row.Day}</td>
      ${models.map((model) => `<td>${row[model] === "" ? "" : Number(row[model]).toFixed(6)}</td>`).join("")}
    </tr>
  `).join("");
}

function firstDatesByModel() {
  const firstDates = new Map();
  state.tokens.forEach((row) => {
    const current = firstDates.get(row.Display_Name);
    if (!current || row.Date < current) {
      firstDates.set(row.Display_Name, row.Date);
    }
  });
  return firstDates;
}

function rowsForModel(model, dropLatest = false) {
  const rows = state.tokens
    .filter((row) => row.Display_Name === model)
    .sort((a, b) => a.Date.localeCompare(b.Date));
  return dropLatest && rows.length > 1 ? rows.slice(0, -1) : rows;
}

function calcEarlySlope(model, windowDays) {
  const rows = rowsForModel(model, true);
  if (!rows.length) return null;
  const startDate = rows[0].Date;
  const windowRows = rows.filter((row) => daysBetween(startDate, row.Date) <= windowDays);
  if (!windowRows.length) return null;
  const cumTokens = windowRows.reduce((sum, row) => sum + row.Total_Tokens, 0);
  const actualDays = Math.max(...windowRows.map((row) => daysBetween(startDate, row.Date)));
  return actualDays <= 0 ? cumTokens : cumTokens / actualDays;
}

function newModels() {
  const firstDates = firstDatesByModel();
  const latestDate = state.tokens.map((row) => row.Date).sort().at(-1);
  const cutoffTime = new Date(`${latestDate}T00:00:00Z`).getTime() - alertConfig.newModelLookbackDays * 86400000;
  return Array.from(firstDates.entries())
    .filter(([, firstDate]) => new Date(`${firstDate}T00:00:00Z`).getTime() >= cutoffTime)
    .map(([model, firstDate]) => ({ model, firstDate }));
}

function detectFastRamp() {
  const slopes = state.models
    .map((model) => calcEarlySlope(model, alertConfig.rampObservationWindow))
    .filter((value) => value !== null && value > 0);
  const threshold = percentile(slopes, alertConfig.rampPercentileThreshold);

  const triggered = newModels().flatMap(({ model, firstDate }) => {
    const slope = calcEarlySlope(model, alertConfig.rampObservationWindow);
    if (slope === null || slope <= threshold) return [];
    const rows = rowsForModel(model, true);
    if (!rows.length) return [];
    const obsDays = Math.max(1, daysBetween(firstDate, rows.at(-1).Date));
    const exceedPct = threshold > 0 && Number.isFinite(threshold) ? ((slope - threshold) / threshold) * 100 : Infinity;
    return [{
      model,
      onlineDate: firstDate,
      observationDays: Math.min(obsDays, alertConfig.rampObservationWindow),
      currentSlope: slope,
      threshold,
      exceedPct,
    }];
  }).sort((a, b) => b.exceedPct - a.exceedPct);

  return { triggered, threshold };
}

function detectSlopeDrop() {
  const triggered = newModels().flatMap(({ model, firstDate }) => {
    const rows = rowsForModel(model, true);
    if (rows.length < alertConfig.slopeMinDays) return [];

    let cumTokens = 0;
    const enriched = rows.map((row) => {
      cumTokens += row.Total_Tokens;
      return {
        ...row,
        dayNum: daysBetween(firstDate, row.Date),
        cumTokens,
      };
    });

    const latest = enriched.at(-1);
    const prev = enriched.at(-2);
    if (!latest || !prev || latest.dayNum <= 0 || prev.dayNum <= 0) return [];

    const slopeLatest = latest.cumTokens / latest.dayNum;
    const slopePrev = prev.cumTokens / prev.dayNum;
    if (slopePrev <= 0) return [];

    const drop = (slopeLatest - slopePrev) / slopePrev;
    if (drop > -alertConfig.slopeDropThreshold) return [];

    return [{
      model,
      onlineDate: firstDate,
      slopePrev,
      slopeLatest,
      dropPct: drop * 100,
    }];
  }).sort((a, b) => a.dropPct - b.dropPct);

  return { triggered };
}

function renderAlerts() {
  const fastRamp = detectFastRamp();
  const slopeDrop = detectSlopeDrop();

  $("#fast-ramp-caption").textContent = `最近 ${alertConfig.newModelLookbackDays} 天新模型，前 ${alertConfig.rampObservationWindow} 天日均累计斜率超过全库 P${alertConfig.rampPercentileThreshold} 即触发。当前阈值：${Number.isFinite(fastRamp.threshold) ? `${fastRamp.threshold.toFixed(6)} B/天` : "暂无"}`;
  $("#slope-drop-caption").textContent = `最近 ${alertConfig.newModelLookbackDays} 天新模型，最新累计斜率相比前一天下降 ${Math.round(alertConfig.slopeDropThreshold * 100)}% 以上即触发。`;

  $("#fast-ramp-alerts").innerHTML = renderAlertList(
    fastRamp.triggered,
    (item) => `
      <strong>${escapeHtml(item.model)}</strong>
      <span>上线 ${item.onlineDate}，观察 ${item.observationDays} 天，当前 ${item.currentSlope.toFixed(6)} B/天，超出阈值 ${Number.isFinite(item.exceedPct) ? `+${item.exceedPct.toFixed(1)}%` : "较多"}</span>
    `,
    "当前没有增长过快的新模型",
  );

  $("#slope-drop-alerts").innerHTML = renderAlertList(
    slopeDrop.triggered,
    (item) => `
      <strong>${escapeHtml(item.model)}</strong>
      <span>上线 ${item.onlineDate}，斜率 ${item.slopePrev.toFixed(6)} → ${item.slopeLatest.toFixed(6)} B/天，变化 ${item.dropPct.toFixed(1)}%</span>
    `,
    "当前没有增长放缓预警",
    true,
  );
}

function renderAlertList(items, renderItem, emptyText, warning = false) {
  if (!items.length) return `<div class="alert-empty">${emptyText}</div>`;
  return `<div class="alert-list">${items.map((item) => `<div class="alert-item${warning ? " warning" : ""}">${renderItem(item)}</div>`).join("")}</div>`;
}

function renderSingleModelOptions() {
  const search = $("#model-search").value.trim().toLowerCase();
  const select = $("#model-select");
  const current = select.value;
  const visibleModels = state.models.filter((model) => model.toLowerCase().includes(search));

  select.innerHTML = "";
  visibleModels.forEach((model, index) => {
    const option = document.createElement("option");
    option.value = model;
    option.textContent = model;
    option.selected = model === current || (!current && index === 0);
    select.append(option);
  });
  renderSingleModel();
}

function filterByRange(rows) {
  const range = $("#range-select").value;
  if (range === "all" || !rows.length) return rows;
  const maxTime = Math.max(...rows.map((row) => new Date(row.Date).getTime()));
  const minTime = maxTime - Number(range) * 24 * 60 * 60 * 1000;
  return rows.filter((row) => new Date(row.Date).getTime() >= minTime);
}

function renderSingleModel() {
  if (!state.tokens.length) return;
  const model = $("#model-select").value || state.models[0];
  const fullRows = state.tokens
    .filter((row) => row.Display_Name === model)
    .sort((a, b) => a.Date.localeCompare(b.Date));
  const rows = filterByRange(fullRows);
  state.filteredTokens = rows;

  const latest = rows.at(-1);
  const total = rows.reduce((sum, row) => sum + row.Total_Tokens, 0);

  $("#metric-models").textContent = fullRows.length ? `${fullRows[0].Date} ~ ${fullRows.at(-1).Date}` : "-";
  $("#metric-range").textContent = rows.length ? `${rows[0].Date} ~ ${rows.at(-1).Date}` : "-";
  $("#metric-latest").textContent = latest ? shortNumber(latest.Total_Tokens) : "-";
  $("#metric-total").textContent = shortNumber(total);

  renderSingleModelChart(rows);
  renderTokenTable(rows);
}

function renderSingleModelChart(rows) {
  const labels = rows.map((row) => row.Date);
  const datasets = [{
    label: "Total Tokens",
    data: rows.map((row) => row.Total_Tokens),
    borderColor: "#0f8b8d",
    backgroundColor: "#0f8b8d",
    tension: 0.2,
    pointRadius: 0,
    borderWidth: 3,
  }];

  if (state.charts.token) state.charts.token.destroy();
  state.charts.token = new Chart($("#token-chart"), {
    type: "line",
    data: { labels, datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      plugins: {
        legend: { position: "bottom" },
        tooltip: { callbacks: { label: (ctx) => `${ctx.dataset.label}: ${shortNumber(ctx.parsed.y || 0)}` } },
      },
      scales: {
        x: { ticks: { maxTicksLimit: 10 } },
        y: { title: { display: true, text: "Tokens (Billion)" } },
      },
    },
  });
}

function renderTokenTable(rows) {
  const tbody = $("#token-table tbody");
  tbody.innerHTML = rows.slice().reverse().slice(0, 500).map((row) => `
    <tr>
      <td>${escapeHtml(row.Date)}</td>
      <td>${escapeHtml(row.Display_Name)}</td>
      <td>${row.Total_Tokens.toFixed(6)}</td>
    </tr>
  `).join("");
}

function renderProductOptions() {
  const vendor = $("#vendor-select").value;
  const productSelect = $("#product-select");
  const products = state.reports.filter((item) => item.vendor === vendor);
  productSelect.innerHTML = "";
  products.forEach((item) => {
    const option = document.createElement("option");
    option.value = item.product;
    option.textContent = item.product;
    productSelect.append(option);
  });
  renderProductReport();
}

function renderProductDownloads() {
  const wrap = $("#product-downloads");
  if (!wrap) return;
  wrap.innerHTML = state.reports.flatMap((item) => [
    `<a href="${escapeHtml(item.report)}" download>${escapeHtml(item.product)} 报告 HTML</a>`,
    `<a href="${escapeHtml(item.dataset)}" download>${escapeHtml(item.product)} 样本 JSON</a>`,
  ]).join("");
}

async function renderProductReport() {
  const vendor = $("#vendor-select").value;
  const product = $("#product-select").value;
  const selected = state.reports.find((item) => item.vendor === vendor && item.product === product);
  if (!selected) return;

  $("#report-frame").src = selected.report;
  $("#report-link").href = selected.report;
  state.currentSamples = await loadJson(selected.dataset);
  populateSampleFilters();
  renderSamples();
}

function normalizeText(value) {
  if (Array.isArray(value)) return value.join(" ");
  return value ? String(value) : "";
}

function valuesFromColumns(row, columns) {
  return columns.flatMap((column) => {
    const value = row[column];
    if (Array.isArray(value)) return value.filter(Boolean).map(String);
    return value ? [String(value)] : [];
  });
}

function topCounts(rows, columns, limit = 6) {
  const targetColumns = Array.isArray(columns) ? columns : [columns];
  const counts = new Map();
  rows.forEach((row) => {
    valuesFromColumns(row, targetColumns).forEach((value) => counts.set(value, (counts.get(value) || 0) + 1));
  });
  return Array.from(counts.entries()).sort((a, b) => b[1] - a[1]).slice(0, limit);
}

function populateSampleFilters() {
  sampleFilters.forEach((filter) => {
    const select = $(`#${filter.id}`);
    const options = Array.from(new Set(state.currentSamples.flatMap((row) => valuesFromColumns(row, filter.columns))))
      .filter(Boolean)
      .sort((a, b) => a.localeCompare(b, "zh-Hans-CN"));
    select.innerHTML = options.map((option) => `<option value="${escapeHtml(option)}">${escapeHtml(option)}</option>`).join("");
  });
  $("#sample-search").value = "";
}

function rowMatchesSampleFilters(row) {
  return sampleFilters.every((filter) => {
    const selected = selectedValues(`#${filter.id}`);
    if (!selected.length) return true;
    const values = valuesFromColumns(row, filter.columns);
    return selected.some((item) => values.includes(item));
  });
}

function resetSampleFilters() {
  $("#sample-search").value = "";
  sampleFilters.forEach((filter) => {
    Array.from($(`#${filter.id}`).options).forEach((option) => {
      option.selected = false;
    });
  });
  renderSamples();
}

function renderSamples() {
  const query = $("#sample-search").value.trim().toLowerCase();
  const rows = state.currentSamples.filter((row) => {
    if (!rowMatchesSampleFilters(row)) return false;
    if (!query) return true;
    return [
      row.handle,
      row.translated,
      row.x,
      row.sentiment,
      row.narrative_bucket,
      row.region_bucket,
      normalizeText(row.scene_tags),
      normalizeText(row.scenario_tags),
      normalizeText(row.praise_tags),
      normalizeText(row.complaint_tags),
      normalizeText(row.competitor_tags),
    ].join(" ").toLowerCase().includes(query);
  });
  state.sampleRows = rows;

  const sortedByDate = rows
    .filter((row) => row.t)
    .slice()
    .sort((a, b) => new Date(b.t) - new Date(a.t));
  const topSentiment = topCounts(rows, "sentiment", 1)[0]?.[0] || "未知";
  const competitorCount = rows.reduce((sum, row) => sum + valuesFromColumns(row, ["competitor_tags"]).length, 0);

  $("#sample-count").textContent = rows.length.toLocaleString();
  $("#sample-latest").textContent = sortedByDate[0]?.t?.slice(0, 10) || "未知";
  $("#sample-sentiment").textContent = topSentiment;
  $("#sample-competitors").textContent = competitorCount.toLocaleString();

  renderSampleBars(rows);
  renderSampleTable(rows);
}

function renderSampleBars(rows) {
  const config = [
    [["sentiment"], "情绪分布"],
    [["narrative_bucket"], "叙事主题"],
    [["region_bucket"], "区域分布"],
    [["scene_tags", "scenario_tags"], "场景"],
    [["praise_tags"], "夸赞指标"],
    [["competitor_tags"], "竞品提及"],
  ];
  const wrap = $("#sample-bars");
  wrap.innerHTML = "";

  config.forEach(([columns, title]) => {
    const counts = topCounts(rows, columns);
    if (!counts.length) return;
    const max = counts[0][1];
    const group = document.createElement("div");
    group.className = "bar-group";
    group.innerHTML = `<h3>${title}</h3>`;
    counts.forEach(([label, count]) => {
      const row = document.createElement("div");
      row.className = "bar-row";
      row.innerHTML = `
        <span title="${escapeHtml(label)}">${escapeHtml(label)}</span>
        <div class="bar-track"><div class="bar-fill" style="width:${Math.max(4, (count / max) * 100)}%"></div></div>
        <strong>${count}</strong>
      `;
      group.append(row);
    });
    wrap.append(group);
  });
}

function renderSampleTable(rows) {
  const tbody = $("#sample-table tbody");
  tbody.innerHTML = rows.slice(0, 160).map((row) => {
    const content = normalizeText(row.translated || row.x).slice(0, 220);
    return `
      <tr>
        <td>${escapeHtml(normalizeText(row.t).slice(0, 10))}</td>
        <td>${escapeHtml(normalizeText(row.handle))}</td>
        <td>${escapeHtml(normalizeText(row.sentiment))}</td>
        <td>${escapeHtml(normalizeText(row.narrative_bucket))}</td>
        <td>${escapeHtml(content)}</td>
      </tr>
    `;
  }).join("");
}

async function init() {
  setupTabs();
  setupCumulativeControls();
  setupTokenControls();
  setupProductControls();

  try {
    showToast("正在加载 Tokens 数据...", true);
    const rows = await loadCsv("history_database.csv");
    state.tokens = rows.map((row) => ({
      ...row,
      Total_Tokens: numberValue(row.Total_Tokens),
      Display_Name: row.Model.includes("/") ? row.Model.split("/").at(-1) : row.Model,
    }));
    const totals = Array.from(groupBy(state.tokens, (row) => row.Display_Name).entries())
      .map(([model, items]) => [model, items.reduce((sum, row) => sum + row.Total_Tokens, 0)])
      .sort((a, b) => b[1] - a[1]);
    state.models = totals.map(([model]) => model);
    renderCumulativeModelOptions();
    renderSingleModelOptions();
    renderAlerts();

    const manifest = await loadJson("product_reports/manifest.json");
    state.reports = manifest.reports;
    const vendors = Array.from(new Set(state.reports.map((item) => item.vendor))).sort();
    $("#vendor-select").innerHTML = vendors.map((vendor) => `<option value="${escapeHtml(vendor)}">${escapeHtml(vendor)}</option>`).join("");
    renderProductDownloads();
    renderProductOptions();

    showToast("数据已更新");
  } catch (error) {
    console.error(error);
    showToast(error.message || "数据加载失败", true);
  }
}

init();
