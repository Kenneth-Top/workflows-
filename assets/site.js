const state = {
  tokens: [],
  filteredTokens: [],
  models: [],
  chart: null,
  reports: [],
  currentSamples: [],
};

const palette = ["#0f8b8d", "#d1495b", "#edae49", "#2e6f95", "#6c5ce7", "#247ba0", "#9a6324", "#008080"];

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

function setupTokenControls() {
  $("#model-search").addEventListener("input", renderModelOptions);
  $("#model-select").addEventListener("change", renderTokens);
  $("#range-select").addEventListener("change", renderTokens);
  $("#download-filtered").addEventListener("click", downloadFilteredTokens);
}

function renderModelOptions() {
  const search = $("#model-search").value.trim().toLowerCase();
  const select = $("#model-select");
  const previous = new Set(Array.from(select.selectedOptions).map((item) => item.value));
  const visibleModels = state.models.filter((model) => model.toLowerCase().includes(search));

  select.innerHTML = "";
  visibleModels.forEach((model, index) => {
    const option = document.createElement("option");
    option.value = model;
    option.textContent = model;
    option.selected = previous.has(model) || (!previous.size && index < 3);
    select.append(option);
  });
  renderTokens();
}

function selectedModels() {
  const selected = Array.from($("#model-select").selectedOptions).map((item) => item.value);
  return selected.length ? selected : state.models.slice(0, 3);
}

function filterByRange(rows) {
  const range = $("#range-select").value;
  if (range === "all" || !rows.length) return rows;
  const maxTime = Math.max(...rows.map((row) => new Date(row.Date).getTime()));
  const minTime = maxTime - Number(range) * 24 * 60 * 60 * 1000;
  return rows.filter((row) => new Date(row.Date).getTime() >= minTime);
}

function renderTokens() {
  if (!state.tokens.length) return;
  const models = selectedModels();
  const baseRows = state.tokens.filter((row) => models.includes(row.Display_Name));
  const rows = filterByRange(baseRows).sort((a, b) => a.Date.localeCompare(b.Date));
  state.filteredTokens = rows;

  const allDates = state.tokens.map((row) => row.Date).sort();
  const latestDate = allDates.at(-1);
  const latestTotal = state.tokens
    .filter((row) => row.Date === latestDate)
    .reduce((sum, row) => sum + row.Total_Tokens, 0);
  const selectedTotal = rows.reduce((sum, row) => sum + row.Total_Tokens, 0);

  $("#metric-models").textContent = state.models.length.toLocaleString();
  $("#metric-range").textContent = `${allDates[0]} ~ ${latestDate}`;
  $("#metric-latest").textContent = shortNumber(latestTotal);
  $("#metric-total").textContent = shortNumber(selectedTotal);

  renderTokenChart(rows, models);
  renderTokenTable(rows);
}

function renderTokenChart(rows, models) {
  const grouped = groupBy(rows, (row) => row.Display_Name);
  const dates = Array.from(new Set(rows.map((row) => row.Date))).sort();
  const datasets = models.map((model, index) => {
    let running = 0;
    const byDate = new Map((grouped.get(model) || []).map((row) => [row.Date, row.Total_Tokens]));
    return {
      label: model,
      data: dates.map((date) => {
        running += byDate.get(date) || 0;
        return Number(running.toFixed(6));
      }),
      borderColor: palette[index % palette.length],
      backgroundColor: palette[index % palette.length],
      tension: 0.25,
      pointRadius: 0,
      borderWidth: 2,
    };
  });

  if (state.chart) state.chart.destroy();
  state.chart = new Chart($("#token-chart"), {
    type: "line",
    data: { labels: dates, datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      plugins: {
        legend: { position: "bottom" },
        tooltip: { callbacks: { label: (ctx) => `${ctx.dataset.label}: ${shortNumber(ctx.parsed.y)}` } },
      },
      scales: {
        x: { ticks: { maxTicksLimit: 9 } },
        y: { title: { display: true, text: "累计 Tokens (Billion)" } },
      },
    },
  });
}

function renderTokenTable(rows) {
  const tbody = $("#token-table tbody");
  tbody.innerHTML = "";
  rows.slice().reverse().slice(0, 500).forEach((row) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${row.Date}</td>
      <td>${row.Display_Name}</td>
      <td>${row.Prompt.toFixed(6)}</td>
      <td>${row.Completion.toFixed(6)}</td>
      <td>${row.Reasoning.toFixed(6)}</td>
      <td>${row.Total_Tokens.toFixed(6)}</td>
    `;
    tbody.append(tr);
  });
}

function downloadFilteredTokens() {
  if (!state.filteredTokens.length) return;
  const headers = ["Date", "Model", "Prompt", "Completion", "Reasoning", "Total_Tokens"];
  const csv = [
    headers.join(","),
    ...state.filteredTokens.map((row) => headers.map((header) => row[header]).join(",")),
  ].join("\n");
  const blob = new Blob([`\ufeff${csv}`], { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "filtered_tokens.csv";
  link.click();
  URL.revokeObjectURL(url);
}

function setupProductControls() {
  $("#vendor-select").addEventListener("change", renderProductOptions);
  $("#product-select").addEventListener("change", renderProductReport);
  $("#sample-search").addEventListener("input", renderSamples);
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

async function renderProductReport() {
  const vendor = $("#vendor-select").value;
  const product = $("#product-select").value;
  const selected = state.reports.find((item) => item.vendor === vendor && item.product === product);
  if (!selected) return;

  $("#report-frame").src = selected.report;
  $("#report-link").href = selected.report;
  state.currentSamples = await loadJson(selected.dataset);
  renderSamples();
}

function normalizeText(value) {
  if (Array.isArray(value)) return value.join(" ");
  return value ? String(value) : "";
}

function topCounts(rows, column, limit = 6) {
  const counts = new Map();
  rows.forEach((row) => {
    const values = Array.isArray(row[column]) ? row[column] : [row[column]];
    values.filter(Boolean).forEach((value) => counts.set(value, (counts.get(value) || 0) + 1));
  });
  return Array.from(counts.entries()).sort((a, b) => b[1] - a[1]).slice(0, limit);
}

function renderSamples() {
  const query = $("#sample-search").value.trim().toLowerCase();
  const rows = state.currentSamples.filter((row) => {
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
      normalizeText(row.competitor_tags),
    ].join(" ").toLowerCase().includes(query);
  });

  const sortedByDate = rows
    .filter((row) => row.t)
    .slice()
    .sort((a, b) => new Date(b.t) - new Date(a.t));
  const topSentiment = topCounts(rows, "sentiment", 1)[0]?.[0] || "未知";
  const competitorCount = rows.reduce((sum, row) => sum + (Array.isArray(row.competitor_tags) ? row.competitor_tags.length : 0), 0);

  $("#sample-count").textContent = rows.length.toLocaleString();
  $("#sample-latest").textContent = sortedByDate[0]?.t?.slice(0, 10) || "未知";
  $("#sample-sentiment").textContent = topSentiment;
  $("#sample-competitors").textContent = competitorCount.toLocaleString();

  renderSampleBars(rows);
  renderSampleTable(rows);
}

function renderSampleBars(rows) {
  const config = [
    ["sentiment", "情绪分布"],
    ["narrative_bucket", "叙事主题"],
    ["region_bucket", "区域分布"],
    ["competitor_tags", "竞品提及"],
  ];
  const wrap = $("#sample-bars");
  wrap.innerHTML = "";

  config.forEach(([column, title]) => {
    const counts = topCounts(rows, column);
    if (!counts.length) return;
    const max = counts[0][1];
    const group = document.createElement("div");
    group.className = "bar-group";
    group.innerHTML = `<h3>${title}</h3>`;
    counts.forEach(([label, count]) => {
      const row = document.createElement("div");
      row.className = "bar-row";
      row.innerHTML = `
        <span title="${label}">${label}</span>
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
  tbody.innerHTML = "";
  rows.slice(0, 120).forEach((row) => {
    const tr = document.createElement("tr");
    const content = normalizeText(row.translated || row.x).slice(0, 180);
    tr.innerHTML = `
      <td>${normalizeText(row.t).slice(0, 10)}</td>
      <td>${normalizeText(row.handle)}</td>
      <td>${normalizeText(row.sentiment)}</td>
      <td>${normalizeText(row.narrative_bucket)}</td>
      <td>${content}</td>
    `;
    tbody.append(tr);
  });
}

async function init() {
  setupTabs();
  setupTokenControls();
  setupProductControls();

  try {
    showToast("正在加载 Tokens 数据...", true);
    const rows = await loadCsv("history_database.csv");
    state.tokens = rows.map((row) => ({
      ...row,
      Prompt: numberValue(row.Prompt),
      Completion: numberValue(row.Completion),
      Reasoning: numberValue(row.Reasoning),
      Total_Tokens: numberValue(row.Total_Tokens),
      Display_Name: row.Model.includes("/") ? row.Model.split("/").at(-1) : row.Model,
    }));
    const totals = Array.from(groupBy(state.tokens, (row) => row.Display_Name).entries())
      .map(([model, items]) => [model, items.reduce((sum, row) => sum + row.Total_Tokens, 0)])
      .sort((a, b) => b[1] - a[1]);
    state.models = totals.map(([model]) => model);
    renderModelOptions();

    const manifest = await loadJson("product_reports/manifest.json");
    state.reports = manifest.reports;
    const vendors = Array.from(new Set(state.reports.map((item) => item.vendor))).sort();
    $("#vendor-select").innerHTML = vendors.map((vendor) => `<option value="${vendor}">${vendor}</option>`).join("");
    renderProductOptions();

    showToast("数据已更新");
  } catch (error) {
    console.error(error);
    showToast(error.message || "数据加载失败", true);
  }
}

init();
