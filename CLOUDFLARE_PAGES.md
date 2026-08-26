# Cloudflare Pages 部署设置

这个仓库现在可以作为静态站点直接部署到 Cloudflare Pages。每天 GitHub Actions 更新 CSV 并 push 后，Cloudflare Pages 会自动重新部署，网站不会因为空闲而睡眠。

## 推荐设置

- Framework preset: `None`
- Build command: 留空
- Build output directory: `.`
- Root directory: 留空或仓库根目录
- Production branch: `main`

## 数据更新

现有 `.github/workflows/daily.yml` 会继续更新这些文件：

- `history_database.csv`
- `openrouter_pricing_provider_records.csv`
- `openrouter_benchmark_records.csv`
- `lmarena_leaderboard_records.csv`

只要 Cloudflare Pages 连接了这个 GitHub 仓库，每次 workflow push 后都会触发一次重新部署。

## 排障提示

如果 GitHub `main` 分支里的 CSV 已经更新，但 `pages.dev` 线上文件仍然停留在旧日期，请先检查 Cloudflare Pages 的最新生产部署是否确实取到了最新一次 `main` push。监控数据文件是静态资源，线上落后通常说明部署没有触发、触发后失败，或者生产分支/根目录配置不一致。

## 产品测评报告

页面读取 `product_reports/manifest.json`。每个产品只登记最新版 HTML 报告和最新版 JSON 数据。
