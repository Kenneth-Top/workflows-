param(
  [string]$ExcelPath = "",
  [string]$SheetName = "market_cap",
  [string]$OutputCsv = "ai_market_cap_history.csv",
  [ValidateSet("YiHKD", "BillionHKD")]
  [string]$WideUnit = "YiHKD",
  [int]$RefreshWaitSeconds = 90,
  [switch]$NoGitPush
)

$ErrorActionPreference = "Stop"

function Resolve-RepoPath([string]$PathValue) {
  if ([System.IO.Path]::IsPathRooted($PathValue)) {
    return $PathValue
  }
  return Join-Path $PSScriptRoot $PathValue
}

function Normalize-Header([string]$Value) {
  if ($null -eq $Value) { $Value = "" }
  return $Value.Trim().ToLowerInvariant().Replace(" ", "").Replace("_", "").Replace("-", "")
}

function Convert-ValueToBillionHKD([string]$Value, [string]$Unit) {
  if (-not $Value) { return $null }
  $clean = $Value.ToString().Trim().Replace(",", "")
  if (-not $clean) { return $null }
  $number = 0.0
  if (-not [double]::TryParse($clean, [ref]$number)) { return $null }
  if ($Unit -eq "YiHKD") {
    return $number / 10.0
  }
  return $number
}

function CompanyIdFromHeader([string]$Header) {
  $normalized = Normalize-Header $Header
  $map = @{
    "minimax" = "minimax"
    "minimaxw" = "minimax"
    "0100hk" = "minimax"
    "智谱" = "zhipu"
    "zhipu" = "zhipu"
    "zai" = "zhipu"
    "2513hk" = "zhipu"
    "kimi" = "kimi"
    "moonshot" = "kimi"
    "阶跃星辰" = "stepfun"
    "stepfun" = "stepfun"
    "阿里" = "alibaba"
    "alibaba" = "alibaba"
    "9988hk" = "alibaba"
    "字节" = "bytedance"
    "bytedance" = "bytedance"
    "腾讯" = "tencent"
    "tencent" = "tencent"
    "0700hk" = "tencent"
    "openai" = "openai"
    "anthropic" = "anthropic"
    "google" = "google"
    "googl" = "google"
    "meta" = "meta"
    "spacex" = "spacex"
  }
  if ($map.ContainsKey($normalized)) {
    return $map[$normalized]
  }
  return $normalized
}

function Convert-WindCsvToHistoryCsv([string]$InputCsv, [string]$OutputCsv, [string]$Unit) {
  $canonicalHeaders = @("Date", "Company_ID", "Ticker", "Market_Cap_Billion_HKD", "Market_Cap_Native_Billion", "Currency", "Close", "Source", "Updated_At")
  $rows = Import-Csv -LiteralPath $InputCsv
  if (-not $rows -or $rows.Count -eq 0) {
    throw "Excel 导出的 CSV 为空。"
  }

  $headers = @($rows[0].PSObject.Properties.Name)
  $normalizedHeaders = $headers | ForEach-Object { Normalize-Header $_ }
  $canonicalNormalized = $canonicalHeaders | ForEach-Object { Normalize-Header $_ }
  $isCanonical = @($canonicalNormalized | Where-Object { $_ -notin $normalizedHeaders }).Count -eq 0

  if ($isCanonical) {
    Copy-Item -LiteralPath $InputCsv -Destination $OutputCsv -Force
    return
  }

  $dateHeader = $headers | Where-Object { (Normalize-Header $_) -in @("date", "日期", "时间") } | Select-Object -First 1
  if (-not $dateHeader) {
    throw "宽表必须有 Date 或 日期 列。推荐格式：Date | minimax | zhipu | ..."
  }

  $companyRows = @{}
  if (Test-Path (Join-Path $PSScriptRoot "ai_market_companies.csv")) {
    Import-Csv -LiteralPath (Join-Path $PSScriptRoot "ai_market_companies.csv") | ForEach-Object {
      $companyRows[$_.company_id] = $_
    }
  }

  $existing = @{}
  if (Test-Path $OutputCsv) {
    Import-Csv -LiteralPath $OutputCsv | ForEach-Object {
      $existing["$($_.Date)|$($_.Company_ID)"] = $_
    }
  }

  $companyHeaders = $headers | Where-Object { $_ -ne $dateHeader -and $_ -and -not $_.StartsWith("Unnamed") }
  $updatedAt = (Get-Date).ToUniversalTime().ToString("s") + "Z"
  foreach ($row in $rows) {
    $date = $row.$dateHeader
    if ($null -eq $date) { $date = "" }
    $date = $date.ToString().Trim()
    if (-not $date) { continue }
    try {
      $date = ([datetime]$date).ToString("yyyy-MM-dd")
    } catch {
      $date = $date.Substring(0, [Math]::Min(10, $date.Length))
    }
    foreach ($header in $companyHeaders) {
      $companyId = CompanyIdFromHeader $header
      $value = Convert-ValueToBillionHKD $row.$header $Unit
      if ($null -eq $value) { continue }
      $company = $companyRows[$companyId]
      $ticker = if ($company) { $company.ticker } else { "" }
      $currency = if ($company) { $company.currency } else { "HKD" }
      $marketCap = "{0:F6}" -f $value
      $existing["$date|$companyId"] = [pscustomobject]@{
        Date = $date
        Company_ID = $companyId
        Ticker = $ticker
        Market_Cap_Billion_HKD = $marketCap
        Market_Cap_Native_Billion = $marketCap
        Currency = $currency
        Close = ""
        Source = "wind_excel"
        Updated_At = $updatedAt
      }
    }
  }

  $existing.Values |
    Sort-Object Date, Company_ID |
    Select-Object Date, Company_ID, Ticker, Market_Cap_Billion_HKD, Market_Cap_Native_Billion, Currency, Close, Source, Updated_At |
    Export-Csv -LiteralPath $OutputCsv -NoTypeInformation -Encoding UTF8
}

if (-not $ExcelPath) {
  $candidates = @(
    (Join-Path $PSScriptRoot "wind_market_cap.xlsx"),
    (Join-Path $PSScriptRoot "ai_market_cap_wind.xlsx")
  )
  $ExcelPath = ($candidates | Where-Object { Test-Path $_ } | Select-Object -First 1)
}

if (-not $ExcelPath -or -not (Test-Path $ExcelPath)) {
  throw "找不到 Wind Excel。请把文件放在仓库根目录并命名为 wind_market_cap.xlsx，或用 -ExcelPath 传入完整路径。"
}

$ExcelPath = Resolve-RepoPath $ExcelPath
$OutputCsv = Resolve-RepoPath $OutputCsv
$TempCsv = Join-Path $env:TEMP ("wind_market_cap_{0}.csv" -f ([guid]::NewGuid().ToString("N")))

Write-Host "Opening Excel: $ExcelPath"
$excel = $null
$workbook = $null
try {
  $excel = New-Object -ComObject Excel.Application
  $excel.Visible = $true
  $excel.DisplayAlerts = $false
  $workbook = $excel.Workbooks.Open($ExcelPath)

  Write-Host "Refreshing Wind formulas..."
  $workbook.RefreshAll()
  try {
    $excel.CalculateUntilAsyncQueriesDone()
  } catch {
    Write-Host "CalculateUntilAsyncQueriesDone unavailable, waiting manually..."
  }
  Start-Sleep -Seconds $RefreshWaitSeconds
  $excel.CalculateFullRebuild()
  $workbook.Save()

  $worksheet = $null
  foreach ($sheet in $workbook.Worksheets) {
    if ($sheet.Name -eq $SheetName) {
      $worksheet = $sheet
      break
    }
  }
  if (-not $worksheet) {
    throw "找不到工作表 '$SheetName'。请确认 Excel 中有这个 sheet。推荐格式：Date | minimax | zhipu | ..."
  }

  Write-Host "Exporting sheet '$SheetName' to CSV..."
  $worksheet.Copy()
  $tempWorkbook = $excel.ActiveWorkbook
  $tempWorkbook.SaveAs($TempCsv, 6)
  $tempWorkbook.Close($false)

  Convert-WindCsvToHistoryCsv -InputCsv $TempCsv -OutputCsv $OutputCsv -Unit $WideUnit
  Write-Host "Updated CSV: $OutputCsv"
} finally {
  if ($workbook) {
    $workbook.Close($true)
  }
  if ($excel) {
    $excel.Quit()
  }
  if (Test-Path $TempCsv) {
    Remove-Item -LiteralPath $TempCsv -Force
  }
  [System.GC]::Collect()
  [System.GC]::WaitForPendingFinalizers()
}

if (-not $NoGitPush) {
  Push-Location $PSScriptRoot
  try {
    Write-Host "Pushing updated market cap CSV..."
    git pull --rebase origin main
    git add ai_market_cap_history.csv
    git commit -m "Refresh Wind market cap data"
    if ($LASTEXITCODE -ne 0) {
      Write-Host "No CSV changes to commit."
    } else {
      git push origin main
    }
  } finally {
    Pop-Location
  }
}

Write-Host "Done."
