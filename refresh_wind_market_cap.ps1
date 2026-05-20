param(
  [string]$ExcelPath = "",
  [string]$SheetName = "market_cap",
  [string]$OutputCsv = "ai_market_cap_history.csv",
  [ValidateSet("Auto", "Wide", "Long", "Sheets")]
  [string]$WorkbookMode = "Auto",
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

function Invoke-ExcelComWithRetry([string]$Description, [scriptblock]$Action, [int]$Attempts = 8) {
  for ($attempt = 1; $attempt -le $Attempts; $attempt++) {
    try {
      return & $Action
    } catch [System.Runtime.InteropServices.COMException] {
      if ($attempt -eq $Attempts) {
        throw
      }
      Write-Host "$Description was busy; retrying ($attempt/$Attempts)..."
      Start-Sleep -Seconds 3
    }
  }
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

function Load-CompanyMaps() {
  $byId = @{}
  $byTicker = @{}
  if (Test-Path (Join-Path $PSScriptRoot "ai_market_companies.csv")) {
    Import-Csv -LiteralPath (Join-Path $PSScriptRoot "ai_market_companies.csv") | ForEach-Object {
      $byId[$_.company_id] = $_
      if ($_.ticker) {
        $byTicker[(Normalize-Header $_.ticker)] = $_.company_id
      }
    }
  }
  return @{ ById = $byId; ByTicker = $byTicker }
}

function Read-ExistingRows([string]$OutputCsv) {
  $existing = @{}
  if (Test-Path $OutputCsv) {
    Import-Csv -LiteralPath $OutputCsv | ForEach-Object {
      $existing["$($_.Date)|$($_.Company_ID)"] = $_
    }
  }
  return $existing
}

function Export-HistoryRows([hashtable]$Rows, [string]$OutputCsv) {
  $headers = @("Date", "Company_ID", "Ticker", "Market_Cap_Billion_HKD", "Market_Cap_Native_Billion", "Currency", "Close", "Source", "Updated_At")
  $lines = New-Object System.Collections.Generic.List[string]
  $lines.Add(($headers -join ","))
  $Rows.Values |
    Sort-Object Date, Company_ID |
    ForEach-Object {
      $values = foreach ($header in $headers) {
        Format-CsvCell $_.$header
      }
      $lines.Add(($values -join ","))
    }
  $utf8WithBom = New-Object System.Text.UTF8Encoding($true)
  [System.IO.File]::WriteAllLines($OutputCsv, $lines, $utf8WithBom)
}

function Format-CsvCell($Value) {
  if ($null -eq $Value) { return "" }
  $text = $Value.ToString()
  if ($text.Contains('"')) {
    $text = $text.Replace('"', '""')
  }
  if ($text.Contains(",") -or $text.Contains('"') -or $text.Contains("`r") -or $text.Contains("`n")) {
    return '"' + $text + '"'
  }
  return $text
}

function Upsert-MarketCapRow(
  [hashtable]$Rows,
  [hashtable]$CompanyRows,
  [string]$Date,
  [string]$CompanyId,
  [double]$MarketCapBillion,
  [string]$Source,
  [string]$UpdatedAt
) {
  $company = $CompanyRows[$CompanyId]
  $ticker = if ($company) { $company.ticker } else { "" }
  $currency = if ($company) { $company.currency } else { "HKD" }
  $marketCap = "{0:F6}" -f $MarketCapBillion
  $Rows["$Date|$CompanyId"] = [pscustomobject]@{
    Date = $Date
    Company_ID = $CompanyId
    Ticker = $ticker
    Market_Cap_Billion_HKD = $marketCap
    Market_Cap_Native_Billion = $marketCap
    Currency = $currency
    Close = ""
    Source = $Source
    Updated_At = $UpdatedAt
  }
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

  $maps = Load-CompanyMaps
  $companyRows = $maps.ById
  $existing = Read-ExistingRows $OutputCsv

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
      Upsert-MarketCapRow -Rows $existing -CompanyRows $companyRows -Date $date -CompanyId $companyId -MarketCapBillion $value -Source "wind_excel" -UpdatedAt $updatedAt
    }
  }

  Export-HistoryRows -Rows $existing -OutputCsv $OutputCsv
}

function Convert-WindWorkbookSheetsToHistoryCsv($Workbook, [string]$OutputCsv, [string]$Unit) {
  $maps = Load-CompanyMaps
  $companyRows = $maps.ById
  $tickerMap = $maps.ByTicker
  $existing = Read-ExistingRows $OutputCsv
  $updatedAt = (Get-Date).ToUniversalTime().ToString("s") + "Z"

  foreach ($sheet in $Workbook.Worksheets) {
    $sheetName = $sheet.Name
    if ($sheetName.StartsWith("_")) { continue }

    $ticker = ""
    $companyId = ""
    for ($row = 1; $row -le [Math]::Min(10, $sheet.UsedRange.Rows.Count); $row++) {
      $key = ($sheet.Cells.Item($row, 1).Text).Trim()
      if ($key -in @("证券代码", "Ticker", "股票代码")) {
        $ticker = ($sheet.Cells.Item($row, 2).Text).Trim()
        break
      }
    }
    if ($ticker) {
      $companyId = $tickerMap[(Normalize-Header $ticker)]
    }
    if (-not $companyId) {
      $companyId = CompanyIdFromHeader $sheetName
    }

    $headerRow = 0
    $dateCol = 0
    $valueCol = 0
    for ($row = 1; $row -le [Math]::Min(20, $sheet.UsedRange.Rows.Count); $row++) {
      for ($col = 1; $col -le [Math]::Min(10, $sheet.UsedRange.Columns.Count); $col++) {
        $header = Normalize-Header ($sheet.Cells.Item($row, $col).Text)
        if ($header -in @("date", "日期", "时间")) {
          $headerRow = $row
          $dateCol = $col
        }
        if ($header -in @("ev", "总市值", "总市值1", "marketcap", "市值", "市值亿港币")) {
          $valueCol = $col
        }
      }
      if ($headerRow -gt 0 -and $dateCol -gt 0 -and $valueCol -gt 0) { break }
    }

    if ($headerRow -eq 0 -or $dateCol -eq 0 -or $valueCol -eq 0) {
      Write-Host "Skipping sheet '$sheetName': cannot find Date/ev columns."
      continue
    }

    for ($row = $headerRow + 1; $row -le $sheet.UsedRange.Rows.Count; $row++) {
      $dateText = ($sheet.Cells.Item($row, $dateCol).Text).Trim()
      $valueText = ($sheet.Cells.Item($row, $valueCol).Text).Trim()
      if (-not $dateText -or -not $valueText) { continue }
      try {
        $date = ([datetime]$dateText).ToString("yyyy-MM-dd")
      } catch {
        if ($dateText.Length -lt 10) { continue }
        $date = $dateText.Substring(0, 10)
      }
      $value = Convert-ValueToBillionHKD $valueText $Unit
      if ($null -eq $value) { continue }
      Upsert-MarketCapRow -Rows $existing -CompanyRows $companyRows -Date $date -CompanyId $companyId -MarketCapBillion $value -Source "wind_excel_sheet" -UpdatedAt $updatedAt
    }
  }

  Export-HistoryRows -Rows $existing -OutputCsv $OutputCsv
}

function Test-WindCompanySheet($Sheet) {
  $hasTicker = $false
  for ($row = 1; $row -le [Math]::Min(10, $Sheet.UsedRange.Rows.Count); $row++) {
    $key = ($Sheet.Cells.Item($row, 1).Text).Trim()
    if ($key -in @("证券代码", "Ticker", "股票代码")) {
      $hasTicker = $true
      break
    }
  }

  $hasDate = $false
  $hasValue = $false
  for ($row = 1; $row -le [Math]::Min(20, $Sheet.UsedRange.Rows.Count); $row++) {
    for ($col = 1; $col -le [Math]::Min(10, $Sheet.UsedRange.Columns.Count); $col++) {
      $header = Normalize-Header ($Sheet.Cells.Item($row, $col).Text)
      if ($header -in @("date", "日期", "时间")) {
        $hasDate = $true
      }
      if ($header -in @("ev", "总市值", "总市值1", "marketcap", "市值", "市值亿港币")) {
        $hasValue = $true
      }
    }
  }

  return ($hasTicker -and $hasDate -and $hasValue)
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
  Invoke-ExcelComWithRetry "RefreshAll" { $workbook.RefreshAll() } | Out-Null
  try {
    Invoke-ExcelComWithRetry "CalculateUntilAsyncQueriesDone" { $excel.CalculateUntilAsyncQueriesDone() } | Out-Null
  } catch {
    Write-Host "CalculateUntilAsyncQueriesDone unavailable, waiting manually..."
  }
  Start-Sleep -Seconds $RefreshWaitSeconds
  Invoke-ExcelComWithRetry "CalculateFullRebuild" { $excel.CalculateFullRebuild() } | Out-Null
  Invoke-ExcelComWithRetry "Workbook.Save" { $workbook.Save() } | Out-Null

  if ($WorkbookMode -eq "Sheets") {
    Write-Host "Converting one-company-per-sheet workbook..."
    Convert-WindWorkbookSheetsToHistoryCsv -Workbook $workbook -OutputCsv $OutputCsv -Unit $WideUnit
  } else {
    $worksheet = $null
    foreach ($sheet in $workbook.Worksheets) {
      if ($sheet.Name -eq $SheetName) {
        $worksheet = $sheet
        break
      }
    }
    if (-not $worksheet) {
      if ($WorkbookMode -eq "Auto") {
        Write-Host "Sheet '$SheetName' not found. Falling back to one-company-per-sheet mode..."
        Convert-WindWorkbookSheetsToHistoryCsv -Workbook $workbook -OutputCsv $OutputCsv -Unit $WideUnit
      } else {
        throw "找不到工作表 '$SheetName'。请确认 Excel 中有这个 sheet。推荐格式：Date | minimax | zhipu | ..."
      }
    } else {
      $convertedWorkbook = $false
      if ($WorkbookMode -eq "Auto" -and (Test-WindCompanySheet $worksheet)) {
        Write-Host "Detected one-company-per-sheet Wind layout. Converting all company sheets..."
        Convert-WindWorkbookSheetsToHistoryCsv -Workbook $workbook -OutputCsv $OutputCsv -Unit $WideUnit
        $convertedWorkbook = $true
      }
      if (-not $convertedWorkbook) {
        Write-Host "Exporting sheet '$SheetName' to CSV..."
        $worksheet.Copy()
        $tempWorkbook = $excel.ActiveWorkbook
        $tempWorkbook.SaveAs($TempCsv, 6)
        $tempWorkbook.Close($false)
        Convert-WindCsvToHistoryCsv -InputCsv $TempCsv -OutputCsv $OutputCsv -Unit $WideUnit
      }
    }
  }
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
