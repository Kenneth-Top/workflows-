param(
  [string]$ExcelPath = "",
  [string]$SheetName = "market_cap",
  [string]$OutputCsv = "ai_market_cap_history.csv",
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
    throw "找不到工作表 '$SheetName'。请确认 Excel 中有这个 sheet，且表头与 ai_market_cap_history.csv 一致。"
  }

  Write-Host "Exporting sheet '$SheetName' to CSV..."
  $worksheet.Copy()
  $tempWorkbook = $excel.ActiveWorkbook
  $tempWorkbook.SaveAs($TempCsv, 6)
  $tempWorkbook.Close($false)

  Copy-Item -LiteralPath $TempCsv -Destination $OutputCsv -Force
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
