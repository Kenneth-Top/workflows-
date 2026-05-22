param(
  [string]$ProtocolName = "wind-market-refresh"
)

$ErrorActionPreference = "Stop"

$repoRoot = $PSScriptRoot
$refreshScript = Join-Path $repoRoot "refresh_wind_market_cap.ps1"

if (-not (Test-Path -LiteralPath $refreshScript)) {
  throw "找不到 refresh_wind_market_cap.ps1：$refreshScript"
}

$powershell = Join-Path $env:WINDIR "System32\WindowsPowerShell\v1.0\powershell.exe"
$command = '"' + $powershell + '" -NoProfile -ExecutionPolicy Bypass -NoExit -File "' + $refreshScript + '"'
$protocolRoot = "Registry::HKEY_CURRENT_USER\Software\Classes\$ProtocolName"

New-Item -Path $protocolRoot -Force | Out-Null
Set-Item -Path $protocolRoot -Value "URL:Wind Market Cap Refresh"
New-ItemProperty -Path $protocolRoot -Name "URL Protocol" -Value "" -PropertyType String -Force | Out-Null

New-Item -Path "$protocolRoot\shell\open\command" -Force | Out-Null
Set-Item -Path "$protocolRoot\shell\open\command" -Value $command

Write-Host "已注册 ${ProtocolName}://run"
Write-Host "网页按钮会打开 PowerShell，并执行："
Write-Host $refreshScript
