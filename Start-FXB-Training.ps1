param(
    [string]$ProjectRoot = $PSScriptRoot,
    [Parameter(Mandatory = $true)][string]$DataDirectory,
    [ValidateRange(1, 200)][int]$Epochs = 10
)
$ErrorActionPreference = 'Stop'
if ([string]::IsNullOrWhiteSpace($ProjectRoot)) { $ProjectRoot = (Get-Location).Path }
$fxbRoot = (Resolve-Path -LiteralPath $ProjectRoot).Path
$fxbData = (Resolve-Path -LiteralPath $DataDirectory).Path
$fxbPython = Join-Path $fxbRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $fxbPython -PathType Leaf)) { throw 'FXB Python environment was not found.' }
$fxbAudit = Get-Content -LiteralPath (Join-Path $fxbData 'audit.json') -Raw | ConvertFrom-Json
$fxbProfile = Get-Content -LiteralPath (Join-Path $fxbData 'profile.json') -Raw | ConvertFrom-Json
$fxbInitial = [double]$fxbProfile.initial_balance_scenario
if ([double]::IsNaN($fxbInitial) -or [double]::IsInfinity($fxbInitial) -or $fxbInitial -le 0) {
    throw 'The prepared profile must declare a positive initial_balance_scenario.'
}
$fxbValidation = @'
import json, sys
from pathlib import Path
from fxb import __version__
from fxb.data import load_bars
if tuple(map(int, __version__.split('.'))) < (0, 2, 3):
    raise SystemExit('Update FXB to 0.2.3 or newer before using prepared data.')
root = Path(sys.argv[1])
audit = json.loads((root / 'audit.json').read_text(encoding='utf-8-sig'))
if audit.get('schema') != 'fxb-market-audit-v1' or not audit.get('symbols'):
    raise SystemExit('Invalid preparation audit.')
for item in audit['symbols']:
    path = root / item['file']
    if path.resolve().parent != root.resolve():
        raise SystemExit('Dataset path must stay inside the prepared directory.')
    bars = load_bars(path)
    if bars.digest() != item['dataset_sha256']:
        raise SystemExit('Prepared data or assumptions changed: ' + item['symbol'])
print('Prepared dataset hashes verified. FXB ' + __version__)
'@
& $fxbPython -c $fxbValidation $fxbData
if ($LASTEXITCODE -ne 0) { throw 'Prepared dataset validation failed.' }
$fxbRunId = 'market-' + [DateTime]::UtcNow.ToString('yyyyMMdd-HHmmss') + '-' + [Guid]::NewGuid().ToString('N').Substring(0, 8)
$fxbRuns = Join-Path $fxbRoot ('runs\' + $fxbRunId)
$fxbReports = Join-Path $fxbRuns 'reports'
New-Item -ItemType Directory -Path $fxbReports -Force | Out-Null
$fxbBalanceText = $fxbInitial.ToString([Globalization.CultureInfo]::InvariantCulture)
Write-Host 'Training the complete connectome on historical data. This command sends no orders.'
Write-Host ('Initial balance scenario: ' + $fxbBalanceText)
Write-Host ('Clock evidence: ' + $fxbProfile.clock_evidence.status)
Push-Location $fxbRoot
try {
    foreach ($fxbItem in $fxbAudit.symbols) {
        $fxbName = [IO.Path]::GetFileNameWithoutExtension([string]$fxbItem.file)
        $fxbOutput = Join-Path $fxbRuns $fxbName
        Write-Host ('Training: ' + $fxbItem.symbol)
        & $fxbPython -u -m fxb train --data (Join-Path $fxbData $fxbItem.file) --out $fxbOutput --full-brain --epochs $Epochs --initial-balance $fxbBalanceText
        if ($LASTEXITCODE -ne 0) { throw ('Training stopped for ' + $fxbItem.symbol + '. Existing output was preserved at ' + $fxbOutput) }
        Copy-Item -LiteralPath (Join-Path $fxbOutput 'report.json') -Destination (Join-Path $fxbReports ($fxbName + '-report.json'))
        Copy-Item -LiteralPath (Join-Path $fxbOutput 'holdout_trades.json') -Destination (Join-Path $fxbReports ($fxbName + '-holdout-trades.json'))
    }
    $fxbDiagnostics = Join-Path $fxbRoot 'diagnostics'
    New-Item -ItemType Directory -Path $fxbDiagnostics -Force | Out-Null
    $fxbReportZip = Join-Path $fxbDiagnostics ('fxb-training-report-' + $fxbRunId + '.zip')
    Compress-Archive -LiteralPath $fxbReports -DestinationPath $fxbReportZip
    Write-Host ('Training complete. Models: ' + $fxbRuns)
    Write-Host ('Reports to review: ' + $fxbReportZip)
} finally {
    Pop-Location
}
