$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
function Invoke-Checked {
    param([string]$Command, [string[]]$Arguments)
    & $Command @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Command failed: $Command" }
}
if (-not (Test-Path '.venv\Scripts\python.exe')) {
    Invoke-Checked 'py' @('-3.12', '-m', 'venv', '.venv')
}
$Python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
Invoke-Checked $Python @('-m', 'pip', 'install', '-e', '.[brain,desktop,mt5]')
if (-not (Test-Path 'data\connectome\manifest.json')) {
    Invoke-Checked $Python @('-m', 'fxb', 'brain-install')
}
Invoke-Checked $Python @('-m', 'fxb', 'serve')
