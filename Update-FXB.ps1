# Update program files from one exact public FXB commit. Keeps data, runs,
# state, diagnostics and the existing Python virtual environment.
[CmdletBinding()]
param(
    [string]$ProjectRoot = '',
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[0-9a-f]{40}$')]
    [string]$Revision
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
if ([string]::IsNullOrWhiteSpace($ProjectRoot)) { $ProjectRoot = (Get-Location).Path }
$fxbRoot = (Resolve-Path -LiteralPath $ProjectRoot).Path
$fxbPython = Join-Path $fxbRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $fxbPython -PathType Leaf)) {
    throw 'Open PowerShell in the installed FXB folder, then use -ProjectRoot .'
}
if (-not (Test-Path -LiteralPath (Join-Path $fxbRoot 'src\fxb\mt5.py') -PathType Leaf)) {
    throw 'This folder is not an FXB source installation.'
}
$fxbProcesses = @(Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" | Where-Object {
    $_.ExecutablePath -and ($_.ExecutablePath -ieq $fxbPython)
})
if ($fxbProcesses.Count -gt 0) {
    throw 'Close FXB using Ctrl+C in its launcher window before updating. Leave the MT5 terminal open.'
}
if (Test-Path -LiteralPath (Join-Path $fxbRoot '.git')) {
    $fxbDirty = & git -C $fxbRoot status --porcelain --untracked-files=no
    if ($LASTEXITCODE -ne 0) { throw 'Cannot read the Git working tree.' }
    if ($fxbDirty) { throw 'Tracked source files have local changes. Preserve those changes before updating.' }
    & git -C $fxbRoot fetch 'https://github.com/sjdpluse/fxb.git' $Revision
    if ($LASTEXITCODE -ne 0) { throw 'Git download failed; installed files were not changed.' }
    & git -C $fxbRoot merge --ff-only $Revision
    if ($LASTEXITCODE -ne 0) { throw 'Fast-forward update refused. No local changes were discarded.' }
}
else {
    $fxbTemp = Join-Path ([System.IO.Path]::GetTempPath()) ('fxb-update-' + [Guid]::NewGuid().ToString('N'))
    $fxbBackup = Join-Path $fxbRoot ('backups\update-' + [DateTime]::UtcNow.ToString('yyyyMMdd-HHmmss-ffff'))
    $fxbProgramItems = @('src', 'pyproject.toml', 'Start-FXB.ps1', 'README.md')
    New-Item -ItemType Directory -Path $fxbTemp | Out-Null
    try {
        $fxbZip = Join-Path $fxbTemp 'release.zip'
        Invoke-WebRequest -UseBasicParsing -Uri "https://github.com/sjdpluse/fxb/archive/$Revision.zip" -OutFile $fxbZip
        Expand-Archive -LiteralPath $fxbZip -DestinationPath (Join-Path $fxbTemp 'release')
        $fxbSource = Join-Path $fxbTemp ("release\fxb-$Revision")
        foreach ($fxbItem in $fxbProgramItems) {
            if (-not (Test-Path -LiteralPath (Join-Path $fxbSource $fxbItem))) {
                throw 'Downloaded archive is incomplete; installed files were not changed.'
            }
        }
        New-Item -ItemType Directory -Path $fxbBackup -Force | Out-Null
        foreach ($fxbItem in $fxbProgramItems) {
            $fxbExisting = Join-Path $fxbRoot $fxbItem
            if (Test-Path -LiteralPath $fxbExisting) {
                Copy-Item -LiteralPath $fxbExisting -Destination $fxbBackup -Recurse -Force
            }
        }
        try {
            foreach ($fxbItem in $fxbProgramItems) {
                if ($fxbItem -eq 'src') {
                    Copy-Item -Path (Join-Path $fxbSource 'src\*') -Destination (Join-Path $fxbRoot 'src') -Recurse -Force
                }
                else {
                    Copy-Item -LiteralPath (Join-Path $fxbSource $fxbItem) -Destination (Join-Path $fxbRoot $fxbItem) -Force
                }
            }
        }
        catch {
            Write-Error "Program copy failed. Original source is preserved at $fxbBackup. Restore it before starting FXB."
            throw
        }
        Write-Host "Previous program files: $fxbBackup"
    }
    finally {
        if (Test-Path -LiteralPath $fxbTemp) { Remove-Item -LiteralPath $fxbTemp -Recurse -Force }
    }
}
Push-Location -LiteralPath $fxbRoot
try {
    & $fxbPython -m pip install -e '.[brain,desktop,mt5]'
    if ($LASTEXITCODE -ne 0) { throw 'Dependencies were not installed completely. Rerun Start-FXB.ps1 before starting the workstation.' }
    & $fxbPython -c 'import fxb; print(fxb.__version__)'
    if ($LASTEXITCODE -ne 0) { throw 'The updated package could not be imported.' }
}
finally { Pop-Location }
Write-Host 'Update complete. Run Start-FXB.ps1 to open the workstation.'
