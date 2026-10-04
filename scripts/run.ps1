param(
    [ValidateSet('smoke', 'pilot')][string]$Preset = 'smoke',
    [ValidateSet('auto', 'cpu', 'cuda')][string]$Device = 'auto',
    [int]$Seed = 17
)
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
$pythonPath = Join-Path (Get-Location) '.venv\Scripts\python.exe'
if (-not (Test-Path $pythonPath)) { throw 'Run scripts\setup.ps1 first' }
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss-fff'
$runPath = "runs/$Preset-$Seed-$stamp"
& $pythonPath -m breadcrumb_memory train --config "configs/$Preset.json" --output $runPath --device $Device --seed $Seed
if ($LASTEXITCODE -ne 0) { throw "Experiment stopped. Inspect $runPath" }
Get-Content "$runPath/REPORT.md"
