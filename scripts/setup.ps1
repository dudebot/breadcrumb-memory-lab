param([switch]$UseExistingTorch)
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
if (-not (Test-Path '.venv\Scripts\python.exe')) {
    if ($UseExistingTorch) {
        python -m venv --system-site-packages .venv
    } else {
        python -m venv .venv
    }
    if ($LASTEXITCODE -ne 0) { throw 'Virtual environment creation failed' }
}
if (-not $UseExistingTorch) {
    & .\.venv\Scripts\python.exe -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
    if ($LASTEXITCODE -ne 0) { throw 'PyTorch installation failed' }
}
& .\.venv\Scripts\python.exe -m breadcrumb_memory doctor
if ($LASTEXITCODE -ne 0) { throw 'Hardware check failed' }
