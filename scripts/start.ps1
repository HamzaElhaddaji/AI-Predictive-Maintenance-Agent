param([switch]$SkipInstall)
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
if (-not (Test-Path .venv/Scripts/python.exe)) { python -m venv .venv }
if (-not $SkipInstall) {
    & .venv/Scripts/python.exe -m pip install -r requirements.lock
    if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed' }
}
if (-not (Test-Path models/maintenance.joblib)) {
    & .venv/Scripts/python.exe -m src.maintenance.cli train
    if ($LASTEXITCODE -ne 0) { throw 'Training failed' }
}
$api = Start-Process -WindowStyle Hidden -PassThru -FilePath .venv/Scripts/python.exe -ArgumentList @(
    '-m','uvicorn','src.maintenance.api:app','--host','127.0.0.1','--port','8000')
try {
    & .venv/Scripts/python.exe -m streamlit run src/maintenance/ui.py --server.address=127.0.0.1
} finally {
    if (-not $api.HasExited) { Stop-Process -Id $api.Id }
}
