$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$PythonPath = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$DataRoot = (Resolve-Path (Join-Path $ProjectRoot "..\cxr_first")).Path

if (-not (Test-Path $PythonPath)) {
    throw "Project environment is missing. Run scripts\setup_windows_cuda.ps1 first."
}
if (-not (Test-Path $DataRoot)) {
    throw "Could not find the image tree at $DataRoot. Keep project_main beside cxr_first or pass --data-root manually."
}

Push-Location $ProjectRoot
try {
    & $PythonPath -m cxr_project.train --config configs/g1_seed42.yaml --data-root $DataRoot --device cuda
    if ($LASTEXITCODE -ne 0) { throw "Training failed with exit code $LASTEXITCODE." }
}
finally {
    Pop-Location
}
