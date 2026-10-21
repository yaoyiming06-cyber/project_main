$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$VenvPath = Join-Path $ProjectRoot ".venv"
$PythonPath = Join-Path $VenvPath "Scripts\python.exe"

if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
    throw "Python Launcher was not found. Install 64-bit Python 3.11 or 3.12 from python.org first."
}

$PythonSelector = "-3.11"
& py $PythonSelector --version
if ($LASTEXITCODE -ne 0) {
    $PythonSelector = "-3.12"
    & py $PythonSelector --version
    if ($LASTEXITCODE -ne 0) {
        throw "Python 3.11 or 3.12 was not found. Install one and enable the Python Launcher."
    }
}

if (-not (Test-Path $PythonPath)) {
    & py $PythonSelector -m venv $VenvPath
    if ($LASTEXITCODE -ne 0) { throw "Could not create the project virtual environment." }
}

& $PythonPath -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "Could not upgrade pip." }

# The PyTorch wheel includes the CUDA 12.8 runtime. A compatible NVIDIA driver
# is required; the full CUDA Toolkit is not needed for this project.
& $PythonPath -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
if ($LASTEXITCODE -ne 0) { throw "Could not install the CUDA-enabled PyTorch wheels." }

& $PythonPath -m pip install -e $ProjectRoot
if ($LASTEXITCODE -ne 0) { throw "Could not install project dependencies." }

& $PythonPath (Join-Path $ProjectRoot "scripts\check_environment.py")
if ($LASTEXITCODE -ne 0) { throw "Environment verification failed." }

Write-Host "Environment ready. Run scripts\download_weights.py and scripts\validate_data.py next."
