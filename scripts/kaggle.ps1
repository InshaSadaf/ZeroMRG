param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$KaggleArgs
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$KaggleExecutable = Join-Path $ProjectRoot "venv\Scripts\kaggle.exe"
$KaggleConfigDirectory = Join-Path $ProjectRoot ".kaggle"

if (-not (Test-Path -LiteralPath $KaggleExecutable)) {
    throw "Project Kaggle executable was not found: $KaggleExecutable"
}

New-Item -ItemType Directory -Force -Path $KaggleConfigDirectory | Out-Null
$env:KAGGLE_CONFIG_DIR = $KaggleConfigDirectory

& $KaggleExecutable @KaggleArgs
exit $LASTEXITCODE
