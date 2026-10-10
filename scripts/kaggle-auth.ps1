param(
    [Parameter(Mandatory = $true)]
    [string]$CredentialFile
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$KaggleConfigDirectory = Join-Path $ProjectRoot ".kaggle"
$Destination = Join-Path $KaggleConfigDirectory "kaggle.json"
$ResolvedCredential = (Resolve-Path -LiteralPath $CredentialFile).Path

$credential = Get-Content -Raw -LiteralPath $ResolvedCredential | ConvertFrom-Json
if (-not $credential.username -or -not $credential.key) {
    throw "The selected file is not a valid Kaggle kaggle.json credential."
}

New-Item -ItemType Directory -Force -Path $KaggleConfigDirectory | Out-Null
Copy-Item -LiteralPath $ResolvedCredential -Destination $Destination -Force

Write-Output "Kaggle credentials installed in the project's private .kaggle directory."
Write-Output "The credential value was not printed."
Write-Output "Run: .\scripts\kaggle.ps1 datasets list --mine"
