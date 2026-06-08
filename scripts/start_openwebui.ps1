param(
    [string]$OpenAiApiBaseUrl = "http://127.0.0.1:11434/v1",
    [string]$OpenAiApiKey = "ollama",
    [string]$BindHost = "127.0.0.1",
    [int]$Port = 3000
)

$repoRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $repoRoot ".venv\Scripts\python.exe"
$launcherPath = Join-Path $repoRoot "scripts\run_openwebui.py"

if (-not (Test-Path $pythonPath)) {
    Write-Error "Missing virtualenv Python at $pythonPath"
    exit 1
}

if (-not (Test-Path $launcherPath)) {
    Write-Error "Missing launcher script at $launcherPath"
    exit 1
}

$env:OPENAI_API_BASE_URL = $OpenAiApiBaseUrl
$env:OPENAI_API_KEY = $OpenAiApiKey

Write-Host "Starting Open WebUI at http://$BindHost`:$Port"
Write-Host "Backend: $env:OPENAI_API_BASE_URL"

Set-Location $repoRoot
& $pythonPath $launcherPath serve --host $BindHost --port $Port