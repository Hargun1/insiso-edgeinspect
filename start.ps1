$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$previewPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $previewPython)) { throw 'Run setup.ps1 first.' }
if (-not (Test-Path -LiteralPath 'web/dist/index.html')) {
    Push-Location web
    try {
        npm ci
        if ($LASTEXITCODE -ne 0) { throw 'npm ci failed.' }
        npm run build
        if ($LASTEXITCODE -ne 0) { throw 'Frontend build failed.' }
    } finally { Pop-Location }
}
Write-Host 'Open http://127.0.0.1:8000. Keep this terminal open; Ctrl+C stops the demo.'
& $previewPython -m uvicorn src.api:app --host 127.0.0.1 --port 8000

