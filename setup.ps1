$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
Write-Host 'Checking internet access...'
try { Resolve-DnsName pypi.org -DnsOnly -QuickTimeout -ErrorAction Stop | Out-Null }
catch { throw 'Cannot resolve pypi.org in this terminal. Check your internet/VPN/firewall or use a network that permits PyPI. No plugin is required.' }
Write-Host 'Creating an isolated Python environment...'
if (-not (Test-Path '.venv/Scripts/python.exe')) {
    python -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'venv creation failed. Try Python 3.12 with: py -3.12 -m venv .venv' }
}
& '.venv/Scripts/python.exe' -m ensurepip --upgrade
if ($LASTEXITCODE -ne 0) {
    Write-Host 'ensurepip failed; trying the installed pip against the isolated environment...'
    python -m pip --python .venv install --upgrade pip
    if ($LASTEXITCODE -ne 0) { throw 'pip bootstrap failed. Install Python 3.12, recreate the incomplete environment, and retry.' }
}
& '.venv/Scripts/python.exe' -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed. Read the error above. Python 3.12 is recommended if wheels are unavailable.' }
Push-Location web
try {
    npm install
    if ($LASTEXITCODE -ne 0) { throw 'npm install failed. Check access to registry.npmjs.org.' }
    npm run build
    if ($LASTEXITCODE -ne 0) { throw 'Frontend build failed.' }
} finally { Pop-Location }
& '.venv/Scripts/python.exe' -m unittest discover -s tests -v
if ($LASTEXITCODE -ne 0) { throw 'Unit tests failed.' }
Write-Host 'Setup complete. Next: download metal_nut from MVTec and follow README.md.'

