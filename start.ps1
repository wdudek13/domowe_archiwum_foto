Set-Location -Path $PSScriptRoot

if (-not (Test-Path ".\venv\Scripts\Activate.ps1")) {
    Write-Error "Wirtualne srodowisko .\venv nie istnieje. Utworz je: python -m venv venv"
    Exit 1
}

& ".\venv\Scripts\Activate.ps1"
Write-Host "Uruchamianie Domowe Archiwum Foto..." -ForegroundColor Cyan
uvicorn main:app --reload --port 8000
