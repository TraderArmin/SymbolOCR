$ErrorActionPreference = "Stop"

# --- Project root ---
$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $Root

# --- Paths ---
$Venv = Join-Path $Root ".venv"
$BuildDir = Join-Path $Root "build"
$DistDir  = Join-Path $Root "dist"
$PwBrowsersDir = Join-Path $Root "pw-browsers"   # will be bundled
$TessDir = Join-Path $Root "tesseract"

# --- Sanity checks ---
if (!(Test-Path $TessDir)) {
  throw "tesseract/ Ordner nicht gefunden. Lege tesseract\... ins Projekt."
}

# --- Create venv if missing ---
if (!(Test-Path $Venv)) {
  py -m venv $Venv
}

# --- Find venv python (Windows layout or POSIX layout) ---
$PyWin = Join-Path $Venv "Scripts\python.exe"
$PyPosix = Join-Path $Venv "bin\python.exe"

if (Test-Path $PyWin) {
  $Py = $PyWin
} elseif (Test-Path $PyPosix) {
  $Py = $PyPosix
} else {
  throw "Konnte venv-python nicht finden. Erwartet: $PyWin oder $PyPosix"
}

# --- Install deps ---
& $Py -m pip install -U pip
& $Py -m pip install -r requirements.txt
& $Py -m pip install pyinstaller
& $Py -m pip install playwright

# --- Ensure browsers dir exists ---
New-Item -ItemType Directory -Force -Path $PwBrowsersDir | Out-Null

# --- Install Playwright browser INTO project folder (portable) ---
$env:PLAYWRIGHT_BROWSERS_PATH = (Resolve-Path $PwBrowsersDir).Path
Write-Host "PLAYWRIGHT_BROWSERS_PATH=$env:PLAYWRIGHT_BROWSERS_PATH"

& $Py -m playwright install chromium

# --- Validate that browsers were installed ---
$BrowserCount = @(Get-ChildItem -Path $PwBrowsersDir -Recurse -Force -ErrorAction SilentlyContinue).Count
if ($BrowserCount -lt 10) {
  throw "Playwright Browser wurden nicht in '$PwBrowsersDir' installiert (zu wenig Dateien: $BrowserCount)."
}

# --- Clean old build artifacts ---
if (Test-Path $BuildDir) { Remove-Item $BuildDir -Recurse -Force }
if (Test-Path $DistDir)  { Remove-Item $DistDir  -Recurse -Force }

# --- Build exe ---
& $Py -m PyInstaller --noconfirm --onefile --windowed `
  --name "SymbolExtractorATLAS" `
  --add-data "$TessDir;tesseract" `
  --add-data "$PwBrowsersDir;pw-browsers" `
  main.py

# --- Create zip artifact (optional) ---
$ExePath = Join-Path $Root "dist\SymbolExtractorATLAS.exe"
if (!(Test-Path $ExePath)) { throw "EXE nicht gefunden: $ExePath" }

$ZipPath = Join-Path $Root "dist\SymbolExtractorATLAS.zip"
if (Test-Path $ZipPath) { Remove-Item $ZipPath -Force }
Compress-Archive -Path $ExePath -DestinationPath $ZipPath

Write-Host "Build fertig:"
Write-Host "  EXE: $ExePath"
Write-Host "  ZIP: $ZipPath"
