# Quick setup script for Windows PowerShell

$VENV_DIR = ".venv"
$PROJECT_FILE = "pyproject.toml"
$DOCKER_COMPOSE_FILE = "docker-compose.yml"
$CHROMA_SERVICE_NAME = "chroma"

Write-Host "==> Checking Python..."
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Error "Python not found. Please install Python 3.9+ and re-run."
    exit 1
}

Write-Host "==> Creating virtual environment..."
if (-not (Test-Path $VENV_DIR)) {
    python -m venv $VENV_DIR
}

Write-Host "==> Activating virtual environment..."
& "$VENV_DIR\Scripts\Activate.ps1"

Write-Host "==> Upgrading pip..."
python -m pip install --upgrade pip wheel

if (Test-Path $PROJECT_FILE) {
    Write-Host "==> Installing CLASSMATE-RAG and its dependencies..."
    # Editable install also creates the `rag` console script declared in
    # pyproject.toml, so there is no .cmd shim to keep in sync.
    pip install -e .
} else {
    Write-Warning "$PROJECT_FILE not found. Skipping install."
}

if ((Test-Path ".env.example") -and -not (Test-Path ".env")) {
    Write-Host "==> Copying .env.example → .env"
    Copy-Item ".env.example" ".env"
}

# --- Start vector DB via Docker ---
if (Get-Command docker -ErrorAction SilentlyContinue) {
    if (Test-Path $DOCKER_COMPOSE_FILE) {
        Write-Host "==> Starting vector DB via docker compose..."
        docker compose up -d $CHROMA_SERVICE_NAME
        if ($LASTEXITCODE -ne 0) {
            docker compose up -d
        }
    } else {
        Write-Host "NOTE: $DOCKER_COMPOSE_FILE not found. Skipping Docker startup."
    }
} else {
    Write-Host "NOTE: Docker not installed. Skipping Docker startup."
}

Write-Host "Setup complete."
Write-Host "To activate the venv in future sessions: .\$VENV_DIR\Scripts\Activate.ps1"
Write-Host "Then run: rag --help"
