#!/usr/bin/env bash
set -euo pipefail

VENV_DIR=".venv"
REQ_FILE="requirements.txt"
DOCKER_COMPOSE_FILE="docker-compose.yml"
CHROMA_SERVICE_NAME="chroma"

echo "==> Checking Python..."
if ! command -v python3 >/dev/null 2>&1; then
  echo "ERROR: python3 not found. Please install Python 3.9+ and re-run." >&2
  exit 1
fi

PY=python3

echo "==> Creating virtual environment..."
if [ ! -d "${VENV_DIR}" ]; then
  ${PY} -m venv "${VENV_DIR}"
fi

source "${VENV_DIR}/bin/activate"

echo "==> Upgrading pip..."
python -m pip install --upgrade pip wheel

if [ -f "${REQ_FILE}" ]; then
  echo "==> Installing dependencies..."
  pip install -r "${REQ_FILE}"
else
  echo "WARNING: ${REQ_FILE} not found. Skipping dependencies."
fi

if [ -f ".env.example" ] && [ ! -f ".env" ]; then
  echo "==> Copying .env.example → .env"
  cp .env.example .env
fi

# --- Create rag shortcut command ---
RAG_BIN="${VENV_DIR}/bin/rag"
echo "==> Creating rag command shortcut..."
cat > "${RAG_BIN}" <<'EOF'
#!/usr/bin/env bash
exec python -m rag.cli "$@"
EOF
chmod +x "${RAG_BIN}"

# --- Start vector DB via Docker ---
if command -v docker >/dev/null 2>&1; then
  if [ -f "${DOCKER_COMPOSE_FILE}" ]; then
    echo "==> Starting vector DB via docker compose..."
    docker compose up -d "${CHROMA_SERVICE_NAME}" || docker compose up -d
  else
    echo "NOTE: ${DOCKER_COMPOSE_FILE} not found. Skipping Docker startup."
  fi
else
  echo "NOTE: Docker not installed. Skipping Docker startup."
fi

echo "Setup complete."
echo "To activate the venv: source ${VENV_DIR}/bin/activate"
echo "Then run: rag --help"
