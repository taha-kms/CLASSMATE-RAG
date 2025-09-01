#!/usr/bin/env bash
set -euo pipefail

# --- Config ---
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

echo "==> Creating virtual environment at ${VENV_DIR} (if missing)..."
if [ ! -d "${VENV_DIR}" ]; then
  ${PY} -m venv "${VENV_DIR}"
fi

# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"

echo "==> Upgrading pip and wheel..."
python -m pip install --upgrade pip wheel

if [ -f "${REQ_FILE}" ]; then
  echo "==> Installing Python dependencies from ${REQ_FILE}..."
  pip install -r "${REQ_FILE}"
else
  echo "WARNING: ${REQ_FILE} not found. Skipping Python dependency install."
fi

# Optional: bootstrap .env
if [ -f ".env.example" ] && [ ! -f ".env" ]; then
  echo "==> Creating .env from .env.example (edit as needed)..."
  cp .env.example .env || true
fi

# --- Start vector DB via Docker Compose ---
start_chroma() {
  local compose_cmd="$1"
  if [ -f "${DOCKER_COMPOSE_FILE}" ]; then
    echo "==> Starting vector DB with ${compose_cmd}..."
    if ${compose_cmd} up -d "${CHROMA_SERVICE_NAME}"; then
      echo "==> '${CHROMA_SERVICE_NAME}' service started."
    else
      echo "==> Service '${CHROMA_SERVICE_NAME}' not found or failed. Bringing up all services..."
      ${compose_cmd} up -d
    fi
  else
    echo "NOTE: ${DOCKER_COMPOSE_FILE} not found. Skipping Docker startup."
  fi
}

echo "==> Checking Docker..."
if command -v docker >/dev/null 2>&1; then
  if docker info >/dev/null 2>&1; then
    if docker compose version >/dev/null 2>&1; then
      start_chroma "docker compose"
    elif command -v docker-compose >/dev/null 2>&1; then
      start_chroma "docker-compose"
    else
      echo "NOTE: 'docker compose'/'docker-compose' not found. Skipping Docker startup."
    fi
  else
    echo "NOTE: Docker daemon not running or permission denied. Skipping Docker startup."
  fi
else
  echo "NOTE: Docker not installed. Skipping Docker startup."
fi

echo ""
echo "Setup complete."
echo "- To activate the venv:  source ${VENV_DIR}/bin/activate"
echo "- To bring services down: docker compose down  (or docker-compose down)"
