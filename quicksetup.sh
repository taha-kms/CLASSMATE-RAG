#!/usr/bin/env bash
set -euo pipefail

VENV_DIR=".venv"
PROJECT_FILE="pyproject.toml"
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
python3 -m pip install --upgrade pip wheel

# --- PyTorch build ------------------------------------------------------
# sentence-transformers needs torch, and the default PyPI wheel is the CUDA
# build: about 4 GB of nvidia libraries a machine without a usable GPU will
# never load. Install the chosen build first so pip keeps it when it later
# resolves sentence-transformers.
#
# Override with CLASSMATE_TORCH=cpu or CLASSMATE_TORCH=gpu.
TORCH_CHOICE="${CLASSMATE_TORCH:-auto}"
if [ "${TORCH_CHOICE}" = "auto" ]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    TORCH_CHOICE="gpu"
  else
    TORCH_CHOICE="cpu"
  fi
fi

TORCH_REQS="requirements-${TORCH_CHOICE}.txt"
if [ -f "${TORCH_REQS}" ]; then
  echo "==> Installing PyTorch (${TORCH_CHOICE} build)..."
  pip install -r "${TORCH_REQS}"
else
  echo "WARNING: ${TORCH_REQS} not found. Letting pip pick a torch build."
fi

if [ -f "${PROJECT_FILE}" ]; then
  echo "==> Installing CLASSMATE-RAG and its dependencies..."
  # Editable install also creates the `rag` console script declared in
  # pyproject.toml, so there is no shell shim to keep in sync.
  pip install -e .
else
  echo "WARNING: ${PROJECT_FILE} not found. Skipping install."
fi

if [ -f ".env.example" ] && [ ! -f ".env" ]; then
  echo "==> Copying .env.example → .env"
  cp .env.example .env
fi

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
