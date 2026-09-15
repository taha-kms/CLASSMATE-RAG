
# Installation Guide

This document explains how to set up the project on **Linux/macOS** and **Windows**.

---

## 1. Prerequisites

Before installing, make sure you have:

- **Python**: version 3.9 or higher  
- **pip**: comes with Python, but you can upgrade it later  
- **Docker**: required for running the vector database (Chroma)  
- **Git**: to clone the repository  

Optional (but recommended):

- **GPU drivers + CUDA/cuBLAS** for faster inference  
- **Make** (Linux/macOS) for convenience  

---

## 2. Clone the Repository

```bash
git clone https://github.com/taha-kms/CLASSMATE-RAG.git
cd CLASSMATE-RAG
```

---

## 3. Quick Setup

We provide helper scripts for Linux/macOS (`quicksetup.sh`) and Windows (`quicksetup.ps1`).
They will:

* Create a `.venv` virtual environment
* Upgrade `pip` and install the project with `pip install -e .`, which pulls the
  dependencies and creates the `rag` command
* Copy `.env.example` to `.env` if missing
* Start the Docker-based vector DB

If you would rather not use the helper script:

```bash
pip install -r requirements-cpu.txt   # or requirements-gpu.txt
pip install -e .
pip install -r requirements-test.txt  # only to run the tests
```

## Which PyTorch build

`sentence-transformers` needs PyTorch, and the default wheel on PyPI is the
CUDA one. On a machine that cannot use it, that is a large download that
never gets loaded:

| | installed size |
| --- | --- |
| CPU build | ~1.2 GB |
| CUDA build plus nvidia libraries and triton | ~5.2 GB |

The helper scripts pick automatically: the CUDA build when `nvidia-smi` is
present, the CPU build otherwise. Override it if the guess is wrong:

```bash
CLASSMATE_TORCH=cpu ./quicksetup.sh
```

```powershell
$env:CLASSMATE_TORCH = "cpu"; .\quicksetup.ps1
```

Having an NVIDIA card is not on its own a reason to take the CUDA build.
What matters is whether it has enough VRAM for the model you plan to run: a
7B Q4 GGUF needs roughly 4.4 GB to offload fully, so a 4 GB card cannot
hold one. Order matters too, the torch build has to be installed before
`pip install -e .`, or pip resolves sentence-transformers first and pulls
the CUDA wheel anyway.

### Linux / macOS

```bash
./quicksetup.sh
```

Activate the environment:

```bash
source .venv/bin/activate
```

### Windows (PowerShell)

```bash
.\quicksetup.ps1
```

Activate the environment:

```bash
.\.venv\Scripts\Activate.ps1
```

---

## 4. Verify Installation

Run:

```bash
rag --help
```

You should see the CLI help menu.

If you see errors related to Docker or the vector DB, check that Docker is installed and running.

---

## 5. Manual Setup (if not using quicksetup)

If you prefer manual steps:

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
.\.venv\Scripts\Activate.ps1   # Windows PowerShell

# Upgrade pip
pip install --upgrade pip wheel

# Install dependencies
pip install -r requirements.txt

# Copy environment variables
cp .env.example .env   # or manually create a .env file

# Start vector DB
docker compose up -d
```

---

## 6. To Start

* Edit `.env` to configure model paths and settings.
* Ingest documents into the system (see [usage.md](usage.md)).
* Run queries with the `rag` CLI.



## Running in a container

```bash
docker build -t classmate-rag .
docker run --rm classmate-rag --help
```

The image carries the application and its dependencies, nothing else.
Models and indexes stay on the host and are mounted in:

```bash
docker run --rm \
  -v "$PWD/models:/app/models" \
  -v "$PWD/indexes:/app/indexes" \
  -v "$PWD/data:/app/data" \
  classmate-rag stats
```

It runs as a non-root user, uid 1000 by default. If your host user has a
different uid, rebuild so bind-mounted files stay writable:

```bash
docker build --build-arg UID=$(id -u) --build-arg GID=$(id -g) -t classmate-rag .
```

The image ships the CPU build of PyTorch. A GPU image is a separate thing
and needs the host to have nvidia-container-toolkit installed.

OCR is not included. `ENABLE_OCR=true` needs poppler-utils and tesseract-ocr,
which are deliberately left out to keep the image small.

Running the whole stack, application and Chroma together, comes with the
compose setup.
