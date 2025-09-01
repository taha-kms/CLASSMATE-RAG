
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
git clone https://github.com/your-org/your-repo.git
cd your-repo
```

---

## 3. Quick Setup

We provide helper scripts for Linux/macOS (`quicksetup.sh`) and Windows (`quicksetup.ps1`).
They will:

* Create a `.venv` virtual environment
* Upgrade `pip` and install dependencies from `requirements.txt`
* Copy `.env.example` to `.env` if missing
* Start the Docker-based vector DB
* Create a shortcut command `rag` (instead of `python -m rag.cli`)

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


