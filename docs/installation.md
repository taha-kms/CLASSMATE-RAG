
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

## Running the whole stack

`docker compose` brings up the database and gives you the CLI against it:

```bash
docker compose up -d --wait chroma      # waits for it to be healthy, not just started
docker compose run --rm rag stats
docker compose run --rm rag add data/notes.pdf --course Maths --unit 3
docker compose run --rm rag ask "What is the chain rule?" --course Maths
```

The `rag` service sits behind a `cli` profile, so `docker compose up` starts
only the database. The CLI is one-shot and is meant to be run, not left
running.

### Matching your user

The image runs as uid 1000. If yours differs, anything the container writes
into `./indexes` or `./models` comes back owned by someone else, and you
cannot delete your own index without `sudo`.

Compose runs the container as your user:

```bash
CLASSMATE_UID=$(id -u) CLASSMATE_GID=$(id -g) docker compose run --rm rag stats
```

Put those two in `.env` and they apply to every command. This is a runtime
override, so a uid other than 1000 needs no rebuild. The build arguments of
the same name only set the default baked into the image, which matters if
you run `docker run` directly rather than through compose:

```bash
CLASSMATE_UID=$(id -u) CLASSMATE_GID=$(id -g) docker compose build
```

Note the names. `UID` is readonly in bash, so the obvious
`UID=$(id -u) ...` fails before docker is even reached.

### What persists

Three host directories are mounted, and between them they hold everything
expensive:

| Path | Holds | Size |
| --- | --- | --- |
| `./models` | GGUF model files you supply | several GB |
| `./models/hf_cache` | the downloaded embedding model | ~1.1 GB |
| `./indexes` | Chroma, BM25 and the embedding cache | grows with the corpus |

`HF_HOME` points at `/app/models/hf_cache`, which is inside the `./models`
mount, so the embedding model is downloaded once rather than on every fresh
container. Losing that mount means re-downloading about a gigabyte before
the next question can be answered.

### Upgrading an existing index

The database image moved from Chroma 0.6 to 1.5, which changed where the
server keeps its files: 1.x reads `/config.yaml` and stores everything under
`/data`, ignoring the `IS_PERSISTENT` and `PERSIST_DIRECTORY` variables 0.6
used. The compose file mounts `./indexes/chroma` at the new path, so there is
nothing to do by hand.

An index written by 0.6 opens in place. No dump and restore is needed, and
`rag stats` should report the same `vector_count` afterwards as before:

```bash
rag stats                                  # note vector_count
docker compose pull chroma
docker compose up -d --wait chroma
rag stats                                  # same vector_count, "consistent": true
```

Rebuild the `rag` image as well, or pull a newer one. The client library is
installed into the image, and `docker compose run` reuses whatever image is
already there rather than noticing that `requirements.txt` moved:

```bash
CLASSMATE_UID=$(id -u) CLASSMATE_GID=$(id -g) docker compose build rag
# or, if you pull rather than build:
docker compose pull rag
```

Skipping this leaves a 0.6 client talking to a 1.5 server, and that pair fails
the same quiet way as a wrong host:

```
vector_count: -1
bm25 count: 2
```

Take a copy first regardless. The server rewrites the SQLite file and the
segment directory on first open, so the upgrade is not something you can
undo by pulling the old image back:

```bash
cp -a indexes/chroma indexes/chroma.bak
```

If the counts do not match, `rag reconcile` reports what is stranded, and the
BM25 catalogue is a complete copy of the corpus:

```bash
rag dump --path corpus.jsonl    # reads BM25, not Chroma
rag restore --path corpus.jsonl # re-embeds and rewrites both stores
```

### Talking to the database

Inside compose the database is `http://chroma:8000`, not `localhost`.
`localhost` in the application container is the application container. The
compose file sets this for you; it only matters if you override it, and the
failure is quiet:

```
vector_count with wrong host: -1
bm25 count: 2
```

`-1` means "could not reach the vector store". A corpus that genuinely holds
nothing reports `0`.

## Pulling a prebuilt image

Building locally means compiling llama-cpp-python, which is the part that
takes minutes and needs a toolchain. Pulling avoids it:

```bash
docker pull ghcr.io/taha-kms/classmate-rag:main
```

| Tag | What it is |
| --- | --- |
| `:main` | the latest commit on main, rebuilt on every push |
| `:sha-<commit>` | one specific commit, if you need to pin |
| `:0.2.0`, `:latest` | published from a version tag, see below |

`:main` moves. Pin to a version tag or a `sha-` tag if you want something
that does not change under you.

Every published image has already passed the smoke test and the
vulnerability scan in CI, and is the same artefact those checks ran
against rather than a rebuild of it.

## Releases

Tagging a version builds the image, pushes it to GHCR and creates a GitHub
release:

```bash
# bump the version in pyproject.toml first, then
git tag v0.2.0
git push origin v0.2.0
```

The tag is the source of truth. The workflow refuses to publish if
`pyproject.toml` disagrees with it, so the two cannot drift apart
silently.

Published images:

```bash
docker pull ghcr.io/taha-kms/classmate-rag:0.2.0
docker pull ghcr.io/taha-kms/classmate-rag:latest
```

A tag containing a hyphen, `v0.2.0-rc1`, is published as a pre-release.

## Supplying a model

No model ships with the image. It would add several gigabytes to something
everyone pulls, and which model suits you depends on your machine and what
you are asking it.

Everything except answering works without one:

```bash
docker compose run --rm rag add data/notes.pdf --course Maths
docker compose run --rm rag preview "the chain rule" --course Maths
docker compose run --rm rag stats
```

So you can ingest a corpus and confirm retrieval is sensible before
committing to a download.

The quickest route is to let a profile choose:

```bash
rag profiles                      # what fits this machine
rag model download --profile light
rag model list                    # what is already here
```

Downloads report their size before starting, show progress, resume if
interrupted, and refuse up front when there is not enough disk rather than
failing at 95%.

To pick a specific model instead, put a `.gguf` in `./models` and point at
it:

```bash
# in .env
LLM_MODEL_PATH=./models/your-model.gguf
```

Or let it fetch one on first use:

```bash
# in .env
LLM_REPO_ID=TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF
LLM_FILENAME=tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
```

TinyLlama is about 640 MB and answers poorly; it is a good way to confirm
the pipeline works end to end before downloading something serious. A 7B
Q4_K_M is around 4.4 GB and is the usual choice.

`./models` is a host mount, so a model survives the container being
recreated and is downloaded once.

## Building a CUDA image yourself

The published image is CPU-only, deliberately. `llama-cpp-python` has to be
**compiled** with CUDA support; it is not a runtime switch, so a GPU image
is a separate build of roughly 5 GB that is useless without
nvidia-container-toolkit installed on the host. One CPU image that runs
everywhere is the better default, and a hosted backend gives you speed
without a GPU at all.

If you do want to build one, read the next section first. It is often not
worth the afternoon.

### Is it worth it on your card

Fully offloading a 7B Q4_K_M needs about 4.4 GB of VRAM, plus room for the
context window:

| VRAM | Realistic outcome |
| --- | --- |
| under 4 GB | do not bother. Stay on CPU or use a hosted backend |
| 4-6 GB | partial offload only. Faster than CPU, not dramatically |
| 8 GB+ | one 7B Q4 fully offloaded, one at a time |
| 16 GB+ | comfortable, and the only range where subject routing makes sense |

Check what you have:

```bash
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
```

### Host prerequisite

The host needs nvidia-container-toolkit, and this is the step most people
get stuck on. Docker alone cannot pass a GPU through. Verify it works
before building anything:

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

If that prints your card, you are ready. If it errors, fix that first,
because nothing below will work until it does.

### The build

Two changes to the Dockerfile. Use a CUDA devel base for the builder stage
so `nvcc` is present, and set `CMAKE_ARGS` before installing
llama-cpp-python so it compiles the CUDA backend:

```dockerfile
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS builder
# ... python, build-essential, cmake ...
ENV CMAKE_ARGS="-DGGML_CUDA=on"
RUN pip install --no-cache-dir llama-cpp-python
```

The runtime stage needs a CUDA runtime base rather than `python:slim`, so
the CUDA shared libraries are present.

Install `requirements-gpu.txt` rather than `requirements-cpu.txt`, or torch
arrives as the CPU build and the GPU sits idle for the embedding step.

### Running it

Give the container the GPU and tell llama.cpp how much to offload:

```yaml
services:
  rag:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    environment:
      LLAMA_GPU_LAYERS: "20"
      ROUTE_N_GPU_LAYERS: "20"
```

`-1` means every layer. It is the setting most likely to fail, and it fails
late, partway through loading, on exactly the machines that cannot afford
the wait. Start with a number well below the layer count and raise it until
the model stops fitting.
