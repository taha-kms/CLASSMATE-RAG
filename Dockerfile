# syntax=docker/dockerfile:1.7

# Two stages. llama-cpp-python compiles from source, which needs gcc and
# cmake, and none of that belongs in the image people actually run.
ARG PYTHON_VERSION=3.12

# ---------------------------------------------------------------- builder ---
FROM python:${PYTHON_VERSION}-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential \
      cmake \
 && rm -rf /var/lib/apt/lists/*

# A venv rather than the system site-packages, so the runtime stage copies
# one self-contained directory.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

WORKDIR /build

# CPU PyTorch first. sentence-transformers depends on torch, and the default
# PyPI wheel is the CUDA build: roughly 4 GB of nvidia libraries this image
# has no use for. Pinning it first means pip leaves it alone later.
COPY requirements-cpu.txt ./
RUN pip install -r requirements-cpu.txt

# Dependencies before source, so editing code does not rebuild llama.cpp.
COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY pyproject.toml README.md ./
COPY rag ./rag
COPY cli ./cli
RUN pip install --no-deps .

# ---------------------------------------------------------------- runtime ---
FROM python:${PYTHON_VERSION}-slim AS runtime

# libgomp is llama.cpp's OpenMP runtime. Without it the compiled extension
# imports fine and fails at model load.
# Upgrade first: the base image lags its own security updates between
# rebuilds, and the scan in CI counts those against us. This cleared three
# CRITICAL advisories in perl-base on its own.
RUN apt-get update \
 && apt-get upgrade -y --no-install-recommends \
 && apt-get install -y --no-install-recommends \
      libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# Non-root, with a configurable uid so bind-mounted indexes do not come back
# owned by root on the host. See #24.
ARG UID=1000
ARG GID=1000
RUN groupadd --gid "${GID}" classmate \
 && useradd --uid "${UID}" --gid "${GID}" --create-home --shell /bin/bash classmate

COPY --from=builder /opt/venv /opt/venv

ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/models/hf_cache

WORKDIR /app

# Mount points. Models and indexes are volumes: the GGUFs alone are several
# GB and nothing here is worth baking into a layer.
RUN mkdir -p /app/models /app/indexes /app/data \
 && chown -R classmate:classmate /app

USER classmate

# The project root is located by walking up for pyproject.toml, so the file
# has to exist here for relative index paths to resolve (see #15).
COPY --chown=classmate:classmate pyproject.toml ./

ENTRYPOINT ["rag"]
CMD ["--help"]
