# CLASSMATE-RAG

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)](https://pytorch.org/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-0.4%2B-brightgreen)](https://www.trychroma.com/)
[![llama.cpp](https://img.shields.io/badge/llama.cpp--python-0.2%2B-orange)](https://github.com/abetlen/llama-cpp-python)
[![tests](https://github.com/taha-kms/CLASSMATE-RAG/actions/workflows/tests.yml/badge.svg)](https://github.com/taha-kms/CLASSMATE-RAG/actions/workflows/tests.yml)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)

A **Retrieval-Augmented Generation (RAG)** system for course materials.
It ingests documents (PDF, DOCX, PPTX, EPUB, HTML, CSV, TXT, MD), indexes them in **BM25** + **Chroma vector DB**, and answers questions with grounded citations using local GGUF models (`llama.cpp` by default) or optional hosted providers.

---

## ✨ Features

* **CLI-first workflow** (`rag` command)
* Ingestion with metadata (course, unit, tags, language, semester, author)
* **Hybrid retrieval** (BM25 keyword + vector embeddings, fused with RRF)
* **Cited answers** generated with local LLMs by default
* **Privacy-preserving local default**: zero data leaves your machine unless a hosted provider is configured
* **Admin tools**: stats, preview, backup/restore, vacuum, rebuild embeddings, reingest
* **Document loaders**: PDF, DOCX, PPTX, EPUB, HTML, CSV, TXT, Markdown
* **Multilingual support** with E5 embeddings (`intfloat/multilingual-e5-base`)

---

## 🔒 Data Privacy & Hosted Providers

By default, CLASSMATE-RAG runs **completely locally and offline**:
* Document processing, vector embeddings, and generation execute on your machine.
* The default backend is `llama_cpp`, ensuring no coursework or user questions leave your device.

When configuring or selecting a hosted backend (e.g. Anthropic, OpenAI, or Google):
* **What leaves the machine:** Only the user's question and the retrieved text chunks/passages included in the prompt context are transmitted to the provider. The rest of your index and unretrieved corpus **never leave the machine**.
* **Per-query cost:** Hosted providers charge based on token volume. In a RAG pipeline, retrieved document chunks in the context window make up the bulk of token usage and cost.
* **Privacy by default:** `llama_cpp` remains the default backend so that the private option is what you get out of the box without manual configuration.

---

## 📦 Installation

See [docs/installation.md](docs/installation.md) for details.
Quick setup (Linux/macOS):

```bash
./quicksetup.sh
source .venv/bin/activate
rag --help
```

Windows (PowerShell):

```powershell
.\quicksetup.ps1
.\.venv\Scripts\Activate.ps1
rag --help
```

---

## 🚀 Usage

Ingest a document:

```bash
rag add path/to/file.pdf --course "Math101" --unit "1" --language "en" --tags exam,week1
```

Ask a question:

```bash
rag ask "What is the chain rule?" --course "Math101"
```

Preview retrieval (no generation):

```bash
rag preview "Explain entropy"
```

See [docs/usage.md](docs/usage.md) for more.

---

## 🛠️ Maintenance

* Show stats: `rag stats`
* Backup: `rag dump --path dumps/corpus.jsonl`
* Restore: `rag restore --path dumps/corpus.jsonl`
* Vacuum: `rag vacuum`
* Rebuild embeddings:
  `rag rebuild --model intfloat/multilingual-e5-large`
* Manage entries: `rag list`, `rag show`, `rag delete`, `rag reingest`

Details in [docs/configuration.md](docs/configuration.md).

---

## 📖 Documentation

* [Installation](docs/installation.md)
* [Usage](docs/usage.md)
* [Configuration](docs/configuration.md)
* [Architecture](docs/architecture.md)

---

## 📝 License

Copyright (C) 2026 Taha Kamalisadeghian &lt;tahakamali14@gmail.com&gt;

CLASSMATE-RAG is free software: you can redistribute it and/or modify it under the terms of the **GNU General Public License v3.0** as published by the Free Software Foundation. This program is distributed in the hope that it will be useful, but **WITHOUT ANY WARRANTY**; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the [LICENSE](LICENSE) file for the full text, or visit &lt;https://www.gnu.org/licenses/gpl-3.0.html&gt;.

---

## 🧩 Project Structure

```
cli/             # CLI entrypoint (argparse)
rag/             # Core RAG system
  admin/         # Backup, restore, manage, inspect
  chunking/      # Sentence-aware text splitting
  embeddings/    # E5 embedder + on-disk cache
  generation/    # llama.cpp runner, prompting, citation post-processing
  loaders/       # File loaders (PDF, DOCX, PPTX, EPUB, HTML, CSV, TXT, MD)
  metadata/      # DocumentMetadata schema + Pydantic CLI validation
  pipeline/      # Ingestion + ask orchestration
  retrieval/     # BM25, Chroma vector store, RRF hybrid fusion, neighbor expansion
  routing/       # Hybrid subject router (math/code/translation/default) + sticky model loader
  utils/         # Language detection, near-duplicate filtering, stable IDs
  config.py      # Env/.env-driven configuration
  model_fetch.py # On-demand GGUF download from HF
docs/            # Documentation
tests/           # pytest suite
tools/           # Benchmark / helper scripts
```

---
