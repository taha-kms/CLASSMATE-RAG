# CLASSMATE-RAG

A **Retrieval-Augmented Generation (RAG)** system for course materials.
It ingests documents (PDF, DOCX, PPTX, EPUB, HTML, CSV, TXT, MD), indexes them in **BM25** + **Chroma vector DB**, and answers questions with grounded citations using LLaMA/Mistral GGUF models.

---

## ✨ Features

* **CLI-first workflow** (`classmate` command)
* Ingestion with metadata (course, unit, tags, language, semester, author)
* **Hybrid retrieval** (BM25 keyword + vector embeddings, fused with RRF)
* **Cited answers** generated with local LLMs
* **Admin tools**: stats, preview, backup/restore, vacuum, rebuild embeddings, reingest
* **Document loaders**: PDF, DOCX, PPTX, EPUB, HTML, CSV, TXT, Markdown
* **Multilingual support** with E5 embeddings (`intfloat/multilingual-e5-base`)

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

## 🛠️ Admin & Maintenance

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

## 🧩 Project Structure

```
cli/           # CLI entrypoint
rag/           # Core RAG system
  admin/       # Backup, restore, manage, inspect
  chunking/    # Text splitting into chunks
  embeddings/  # Embedding models & cache
  generation/  # LLM runner, prompting, postprocessing
  loaders/     # File loaders
  retrieval/   # BM25, Chroma, hybrid fusion
  pipeline/    # Ingestion, query orchestration
docs/          # Documentation
tests/         # Test suite
tools/         # Benchmark scripts
```

---
