# CLASSMATE-RAG

A local, privacy-friendly **RAG** (Retrieval-Augmented Generation) CLI to help students study across **English** and **Italian** materials.

- **Embeddings:** [`intfloat/multilingual-e5-base`]
- **Vector DB:** Chroma (HTTP thin client) + **BM25** hybrid retrieval
- **Generator:** Local **Llama 3.1** via `llama.cpp` (`llama-cpp-python`)
- **OS:** Windows/macOS/Linux (Windows supported & tested)

---

## Quickstart

### 1) Prerequisites

- Python 3.10–3.12
- [Docker Desktop] running (for the Chroma server)
- (Windows) WSL2 enabled is recommended

### 2) Install

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Unix:    source .venv/bin/activate
pip install -r requirements.txt
````

### 3) Configure `.env`

Create `.env` at repo root (example):

```dotenv
# --- Vector DB (Chroma thin HTTP client) ---
CHROMA_HTTP_URL=http://localhost:8000

# --- Embeddings (Hugging Face) ---
HF_TOKEN= # your HF access token (optional but recommended for faster downloads)

# --- Llama.cpp model selection ---
# Either set a local path to a .gguf file:
LLAMA_GGUF=./models/Llama-3.1-8B-Instruct-Q4_K_M.gguf

# Or set remote repo+filename for auto-download into ./models/ (requires HF_TOKEN):
LLAMA_REPO_ID=TheBloke/Llama-3.1-8B-Instruct-GGUF
LLAMA_FILENAME=llama-3.1-8b-instruct.Q4_K_M.gguf

# Optional OCR toggle for scanned PDFs
ENABLE_OCR=false
```

> You can point `LLAMA_GGUF` to any **.gguf** file; if missing and `LLAMA_REPO_ID`+`LLAMA_FILENAME` are set, the app will attempt to download it to `./models/` using your `HF_TOKEN`.

### 4) Start Chroma (Docker)

```bash
docker compose up -d
# verifies a container on localhost:8000
```

### 5) Ingest some files

```bash
python -m cli.main add .\data\java.txt --course oop --unit "Week 01" --tags exam,week3 --language en
# Paths with spaces must be quoted:
python -m cli.main add ".\data\Getting Started with Java - Dev.pdf" --course oop --unit "Week 02" --language auto --tags exam,week2
python -m cli.main add .\data\oracle.docx --language auto
python -m cli.main add .\data\slides.pptx --course ds101 --unit "Week 03" --language auto --tags lecture
```

### 6) Ask questions

```bash
python -m cli.main ask "What is polymorphism in Java?" --course oop --language en --k 6
```

* If the answer is grounded in retrieved context, you’ll see citations like `[1][2]`.
* If the corpus lacks the answer, the app returns a brief **general-knowledge fallback** (clearly labeled).

### 7) Observe what retrieval is doing

```bash
# Preview retrieval only (no generation), with scores & provenance
python -m cli.main preview "What is polymorphism in Java?" --course oop --k 6

# Index stats (vector count + disk sizes)
python -m cli.main stats
```

---

## What’s implemented (so far)

### Ingestion & loaders

* **Loaders:** `txt`, `md`, `pdf` (text), `docx`, `pptx`
* **Chunking:** Sentence-aware chunking with overlap (robust EN/IT heuristics)
* **Metadata:** `course`, `unit`, `language (en|it|auto)`, `doc_type`, `author`, `semester`, `tags`

  * **Tags** are normalized to boolean flags in the index: e.g. `exam,week3` → `tag_exam=True`, `tag_week3=True`
  * `doc_type="other"` is treated as a **placeholder** and **ignored** in filters

### Retrieval

* **Hybrid:** Vector (Chroma) + BM25 (rank-bm25)
* **Diversification:** **MMR** on vector candidates before **RRF** fusion
* **Filters:** Equality filters on metadata + tag flags (ANDed)
* **Context budget:** Compact, numbered blocks with a **total budget ≈ 3.5k chars** to avoid truncation

### Generation

* **Local LLM:** `llama-cpp-python` runs your `.gguf`
* **Bilingual prompts:** English/Italian, auto-detect or forced via CLI
* **Grounded-by-default:** Uses retrieved blocks only; cites as `[n]`
* **Fallback:** If grounded answer is “I don’t know,” a short **general** (no citations) answer is produced and **labeled**

### Chroma & Windows stability

* We use **chromadb-client (thin HTTP client)** and a **Dockerized Chroma server** to avoid `onnxruntime` issues
* Host normalization & settings enforced: `chroma_api_impl="chromadb.api.fastapi.FastAPI"`

---

## CLI usage

```
classmate add <path> [--course C] [--unit U] [--language en|it|auto] [--doc-type] [--author] [--semester] [--tags t1,t2]
classmate ask "<question>" [--course] [--unit] [--language] [--doc-type] [--author] [--semester] [--tags] [--k 8] [--hybrid on|off]
classmate preview "<question>" [...]  # retrieval only
classmate stats
```

**Examples**

```bash
# Ingest markdown
python -m cli.main add .\notes\oop.md --course oop --unit "Week 01" --language auto --tags reading

# Ask in Italian, hybrid off (vector-only), top-5
python -m cli.main ask "Spiega l'incapsulamento in Java." --course oop --language it --k 5 --hybrid off

# Preview with tag filter
python -m cli.main preview "Collections framework" --course oop --tags exam,week2 --k 6
```

---

## Configuration

Most knobs live in `.env`:

| Key                                | Meaning                                                                         |
| ---------------------------------- | ------------------------------------------------------------------------------- |
| `CHROMA_HTTP_URL`                  | e.g. `http://localhost:8000` (Docker service)                                   |
| `HF_TOKEN`                         | Hugging Face token (recommended for model downloads)                            |
| `LLAMA_GGUF`                       | Path to local `.gguf`                                                           |
| `LLAMA_REPO_ID` / `LLAMA_FILENAME` | If set, auto-download GGUF into `./models/`                                     |
| `ENABLE_OCR`                       | `true/false` — try OCR for scanned PDFs (requires Tesseract/Poppler on Windows) |

**Indexes**

* **Chroma (vectors):** `./indexes/chroma` (mounted into the container)
* **BM25 (lexical):** `./indexes/bm25`

---

## Directory layout

```
CLASSMATE-RAG/
├─ cli/
│  └─ main.py                     # CLI entrypoint (add / ask / preview / stats)
├─ rag/
│  ├─ config/                     # config loader (.env, paths, defaults)
│  ├─ loaders/                    # txt, md, pdf, docx, pptx
│  ├─ chunking/                   # sentence chunker
│  ├─ embeddings/                 # e5 embedder wrapper
│  ├─ retrieval/
│  │  ├─ vector_chroma.py         # thin-client Chroma wrapper (HTTP / local)
│  │  ├─ bm25.py                  # rank-bm25 store
│  │  └─ fusion.py                # MMR + RRF hybrid retriever
│  ├─ generation/
│  │  ├─ llama_cpp_runner.py      # llama.cpp chat wrapper
│  │  └─ prompting.py             # grounded + fallback prompts
│  ├─ admin/
│  │  └─ inspect.py               # preview & stats helpers
│  ├─ pipeline/
│  │  └─ rag.py                   # ingest_file(), ask_question()
│  └─ utils/                      # ids, lang detect, helpers
├─ indexes/
│  ├─ chroma/                     # persisted Chroma DB (Docker bind-mount)
│  └─ bm25/                       # BM25 store files
├─ models/                        # your .gguf lives here (if local)
├─ data/                          # sample docs
├─ docker-compose.yml             # launches Chroma server on port 8000
└─ requirements.txt
```

---

## Troubleshooting

* **Docker pipe error on Windows**: start **Docker Desktop**; it must say **“Engine running”**. Then `docker compose up -d`.
* **Chroma “host mismatch”**: we normalize to `localhost`. Set `CHROMA_HTTP_URL=http://localhost:8000`.
* **ONNX / onnxruntime errors**: we do **not** use onnxruntime; the thin client avoids it.
* **Paths with spaces**: quote your file paths (CMD and PowerShell have slightly different quoting rules).
* **HF symlink warning on Windows**: harmless; enable Developer Mode to silence.

---

## Security & privacy

* Everything runs locally.
* Your Hugging Face token (`HF_TOKEN`) is used only for downloads if you enable it.
* Documents and embeddings stay on your machine.

---

## Roadmap (next)

* **Step 15**: Ingestion management (`list`, `show`, `delete`, `reingest`)
* **Step 15A**: Bootstrap scripts (`setup`, `run_chroma`, `ingest-sample`, `reset`, …)
* **Step 16–25**: Validation, eval suite, citation integrity, performance/caching, multilingual robustness, more loaders, backup/export, CI, DX polish.

