# Configuration Guide

This document explains how to configure the RAG system and use Maintenance commands
for maintaining the indexes.

---

## 1. Environment Variables

Configuration is usually stored in a `.env` file at the project root.  
A `.env.example` file is included — copy it as a starting point:

```bash
cp .env.example .env
````

### Embeddings and models

| Variable | Description | Default |
| --- | --- | --- |
| `EMBEDDING_MODEL_NAME` | Sentence-Transformers model used for embeddings | `intfloat/multilingual-e5-base` |
| `LLM_BACKEND` | Generation backend | `llama_cpp` |
| `LLM_MODEL_PATH` | Local `.gguf` file used when routing is off | `./models/Llama-3.1-8B-Instruct.Q4_K_M.gguf` |
| `LLM_REPO_ID` | Hugging Face repo to download the model from if it is missing | unset |
| `LLM_FILENAME` | File to fetch from that repo | unset |
| `LLAMA_GPU_LAYERS` | Layers to offload to the GPU on the non-routed path. `0` is CPU only | `0` |
| `HF_TOKEN` | Token for private or gated repos. `HUGGINGFACE_HUB_TOKEN` and `CLASSMATE_RAG_HF_TOKEN` are accepted too | unset |

### Storage

Relative paths resolve against the project root, not your working directory,
so the CLI finds the same corpus from anywhere.

| Variable | Description | Default |
| --- | --- | --- |
| `CHROMA_PERSIST_DIRECTORY` | Where the embedded Chroma database lives | `./indexes/chroma` |
| `CHROMA_COLLECTION_NAME` | Collection name inside Chroma | `classmate_rag` |
| `CHROMA_HTTP_URL` | Chroma server endpoint. Unset means embedded mode | unset |
| `BM25_DIRECTORY` | Where the BM25 lexical index lives | `./indexes/bm25` |
| `EMB_CACHE_DIR` | Where cached embeddings are stored | `./indexes/emb_cache` |

`CHROMA_BIND_HOST` and `CHROMA_HOST_PORT` are read by docker-compose rather
than by the application. See "Changing the Chroma port" below.

The Hugging Face cache variables (`HF_HOME`, `HUGGINGFACE_HUB_CACHE`,
`SENTENCE_TRANSFORMERS_HOME`) are read by those libraries directly. They are
set in `.env.example` so downloaded models land under `./models/hf_cache`
instead of your home directory.

### Ingestion

| Variable | Description | Default |
| --- | --- | --- |
| `CHUNK_SIZE` | Characters per chunk | `1000` |
| `CHUNK_OVERLAP` | Overlap between neighbouring chunks | `150` |
| `INGEST_THREADS` | Threads used to chunk pages. Defaults to half your cores | half of `os.cpu_count()` |
| `DEDUP_CHUNKS` | Drop near-duplicate chunks at ingest time | `false` |
| `DEDUP_THRESHOLD` | Jaccard similarity above which two chunks count as duplicates | `0.92` |
| `ENABLE_OCR` | Run OCR over PDFs that have no text layer | `false` |
| `ENABLE_LANGUAGE_DETECTION` | Detect chunk language when metadata says `auto` | `true` |

### Retrieval

| Variable | Description | Default |
| --- | --- | --- |
| `USE_HYBRID` | Combine BM25 and vector results with RRF | `true` |
| `K_VECTOR` | Candidates fetched from the vector store | `8` |
| `K_BM25` | Candidates fetched from the lexical index | `8` |
| `ENABLE_NEIGHBOR_EXPANSION` | Pull in chunks adjacent to each hit | `true` |
| `NEIGHBOR_RADIUS` | How many neighbouring chunks either side | `1` |
| `DOC_DIVERSITY_CAP` | Most chunks kept from any single document | `3` |

### Answers

| Variable | Description | Default |
| --- | --- | --- |
| `DEFAULT_LANGUAGE` | `en`, `it`, or `auto` to follow the question | `auto` |
| `STRICT_CITATIONS` | Clean up `[n]` markers after generation | `false` |
| `APPEND_SOURCES_BLOCK` | Append a Sources list when strict citations is on | `false` |
| `TRANSLATE_ON_MISS` | Translate an answer that came back in the wrong language | `false` |
| `LOG_LEVEL` | Verbosity on stderr. See "Logging" below | `INFO` |

### Subject routing

Off by default. See the model size requirements before enabling it.

| Variable | Description | Default |
| --- | --- | --- |
| `ENABLE_ROUTING` | Pick a model per question subject | `false` |
| `ROUTE_MATH_MODEL_PATH` | GGUF for the maths route | `./models/DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf` |
| `ROUTE_CODE_MODEL_PATH` | GGUF for the code route | `./models/Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf` |
| `ROUTE_TRANSLATION_MODEL_PATH` | GGUF for the translation route | `./models/salamandraTA-7B-instruct.Q4_K_M.gguf` |
| `ROUTE_DEFAULT_MODEL_PATH` | GGUF for everything else | `./models/Qwen3-8B-Q4_K_M.gguf` |
| `ROUTE_N_CTX` | Context window for routed models | `4096` |
| `ROUTE_N_GPU_LAYERS` | Layers offloaded to the GPU. `0` is CPU, `-1` is all | `0` |
| `ROUTE_MAX_TOKENS` | Cap on generated tokens | `768` |
| `ROUTE_TEMPERATURE` | Sampling temperature | `0.2` |
| `ROUTE_TOP_P` | Nucleus sampling cutoff | `0.95` |
| `ROUTE_QUERY_MARGIN` | Score gap below which a question counts as ambiguous | `0.10` |
| `ROUTE_METADATA_THRESHOLD` | Share of retrieved chunks that must agree on a subject | `0.60` |
| `ROUTE_TRANSLATION_REQUIRES_INTENT` | Require an explicit translate keyword for that route | `true` |

> 💡 After changing `.env`, restart your environment to apply settings.

---

## 2. Index Storage

The system uses two types of indexes:

* **Chroma Vector Store** (`CHROMA_PERSIST_DIRECTORY`):
  Stores embeddings for semantic search.

* **BM25 Store** (`./indexes/bm25`):
  Stores text chunks for keyword search.

Both are used together in **hybrid retrieval**.

---

## 3. Maintenance Commands

The `rag ` subcommands help you **inspect, backup, restore, and clean** your indexes.

### Show Index Stats

Check how many vectors are stored and disk usage:

```bash
rag stats
```

Output includes:

* vector count
* storage paths for Chroma & BM25
* current embedding model

---

### Preview Retrieval

See what would be retrieved (without LLM generation):

```bash
rag preview "What is the chain rule in calculus?"
```

Shows snippets, provenance, and scores.

---

### Backup Index

Export all chunks to a JSONL file:

```bash
rag dump --path backup.jsonl
```

You can include checksums for integrity.

---

### Restore from Backup

Load chunks back into BM25 and Chroma:

```bash
rag restore backup.jsonl
```

---

### Vacuum (Clean Indexes)

Compact and save the indexes:

```bash
rag vacuum
```

---

### Rebuild Embeddings

Recompute embeddings with a new model:

```bash
rag rebuild --model intfloat/multilingual-e5-large
```

This will keep BM25 intact but update the Chroma store.

---

### Manage Entries

#### List Entries

```bash
rag list --course "Math101"
```

#### Delete by ID

```bash
rag delete <chunk_id>
```

#### Reingest Files

```bash
rag reingest path/to/file.pdf
```

#### List Source Paths

```bash
rag sources
```

---

## Changing the Chroma port

`CHROMA_HOST_PORT` moves the published port when something else on the machine
already uses 8000:

```bash
CHROMA_HOST_PORT=8001 docker compose up -d chroma
```

Set it in `.env` instead to make it stick. `CHROMA_HTTP_URL` has to use the same
port, otherwise the app starts normally and then fails on the first query.

`CHROMA_BIND_HOST` defaults to `127.0.0.1`. Chroma has no authentication in front
of it, so only change this if you understand what you are exposing.

## Logging

`LOG_LEVEL` controls how much the CLI reports. Records go to stderr, so
stdout stays parseable:

```bash
rag stats | jq           # JSON only
LOG_LEVEL=DEBUG rag ask "..."   # progress on stderr
```

At `INFO`, the default, only this project's own messages appear, which is
mostly model loads and evictions. Chatty third-party loggers (httpx,
chromadb, transformers and friends) are held at `WARNING` so a single query
does not produce a line per HTTP request. `DEBUG` lifts that and shows
everything.

## Where settings come from

Precedence, highest first:

1. Variables exported in your shell
2. The project `.env`
3. The defaults in `rag/config.py`

So a one-off override works as you'd expect, without editing `.env`:

```bash
LOG_LEVEL=DEBUG rag ask "..."
CHROMA_HOST_PORT=8001 docker compose up -d chroma
```
