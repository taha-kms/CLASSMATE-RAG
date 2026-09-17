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
| `LLM_PROVIDER` | Which provider generates answers. `llama_cpp` runs locally | `llama_cpp` |
| `ANTHROPIC_API_KEY` | Key for `LLM_PROVIDER=anthropic`. Prefer `rag config set` | unset |
| `ANTHROPIC_MODEL` | Model for that provider | `claude-opus-5` |
| `MODEL_PROFILE` | Sized model set: `light`, `balanced`, `heavy` or `custom`. See below | `custom` |
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

Off by default, and the defaults are heavy. Read the next section before
turning it on.

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

## Model profiles

`MODEL_PROFILE` picks a set of models sized for your machine, instead of
naming four GGUFs by hand.

```bash
rag profiles
```

reports what it found and which profiles fit:

```
hardware   : VRAM 4.0 GB, RAM 15.4 GB, free disk 18.0 GB
recommended: light
  light       4.0 GB dl   2.8 GB vram  fits=True
  balanced    8.8 GB dl   5.2 GB vram  fits=False
       ! 5.2 GB of VRAM needed to offload fully, 4.0 GB present.
  heavy      17.8 GB dl   9.7 GB vram  fits=False
```

Once chosen, fetch what it needs:

```bash
rag model download --profile light
```

| Profile | Models | Download | VRAM to offload |
| --- | --- | --- | --- |
| `light` | 3B | ~4 GB | ~2.8 GB |
| `balanced` | 7B | ~8.8 GB | ~5.2 GB |
| `heavy` | 14B | ~17.8 GB | ~9.7 GB |
| `custom` | whatever `ROUTE_*_MODEL_PATH` says | — | — |

`custom` is the default and preserves the previous behaviour exactly.

"Does not fit" is a warning, not a refusal. A model too large for the GPU
still runs on CPU; it is just slow, and that is your call to make. Only one
model is resident at a time, so the VRAM figure is the largest single
model rather than the sum.

The recommendation is deliberately conservative: with no GPU detected it
suggests `light`, because everything runs on CPU eventually and pointing
someone at a 14B model they will wait minutes for is not helpful.

## What subject routing actually costs

`ENABLE_ROUTING=true` swaps models per question subject. The defaults are
four separate 7-8B models:

| Route | Default model | Download |
| --- | --- | --- |
| math | DeepSeek-R1-Distill-Qwen-7B Q4_K_M | ~4.4 GB |
| code | Qwen2.5-Coder-7B-Instruct Q4_K_M | ~4.4 GB |
| translation | salamandraTA-7B-instruct Q4_K_M | ~4.4 GB |
| default | Qwen3-8B Q4_K_M | ~4.9 GB |
| | **total** | **~18 GB** |

Nothing downloads them for you, and a missing one falls back to the default
route rather than failing, so an unconfigured install quietly answers every
question with the same model.

### Memory

Only one model is resident at a time. The loader evicts the current one
before loading the next, which is the only way this fits on a normal
machine, but it means a question that changes route pays a full model load
first.

Fully offloading a 7B Q4 to the GPU needs about 4.4 GB of VRAM on top of
what the context window uses. So:

| VRAM | What works |
| --- | --- |
| under 4 GB | CPU only. Leave `ROUTE_N_GPU_LAYERS` at `0` |
| 4-6 GB | partial offload. Raise `ROUTE_N_GPU_LAYERS` gradually; `-1` will run out of memory |
| 8 GB+ | one 7B Q4 fully offloaded, one at a time |

`ROUTE_N_GPU_LAYERS=-1` means "all layers" and is the setting most likely
to fail: it succeeds on a machine with headroom and dies partway through
loading on one without.

### Is it worth it

Routing helps when the subjects are genuinely different and you have the
disk and memory for several specialist models. On a laptop it usually is
not worth 18 GB, and one good general model on the default route is the
better trade. A hosted backend avoids the question entirely.

## Credentials

API keys are not ordinary settings and are not stored in `.env`.

```bash
rag config set ANTHROPIC_API_KEY sk-ant-...
rag config get ANTHROPIC_API_KEY     # ********1234
```

They go in `.secrets.env`, written `0600` and gitignored. Reads give a
masked form: enough to tell which key is configured, not enough to use it.
Nothing returns a stored credential in full, so one cannot end up in a
screenshot, a shell history or a bug report by someone simply asking what
is set.

Precedence is the same as every other setting: a variable exported in your
shell beats the stored one, so a one-off override works.

The file lives in the project rather than the image, so a container mounts
it and `docker rm` does not lose it, and no image layer ever contains a
credential.

If a key does reach a file it should not, the gitleaks check in CI covers
Anthropic, OpenAI, Hugging Face and Google shapes.

## Hosted providers and your documents

`LLM_PROVIDER` chooses what generates answers. `llama_cpp`, the default,
runs on your machine and nothing leaves it.

Any other provider runs on someone else's computers, and each question
sends them:

- the question
- the passages retrieved from your documents to answer it

Not the corpus, and not the documents those passages came from. But the
passages are your coursework, and they do leave the machine. Queries are
also billed per token by the provider.

The first time a hosted provider is used in a session, the application
prints what it is about to send, on stderr so it cannot end up in piped
JSON. Every answer carries a `backend` field, so whether a given answer was
local is never a question about configuration.

## Using Anthropic

```bash
pip install -e ".[anthropic]"
rag config set ANTHROPIC_API_KEY sk-ant-...
LLM_PROVIDER=anthropic rag ask "What is the chain rule?" --course Maths
```

The SDK is an optional extra, so a local-only install does not carry it.
The key belongs in the credential store rather than `.env`; see
Credentials above.

Read "Hosted providers and your documents" first. This sends your retrieved
coursework to Anthropic and is billed per token.

`ANTHROPIC_MODEL` picks the model. Subject routing has no effect here:
routes exist to swap local GGUFs, and a hosted provider has one model per
request instead.
