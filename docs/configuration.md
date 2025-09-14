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

### Common Settings

| Variable                   | Description                                     | Example                           |
| -------------------------- | ----------------------------------------------- | --------------------------------- |
| `EMBEDDING_MODEL_NAME`     | Name of embedding model to use                  | `intfloat/multilingual-e5-base`   |
| `CHROMA_PERSIST_DIRECTORY` | Path where Chroma vector DB is stored           | `./indexes/chroma`                |
| `CHROMA_COLLECTION_NAME`   | Collection name inside Chroma                   | `rag_collection`                  |
| `HF_TOKEN`                 | (optional) HuggingFace token for private models | `<your_token>`                    |
| `LLAMA_MODEL_PATH`         | Path to local `.gguf` model file                | `./models/mistral-7b.Q4_K_M.gguf` |
| `LLAMA_CONTEXT_SIZE`       | Context window size for the model               | `2048`                            |
| `LLAMA_GPU_LAYERS`         | Number of layers to run on GPU                  | `35`                              |

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
