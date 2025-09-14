# System Architecture

This document explains the internal architecture of the RAG (Retrieval-Augmented Generation) system.  
It covers all major components, their roles, and how data flows through the system.

---

## 1. High-Level Overview

The system combines **document ingestion, retrieval, and generation** to answer questions with sources.


```
            ┌─────────────┐
            │   Ingestion │
            └──────┬──────┘
                   │
                   ▼
    ┌─────────────────────────────┐
    │   Indexing (BM25 + Chroma)  │
    └──────┬──────────────┬───────┘
           │              │
           ▼              ▼
    ┌──────────┐    ┌─────────────┐
    │ BM25     │    │ Chroma DB   │
    │ (keyword)│    │ (embeddings)│
    └────┬─────┘    └──────┬──────┘
         │                 │
         └───┬──────────┬──┘
             ▼          ▼
          ┌────────────────┐
          │ Hybrid Fusion  │
          │ (retriever)    │
          └──────┬─────────┘
                 │
                 ▼
           ┌──────────┐
           │ Generator │ (LLM)
           └────┬─────┘
                ▼
          ┌────────────┐
          │ Final Answer│
          │ + Citations │
          └────────────┘
```


---

## 2. Components

### 2.1 Chunking

- Splits raw documents into **sentences** and then into **chunks**.  
- Uses overlap between chunks for better context.  
- Implemented in `rag/chunking/chunker.py`.  
- Output: `RagChunk(page, chunk_id, text)` objects.

### 2.2 Embeddings

- Uses **E5 multilingual model** (`intfloat/multilingual-e5-base`) for encoding.  
- Adds prefixes: `"query: ..."`, `"passage: ..."`.  
- Vectors are normalized (L2).  
- Implemented in `rag/embeddings/`.  
- Supports caching with `.npy` files for speed.

### 2.3 Indexing

Two complementary indexes:

- **BM25 Store** (`indexes/bm25/`)  
  - Keyword search on raw text.  
  - Stored in JSONL format.  
  - Fast, precise for exact matches.  

- **Chroma Vector Store** (`indexes/chroma/`)  
  - Stores embeddings in a persistent vector DB.  
  - Allows semantic similarity search.  

### 2.4 Retrieval

- **HybridRetriever** combines BM25 + Chroma results.  
- Uses **Reciprocal Rank Fusion (RRF)** to merge results.  
- Tunable weights: `weight_vector`, `weight_bm25`.  
- Outputs ranked list of chunks with metadata + scores.

### 2.5 Prompting

- Formats retrieved chunks with labels:  
```

\[1] chunk text...
\[2] chunk text...

```
- Builds system + user messages:
- **Grounded mode**: forces citations.  
- **General mode**: open-ended Q&A.  
- Implemented in `rag/generation/prompting.py`.

### 2.6 Generation

- Uses **llama-cpp-python** to run GGUF models (e.g., LLaMA, Mistral).  
- Supports:
- `max_tokens`, `temperature`, `top_p`, `top_k`, `stop` tokens.  
- Implemented in `rag/generation/llama_cpp_runner.py`.

### 2.7 Post-Processing

- Cleans and validates citations (`[1]`, `[2]`).  
- Removes invalid references, merges adjacent citations.  
- Optionally appends a plain **Sources** section.  
- Implemented in `rag/generation/post.py`.

---

## 3. Maintenance Tools

The `rag ` commands allow inspection and maintenance.

- **Stats**: Show vector count and disk usage.  
- **Preview**: See retrieval results without generation.  
- **Backup/Restore**: Export and reload index into JSONL.  
- **Vacuum**: Compact and save indexes.  
- **Rebuild-Embeddings**: Recompute vectors with a new model.  
- **Manage Entries**:
- List by filters (course, unit, tags).  
- Delete by ID.  
- Reingest files with preserved metadata.  

These tools ensure **index consistency** and easy **migration**.

---

## 4. Data Flow

1. **Ingest**  
 - Loader extracts pages.  
 - Pages → sentence splitting → chunks.  
 - Chunks encoded into embeddings.  
 - Stored in BM25 + Chroma.  

2. **Ask**  
 - User question encoded.  
 - HybridRetriever queries BM25 + Chroma.  
 - Results fused with RRF.  
 - Context formatted as `[n] chunk`.  
 - LLM generates answer with citations.  
 - Post-processing cleans output.  

---

## 5. Design Choices

- **Hybrid retrieval**: combines strengths of BM25 (precision) and embeddings (recall).  
- **Sentence-aware chunking**: avoids cutting text in unnatural places.  
- **Embeddings cache**: speeds up repeated indexing.  
- **Citations enforcement**: ensures transparency in answers.  
- **CLI-first design**: simplifies usage (`rag` command).  
- **Dockerized vector DB**: easy to run and persist across sessions.  

---

## 6. Future Extensions

- Support multiple embedding models (configurable).  
- Add structured data loaders (CSV, SQL).  
- Web UI for ingestion and querying.  
- Cloud storage integration (S3, GCS).  
- Model fine-tuning for domain-specific tasks.  

---

## 7. Related Files

- `rag/chunking/` → text splitting  
- `rag/embeddings/` → vectorization  
- `rag/retrieval/` → BM25, Chroma, HybridRetriever  
- `rag/generation/` → LLM + prompt building  
- `rag/admin/` → backup, restore, stats, manage entries  
- `rag/loaders/` → file loaders (PDF, text, etc.)  

---
