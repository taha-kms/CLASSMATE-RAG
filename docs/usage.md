# Usage Guide

This document explains how to use the **`rag`** CLI after installation.  
(See [installation.md](installation.md) if you haven’t set up the project yet.)

---

## 1. Basic Command

The CLI can be run with:

```bash
rag --help
````

This shows all available commands and options.

---

## 2. Ingest Documents

Before asking questions, you need to ingest documents into the vector database.

```bash
rag add path/to/file.pdf --course "Math101" --unit "1" --language "en"
```

Options you can use during ingestion:

* `--course`: course name or ID
* `--unit`: unit/chapter identifier
* `--language`: document language
* `--doc-type`: type of document (slides, notes, book, etc.)
* `--tags`: comma-separated tags

You can ingest multiple files at once:

```bash
rag add data/*.pdf --course "CS50" --language "en"
```

---

## 3. Ask Questions

Once documents are ingested, you can query them:

```bash
rag ask "What is the definition of entropy?"
```

You can filter results by metadata:

```bash
rag ask "Explain Newton's second law" --course "Physics101" --unit "2"
```

---

## 4. Maintenance Commands

For maintenance and debugging, use these commands.

### Show Index Stats

```bash
rag stats
```

### Preview Retrieval (no generation)

```bash
rag preview "What is machine learning?"
```

### Backup Index

```bash
rag backup backup.jsonl
```

### Restore from Backup

```bash
rag restore backup.jsonl
```

---

## 5. Managing Data

### List Entries

```bash
rag list --course "Math101"
```

### Delete by ID

```bash
rag delete <chunk_id>
```

### Reingest Files

```bash
rag reingest path/to/file.pdf
```

---

## 6. Tips

* Always activate the virtual environment before using `rag`:

  ```bash
  source .venv/bin/activate   # Linux/macOS
  .\.venv\Scripts\Activate.ps1   # Windows PowerShell
  ```

* If Docker isn’t running, the vector DB will not work. Start it with:

  ```bash
  docker compose up -d
  ```

* You can inspect `.env` to adjust model paths, embedding settings, and DB configuration.

---