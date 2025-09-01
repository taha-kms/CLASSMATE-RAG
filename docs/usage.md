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
rag ingest path/to/file.pdf --course "Math101" --unit "1" --language "en"
```

Options you can use during ingestion:

* `--course`: course name or ID
* `--unit`: unit/chapter identifier
* `--language`: document language
* `--doc-type`: type of document (slides, notes, book, etc.)
* `--tags`: comma-separated tags

You can ingest multiple files at once:

```bash
rag ingest data/*.pdf --course "CS50" --language "en"
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

## 4. Admin Commands

For maintenance and debugging, there are admin commands.

### Show Index Stats

```bash
rag admin stats
```

### Preview Retrieval (no generation)

```bash
rag admin preview "What is machine learning?"
```

### Backup Index

```bash
rag admin backup backup.jsonl
```

### Restore from Backup

```bash
rag admin restore backup.jsonl
```

---

## 5. Managing Data

### List Entries

```bash
rag admin list --course "Math101"
```

### Delete by ID

```bash
rag admin delete <chunk_id>
```

### Reingest Files

```bash
rag admin reingest path/to/file.pdf
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