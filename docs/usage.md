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
* `--language`: document language (`en`, `it`, or `auto` to detect per chunk)
* `--doc-type`: one of `pdf`, `docx`, `pptx`, `md`, `txt`, `html`, `csv`, `epub`, `other` (inferred from file extension by default)
* `--author`: author or source
* `--semester`: semester label (e.g., `2025S`)
* `--tags`: comma-separated tags
* `--fixup`: trim/slugify field values automatically

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
## Grounded and ungrounded answers

`rag ask` reports whether the answer actually cited your material:

```json
{
  "grounded": true,
  "sources": [{ "n": 1, "ref": "/path/to/notes.md" }]
}
```

`sources` lists only the blocks the answer cited, and `n` matches the `[n]`
markers in the text, so a reader can follow a claim back to its source.

When the model answers without citing anything, the answer is still
returned, but with no sources and a short notice:

```json
{
  "grounded": false,
  "sources": [],
  "notice": "From the model's own knowledge, not your documents."
}
```

That case is common with small models, which often ignore the instruction
to cite. Treat those answers as you would any ungrounded model output.
