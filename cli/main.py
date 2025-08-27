"""
CLASSMATE-RAG CLI

Commands:
  - add <path> [metadata flags]
  - ask "<question>" [filter flags]
  - preview "<question>" [filter flags]   -> retrieval-only, shows contexts/scores
  - stats                                  -> index health (counts + disk usage)
  - dump --path dumps/corpus.jsonl [--no-emb]
  - restore --path dumps/corpus.jsonl
  - vacuum
  - rebuild --model intfloat/multilingual-e5-base
  - list [filter flags] [--limit N --offset M]
  - show (--id ID ... | --path PATH)
  - delete [--id ID ... | --path PATH | filter flags] [--dry-run]
  - reingest (--path PATH ... | --id ID ... | filter flags)

Notes:
- Pass --fixup to auto-trim fields and slug tags at the CLI boundary (Step 16).
"""

from __future__ import annotations

# --- LOAD .env EARLY (so HF cache vars take effect before imports) ---
from pathlib import Path

try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env", override=True)
except Exception:
    pass
# --------------------------------------------------------------------

import argparse
import json
import sys
from pathlib import Path as _Path
from typing import Optional, List

from rag.metadata import normalize_cli_metadata, DocumentMetadata
from rag.metadata.validation import validate_cli_metadata
from rag.loaders import infer_doc_type_from_path
from rag.pipeline import ingest_file, ask_question, retrieve_preview, index_stats
from rag.admin.manage import (
    list_entries,
    show_entries_by_id,
    resolve_ids,
    delete_by_ids,
    reingest_paths,
    list_source_paths,
)
from rag.admin.backup import dump_index, restore_dump, vacuum_indexes, rebuild_embeddings


def _detect_doc_type_from_ext(path: _Path) -> str:
    return infer_doc_type_from_path(path)


def _validated_meta_from_args(
    *,
    args: argparse.Namespace,
    inferred_doc_type: str | None = None,
    explicit_doc_type: bool = False,
) -> DocumentMetadata:
    raw = {
        "course": args.course,
        "unit": args.unit,
        "language": args.language,
        "doc_type": args.doc_type,
        "author": args.author,
        "semester": args.semester,
        "tags": args.tags,
    }
    clean = validate_cli_metadata(
        raw,
        fixup=bool(getattr(args, "fixup", False)),
        inferred_doc_type=inferred_doc_type,
        explicit_doc_type=explicit_doc_type,
    )
    # Hand off to existing normalizer to produce DocumentMetadata (enums etc.)
    meta: DocumentMetadata = normalize_cli_metadata(
        course=clean.get("course"),
        unit=clean.get("unit"),
        language=clean.get("language"),
        doc_type=clean.get("doc_type"),
        author=clean.get("author"),
        semester=clean.get("semester"),
        tags=clean.get("tags"),
    )
    return meta


def cmd_add(args: argparse.Namespace) -> int:
    path = _Path(args.path)
    if not path.exists():
        print(f"ERROR: file not found: {path}", file=sys.stderr)
        return 2

    inferred_dt = _detect_doc_type_from_ext(path)
    meta = _validated_meta_from_args(
        args=args,
        inferred_doc_type=inferred_dt,
        explicit_doc_type=(args.doc_type is not None),
    )

    try:
        res = ingest_file(path=path, doc_meta=meta)
    except Exception as e:
        print(json.dumps({"action": "ingest", "file": str(path), "error": str(e)}), file=sys.stderr)
        return 1

    print(json.dumps({
        "action": "ingest",
        "file": str(path),
        "doc_type": res.doc_type,
        "total_pages": res.total_pages,
        "total_chunks": res.total_chunks,
        "upserted": res.upserted,
        "created_at": res.created_at,
        "metadata": meta.to_dict(),
    }, ensure_ascii=False, indent=2))
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    question = args.question.strip()
    if not question:
        print("ERROR: question cannot be empty", file=sys.stderr)
        return 2

    meta_filters: DocumentMetadata = _validated_meta_from_args(args=args)

    hybrid = (args.hybrid == "on")
    try:
        res = ask_question(
            question=question,
            filters=meta_filters,
            top_k=int(args.k),
            hybrid=hybrid,
        )
    except Exception as e:
        print(json.dumps({"action": "query", "error": str(e)}), file=sys.stderr)
        return 1

    output = {
        "action": "query",
        "question": res.question,
        "answer": res.answer,
        "language": res.language,
        "top_k": res.top_k,
        "hybrid": res.hybrid,
        "sources": [{"n": i + 1, "ref": ref} for i, ref in enumerate(res.sources)],
        "filters": res.filters_applied,
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


def cmd_preview(args: argparse.Namespace) -> int:
    question = args.question.strip()
    if not question:
        print("ERROR: question cannot be empty", file=sys.stderr)
        return 2

    meta_filters: DocumentMetadata = _validated_meta_from_args(args=args)

    hybrid = (args.hybrid == "on")
    try:
        items = retrieve_preview(
            question=question,
            filters=meta_filters.to_dict(),
            top_k=int(args.k),
            hybrid=hybrid,
        )
    except Exception as e:
        print(json.dumps({"action": "preview", "error": str(e)}), file=sys.stderr)
        return 1

    print(json.dumps({
        "action": "preview",
        "question": question,
        "top_k": int(args.k),
        "hybrid": hybrid,
        "results": items,
        "filters": meta_filters.to_dict(),
    }, ensure_ascii=False, indent=2))
    return 0


def cmd_stats(_args: argparse.Namespace) -> int:
    try:
        s = index_stats()
    except Exception as e:
        print(json.dumps({"action": "stats", "error": str(e)}), file=sys.stderr)
        return 1
    print(json.dumps({"action": "stats", **s}, ensure_ascii=False, indent=2))
    return 0


# ---------- Backup / Export / Migration ----------

def cmd_dump(args: argparse.Namespace) -> int:
    include_emb = not bool(args.no_emb)
    try:
        n = dump_index(args.path, include_embedding_checksum=include_emb, batch_size=int(args.batch_size))
    except Exception as e:
        print(json.dumps({"action": "dump", "error": str(e)}), file=sys.stderr)
        return 1
    print(json.dumps({"action": "dump", "path": args.path, "wrote": n, "include_embedding_checksum": include_emb}, ensure_ascii=False, indent=2))
    return 0


def cmd_restore(args: argparse.Namespace) -> int:
    try:
        n = restore_dump(args.path, batch_size=int(args.batch_size))
    except Exception as e:
        print(json.dumps({"action": "restore", "error": str(e)}), file=sys.stderr)
        return 1
    print(json.dumps({"action": "restore", "path": args.path, "restored": n}, ensure_ascii=False, indent=2))
    return 0


def cmd_vacuum(_args: argparse.Namespace) -> int:
    try:
        status = vacuum_indexes()
    except Exception as e:
        print(json.dumps({"action": "vacuum", "error": str(e)}), file=sys.stderr)
        return 1
    print(json.dumps({"action": "vacuum", **status}, ensure_ascii=False, indent=2))
    return 0


def cmd_rebuild(args: argparse.Namespace) -> int:
    try:
        out = rebuild_embeddings(args.model, batch_size=int(args.batch_size))
    except Exception as e:
        print(json.dumps({"action": "rebuild", "error": str(e)}), file=sys.stderr)
        return 1
    print(json.dumps({"action": "rebuild", **out}, ensure_ascii=False, indent=2))
    return 0


# ---------- Ingestion management + validation ----------

def _filters_from_args(args: argparse.Namespace) -> dict:
    return _validated_meta_from_args(args=args).to_dict()


def cmd_list(args: argparse.Namespace) -> int:
    where = _filters_from_args(args)
    entries = list_entries(where=where, limit=args.limit, offset=args.offset)
    out = {
        "action": "list",
        "count": len(entries),
        "filters": where,
        "items": [
            {
                "id": e.id,
                "source_path": e.metadata.get("source_path"),
                "page": e.metadata.get("page"),
                "chunk_id": e.metadata.get("chunk_id"),
                "language": e.metadata.get("language"),
                "doc_type": e.metadata.get("doc_type"),
                "author": e.metadata.get("author"),
                "course": e.metadata.get("course"),
                "unit": e.metadata.get("unit"),
                "semester": e.metadata.get("semester"),
            }
            for e in entries
        ],
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    if not args.id and not args.path:
        print("ERROR: show requires --id or --path", file=sys.stderr)
        return 2

    if args.id:
        entries = show_entries_by_id(args.id)
    else:
        ids = resolve_ids(path=args.path)
        entries = show_entries_by_id(ids)

    out = {
        "action": "show",
        "items": [
            {
                "id": e.id,
                "metadata": e.metadata,
                "snippet": (e.text or "")[:1000],
            }
            for e in entries
        ],
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


def cmd_delete(args: argparse.Namespace) -> int:
    # Determine target IDs
    ids: List[str] = []
    if args.id:
        ids = list(args.id)
    elif args.path:
        ids = resolve_ids(path=args.path)
    else:
        where = _filters_from_args(args)
        ids = resolve_ids(where=where)

    if not ids:
        print(json.dumps({"action": "delete", "deleted": 0, "vectors": 0, "bm25": 0, "ids": []}, indent=2))
        return 0

    if args.dry_run:
        print(json.dumps({"action": "delete", "dry_run": True, "would_delete": len(ids), "ids": ids}, ensure_ascii=False, indent=2))
        return 0

    vec_n, bm25_n = delete_by_ids(ids)
    print(json.dumps({"action": "delete", "deleted": len(ids), "vectors": vec_n, "bm25": bm25_n, "ids": ids}, ensure_ascii=False, indent=2))
    return 0


def cmd_reingest(args: argparse.Namespace) -> int:
    targets: List[str] = []

    if args.path:
        # accept multiple --path
        targets = [str(Path(p).expanduser().resolve()) for p in args.path]
    elif args.id:
        # map IDs -> source paths
        ids = list(args.id)
        show = show_entries_by_id(ids)
        spaths = {str(e.metadata.get("source_path") or "") for e in show if e.metadata.get("source_path")}
        targets = sorted(str(Path(p).expanduser().resolve()) for p in spaths if p)
    else:
        # resolve by filters -> source paths
        where = _filters_from_args(args)
        targets = list_source_paths(where=where)

    if not targets:
        print(json.dumps({"action": "reingest", "reingested": 0, "paths": []}, ensure_ascii=False, indent=2))
        return 0

    if args.dry_run:
        print(json.dumps({"action": "reingest", "dry_run": True, "would_reingest": len(targets), "paths": targets}, ensure_ascii=False, indent=2))
        return 0

    try:
        res = reingest_paths(targets)
    except Exception as e:
        print(json.dumps({"action": "reingest", "error": str(e)}), file=sys.stderr)
        return 1

    print(json.dumps({"action": "reingest", "reingested": len(res), "results": res}, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="classmate", description="CLASSMATE-RAG CLI")
    sub = p.add_subparsers(dest="command", required=True)

    # add
    pa = sub.add_parser("add", help="Ingest a file with metadata")
    pa.add_argument("path", help="Path to the document to ingest")
    pa.add_argument("--course", type=str, help="Course code or name")
    pa.add_argument("--unit", type=str, help="Unit/module name")
    pa.add_argument("--language", type=str, choices=["en", "it", "auto"], default="auto", help="Language of the document (or auto)")
    pa.add_argument("--doc-type", type=str, choices=["pdf", "docx", "pptx", "md", "txt", "html", "csv", "epub", "other"], help="Document type (inferred by default)")
    pa.add_argument("--author", type=str, help="Author or source")
    pa.add_argument("--semester", type=str, help="Semester label (e.g., 2025S)")
    pa.add_argument("--tags", type=str, help="Comma-separated tags (e.g., exam,week1,lab)")
    pa.add_argument("--fixup", action="store_true", help="Auto-trim fields and slug tags if needed")
    pa.set_defaults(func=cmd_add)

    # ask
    pq = sub.add_parser("ask", help="Ask a question with optional metadata filters")
    pq.add_argument("question", help="The user question in English or Italian (use quotes)")
    pq.add_argument("--course", type=str, help="Filter by course")
    pq.add_argument("--unit", type=str, help="Filter by unit/module")
    pq.add_argument("--language", type=str, choices=["en", "it", "auto"], default="auto", help="Answer/query language (auto=match question)")
    pq.add_argument("--doc-type", type=str, choices=["pdf", "docx", "pptx", "md", "txt", "html", "csv", "epub", "other"], help="Filter by document type")
    pq.add_argument("--author", type=str, help="Filter by author/source")
    pq.add_argument("--semester", type=str, help="Filter by semester")
    pq.add_argument("--tags", type=str, help="Filter by comma-separated tags")
    pq.add_argument("--k", type=int, default=8, help="Top-K results after fusion")
    pq.add_argument("--hybrid", type=str, choices=["on", "off"], default="on", help="Use hybrid retrieval (vector+BM25)")
    pq.add_argument("--fixup", action="store_true", help="Auto-trim fields and slug tags if needed")
    pq.set_defaults(func=cmd_ask)

    # preview
    pp = sub.add_parser("preview", help="Preview retrieval results (no generation)")
    pp.add_argument("question", help="The query/question")
    pp.add_argument("--course", type=str, help="Filter by course")
    pp.add_argument("--unit", type=str, help="Filter by unit/module")
    pp.add_argument("--language", type=str, choices=["en", "it", "auto"], default="auto", help="Query/answer language (affects nothing here, just filter)")
    pp.add_argument("--doc-type", type=str, choices=["pdf", "docx", "pptx", "md", "txt", "html", "csv", "epub", "other"], help="Filter by document type")
    pp.add_argument("--author", type=str, help="Filter by author/source")
    pp.add_argument("--semester", type=str, help="Filter by semester")
    pp.add_argument("--tags", type=str, help="Filter by comma-separated tags")
    pp.add_argument("--k", type=int, default=8, help="Top-K to preview")
    pp.add_argument("--hybrid", type=str, choices=["on", "off"], default="on", help="Use hybrid retrieval")
    pp.add_argument("--fixup", action="store_true", help="Auto-trim fields and slug tags if needed")
    pp.set_defaults(func=cmd_preview)

    # stats
    ps = sub.add_parser("stats", help="Show index health and disk usage")
    ps.set_defaults(func=cmd_stats)

    # dump
    pd = sub.add_parser("dump", help="Export corpus to a JSONL dump")
    pd.add_argument("--path", required=True, help="Output JSONL path (e.g., dumps/corpus.jsonl)")
    pd.add_argument("--batch-size", type=int, default=256, help="Batch size for embedding checksum")
    pd.add_argument("--no-emb", action="store_true", help="Do not compute embedding checksums (faster, smaller)")
    pd.set_defaults(func=cmd_dump)

    # restore
    pr = sub.add_parser("restore", help="Restore indexes from a JSONL dump")
    pr.add_argument("--path", required=True, help="Input JSONL path")
    pr.add_argument("--batch-size", type=int, default=256, help="Batch size for re-embedding")
    pr.set_defaults(func=cmd_restore)

    # vacuum
    pv = sub.add_parser("vacuum", help="Compact/housekeep indexes (best-effort)")
    pv.set_defaults(func=cmd_vacuum)

    # rebuild
    prebuild = sub.add_parser("rebuild", help="Re-embed all texts with a new embedding model")
    prebuild.add_argument("--model", required=True, help="New embedding model name/path")
    prebuild.add_argument("--batch-size", type=int, default=256, help="Batch size for re-embedding")
    prebuild.set_defaults(func=cmd_rebuild)

    # list
    pl = sub.add_parser("list", help="List indexed chunks by metadata filters")
    pl.add_argument("--course", type=str, help="Filter by course")
    pl.add_argument("--unit", type=str, help="Filter by unit")
    pl.add_argument("--language", type=str, choices=["en", "it", "auto"], help="Filter by language")
    pl.add_argument("--doc-type", type=str, choices=["pdf", "docx", "pptx", "md", "txt", "html", "csv", "epub", "other"], help="Filter by doc type")
    pl.add_argument("--author", type=str, help="Filter by author")
    pl.add_argument("--semester", type=str, help="Filter by semester")
    pl.add_argument("--tags", type=str, help="Filter by tags (comma-separated)")
    pl.add_argument("--limit", type=int, default=50, help="Max items")
    pl.add_argument("--offset", type=int, default=0, help="Offset for paging")
    pl.add_argument("--fixup", action="store_true", help="Auto-trim fields and slug tags if needed")
    pl.set_defaults(func=cmd_list)

    # show
    pshow = sub.add_parser("show", help="Show detailed info for one or more chunk IDs, or all chunks of a path")
    pshow.add_argument("--id", nargs="+", help="One or more chunk IDs")
    pshow.add_argument("--path", type=str, help="Source file path")
    pshow.set_defaults(func=cmd_show)

    # delete
    pdel = sub.add_parser("delete", help="Delete chunks from BOTH vector + BM25 indexes")
    pdel.add_argument("--id", nargs="+", help="One or more chunk IDs to delete")
    pdel.add_argument("--path", type=str, help="Delete all chunks for a given source file")
    pdel.add_argument("--course", type=str, help="Filter by course")
    pdel.add_argument("--unit", type=str, help="Filter by unit")
    pdel.add_argument("--language", type=str, choices=["en", "it", "auto"], help="Filter by language")
    pdel.add_argument("--doc-type", type=str, choices=["pdf", "docx", "pptx", "md", "txt", "html", "csv", "epub", "other"], help="Filter by doc type")
    pdel.add_argument("--author", type=str, help="Filter by author")
    pdel.add_argument("--semester", type=str, help="Filter by semester")
    pdel.add_argument("--tags", type=str, help="Filter by tags (comma-separated)")
    pdel.add_argument("--dry-run", action="store_true", help="Preview what would be deleted")
    pdel.add_argument("--fixup", action="store_true", help="Auto-trim fields and slug tags if needed")
    pdel.set_defaults(func=cmd_delete)

    # reingest
    pre = sub.add_parser("reingest", help="Reingest by path(s), by id(s), or by filters (affects WHOLE files)")
    pre.add_argument("--path", nargs="+", help="One or more file paths to reingest")
    pre.add_argument("--id", nargs="+", help="One or more chunk IDs (their source file(s) will be reingested)")
    pre.add_argument("--course", type=str, help="Filter by course")
    pre.add_argument("--unit", type=str, help="Filter by unit")
    pre.add_argument("--language", type=str, choices=["en", "it", "auto"], help="Filter by language")
    pre.add_argument("--doc-type", type=str, choices=["pdf", "docx", "pptx", "md", "txt", "html", "csv", "epub", "other"], help="Filter by doc type")
    pre.add_argument("--author", type=str, help="Filter by author")
    pre.add_argument("--semester", type=str, help="Filter by semester")
    pre.add_argument("--tags", type=str, help="Filter by tags (comma-separated)")
    pre.add_argument("--dry-run", action="store_true", help="Preview which files would be reingested")
    pre.add_argument("--fixup", action="store_true", help="Auto-trim fields and slug tags if needed")
    pre.set_defaults(func=cmd_reingest)

    return p


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
