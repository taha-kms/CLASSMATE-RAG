"""
CLASSMATE-RAG CLI

Commands:
  - add <path> [metadata flags]
  - ask "<question>" [filter flags]
  - preview "<question>" [filter flags]   -> retrieval-only, shows contexts/scores
  - stats                                  -> index health (counts + disk usage)
"""

from __future__ import annotations

# --- LOAD .env EARLY (so HF cache vars take effect before imports) ---
import os
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
from typing import Optional

from rag.metadata import normalize_cli_metadata, DocumentMetadata
from rag.loaders import infer_doc_type_from_path
from rag.pipeline import ingest_file, ask_question, retrieve_preview, index_stats


def _detect_doc_type_from_ext(path: _Path) -> str:
    return infer_doc_type_from_path(path)


def cmd_add(args: argparse.Namespace) -> int:
    path = _Path(args.path)
    if not path.exists():
        print(f"ERROR: file not found: {path}", file=sys.stderr)
        return 2

    doc_type = args.doc_type or _detect_doc_type_from_ext(path)

    meta: DocumentMetadata = normalize_cli_metadata(
        course=args.course,
        unit=args.unit,
        language=args.language,
        doc_type=doc_type,
        author=args.author,
        semester=args.semester,
        tags=args.tags,
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

    meta_filters: DocumentMetadata = normalize_cli_metadata(
        course=args.course,
        unit=args.unit,
        language=args.language,
        doc_type=args.doc_type,
        author=args.author,
        semester=args.semester,
        tags=args.tags,
    )

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

    meta_filters: DocumentMetadata = normalize_cli_metadata(
        course=args.course,
        unit=args.unit,
        language=args.language,
        doc_type=args.doc_type,
        author=args.author,
        semester=args.semester,
        tags=args.tags,
    )

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


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="classmate", description="CLASSMATE-RAG CLI")
    sub = p.add_subparsers(dest="command", required=True)

    # add
    pa = sub.add_parser("add", help="Ingest a file with metadata")
    pa.add_argument("path", help="Path to the document to ingest")
    pa.add_argument("--course", type=str, help="Course code or name")
    pa.add_argument("--unit", type=str, help="Unit/module name")
    pa.add_argument("--language", type=str, choices=["en", "it", "auto"], default="auto", help="Language of the document (or auto)")
    pa.add_argument("--doc-type", type=str, choices=["pdf", "docx", "pptx", "md", "txt", "html", "csv", "other"], help="Document type (inferred by default)")
    pa.add_argument("--author", type=str, help="Author or source")
    pa.add_argument("--semester", type=str, help="Semester label (e.g., 2025S)")
    pa.add_argument("--tags", type=str, help="Comma-separated tags (e.g., exam,week1,lab)")
    pa.set_defaults(func=cmd_add)

    # ask
    pq = sub.add_parser("ask", help="Ask a question with optional metadata filters")
    pq.add_argument("question", help="The user question in English or Italian (use quotes)")
    pq.add_argument("--course", type=str, help="Filter by course")
    pq.add_argument("--unit", type=str, help="Filter by unit/module")
    pq.add_argument("--language", type=str, choices=["en", "it", "auto"], default="auto", help="Answer/query language (auto=match question)")
    pq.add_argument("--doc-type", type=str, choices=["pdf", "docx", "pptx", "md", "txt", "html", "csv", "other"], help="Filter by document type")
    pq.add_argument("--author", type=str, help="Filter by author/source")
    pq.add_argument("--semester", type=str, help="Filter by semester")
    pq.add_argument("--tags", type=str, help="Filter by comma-separated tags")
    pq.add_argument("--k", type=int, default=8, help="Top-K results after fusion")
    pq.add_argument("--hybrid", type=str, choices=["on", "off"], default="on", help="Use hybrid retrieval (vector+BM25)")
    pq.set_defaults(func=cmd_ask)

    # preview
    pp = sub.add_parser("preview", help="Preview retrieval results (no generation)")
    pp.add_argument("question", help="The query/question")
    pp.add_argument("--course", type=str, help="Filter by course")
    pp.add_argument("--unit", type=str, help="Filter by unit/module")
    pp.add_argument("--language", type=str, choices=["en", "it", "auto"], default="auto", help="Query/answer language (affects nothing here, just filter)")
    pp.add_argument("--doc-type", type=str, choices=["pdf", "docx", "pptx", "md", "txt", "html", "csv", "other"], help="Filter by document type")
    pp.add_argument("--author", type=str, help="Filter by author/source")
    pp.add_argument("--semester", type=str, help="Filter by semester")
    pp.add_argument("--tags", type=str, help="Filter by comma-separated tags")
    pp.add_argument("--k", type=int, default=8, help="Top-K to preview")
    pp.add_argument("--hybrid", type=str, choices=["on", "off"], default="on", help="Use hybrid retrieval")
    pp.set_defaults(func=cmd_preview)

    # stats
    ps = sub.add_parser("stats", help="Show index health and disk usage")
    ps.set_defaults(func=cmd_stats)

    return p


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
