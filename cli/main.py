"""
CLASSMATE-RAG CLI (Step 12: end-to-end with provenance)

Commands:
  - add <path> [metadata flags]     -> ingest and index the file
  - ask "<question>" [filter flags] -> retrieve, fuse, and generate an answer with citations
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional

from rag.metadata import normalize_cli_metadata, DocumentMetadata
from rag.loaders import infer_doc_type_from_path
from rag.pipeline import ingest_file, ask_question


def _detect_doc_type_from_ext(path: Path) -> str:
    # Keep as a soft helper; prefer loader inference for consistency
    return infer_doc_type_from_path(path)


def cmd_add(args: argparse.Namespace) -> int:
    path = Path(args.path)
    if not path.exists():
        print(f"ERROR: file not found: {path}", file=sys.stderr)
        return 2

    # If doc_type not provided, infer from extension/loaders
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

    # Pretty console output with answer + citations
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

    return p


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
