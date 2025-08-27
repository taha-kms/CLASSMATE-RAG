"""
Quick ingest benchmark.

Examples:
    python -m tools.bench_ingest ./data/*.pdf
    python -m tools.bench_ingest --repeat 3 ./data/sample1.pdf ./data/sample2.docx
"""

from __future__ import annotations

import argparse
import glob
import time
from pathlib import Path
from typing import List

from rag.metadata import normalize_cli_metadata, DocumentMetadata
from rag.pipeline import ingest_file


def run(paths: List[str], repeat: int = 1) -> None:
    files: List[Path] = []
    for p in paths:
        if any(ch in p for ch in "*?[]"):
            files.extend(Path(".").glob(p))
        else:
            files.append(Path(p))
    files = [f.resolve() for f in files if f.exists()]
    if not files:
        print("No files found.")
        return

    meta: DocumentMetadata = normalize_cli_metadata(language="auto", tags="bench")
    t0 = time.perf_counter()
    up = 0
    for _ in range(repeat):
        for f in files:
            res = ingest_file(path=str(f), doc_meta=meta)
            up += res.upserted
    dt = time.perf_counter() - t0
    tps = up / dt if dt > 0 else 0.0
    print(f"Ingested chunks: {up} in {dt:.2f}s  ->  {tps:.1f} chunks/sec")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="Files or globs")
    ap.add_argument("--repeat", type=int, default=1, help="Repeat count")
    args = ap.parse_args(argv)

    run(args.paths, repeat=int(args.repeat))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
