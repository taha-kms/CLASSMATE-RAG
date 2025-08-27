"""
Quick ask benchmark (latency & p95).

Examples:
    python -m tools.bench_ask "What is polymorphism in Java?" --n 20
"""

from __future__ import annotations

import argparse
import statistics
import time
from typing import List

from rag.metadata import normalize_cli_metadata
from rag.pipeline import ask_question


def bench(question: str, n: int, k: int) -> None:
    latencies: List[float] = []
    filters = normalize_cli_metadata(language="auto")

    for i in range(n):
        t0 = time.perf_counter()
        _ = ask_question(question=question, filters=filters, top_k=k, hybrid=True)
        dt = time.perf_counter() - t0
        latencies.append(dt)
        print(f"{i+1:>3}/{n}: {dt*1000:.1f} ms")

    mean = statistics.mean(latencies)
    p95 = statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else max(latencies)
    print(f"\nMean: {mean*1000:.1f} ms   p95: {p95*1000:.1f} ms   (n={n})")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("question", type=str)
    ap.add_argument("--n", type=int, default=10, help="Number of runs")
    ap.add_argument("--k", type=int, default=8, help="Top-K retrieval")
    args = ap.parse_args(argv)

    bench(args.question, n=int(args.n), k=int(args.k))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
