#!/usr/bin/env python3
"""Phase 1 — Data cleanup. CPU only.

Reads raw JSONL shards, applies the filter pipeline from
`sota_model.training.data`, writes cleaned JSONL shards.
"""

from __future__ import annotations

import argparse
import glob
import json
import time
from collections import Counter
from pathlib import Path

from sota_model.training.data import (
    BenchmarkContaminationFilter,
    DuplicateRemover,
    LanguageDetector,
    MinLengthFilter,
    PIIRedactor,
    QualityScorer,
    ToxicityFilter,
)
from sota_model.training.sample_loader import iter_jsonl


def build_filter_chain():
    return [
        ("language", LanguageDetector(accepted=("en", "fr", "de", "es", "hi", "zh"))),
        ("min_length", MinLengthFilter(min_chars=100)),
        ("dedup", DuplicateRemover()),
        ("quality", QualityScorer(threshold=0.7)),
        ("toxicity", ToxicityFilter()),
        ("pii", PIIRedactor()),
        ("contamination", BenchmarkContaminationFilter()),
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Glob of raw JSONL files")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--shard-size", type=int, default=100_000)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    chain = build_filter_chain()
    rejected: Counter = Counter()
    accepted = total = 0
    shard_idx = 0
    out_buffer: list[dict] = []
    started = time.time()

    def flush(buf, idx):
        if not buf:
            return idx
        out = args.output / f"cleaned_{idx:05d}.jsonl"
        with out.open("w") as f:
            for d in buf:
                f.write(json.dumps(d) + "\n")
        print(f"  wrote {out} ({len(buf)} docs)")
        return idx + 1

    files = sorted(glob.glob(args.input))
    if not files:
        raise SystemExit(f"no files matched {args.input!r}")

    for path in files:
        print(f"reading {path}")
        for doc in iter_jsonl(Path(path)):
            total += 1
            for stage, f in chain:
                doc = f(doc)
                if doc is None:
                    rejected[stage] += 1
                    break
            if doc is not None:
                accepted += 1
                out_buffer.append(doc)
                if len(out_buffer) >= args.shard_size:
                    shard_idx = flush(out_buffer, shard_idx)
                    out_buffer = []
            if total % 10_000 == 0:
                rate = total / max(0.1, time.time() - started)
                print(f"  {total:>10,} read, {accepted:>10,} accepted, {rate:.0f} docs/s")
    shard_idx = flush(out_buffer, shard_idx)

    summary = {
        "total_in": total, "accepted": accepted,
        "rejected_by_stage": dict(rejected),
        "shards_written": shard_idx,
        "elapsed_s": round(time.time() - started, 1),
    }
    print("\n" + json.dumps(summary, indent=2))
    if args.report:
        args.report.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
