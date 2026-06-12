#!/usr/bin/env python3
"""Phase 2 — Tokenize and pack. CPU only.

Reads cleaned JSONL, encodes via the production BPE, packs into fixed-length
blocks, writes binary `.npy` shards ready for memory-mapped training.
"""

from __future__ import annotations

import argparse
import glob
import json
import time
from pathlib import Path

import numpy as np

from sota_model.training.data import BlockPacker
from sota_model.training.sample_loader import iter_jsonl


def load_tokenizer(name: str):
    """Accepts a path to a trained `tokenizer.json` or a HF AutoTokenizer name."""
    p = Path(name)
    if p.exists() or name.endswith("tokenizer.json"):
        from sota_model.tokenizer import load_tokenizer as load_sota
        return load_sota(name)
    try:
        from transformers import AutoTokenizer
    except ImportError as e:
        raise SystemExit("Install transformers: pip install -e '.[dev]'") from e
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tokenizer-name", default="gpt2")
    parser.add_argument("--seq-len", type=int, default=8192)
    parser.add_argument("--shard-tokens", type=int, default=1_073_741_824)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    tok = load_tokenizer(args.tokenizer_name)
    packer = BlockPacker(seq_len=args.seq_len, reset_mask=True)

    files = sorted(glob.glob(args.input))
    if not files:
        raise SystemExit(f"no files matched {args.input!r}")

    started = time.time()
    total_docs = total_tokens = shards_written = 0
    current: list[np.ndarray] = []
    current_tokens = 0

    def token_streams():
        nonlocal total_docs
        for p in files:
            print(f"reading {p}")
            for doc in iter_jsonl(Path(p)):
                total_docs += 1
                yield tok.encode(doc["text"])

    def flush(idx, blocks):
        if not blocks:
            return idx
        arr = np.concatenate(blocks, axis=0).astype(np.int32)
        out = args.output / f"packed_{idx:05d}.npy"
        np.save(out, arr)
        print(f"  wrote {out} ({arr.size:,} tokens)")
        return idx + 1

    for block in packer.pack(token_streams()):
        ids = np.asarray(block["input_ids"], dtype=np.int32)
        if ids.size != args.seq_len:
            continue
        current.append(ids[None, :])
        current_tokens += ids.size
        total_tokens += ids.size
        if current_tokens >= args.shard_tokens:
            shards_written = flush(shards_written, current)
            current, current_tokens = [], 0
    shards_written = flush(shards_written, current)

    summary = {
        "input_files": files,
        "tokenizer": args.tokenizer_name,
        "seq_len": args.seq_len,
        "docs": total_docs,
        "tokens": total_tokens,
        "shards": shards_written,
        "elapsed_s": round(time.time() - started, 1),
    }
    print("\n" + json.dumps(summary, indent=2))
    if args.report:
        args.report.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
