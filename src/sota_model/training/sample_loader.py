"""JSONL sample loader for tests and end-to-end pipeline smoke runs.

Real production training reads parquet shards from object storage; this module
exists so unit tests and onboarding scripts have a no-dependency path to working
data.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator


def iter_jsonl(path: str | Path) -> Iterator[dict]:
    p = Path(path)
    with p.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_corpus_samples(samples_dir: str | Path = "data/samples") -> dict[str, list[dict]]:
    """Per-source docs from the corpus tree: `<samples_dir>/corpus/<source>/*.jsonl`.

    Subdirectory names mirror `implied_training_corpus.source_mix_pct` keys, so
    the tree doubles as a fixture for the real CorpusLoader path:
    `resolve_sources_from_yaml(config, samples_dir + "/corpus")`.
    """
    root = Path(samples_dir) / "corpus"
    out: dict[str, list[dict]] = {}
    if root.exists():
        for sub in sorted(p for p in root.iterdir() if p.is_dir()):
            docs = [doc for f in sorted(sub.glob("*.jsonl")) for doc in iter_jsonl(f)]
            if docs:
                out[sub.name] = docs
    return out


def load_pretrain_samples(samples_dir: str | Path = "data/samples") -> list[dict]:
    """Flat legacy files plus everything in the corpus tree."""
    p = Path(samples_dir)
    docs: list[dict] = []
    for name in ("pretrain.jsonl", "code.jsonl", "multilingual.jsonl"):
        f = p / name
        if f.exists():
            docs.extend(iter_jsonl(f))
    for source_docs in load_corpus_samples(samples_dir).values():
        docs.extend(source_docs)
    return docs


def load_chat_samples(samples_dir: str | Path = "data/samples") -> list[dict]:
    p = Path(samples_dir) / "chat.jsonl"
    return list(iter_jsonl(p)) if p.exists() else []


def load_tool_use_samples(samples_dir: str | Path = "data/samples") -> list[dict]:
    p = Path(samples_dir) / "tool_use.jsonl"
    return list(iter_jsonl(p)) if p.exists() else []


def load_contamination_samples(samples_dir: str | Path = "data/samples") -> list[dict]:
    p = Path(samples_dir) / "contamination.jsonl"
    return list(iter_jsonl(p)) if p.exists() else []
