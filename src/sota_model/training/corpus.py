"""Corpus loader.

Replaces the `dummy_iter` random-token emitter in `pretrain.py` with a real
loader that:

  - reads JSONL or parquet shards from a list of paths (per source);
  - applies the documented `PretrainingPipeline` filter chain;
  - tokenizes via `SOTATokenizer` (or fallback for tests);
  - packs into `seq_len`-token blocks via `BlockPacker`;
  - mixes sources at the ratios from `implied_training_corpus.source_mix_pct`
    in the config.

The mix is *target* — exact realization depends on shard sizes — but the
loader honors the ratio in expectation by interleaving streams with weights.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Iterator, Optional, Sequence

import torch

from sota_model.config import load_implied
from sota_model.training.data import (
    BenchmarkContaminationFilter,
    BlockPacker,
    DuplicateRemover,
    Filter,
    LanguageDetector,
    MinLengthFilter,
    PIIRedactor,
    QualityScorer,
    ToxicityFilter,
)


# ---- Shard readers ----------------------------------------------------------


def iter_jsonl_shard(path: Path) -> Iterator[dict]:
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def iter_jsonl_dir(directory: Path) -> Iterator[dict]:
    for p in sorted(directory.glob("*.jsonl")):
        yield from iter_jsonl_shard(p)


def iter_parquet_shard(path: Path) -> Iterator[dict]:
    try:
        import pyarrow.parquet as pq
    except ImportError as e:
        raise SystemExit(
            "Install `pyarrow` to read parquet shards (or use JSONL)."
        ) from e
    table = pq.read_table(str(path))
    for batch in table.to_batches():
        for row in batch.to_pylist():
            yield row


# ---- Mix ratio interleaver --------------------------------------------------


@dataclass
class CorpusSource:
    name: str
    iter_fn: Callable[[], Iterable[dict]]
    weight: float = 1.0


def weighted_interleave(sources: Sequence[CorpusSource], seed: int = 0) -> Iterator[dict]:
    """Yield documents from `sources` in a weighted-random order.

    Ratios match `weight / sum(weights)` in expectation. Sources that exhaust
    are dropped; iteration ends when all sources are exhausted.
    """
    rng = random.Random(seed)
    iterators = [iter(s.iter_fn()) for s in sources]
    weights = [s.weight for s in sources]
    while iterators:
        idx = _weighted_pick(rng, weights)
        try:
            yield next(iterators[idx])
        except StopIteration:
            iterators.pop(idx)
            weights.pop(idx)


def _weighted_pick(rng: random.Random, weights: Sequence[float]) -> int:
    total = sum(weights)
    pick = rng.random() * total
    cumulative = 0.0
    for i, w in enumerate(weights):
        cumulative += w
        if pick < cumulative:
            return i
    return len(weights) - 1


# ---- Top-level loader -------------------------------------------------------


@dataclass
class CorpusLoaderConfig:
    seq_len: int = 8_192
    pad_token_id: int = 0
    contamination_filter: bool = True
    quality_filter: bool = True
    toxicity_filter: bool = True
    pii_redactor: bool = True
    min_length: int = 100
    drop_duplicates: bool = True
    accepted_languages: Optional[tuple[str, ...]] = None  # None = pass-through
    seed: int = 0
    shuffle_buffer: int = 1024


def build_filter_chain(cfg: CorpusLoaderConfig) -> list[Filter]:
    chain: list[Filter] = []
    if cfg.accepted_languages is not None:
        chain.append(LanguageDetector(accepted=cfg.accepted_languages))
    chain.append(MinLengthFilter(min_chars=cfg.min_length))
    if cfg.drop_duplicates:
        chain.append(DuplicateRemover())
    if cfg.quality_filter:
        chain.append(QualityScorer())
    if cfg.toxicity_filter:
        chain.append(ToxicityFilter())
    if cfg.pii_redactor:
        chain.append(PIIRedactor())
    if cfg.contamination_filter:
        chain.append(BenchmarkContaminationFilter())
    return chain


class CorpusLoader:
    """End-to-end pretraining batch iterator.

    Expected use (replacing dummy_iter in pretrain.py):

        loader = CorpusLoader(
            sources=resolve_sources_from_yaml("configs/sota_ultra_5.yaml"),
            tokenizer=load_tokenizer("path/to/tokenizer"),
            cfg=CorpusLoaderConfig(seq_len=8192),
        )
        train_one_stage(model_cfg, train_cfg, loader.batches(batch_size=1), ...)
    """

    def __init__(
        self,
        sources: Sequence[CorpusSource],
        tokenizer,
        cfg: Optional[CorpusLoaderConfig] = None,
    ):
        self.sources = list(sources)
        self.tokenizer = tokenizer
        self.cfg = cfg or CorpusLoaderConfig()
        self._filters = build_filter_chain(self.cfg)
        self._packer = BlockPacker(seq_len=self.cfg.seq_len, reset_mask=True)

    def documents(self) -> Iterator[dict]:
        for doc in weighted_interleave(self.sources, seed=self.cfg.seed):
            for f in self._filters:
                doc = f(doc)
                if doc is None:
                    break
            if doc is not None:
                yield doc

    def token_streams(self) -> Iterator[list[int]]:
        for doc in self.documents():
            text = doc.get("text", "")
            if not text:
                continue
            yield self.tokenizer.encode(text)

    def _shuffled_blocks(self) -> Iterator[dict]:
        """Reservoir shuffle over packed blocks (size `cfg.shuffle_buffer`).

        Streaming-friendly decorrelation: adjacent shards of one source would
        otherwise arrive in file order. Buffer of N blocks ≈ N × seq_len tokens
        of look-ahead, seeded for reproducible data order.
        """
        if self.cfg.shuffle_buffer <= 1:
            yield from self._packer.pack(self.token_streams())
            return
        rng = random.Random(self.cfg.seed + 1)
        reservoir: list[dict] = []
        for block in self._packer.pack(self.token_streams()):
            reservoir.append(block)
            if len(reservoir) >= self.cfg.shuffle_buffer:
                idx = rng.randrange(len(reservoir))
                reservoir[idx], reservoir[-1] = reservoir[-1], reservoir[idx]
                yield reservoir.pop()
        rng.shuffle(reservoir)
        yield from reservoir

    def batches(self, batch_size: int = 1) -> Iterator[dict]:
        """Yield {"input_ids": LongTensor(B, T)} ready for the trainer.

        Pad short blocks at the tail. `batch_size` is the per-replica
        micro-batch — wire it from `train_cfg.micro_batch_size` so the
        global-batch identity (seq_len × micro × accum × dp) holds.
        """
        buffer: list[list[int]] = []
        for block in self._shuffled_blocks():
            ids = block["input_ids"]
            if len(ids) < self.cfg.seq_len:
                ids = ids + [self.cfg.pad_token_id] * (self.cfg.seq_len - len(ids))
            buffer.append(ids)
            if len(buffer) >= batch_size:
                yield {"input_ids": torch.tensor(buffer, dtype=torch.long)}
                buffer = []
        if buffer:
            yield {"input_ids": torch.tensor(buffer, dtype=torch.long)}


# ---- Helper to wire corpus from a YAML config ------------------------------


def resolve_sources_from_yaml(
    config_path: str | Path,
    data_root: str | Path,
) -> list[CorpusSource]:
    """Build CorpusSource objects from `implied_training_corpus.source_mix_pct`.

    Looks under `data_root/<source>/*.jsonl` for each named source and
    weights by `source_mix_pct`. Operators with a different layout pass
    their own list of CorpusSource.
    """
    implied = load_implied(config_path).get("implied_training_corpus", {})
    mix = implied.get("source_mix_pct", {})
    data_root = Path(data_root)
    sources: list[CorpusSource] = []
    for name, pct in mix.items():
        directory = data_root / name
        if not directory.exists():
            continue
        sources.append(
            CorpusSource(
                name=name,
                iter_fn=lambda d=directory: iter_jsonl_dir(d),
                weight=float(pct),
            )
        )
    return sources


def loader_config_from_yaml(
    config_path: str | Path,
    *,
    seq_len: int,
    pad_token_id: int = 0,
    seed: int = 0,
) -> CorpusLoaderConfig:
    """Build a CorpusLoaderConfig from the YAML's operational data knobs.

    Reads `implied_training_corpus.pipeline` (filter-chain settings) and, when
    `restrict_to_coverage_languages` is set, derives the accepted-language set
    from `implied_multilingual_coverage` — so the data pipeline is driven by
    the same config file as everything else instead of hardcoded defaults.
    """
    implied = load_implied(config_path)
    pipe = (implied.get("implied_training_corpus") or {}).get("pipeline") or {}

    accepted: Optional[tuple[str, ...]] = None
    if pipe.get("restrict_to_coverage_languages"):
        coverage = implied.get("implied_multilingual_coverage") or {}
        langs: list[str] = []
        if coverage.get("english_baseline"):
            langs.append(coverage["english_baseline"])
        for tier in ("high_resource", "mid_resource", "low_resource", "indic_milu"):
            langs.extend(coverage.get(tier) or [])
        accepted = tuple(dict.fromkeys(langs))  # dedupe, keep order

    return CorpusLoaderConfig(
        seq_len=seq_len,
        pad_token_id=pad_token_id,
        contamination_filter=bool(pipe.get("contamination_filter", True)),
        quality_filter=bool(pipe.get("quality_filter", True)),
        toxicity_filter=bool(pipe.get("toxicity_filter", True)),
        pii_redactor=bool(pipe.get("pii_redactor", True)),
        min_length=int(pipe.get("min_doc_chars", 100)),
        drop_duplicates=bool(pipe.get("drop_duplicates", True)),
        accepted_languages=accepted,
        seed=seed,
        shuffle_buffer=int(pipe.get("shuffle_buffer_blocks", 1024)),
    )
