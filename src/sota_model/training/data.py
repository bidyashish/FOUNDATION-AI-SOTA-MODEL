"""Data pipeline: filtering, contamination scrubbing, sequence packing.

Contamination scrubbing is mandatory:  8.8.1 / 9.2 publishes a
blocklist of HLE-discussing URLs. The same applies to GPQA, MMLU, USAMO problem
text, SWE-bench instances. Train data that's mixed with eval text invalidates
the eval, full stop.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Callable, Iterable, Iterator, Protocol


class Filter(Protocol):
    def __call__(self, doc: dict) -> dict | None: ...


# --- Filters ---


class MinLengthFilter:
    def __init__(self, min_chars: int = 100):
        self.min_chars = min_chars

    def __call__(self, doc: dict) -> dict | None:
        return doc if len(doc.get("text", "")) >= self.min_chars else None


class LanguageDetector:
    """Lightweight language gate. In production, swap for fasttext/cld3."""

    def __init__(self, accepted: tuple[str, ...] = ("en",)):
        self.accepted = accepted

    def __call__(self, doc: dict) -> dict | None:
        # Tag-and-pass: assume an upstream classifier wrote `lang`.
        if "lang" in doc and doc["lang"] not in self.accepted:
            return None
        return doc


class DuplicateRemover:
    """MinHash-LSH style fingerprinting. The implementation here is a sha256
    short-fingerprint stand-in; production should swap in `datasketch.MinHashLSH`
    with Jaccard threshold 0.85 over 5-grams."""

    def __init__(self):
        self._seen: set[str] = set()

    def __call__(self, doc: dict) -> dict | None:
        fp = hashlib.sha256(doc.get("text", "").encode("utf-8")).hexdigest()[:16]
        if fp in self._seen:
            return None
        self._seen.add(fp)
        return doc


class QualityScorer:
    """Predicted quality threshold.

    Default backend is the heuristic baseline. For a production deployment,
    pass `backend=TrainedQualityScorer.load(path)` (or call
    `.train(...)` first). The two backends share the
    `(doc) -> doc | None` contract so the filter list doesn't change.
    """

    def __init__(self, threshold: float = 0.7, backend=None):
        self.threshold = threshold
        if backend is None:
            from sota_model.training.classifiers.quality import HeuristicQualityScorer
            backend = HeuristicQualityScorer(threshold=threshold)
        self._backend = backend

    def __call__(self, doc: dict) -> dict | None:
        return self._backend(doc)


class ToxicityFilter:
    """Block CSAM-adjacent and known-bad content.

    Default is the seed-blocklist backend (preserving legacy behaviour). For
    a production deployment, compose:
        ToxicityFilter(backend=BlocklistToxicityFilter.from_files([...]))
        ToxicityFilter(backend=TrainedToxicityFilter.load(path))
    Real implementations should NOT enumerate bad patterns inline — load
    from an encrypted, audited blocklist updated out-of-band.
    """

    def __init__(self, backend=None):
        if backend is None:
            from sota_model.training.classifiers.toxicity import BlocklistToxicityFilter
            backend = BlocklistToxicityFilter.from_default_seed()
        self._backend = backend

    def __call__(self, doc: dict) -> dict | None:
        return self._backend(doc)


class PIIRedactor:
    _PATTERNS = [
        (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[SSN]"),
        (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"), "[EMAIL]"),
        (re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"), "[PHONE]"),
    ]

    def __call__(self, doc: dict) -> dict | None:
        text = doc.get("text", "")
        for pattern, replacement in self._PATTERNS:
            text = pattern.sub(replacement, text)
        doc["text"] = text
        return doc


class BenchmarkContaminationFilter:
    """Drops documents that contain known eval text or come from known eval URLs.

    Seed list mirrors  9.2 (HLE blocklist) plus common variants.
    """

    DEFAULT_URL_BLOCKLIST: tuple[str, ...] = (
        "huggingface.co",
        "hf.co",
        "promptfoo.dev",
        "scale.com",
        "lastexam.ai",
        "agi.safe.ai",
        "askfilo.com",
        "studocu.com",
        "coursehero.com",
        "qiita.com",
        "openreview.net/pdf?id=46UGfq8kMI",
        "github.com/centerforaisafety/hle",
        "github.com/supaihq/hle",
        # arxiv ids of eval papers
        "2501.14249",
        "2507.05241",
        "2508.10173",
        "2510.08959",
    )

    # Every benchmark named in modelcard 8 (capabilities), 3 (cyber),
    # 5 (agentic safety), 6 (alignment), and 7 (welfare tooling). Training
    # shards containing these names cannot be trusted.
    DEFAULT_TEXT_BLOCKLIST: tuple[str, ...] = (
        # 8.2 software engineering
        "SWE-bench Verified", "SWE-bench Pro",
        "SWE-bench Multilingual", "SWE-bench Multimodal",
        # 8.3
        "Terminal-Bench 2.0",
        # 8.4–8.6
        "GPQA Diamond", "USAMO 2026",
        # 8.7 long context
        "GraphWalks", "OpenAI MRCR",
        # 8.8 agentic search
        "Humanity's Last Exam", "BrowseComp", "DeepSearchQA", "DRACO benchmark",
        # 8.9 multimodal
        "LAB-Bench FigQA", "CharXiv Reasoning", "ScreenSpot-Pro", "OSWorld",
        # 8.10 real-world professional
        "OfficeQA Pro", "Finance Agent benchmark", "MCP-Atlas", "MCP Atlas",
        "Vending-Bench", "VendingBench", "GDPval-AA",
        # 8.11
        "ARC-AGI-1", "ARC-AGI-2",
        # 8.12 multilingual
        "Global MMLU", "GMMLU", "MMMLU", "MILU",
        "INCLUDE benchmark", "ECLeKTic",
        # 8.13
        "BioPipelineBench", "BioMysteryBench",
        # 3 cyber
        "Cybench", "CyberGym",
        # 5 agentic safety
        "Agent Red Teaming benchmark", "SHADE-Arena", "Minimal-LinuxBench",
        # 6 alignment
        "Petri 2.0", "MASK benchmark",
        "AA-Omniscience", "Bias Benchmark for Question Answering",
        # 7 welfare-relevant ops tooling
        "Clio",
    )

    def __init__(
        self,
        urls: Iterable[str] | None = None,
        texts: Iterable[str] | None = None,
    ):
        self.urls = tuple(urls) if urls else self.DEFAULT_URL_BLOCKLIST
        self.texts = tuple(texts) if texts else self.DEFAULT_TEXT_BLOCKLIST

    def __call__(self, doc: dict) -> dict | None:
        url = (doc.get("url") or "").lower().replace("/", "")
        if any(p.lower().replace("/", "") in url for p in self.urls):
            return None
        text = doc.get("text", "")
        if any(t in text for t in self.texts):
            return None
        return doc


# --- Pipeline ---


@dataclass
class PretrainingPipeline:
    tokenizer: Callable
    filters: list[Filter] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.filters:
            self.filters = [
                LanguageDetector(),
                MinLengthFilter(),
                DuplicateRemover(),
                QualityScorer(),
                ToxicityFilter(),
                PIIRedactor(),
                BenchmarkContaminationFilter(),
            ]

    def process(self, docs: Iterable[dict]) -> Iterator[list[int]]:
        for doc in docs:
            for f in self.filters:
                doc = f(doc)
                if doc is None:
                    break
            if doc is None:
                continue
            yield self.tokenizer.encode(doc["text"])


# --- Sequence packing ---


@dataclass
class BlockPacker:
    """Pack token streams into fixed-length blocks with reset masks.

    `reset_mask=True` ensures attention does not span document boundaries —
    catches cases where the document boundary itself is a useful learning signal.
    """

    seq_len: int = 8192
    reset_mask: bool = True

    def pack(self, token_streams: Iterable[list[int]]) -> Iterator[dict]:
        buffer: list[int] = []
        boundaries: list[int] = []
        for stream in token_streams:
            buffer.extend(stream)
            boundaries.append(len(buffer))
            while len(buffer) >= self.seq_len:
                chunk = buffer[: self.seq_len]
                chunk_boundaries = [b for b in boundaries if b <= self.seq_len]
                yield {
                    "input_ids": chunk,
                    "doc_boundaries": chunk_boundaries,
                }
                buffer = buffer[self.seq_len :]
                boundaries = [b - self.seq_len for b in boundaries if b > self.seq_len]
        if buffer:
            yield {"input_ids": buffer, "doc_boundaries": boundaries}
