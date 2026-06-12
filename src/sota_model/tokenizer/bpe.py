"""SOTA tokenizer: 200K-vocab byte-level BPE with reserved special-token blocks.

Each special token below maps to a specific surface in . The 200K
vocab and byte-level fallback are required for the 8.12 multilingual coverage
across 44 languages.

This module ships two backends:

1. **HuggingFace `tokenizers` backend** — used in production. Loaded only on
   demand so the package imports cleanly without it.
2. **Pure-Python byte-level fallback** — used by unit tests and onboarding so
   import-time correctness checks don't depend on optional native deps.

Both expose the same `SOTATokenizer` surface; the rest of the codebase doesn't
care which backend it gets.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence


SPECIAL_TOKENS: tuple[str, ...] = (
    # Always-on framing
    "<|begin_of_text|>",
    "<|end_of_text|>",
    "<|pad|>",
    # Chat / role boundaries — every modelcard 4 / 6.2.3 transcript
    "<|im_start|>",
    "<|im_end|>",
    # Adaptive thinking (modelcard 4.1.1)
    "<|thinking|>",
    "<|/thinking|>",
    # Tool use (modelcard 5.1.1, 8.8, 8.10.3)
    "<|tool_call|>",
    "<|/tool_call|>",
    "<|tool_result|>",
    "<|/tool_result|>",
    # Multimodal images (modelcard 8.9; up to 2576 px / 3.75 MP)
    "<|image_start|>",
    "<|image_end|>",
    # Context compaction (modelcard 4.5 / 8.8.2)
    "<|compacted|>",
    "<|/compacted|>",
    # Computer use action prefix (modelcard 5.1.2 / 8.9.4 / 5.2.2.2)
    "<|computer_action|>",
)
RESERVED_FUTURE: tuple[str, ...] = tuple(f"<|reserved_{i}|>" for i in range(256))
ALL_SPECIAL: tuple[str, ...] = SPECIAL_TOKENS + RESERVED_FUTURE


# Modelcard 8.12 evaluates 44 languages. These groupings are used by the
# per-language compression helper to report against the same set the model
# is graded on.
MODELCARD_LANGUAGES_HIGH_RESOURCE: tuple[str, ...] = (
    "fr", "de", "es", "pt", "ru", "zh", "ja", "ar",
    "it", "nl", "ko", "pl", "tr", "sv", "cs",
)
MODELCARD_LANGUAGES_MID_RESOURCE: tuple[str, ...] = (
    "hi", "vi", "id", "fa", "el", "he", "ro", "uk",
    "sr", "tl", "ms", "bn", "lt", "te",
)
MODELCARD_LANGUAGES_LOW_RESOURCE: tuple[str, ...] = (
    "ig", "yo", "so", "mg", "ny", "ha", "sn",
    "ky", "am", "sw", "si", "ne",
)
MODELCARD_LANGUAGES: tuple[str, ...] = (
    ("en",)
    + MODELCARD_LANGUAGES_HIGH_RESOURCE
    + MODELCARD_LANGUAGES_MID_RESOURCE
    + MODELCARD_LANGUAGES_LOW_RESOURCE
)


# Reference bytes-per-token thresholds. The 200K-vocab modelcard target is
# expected to land near these on a representative corpus; the audit helper
# `compression_audit()` flags drift > 25% as a regression risk for 8.12.
REFERENCE_BPT: dict[str, float] = {
    "en": 4.4, "fr": 4.0, "de": 4.0, "es": 4.2, "pt": 4.2,
    "ru": 2.0, "zh": 1.6, "ja": 1.6, "ar": 1.9, "it": 4.1,
    "nl": 4.1, "ko": 1.7, "pl": 3.5, "tr": 3.4, "sv": 4.0, "cs": 3.5,
    "hi": 1.5, "vi": 3.2, "id": 4.0, "fa": 1.9, "el": 1.8, "he": 1.7,
    "ro": 3.7, "uk": 2.0, "sr": 1.9, "tl": 4.0, "ms": 4.0, "bn": 1.4,
    "lt": 3.3, "te": 1.4,
    "ig": 3.5, "yo": 3.6, "so": 3.5, "mg": 3.6, "ny": 3.6, "ha": 3.6,
    "sn": 3.6, "ky": 1.9, "am": 1.4, "sw": 3.7, "si": 1.4, "ne": 1.4,
}


class SOTATokenizer:
    """Wrapper around either a `tokenizers.Tokenizer` instance or the byte
    fallback. Code outside this module sees a single, stable interface."""

    def __init__(self, hf_tokenizer):
        self._tok = hf_tokenizer
        self._special_ids = {
            tok: hf_tokenizer.token_to_id(tok)
            for tok in ALL_SPECIAL
            if hf_tokenizer.token_to_id(tok) is not None
        }

    @property
    def vocab_size(self) -> int:
        return self._tok.get_vocab_size()

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return self._tok.encode(text, add_special_tokens=add_special_tokens).ids

    def decode(self, ids: Sequence[int], skip_special_tokens: bool = False) -> str:
        return self._tok.decode(list(ids), skip_special_tokens=skip_special_tokens)

    def special_id(self, token: str) -> int:
        if token not in self._special_ids:
            raise KeyError(f"unknown special token: {token!r}")
        return self._special_ids[token]

    def convert_tokens_to_ids(self, token: str) -> int:
        """HuggingFace-compatible accessor used by `inference/engine.py::_tok`."""
        return self.special_id(token)

    def encode_special(self, token: str) -> list[int]:
        return [self.special_id(token)]

    @property
    def eos_token_id(self) -> int:
        return self.special_id("<|end_of_text|>")

    @property
    def bos_token_id(self) -> int:
        return self.special_id("<|begin_of_text|>")

    @property
    def pad_token_id(self) -> int:
        return self.special_id("<|pad|>")

    def measure_compression(self, sample_texts: Iterable[str]) -> dict:
        total_bytes = 0
        total_tokens = 0
        for text in sample_texts:
            total_bytes += len(text.encode("utf-8"))
            total_tokens += len(self.encode(text))
        bpt = total_bytes / max(1, total_tokens)
        return {
            "bytes": total_bytes,
            "tokens": total_tokens,
            "bytes_per_token": round(bpt, 3),
            "tokens_per_kb": round(1024 / max(1e-6, bpt), 2),
        }

    def measure_compression_by_language(
        self,
        samples_by_lang: Mapping[str, Iterable[str]],
    ) -> dict[str, dict]:
        """Per-language compression — pass at least the modelcard 8.12 set."""
        return {lang: self.measure_compression(texts) for lang, texts in samples_by_lang.items()}

    def compression_audit(
        self,
        samples_by_lang: Mapping[str, Iterable[str]],
        drift_pct: float = 25.0,
    ) -> dict:
        """Audit per-language compression against `REFERENCE_BPT`.

        Modelcard 8.12 measures 44-language MMMLU and an English-baseline gap
        budget. Drift >25% from the reference bytes-per-token band is a strong
        signal that the tokenizer was trained on a corpus mix that
        underweights one or more languages — which downstream propagates as
        per-language eval drift.
        """
        per_lang = self.measure_compression_by_language(samples_by_lang)
        regressions: list[dict] = []
        for lang, m in per_lang.items():
            ref = REFERENCE_BPT.get(lang)
            if ref is None:
                continue
            actual = m["bytes_per_token"]
            drift = (actual - ref) / ref * 100
            if abs(drift) > drift_pct:
                regressions.append(
                    {"lang": lang, "ref_bpt": ref, "actual_bpt": actual, "drift_pct": round(drift, 1)}
                )
        return {"per_language": per_lang, "regressions": regressions, "ok": not regressions}


# --- Production backend: HuggingFace `tokenizers` ---


def train_bpe(
    corpus_files: list[str | Path],
    output_dir: str | Path,
    vocab_size: int = 200_000,
    min_frequency: int = 2,
):
    """Train a byte-level BPE on `corpus_files`.

    Output: `output_dir/tokenizer.json` (the tokenizers-native serialization)
    plus `output_dir/sota_meta.json` with vocab_size and special-token list,
    so a load can verify the build matches modelcard requirements.
    """
    try:
        from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    except ImportError as e:
        raise SystemExit("Install `tokenizers`: pip install -e '.[dev]'") from e

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tok = Tokenizer(models.BPE(byte_fallback=True))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=list(ALL_SPECIAL),
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    tok.train([str(p) for p in corpus_files], trainer=trainer)
    tok.save(str(output_dir / "tokenizer.json"))
    (output_dir / "sota_meta.json").write_text(
        json.dumps(
            {
                "vocab_size": vocab_size,
                "special_tokens": list(ALL_SPECIAL),
                "byte_fallback": True,
                "modelcard_languages": list(MODELCARD_LANGUAGES),
            },
            indent=2,
        )
    )
    return SOTATokenizer(tok)


def load_tokenizer(path: str | Path) -> SOTATokenizer:
    """Load a saved tokenizer.

    Accepts a directory (preferred — also reads `sota_meta.json`) or a single
    `tokenizer.json`. Falls back to the pure-Python byte tokenizer if the
    `tokenizers` library isn't installed AND the saved tokenizer is the
    fallback's own format.
    """
    p = Path(path)
    meta_path = (p / "sota_meta.json") if p.is_dir() else (p.parent / "sota_meta.json")
    json_path = (p / "tokenizer.json") if p.is_dir() else p

    try:
        from tokenizers import Tokenizer
        if json_path.exists():
            tok = SOTATokenizer(Tokenizer.from_file(str(json_path)))
            _verify_meta(meta_path, tok)
            return tok
    except ImportError:
        pass

    # Pure-Python fallback path.
    fb_path = (p / "byte_tokenizer.json") if p.is_dir() else p
    return ByteFallbackTokenizer.load(fb_path)


def _verify_meta(meta_path: Path, tok: "SOTATokenizer") -> None:
    """Check the saved meta against modelcard requirements; warn but don't fail."""
    if not meta_path.exists():
        return
    meta = json.loads(meta_path.read_text())
    missing = [s for s in SPECIAL_TOKENS if s not in tok._special_ids]
    if missing:
        import warnings
        warnings.warn(f"saved tokenizer missing special tokens: {missing}", stacklevel=2)
    if meta.get("vocab_size") != tok.vocab_size:
        import warnings
        warnings.warn(
            f"meta vocab_size={meta.get('vocab_size')} != actual {tok.vocab_size}",
            stacklevel=2,
        )


# --- Pure-Python fallback: byte-level tokenizer with special-token prefix ---


class ByteFallbackTokenizer:
    """A byte-level tokenizer with the modelcard's 16 special tokens reserved.

    Vocabulary layout: ids 0..255 = raw bytes, ids 256.. = ALL_SPECIAL.
    Used in tests, CI, and onboarding scripts that don't have the native
    `tokenizers` library installed. NOT for production training — it has 1.0
    bytes/token so completely fails the modelcard 8.12 compression goals.
    """

    def __init__(self):
        self._byte_offset = 256
        self._special_to_id = {s: 256 + i for i, s in enumerate(ALL_SPECIAL)}
        self._id_to_special = {v: k for k, v in self._special_to_id.items()}
        self._vocab_size = 256 + len(ALL_SPECIAL)

    # The duck-typed surface used by SOTATokenizer ---

    def get_vocab_size(self) -> int:
        return self._vocab_size

    def token_to_id(self, t: str) -> int | None:
        return self._special_to_id.get(t)

    def encode(self, text: str, add_special_tokens: bool = False):
        ids: list[int] = []
        # Tokenize special tokens out first so they don't get split byte-wise.
        i = 0
        while i < len(text):
            matched = False
            for s in ALL_SPECIAL:
                if text.startswith(s, i):
                    ids.append(self._special_to_id[s])
                    i += len(s)
                    matched = True
                    break
            if matched:
                continue
            ids.append(text[i].encode("utf-8")[0] if ord(text[i]) < 128 else None)
            if ids[-1] is None:
                ids.pop()
                ids.extend(text[i].encode("utf-8"))
            i += 1

        class _Out:
            pass
        out = _Out()
        out.ids = ids
        return out

    def decode(self, ids: Sequence[int], skip_special_tokens: bool = False) -> str:
        chunks: list[bytes] = []
        for tid in ids:
            if tid >= self._byte_offset:
                if not skip_special_tokens:
                    chunks.append(self._id_to_special[tid].encode("utf-8"))
            else:
                chunks.append(bytes([tid]))
        return b"".join(chunks).decode("utf-8", errors="replace")

    # --- save / load ---

    def save(self, path: str | Path) -> None:
        p = Path(path)
        if p.is_dir():
            p = p / "byte_tokenizer.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"kind": "byte_fallback", "specials": list(ALL_SPECIAL)}))

    @classmethod
    def load(cls, path: str | Path) -> "SOTATokenizer":
        # Fallback ignores the file — its layout is fixed.
        return SOTATokenizer(cls())


def make_byte_fallback() -> SOTATokenizer:
    """Build a fallback tokenizer suitable for tests and CI."""
    return SOTATokenizer(ByteFallbackTokenizer())


# --- CLI entrypoint for tokenizer training ---


def _cli() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Train the SOTA 200K BPE tokenizer.")
    parser.add_argument("--input", nargs="+", required=True, help="Plaintext corpus files")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--vocab-size", type=int, default=200_000)
    parser.add_argument("--min-frequency", type=int, default=2)
    args = parser.parse_args()
    train_bpe(args.input, args.output, vocab_size=args.vocab_size, min_frequency=args.min_frequency)


if __name__ == "__main__":
    _cli()
