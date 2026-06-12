# `sota_model/tokenizer/`

Separate folder because tokenizer choices show up everywhere: data-packing throughput, training wall-clock, inference TTFT, KV-cache memory, serving cost-per-token. Getting it wrong invalidates everything else.

```
__init__.py    re-exports
bpe.py         SOTATokenizer + ByteFallbackTokenizer + SPECIAL_TOKENS + MODELCARD_LANGUAGES
               + REFERENCE_BPT + train_bpe + load_tokenizer + compression_audit + CLI
```

## What ships

- `SOTATokenizer` — thin wrapper around either the HuggingFace `tokenizers` library OR the pure-Python `ByteFallbackTokenizer`. Code outside this module sees a single, stable interface.
- `ByteFallbackTokenizer` — pure-Python byte-level tokenizer with the modelcard's 16 special tokens reserved. **Not for production training** (1.0 bytes/token, completely fails 8.12), but it lets unit tests, CI, and onboarding scripts import the package without a native deps install.
- `MODELCARD_LANGUAGES` — the 44-language set from modelcard 8.12, split into high / mid / low resource tiers.
- `REFERENCE_BPT` — expected `bytes_per_token` band per language for a 200K-vocab tokenizer; used by `compression_audit` to flag drift > 25% as a regression risk.
- `train_bpe(corpus_files, output_dir, vocab_size=200_000)` — production training. Saves `tokenizer.json` plus `sota_meta.json` (modelcard-required special-token list, vocab size, language coverage) so loaders can verify the artifact matches expectations.
- `load_tokenizer(path)` — accepts a directory (preferred — also reads `sota_meta.json`) or a single `tokenizer.json`. Falls back to `ByteFallbackTokenizer` when `tokenizers` isn't installed.
- `measure_compression`, `measure_compression_by_language`, `compression_audit` — cost-analysis helpers.
- `make_byte_fallback()` — explicit factory for the pure-Python fallback.

## CLI

```
python -m sota_model.tokenizer.bpe \
    --input shard1.txt shard2.txt ... \
    --output ./tokenizer \
    --vocab-size 200000 \
    --min-frequency 2
```

## Why 200K vocab

| Vocab | Embed+Head bytes | Notes |
|---|---|---|
| 50K  | ~3.3 GB | bad multilingual compression — fails 8.12 target |
| 128K | ~8.4 GB | underweights low-resource languages |
| **200K** | **~13.1 GB** | **modelcard target — best multilingual ROI** |
| 256K | ~16.8 GB | diminishing returns |

The 200K target is implied by modelcard 8.12 (44-language eval) and the published low-resource gap-to-English in 8.12.1.

## Cost in training

Compression ratio multiplies straight into FLOPs:

- 200K on a mixed 60 TB corpus: ~3.8 bytes/token → ~15.8T tokens
- 50K on the same corpus: ~3.5 bytes/token → ~17.1T tokens (+8.5%)

That extra 8.5% is per-stage; across the 22-week schedule it costs **tens of thousands of dollars** of GPU time wasted on a worse tokenizer.

## Cost in inference

Compression ratio is the **dominant inference cost** of a worse tokenizer:

- TTFT — tokenizer encode is <1% of TTFT; not the bottleneck.
- Throughput — a 50%-worse compression ratio means **50% more output tokens per user-visible character**, which translates straight into latency and dollars.
- KV-cache memory — sized in tokens, but better compression means more user text fits in the same 1M-token budget.

## Operator checklist

- [ ] Train on the cleaned corpus from `scripts/pipelines/01_clean_data.py`, not raw.
- [ ] Vocab size = 200,000.
- [ ] All 16 SPECIAL_TOKENS present in saved tokenizer (the saved `sota_meta.json` lists them — `load_tokenizer` warns on mismatch).
- [ ] Byte-fallback enabled in `train_bpe(...)` (it is by default).
- [ ] `compression_audit(samples_by_lang)` returns `ok=True` for the modelcard-8.12 sample set; investigate any per-language `regressions` entries.
- [ ] `serving/server.py` and `scripts/pipelines/02_tokenize_and_pack.py` updated to call `load_tokenizer(path)` with the new tokenizer directory (replacing the `gpt2` placeholder).
- [ ] Confirm `ByteFallbackTokenizer` is NOT being used in production paths — its 1.0 bytes/token would silently inflate KV-cache cost and tank 8.12 numbers.
