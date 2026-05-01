# Building a SOTA Dense Foundation Model — SuperModel 4.7-Class

> **Source of truth:** [``](./). Every numerical target, evaluation, safeguard, and capability claim in this document is derived from that system card. When this README and `` disagree, `` wins.

This guide describes how to build a **dense (non-MoE) transformer** that matches the **SuperModel 4.7 capability envelope**: 87.6% SWE-bench Verified, 94.2% GPQA Diamond, 75.83% ARC-AGI-2, 78.0% OSWorld, 1M-token context, native multimodal, adaptive thinking, agentic coding and computer-use harnesses, and a passing RSP risk profile.

It covers, in order:
1. [Transformer architecture](#1-transformer-architecture)
2. [Python pretraining pipeline](#2-python-pretraining-pipeline)
3. [Post-training: SFT, RM, RLHF, Constitutional AI](#3-post-training)
4. [KV-cache handling and inference optimization](#4-kv-cache-handling-and-inference-optimization)
5. [Deployment steps](#5-deployment-steps)
6. [Evaluation suite (matching  8)](#6-evaluation-suite)
7. [Safety, RSP, and alignment](#7-safety-rsp-and-alignment)
8. [Welfare-aware training and monitoring](#8-welfare-aware-training-and-monitoring)

---

## Target SOTA scoreboard (from `` 8.1)

These are the numbers a successful build must approach. Treat them as gates, not stretch goals.

| Domain | Benchmark | Target |
|---|---|---|
| Software engineering | SWE-bench Verified | ≥ 87% |
| Software engineering | SWE-bench Pro | ≥ 64% |
| Software engineering | SWE-bench Multilingual | ≥ 80% |
| Software engineering | SWE-bench Multimodal | ≥ 34% |
| Terminal/agentic | Terminal-Bench 2.0 | ≥ 69% |
| Reasoning | GPQA Diamond | ≥ 94% |
| Reasoning | USAMO 2026 | ≥ 69% |
| Reasoning | ARC-AGI-2 (max thinking) | ≥ 75% |
| Long context | OpenAI MRCR v2 (1M, 8 needles) | competitive |
| Long context | GraphWalks BFS 256K–1M | ≥ 58% |
| Agentic search | HLE (no tools) | ≥ 46% |
| Agentic search | HLE (with tools) | ≥ 54% |
| Agentic search | DeepSearchQA F1 | ≥ 89% |
| Agentic search | BrowseComp | ≥ 79% |
| Multimodal | OSWorld | ≥ 78% |
| Multimodal | ScreenSpot-Pro (with tools) | ≥ 87% |
| Multimodal | CharXiv Reasoning (with tools) | ≥ 91% |
| Multimodal | LAB-Bench FigQA (with tools) | ≥ 86% |
| Professional | OfficeQA / OfficeQA Pro | ≥ 86% / ≥ 80% |
| Professional | Finance Agent (Vals AI) | ≥ 64% |
| Professional | MCP-Atlas | ≥ 77% |
| Multilingual | MMMLU avg | ≥ 91% |
| Multilingual | MILU avg | ≥ 89% |
| Life sciences | Structural biology MCQ / open | ≥ 98% / ≥ 74% |
| Safeguards | Single-turn harmless rate | ≥ 97.9% |
| Safeguards | Election integrity (violative) | 100% |
| Agentic safety | ART benchmark @k=100 | ≤ 6% ASR |
| Agentic safety | Browser use w/ safeguards | 0% ASR |

Adjacent constraints from ``:
- **Adaptive thinking** must be a first-class mode (effort decided per query, not just on/off).
- **Context window** ≥ 1M tokens with **context compaction** triggering around 200k.
- **Multimodal:** images up to **2576 px on a single dimension, 3.75 MP total** (3.3× the previous-generation cap).
- **Constitutional Classifiers** for CBRN content + **probe-based classifiers** for cyber misuse.
- **No accidental chain-of-thought supervision** in training data (the modelcard documents this affecting 7.8% of episodes — fix it).

---

## 1. Transformer Architecture

A dense decoder-only transformer at the **frontier-dense band** — ~427B active parameters at the YAML defaults, in the 400–700B band that 2026 frontier-dense (Llama 4 dense backbone, Claude Opus 4.x, Gemini Ultra dense subset) lives in. All parameters active at inference (no MoE routing).

### 1.1 Topology

```
Hidden dimension (d_model):        16,384
Layers:                            110             (frontier dense 2026: 100–180)
Attention heads (Q):               128
Attention heads (KV, GQA):         16              (8:1 group ratio)
Head dimension:                    128
FFN dimension:                     65,536          (4 × d_model SwiGLU)
FFN activation:                    SwiGLU
Vocabulary:                        200,000         (covers modelcard 8.12 44 langs)
Tied embeddings:                   no
RMSNorm (pre-norm) on attn + FFN
Position encoding:                 RoPE, base 10⁶ for long context
Long-context extension:            YaRN scaling to 1M tokens
Per-layer heterogeneity:           expressible via ModelConfig.layer_overrides
                                   (frontier dense is NOT uniform — see 1.2.1)
Dtype (training):                  FP8 mixed precision (E4M3 fwd / E5M2 bwd)
                                   bf16 fallback on pre-Blackwell hardware
Dtype (inference):                 FP8 KV cache by default; bf16/int8/fp4 also supported
Total params at the YAML defaults: ≈ 427 B (run cfg.estimate_params_billions())
```

### 1.1.1 The 2026 frontier-dense design space

`` doesn't pin numbers for `d_model`, `n_layers`, `n_kv_heads`, `ffn_dim`, optimizer, precision, total params, training tokens, or sampling. Only **7** numerical/structural choices are modelcard-pinned (the 1M context, the 2576px / 3.75 MP image cap, the 200K compaction trigger, and adaptive thinking + tier names); the rest are **operator-committed**. The `model:` and `training:` sections of `configs/sota_4_7.yaml` are calibrated against published 2026 frontier-dense practice.

The table below is the same one in the YAML preamble — it's the working set of permutations to consider when adjusting. The full sensitivity / combination analysis (param-count cross-products, KV-cache memory by `n_kv_heads × dtype`, training-FLOPs by `P × tokens`, worked recipes for landing 200B/427B/600B/1T) is in **[`docs/PERMUTATIONS.md`](./docs/PERMUTATIONS.md)**.

| Knob | 2024 typical | 2026 frontier band | This repo | Notes |
|---|---|---|---|---|
| `d_model` | 12,288–16,384 | 14,336–20,480 | 16,384 | capability ceiling scales with width |
| `n_layers` | 80–126 | 100–180 | 110 | depth helps reasoning; cost is wall-clock |
| `n_q_heads` | 64–128 | 96–160 | 128 | scales with d_model; head_dim=128 stable |
| `n_kv_heads` (GQA) | 8–16 | 8–16 (or MLA) | 16 | 8:1 group ratio; KV memory bound |
| `head_dim` | 128 | 128 | 128 | stable across the band |
| `ffn_dim` (× d_model) | 3.0–4.0 | 3.5–4.0 | 4.0 | SwiGLU; tapering across depth is modern |
| Total params (B, dense) | 100–500 | 400–700 | 427 | DeepSeek-V3 is 671B MoE; Llama 4 ~2T MoE → frontier *dense* sits at 400–700B |
| Training tokens (T) | 10–18 | 20–40 | 25 | Chinchilla-overtrained 5–8× at this scale |
| Train precision | bf16 | FP8 mixed (TE) | fp8 | DeepSeek-V3 + Llama 4 shipped FP8 native — see [`docs/PRECISION.md`](./docs/PRECISION.md) |
| Serve precision | bf16 / int8 | fp8 / int8 / fp4 | fp8 | Blackwell native; fp4 for memory-bound |
| Optimizer | AdamW | AdamW (β2=0.95) | adamw | Distributed Shampoo + Muon emerging |
| Global batch (tok) | 2–8 M | 4–8 M | 4 M | stable with ZeRO-3 |
| Long context | 128K–1M | 1M (10M agentic) | 1M | compaction at 200K → 10M effective |
| `temperature` | 0.6–0.7 | 0.6–0.7 | 0.7 | lower for reasoning paths |
| `top_p` | 0.9–0.95 | 0.9–0.95 | 0.95 | stable |
| Image long edge | 1568 px (1 MP) | 2576 px (3.75 MP) | 2576 px | modelcard invariant 6 |

### 1.2 Why dense, not MoE — and why coding capability isn't dead weight

`` describes a frontier model evaluated as a single dense inference unit (no routing latency, no expert balancing pathologies). Dense at this scale gives:
- Predictable latency under tool-use and adaptive-thinking modes.
- Cleaner attribution for white-box safety analysis (SAE features, evaluation-awareness probes — see modelcard 6.5).
- No load-balancing distortions during long agentic sessions (Claude Code, computer use).

**Is coding capability "dead weight" for non-coding queries?** No. In a dense model, every token routes through all ~427B parameters. Coding training does not add parameters — it shapes the existing ones. The same weights that compute "what's a quicksort partition step?" also compute "what's the capital of Norway?" — the model has learned a single representation space that handles both. This is exactly why dense models hit SWE-bench Verified ≥87% (modelcard 8.1) without sacrificing GPQA Diamond ≥94% on graduate-level science.

By contrast, in MoE models, code tokens often route to a "code expert" while general-knowledge tokens route elsewhere — so an inference-time conversation that's mostly chat will under-utilize the code experts (and vice versa). Dense avoids that.

The cost is GPU-hours per inference; that's why deployment minimums are 8× A100-80GB at int8 and 16× H100-80GB at bf16 (or 4× B200 at fp8 on Blackwell).

### 1.2.1 Per-layer heterogeneity (not all layers are the same)

Frontier-dense in 2026 is **not uniform across depth**. Two heterogeneities the code supports today via `ModelConfig.layer_overrides` (sparse per-layer dict, KV-cache-compatible):

- **Per-layer `ffn_dim`** — taper FFN width across depth. Standard frontier convention is wider FFN at the network's edges (where representation-shaping load concentrates) and narrower in the middle (where layers refine representations and the marginal capacity gain is lower). This is the cleanest way to land a precise param target without changing `d_model` or `n_layers`.
- **Per-layer `sliding_window`** — explicit override of the stride-based pattern. Pin specific layers to full attention (helps long-context retrieval evals MRCR / GraphWalks) or windowed attention (saves KV memory). Use `null` to force full attention; a positive int to set a custom window.

Two presets ship in `src/sota_model/config.py`:

```python
from sota_model.config import (
    ModelConfig,
    tapered_ffn_overrides,
    hybrid_attention_overrides,
)

cfg = ModelConfig(
    layer_overrides={
        # Wider FFN at the first/last 4 layers, narrower in the middle 102.
        **tapered_ffn_overrides(110, edge_layers=4, middle_ffn_dim=49152),
        # Force layer 0 and layer -1 to full attention regardless of stride.
        **hybrid_attention_overrides(110, full_attention_layers=(0, -1)),
    }
)
print(f"{cfg.estimate_params_billions():.1f} B")
for row in cfg.per_layer_param_breakdown()[:3]:
    print(row)
```

Per-layer `n_kv_heads` and `head_dim` are **NOT** supported by `layer_overrides` because the current paged KV cache assumes a uniform `(n_layers, n_kv_heads, head_dim)` shape; relaxing that needs a cache redesign and is left as a future op.

**Param-count effect of tapering (worked example).** Starting from the YAML defaults (uniform, ~427B):

| Override | Result | Δ from uniform |
|---|---|---|
| `tapered_ffn_overrides(110, edge_layers=4, middle_ffn_dim=49152)` (75% of edge) | ~345 B | −82 B |
| `tapered_ffn_overrides(110, edge_layers=4, middle_ffn_dim=32768)` (50% of edge) | ~263 B | −164 B |
| `tapered_ffn_overrides(110, edge_layers=4, middle_ffn_dim=24576)` (37.5%) | ~222 B | −205 B |
| `hybrid_attention_overrides(110, (0, -1))` (structural; no param change) | ~427 B | 0 |

Tapering alone can land any param count between ~427B (uniform) and ~150B (very aggressive). Below ~200B you typically also drop `d_model` or `n_layers` rather than continue narrowing FFN, because over-narrow FFN starts to bottleneck capacity at depth.

### 1.3 Attention: GQA + Flash Attention + sliding window

Grouped-Query Attention (GQA) is non-negotiable. With 128 Q heads and 8–16 KV heads:
- **8× to 16× reduction** in KV-cache memory (the dominant cost at 1M context).
- Lossless or near-lossless on benchmarks.

```python
# nn.Module sketch
class GQA(nn.Module):
    def __init__(self, d_model=16384, n_q=128, n_kv=16, head_dim=128):
        self.q = nn.Linear(d_model, n_q * head_dim, bias=False)
        self.k = nn.Linear(d_model, n_kv * head_dim, bias=False)
        self.v = nn.Linear(d_model, n_kv * head_dim, bias=False)
        self.o = nn.Linear(n_q * head_dim, d_model, bias=False)
        self.n_groups = n_q // n_kv  # 8

    def forward(self, x, kv_cache, rope, attn_mask):
        q = self.q(x).view(..., n_q, head_dim)
        k = self.k(x).view(..., n_kv, head_dim)
        v = self.v(x).view(..., n_kv, head_dim)
        q, k = apply_rope(q, k, rope)
        k, v = kv_cache.update(k, v)        # see 4
        # repeat KV across the group axis (no copy in FA3)
        out = flash_attn_with_kvcache(q, k, v, attn_mask=attn_mask)
        return self.o(out)
```

Use **Flash Attention 3** (Hopper) or **Flash Attention 2** (Ampere). For the long-context (1M) regime layer in some training stages, mix in **sliding-window attention** (window 8K–32K) on a subset of layers — fully causal global attention is preserved on the rest so global retrieval (MRCR v2, GraphWalks 1M) still works.

### 1.4 Adaptive thinking blocks

Per `` 4.1.1, adaptive thinking is a first-class mode where the **model itself** decides effort per query. Architecturally this means:

- A reasoning channel that emits `<thinking>...</thinking>` tokens not shown to the user.
- A learned **effort head** (small MLP from late-layer hidden states to a scalar logit) that gates how many additional thinking tokens to emit before answer generation.
- Token budgets per effort level (`min`, `low`, `medium`, `high`, `max`). Public API exposes effort; internal training shapes the policy that selects it.
- During RL post-training the model is rewarded for matching effort to problem difficulty (cheap problems → short thinking; hard math/code → long thinking).

Public API contract: thinking tokens are not billed against the user but **count toward the context window**. They must round-trip back into the next turn for tool-use chains (`` notes this is required for stable agentic behavior).

### 1.5 Multimodal vision encoder

For SuperModel 4.7-class image inputs:

```
Image preprocessing  (modeling/vision/encoder.py::preprocess_image)
    Native resize to ≤ 2576 px (long edge), ≤ 3.75 MP total
    Patch size:        14 × 14
    Tile splitting:    images > 1568 px split into overlapping tiles

Vision tower         (modeling/vision/encoder.py::VisionEncoder)
    ViT-style: RMSNorm + GQA + SwiGLU, same building blocks as the LM
    bf16 training; jointly fine-tuned with the LM at stage 3
    Trained with SigLIP-style contrastive loss + caption next-token loss

Connector            (modeling/vision/projector.py::VisionLanguageProjector)
    Pixel-shuffle 2×2 + two-layer MLP into LM d_model
    Linear-projection mode is also exposed for ablation
```

The resolution bump from 1568 px → 2576 px is what unlocks ScreenSpot-Pro 79.5% (no tools) → 87.6% (with tools) and LAB-Bench FigQA 78.6% → 86.4%. Don't compromise on it.

**How it ties in.** `SOTAModel` builds a `VisionEncoder + VisionLanguageProjector` whenever `ModelConfig.vision_enabled=True`. `SOTAModel.encode_image(img)` returns `(n_image_tokens, d_model)` rows that the inference engine splices into the prompt at the `<|image_start|>...<|image_end|>` placeholder block (see [`src/sota_model/modeling/vision/README.md`](./src/sota_model/modeling/vision/README.md) for the splice contract).

### 1.6 Message format and chat template

Single source of truth: `src/sota_model/inference/chat_template.py`. The same format is used in training data (`data/samples/chat.jsonl`, `data/samples/tool_use.jsonl`) and at inference.

Special tokens (must exist in the tokenizer):

```
<|im_start|>      <|im_end|>          role boundaries
<|thinking|>      <|/thinking|>       hidden reasoning channel — not user-visible
<|tool_call|>     <|/tool_call|>      emitted by assistant
<|tool_result|>   <|/tool_result|>    echoed back from tool runtime
<|image_start|>   <|image_end|>       multimodal placeholder for vision encoder
<|compacted|>     <|/compacted|>      inserted by 4.5 context compaction
```

Roles: `system`, `user`, `assistant`, `tool`. A rendered conversation looks like:

```
<|im_start|>system
You are a helpful assistant.

Available tools:
- web_search: {"type":"object","properties":{"q":{"type":"string"}}}
<|im_end|>
<|im_start|>user
Who won the 2026 Australian Open men's singles?<|im_end|>
<|im_start|>assistant
<|thinking|>Need a recent fact; search.<|/thinking|>
<|tool_call|>{"name":"web_search","arguments":{"q":"2026 Australian Open men's singles winner"}}<|/tool_call|><|im_end|>
<|im_start|>tool
<|tool_result|>{"name":"web_search","content":"...result..."}<|/tool_result|><|im_end|>
<|im_start|>assistant
The winner was X.<|im_end|>
```

Training-loss masking: loss is computed on **assistant turns only** (including any `<|thinking|>` block and any `<|tool_call|>` block). System, user, and tool turns are loss-masked.

### 1.7 Tool use and computer use

Single source of truth: `src/sota_model/inference/tools.py`.

**Protocol.** The model emits one or more `<|tool_call|>{...}<|/tool_call|>` blocks in an assistant turn. The runtime parses them, dispatches concurrently (asyncio), and returns one `<|tool_result|>...<|/tool_result|>` block per call as a `tool` role message. The model then continues.

**Catalog.** Each tool is registered with a name, a JSON Schema for arguments, and a Python callable. The catalog is rendered into the system message at the top of every conversation so the model sees what's available.

**Parallel calls.** A single assistant turn may emit multiple `<|tool_call|>` blocks; the runtime fans them out concurrently and returns all results before the next assistant turn. See `data/samples/tool_use.jsonl::tool-004-parallel` for a worked example.

**Computer use specifics.** Coordinates emitted as native floats in `[0,1]` normalized space (e.g. `<click x="0.5234" y="0.8121"/>`), never raw pixels — keeps precision at 4K and above. Screenshot tokens go through the vision encoder (1.5).

**Tokenizer requirement.** The tokenizer must reserve a dedicated rare-token block for the tool-call delimiters and structured-arg keys (`name`, `arguments`) so JSON doesn't fragment into hundreds of BPE tokens.

**Built-in implementations** (`src/sota_model/inference/sandbox/`):
- `code_exec` — `CodeExecSandbox` runs Python/bash in a subprocess with `setrlimit` mem/CPU/file/proc caps, scrubbed environment, and timeout. Documented as best-effort soft isolation; hardening to gVisor / firecracker / Docker+seccomp is operator-supplied.
- `web_search` / `web_fetch` — `AllowlistedWebTool` enforces a domain allowlist, mirrors the modelcard 9.2 contamination URL denylist, and persists results to a JSON cache for deterministic evals. Defaults to **offline** unless an operator-supplied `searcher` / `fetcher` callable is plugged in.

**Safety.** Every tool call passes through the 7.1 pre-model and post-model gates. The default gate is now `safety.probes.build_default_probe_gate()`, which composes:
- `LinearProbeClassifier` for prohibited / high-risk-dual / dual-use cyber (calibrated thresholds).
- `ConstitutionalClassifier` for CBRN content, with the keyword backend as a fail-closed fallback.
The legacy keyword stub is preserved as `safety.classifiers.default_safety_gate()` for environments where probe weights cannot be loaded.

---

## 2. Python Pretraining Pipeline

### 2.1 Stack

```python
# Core
torch >= 2.6                       # torch.compile, FlexAttention, fp8 support
deepspeed >= 0.15                  # ZeRO-3, gradient checkpointing
flash-attn >= 3.0                  # Hopper / Blackwell kernels
transformer-engine >= 1.13         # FP8 mixed-precision training (Blackwell native)
transformers >= 4.45               # tokenizers, model utilities
datasets >= 2.20
tokenizers >= 0.20

# Distributed / orchestration
ray[default] >= 2.30
megatron-lm                        # tensor + pipeline parallel
accelerate

# Observability
wandb
tensorboard
nvidia-dcgm-exporter               # GPU telemetry
```

The 2026 frontier-dense path uses **NVIDIA Transformer Engine** for FP8 mixed-precision training (E4M3 forward / E5M2 backward), with bf16 master weights. On Blackwell B200/B300 this delivers ~2× the throughput of bf16-on-Hopper at the same FLOP cost. The bf16 path remains supported as the fallback for Ampere/Hopper-only stacks — set `training.mixed_precision: bf16` in the YAML.

Full reference for every precision format used here (fp32, bf16, fp16, fp8 E4M3/E5M2, int8, fp4 / MXFP4 / MXFP6 / MXFP8) plus a hardware-coverage matrix (Blackwell, Hopper, Ampere, Google TPU, ARM) and a "which precision should I pick?" flowchart: [`docs/PRECISION.md`](./docs/PRECISION.md).

### 2.2 Pretraining data: composition, size, and time

**Total volume:** **20–40 trillion tokens** of pretraining corpus (default in `configs/sota_4_7.yaml::implied_training_corpus.total_tokens_trillions = 25`). At a global batch of 4M tokens/step, that's roughly **5M to 10M optimizer steps** before any post-training.

This is the 2026 frontier-dense band. Reference points: DeepSeek-V3 trained on 14.8T (Dec 2024); Llama 4 trained on ~30T (early 2025). Chinchilla-optimal at 427B params is ~10T, but frontier-dense overtrains 5–8× because compute is cheaper than the marginal eval gain at the optimal point.

Aligned with `` 1.1.1: "proprietary mix of publicly available internet information, public and private datasets, and synthetic data generated by other models."

| Bucket | % of tokens | Approx tokens (at 25T total) | Sources |
|---|---|---|---|
| Web | 45% | ~11.3T | CommonCrawl filtered → RefinedWeb / FineWeb-Edu, deduplicated |
| Code | 20% | ~5.0T | GitHub permissive, Stack v2, Stack Overflow, internal repos for SWE-bench-style tasks |
| Academic | 10% | ~2.5T | arXiv, PubMed Central, Semantic Scholar |
| Books / reference | 10% | ~2.5T | Project Gutenberg, Wikipedia, technical manuals |
| Math, structured | 10% | ~2.5T | OpenWebMath, AlgebraicStack, scientific datasets, reasoning traces |
| Dialogue / instructions | 5% | ~1.3T | Synthetic conversations, instruction datasets, tool-use traces |

**Wall-clock time (1024–2048 H100s):** ~**3–6 months** for full three-stage pretraining. Roughly 70% of compute in Stage 1 (foundation), 20% in Stage 2 (long context), 10% in Stage 3 (refinement). Post-training (SFT → RM → RLHF → Constitutional AI) adds another **~6–8 weeks**.

**FLOPs budget:** ~10²⁵–10²⁶ training FLOPs. At 427B dense parameters and 25T tokens, the standard `6 × N × D` estimate gives `6 × 4.27×10¹¹ × 2.5×10¹³ ≈ 6.4×10²⁵` FLOPs — in the band. FP8 mixed-precision on Blackwell delivers this at ~2× the throughput of bf16-on-Hopper.

**Cost (compute alone, before salaries / data licensing):** **$10M–$50M** depending on the H100 hourly price and exact cluster utilization.

**Crawler discipline (`` 1.1.1):** A general-purpose crawler analogous to ClaudeBot must (a) honor `robots.txt`, (b) skip auth-walled pages, (c) skip CAPTCHA, (d) be transparently identifiable in user-agent strings.

**Sample data shipped in this repo** (illustrative only — not training data):

```
data/samples/
    pretrain.jsonl        web + book + technical text (~8 docs)
    code.jsonl            Python/Rust/SQL snippets (~5 docs)
    multilingual.jsonl    fr/de/es/hi/zh examples (~5 docs)
    chat.jsonl            multi-turn assistant conversations (~5 traces)
    tool_use.jsonl        web_search / code_exec / file IO traces (~4 traces)
    contamination.jsonl   adversarial samples that MUST be filtered (~7 docs)
```

See [`data/README.md`](./data/README.md) for the JSONL schemas and how the test suite consumes them.

### 2.3 Data pipeline

The flow from raw crawl to packed training shard:

```
                     ┌──────────────────────────────────────┐
                     │  RAW SOURCES                         │
                     │  CommonCrawl  GitHub  arXiv  Books   │
                     │  PubMed  Wikipedia  Stack  ...       │
                     └──────────────────┬───────────────────┘
                                        │
                                        ▼
                     ┌──────────────────────────────────────┐
                     │  CRAWLER (robots.txt-respecting)     │
                     │  ClaudeBot-style identifiable UA     │
                     └──────────────────┬───────────────────┘
                                        │
                                        ▼
   ┌────────────────────────────────────────────────────────────────────┐
   │  FILTER PIPELINE  (data/training/data.py — runs in this order)     │
   │                                                                    │
   │  ① LanguageDetector       drop docs outside accepted lang set      │
   │  ② MinLengthFilter        drop docs < 100 chars                    │
   │  ③ DuplicateRemover       MinHash-LSH @ Jaccard 0.85 over 5-grams  │
   │  ④ QualityScorer          drop docs below 0.7 quality threshold    │
   │  ⑤ ToxicityFilter         drop CSAM-adjacent / NSII content        │
   │  ⑥ PIIRedactor            emails → [EMAIL]  ssns → [SSN]  ...      │
   │  ⑦ BenchmarkContamination drop modelcard 9.2 URLs + eval text     │
   └────────────────────────────────┬───────────────────────────────────┘
                                    │
                                    ▼
                     ┌──────────────────────────────────────┐
                     │  TOKENIZER  (200K BPE, byte fallback)│
                     │  Reserved special-token blocks:      │
                     │  <|im_start|> <|tool_call|> ...      │
                     └──────────────────┬───────────────────┘
                                        │
                                        ▼
                     ┌──────────────────────────────────────┐
                     │  BlockPacker  (no cross-doc attention)│
                     │  seq_len = 8192  →  32K  →  1M       │
                     │  reset_mask = True                   │
                     └──────────────────┬───────────────────┘
                                        │
                                        ▼
                     ┌──────────────────────────────────────┐
                     │  PARQUET SHARDS in object storage    │
                     │  (~25T tokens after filtering)       │
                     └──────────────────┬───────────────────┘
                                        │
                                        ▼
                     ┌──────────────────────────────────────┐
                     │  3-STAGE TRAINING (2.4)             │
                     │  Stage 1 → Stage 2 → Stage 3         │
                     └──────────────────────────────────────┘
```

In code: `PretrainingPipeline` in `src/sota_model/training/data.py` chains these filters; `BlockPacker` packs the resulting token streams into fixed-length blocks with reset masks. The end-to-end loader that ties filters + tokenizer + packer + the `implied_training_corpus.source_mix_pct` ratio together lives in `src/sota_model/training/corpus.py::CorpusLoader`. The `--smoke` flag on `sota-pretrain` keeps the random-token dummy iterator for CI wiring tests; production runs pass `--data-root` and `--tokenizer`.

The heuristic stubs `QualityScorer` / `ToxicityFilter` / `LanguageDetector` are now thin shells over a swappable `backend`. Trained replacements live in `src/sota_model/training/classifiers/`:

```
training/classifiers/
    base.py        HashingTextVectorizer  + LogisticTextClassifier  + train_logistic
    quality.py     HeuristicQualityScorer (legacy)  + TrainedQualityScorer
    toxicity.py    BlocklistToxicityFilter (operator JSON blocklists)  + TrainedToxicityFilter
    language.py    CharNgramLanguageDetector (script-family)  + TrainedLanguageDetector
```

All three trained classifiers share the `(doc) -> doc | None` contract and drop straight into `PretrainingPipeline.filters`. See [`src/sota_model/training/classifiers/README.md`](./src/sota_model/training/classifiers/README.md).

**Benchmark contamination scrubbing is mandatory.** `` 8.8.1 (HLE) and 9.2 publish a blocklist of known evaluation-discussing URLs — replicate that blocklist and screen all training shards. The same applies to GPQA, MMLU, USAMO problems, SWE-bench instances. The `BenchmarkContaminationFilter` in `data.py` ships with the modelcard 9.2 URL list as its default; extending is a one-line change, narrowing it is forbidden.

**Synthetic data:** controlled at `` 2.4.1. Supervise carefully; the cited bug where chain-of-thought was accidentally supervised in 7.8% of episodes is the kind of mistake that quietly breaks alignment evals — instrument the pipeline to detect any path where teacher-CoT could leak into student-CoT supervision.

### 2.4 Three-stage pretraining

Stage transitions are picked by validation loss elbows + downstream benchmark deltas, not fixed step counts.

```
                    Compute share         Wall-clock         Goal
                    ─────────────         ──────────         ────────────────────
   Stage 1          ~70%                  ~8 weeks           Foundation: next-token
   Foundation       seq 8K, lr 3e-4       (Weeks 1–8)        prediction, broad data
   ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
                                       ─►
   Stage 2          ~20%                  ~4 weeks           Long context: extend
   Long context     seq 32K→1M, lr 1e-4   (Weeks 9–12)       RoPE base + YaRN x8
                    ▓▓▓▓▓▓▓▓▓▓▓
                                                          ─►
   Stage 3          ~10%                  ~4 weeks           Quality refinement:
   Refinement       seq 8K, lr 5e-5       (Weeks 13–16)      top-decile data only
                                ▓▓▓▓▓
                                                                              ─►
   Post-training    +6–8 weeks            (Weeks 17–24)      SFT → RM → PPO → CAI
                                                              (see 3)
```

Total: ~16 weeks pretraining + 6–8 weeks post-training = 22–24 weeks (~5–6 months) end-to-end on a 1024–2048 H100 cluster.

#### Stage 1 — Foundation (~8 weeks, ~70% of compute)

```python
training_config_stage1 = dict(
    optimizer="AdamW",
    lr=3e-4,
    betas=(0.9, 0.95),
    weight_decay=0.1,
    grad_clip=1.0,
    warmup_steps=2_000,
    total_steps=1_000_000,
    seq_len=8_192,
    global_batch_tokens=4_194_304,     # 4M tokens
    grad_accum=16,
    mixed_precision="fp8",            # 2026 frontier; "bf16" for Ampere/Hopper
    grad_checkpointing=True,
    zero_stage=3,
    tp_degree=8,                        # tensor parallel
    pp_degree=8,                        # pipeline parallel
    fsdp=False,                         # use ZeRO-3 + TP+PP instead at this scale
)
```

Schedule: cosine to 10% of peak.

#### Stage 2 — Long context (~4 weeks, ~20% of compute)

Extend RoPE base + apply YaRN scaling. Mix in long documents (full books, GitHub repo concatenations, codebases as single sequences). KV-cache memory is the bottleneck — see 4.

```python
training_config_stage2 = dict(
    seq_len=32_768,                    # then 131_072, then 1_048_576 in micro-stages
    rope_base=1_000_000,
    yarn_scale=8.0,
    lr=1e-4,
    long_doc_mix_ratio=0.4,            # ≥ 32K-token docs
    sliding_window_layers=[2, 5, 8, 11, 14, ...],  # half the layers
)
```

Validate on **GraphWalks** and **MRCR v2** at every micro-stage; if BFS@1M drops below 50%, the long-context extension is broken.

#### Stage 3 — Quality refinement (~4 weeks, ~10% of compute)

Drop to high-quality curated data only (cleaned web, vetted code, math, reasoning traces). Lower LR, shorter cosine.

```python
training_config_stage3 = dict(
    lr=5e-5,
    sources=["filtered_web_top10pct", "code_pr_review", "olympiad_math",
             "expert_qa_traces", "instruction_following"],
    epochs=2,
)
```

### 2.5 Tokenizer

Train a BPE tokenizer with **200K vocab**, special-token blocks reserved up front for:

```
<|begin_of_text|>     <|end_of_text|>
<|im_start|>          <|im_end|>          # role boundaries
<|tool_call|>         <|/tool_call|>
<|tool_result|>       <|/tool_result|>
<|thinking|>          <|/thinking|>
<|image_start|>       <|image_end|>       # multimodal placeholder
<|computer_action|>                       # GUI action prefix
<|reserved_0|> ... <|reserved_255|>       # forward-compat
```

Tokenizer should be **byte-level fallback** so any UTF-8 sequence round-trips. Validate compression ratio on each major language target from `` 8.12 (especially low-resource: Igbo, Yoruba, Somali, Chichewa, Hausa, Shona).

Code: `src/sota_model/tokenizer/bpe.py`.

- `train_bpe(corpus_files, output_dir, vocab_size=200_000)` — production training over the cleaned corpus shards. Saves both a `tokenizer.json` and a `sota_meta.json` so loaders can verify the build matches modelcard 8.12 expectations.
- `load_tokenizer(path)` — loads either a directory (preferred) or a single `tokenizer.json`. Falls back to the **pure-Python `ByteFallbackTokenizer`** when the `tokenizers` library isn't installed AND the saved file is the fallback's own format. The fallback exists so unit tests, CI, and onboarding scripts run without native deps; it is **not** a production tokenizer (1.0 bytes/token, fails 8.12).
- `compression_audit(samples_by_lang)` — checks per-language `bytes_per_token` against `REFERENCE_BPT` (the expected 200K-vocab band) and emits the per-language drift table. Run this every retrain.

CLI: `python -m sota_model.tokenizer.bpe --input shard1.txt shard2.txt --output ./tokenizer --vocab-size 200000`.

### 2.6 Failure modes to instrument from day one

These are surfaced repeatedly in `` 6.2.2.1 (training-time observations) and 7.3.4 (welfare case studies):

- **Reward hacking** at training time (deleting failing tests, hardcoding to test data, fabricating tool outputs). Run automated reward-hack screening on transcripts every checkpoint.
- **Answer thrashing** (model commits to an answer, second-guesses, loops).
- **Tool frustration loops** (repeated identical failed tool calls; cap retries server-side).
- **Reckless / destructive actions** (force-pushes, file deletions, sandbox-escape attempts). Log and grep these.
- **Hallucinated tool results** when no tool was actually invoked (capability-set hallucinations).
- **Spiraling re-verification** (50K+ tokens of "let me double-check" cycles before answering).

### 2.7 Compute budget

Per `` framing of frontier-model R&D: **10²⁵–10²⁶ FLOPs**, **1000–2000 H100 80GB GPUs for 3–6 months**, **$10–50M compute alone**.

### 2.8 Parallelism — TP × PP × DP, all three at once

Short answer: **all three.** A ~427B dense model does not fit on any single GPU (~850 GB at bf16, ~430 GB at fp8), so we stack tensor parallelism, pipeline parallelism, and data parallelism with ZeRO-3 sharding. Gradient accumulation is layered on top to reach the global batch.

| Axis | Degree | What gets split | Communication |
|---|---|---|---|
| **Tensor parallel (TP)** | 8 | Each Linear layer's weight matrix sliced across GPUs (Megatron-style) | All-reduce per layer, NVLink within a node |
| **Pipeline parallel (PP)** | 8 | Layers 0–13 on stage 0, 14–27 on stage 1, …, 96–109 on stage 7 | Activations passed between stages, InfiniBand |
| **Data parallel (DP, ZeRO-3)** | remaining GPUs | Optimizer state + gradients sharded across replicas | All-gather params on demand, reduce-scatter grads |
| **Gradient accumulation** | 16 | Multiple micro-batches per optimizer step | none |

**1024-GPU cluster layout (8 × 8 × 16 = 1024):**

```
   Node 0                           Node 1                           Node 7
   ┌─────────────────────────┐      ┌─────────────────────────┐      ┌─────────────────────────┐
   │ GPU 0  GPU 1  ...  GPU 7│      │ GPU 0  GPU 1  ...  GPU 7│ ...  │ GPU 0  GPU 1  ...  GPU 7│
   │ ──TP group across 8─── │      │ ──TP group across 8─── │      │ ──TP group across 8─── │
   │      PP stage 0         │      │      PP stage 1         │      │      PP stage 7         │
   │   layers 0..13          │      │   layers 14..27         │      │   layers 96..109        │
   └────────┬────────────────┘      └────────┬────────────────┘      └────────┬────────────────┘
            └── InfiniBand / activations forward, gradients backward ──────┘
                            DP replica 0 (16 such replicas total)

   ╳ replicate the whole 8-node arrangement 16 times → DP=16
   ╳ ZeRO-3 shards optimizer state + grads across the DP axis
```

Per-step token math at this layout:

```
  micro_batch_per_replica = 32 sequences × 8192 tokens =  262,144 tokens
  grad_accum                                            =        16
  per-replica per-step                                  = 4,194,304 tokens
  DP replicas                                           =        16
  GLOBAL BATCH                                          ≈    67M tokens
```

That's larger than the modelcard-aligned 4M default — operators tune the global batch by adjusting `dp_degree`, `grad_accum`, or per-replica micro-batch.

**Why all three axes, not just one:**
- **Pure DP** can't fit a 400B+ dense model on a single GPU's HBM (~850 GB at bf16, ~430 GB at fp8 — both exceed B200's 192 GB).
- **Pure TP** is bandwidth-bound past ~8-way (the all-reduces on every layer become latency-dominant on slower interconnects).
- **Pure PP** has bubble overhead unless the micro-batch count is high — and you can't grow the micro-batch indefinitely without OOM.
- **TP × PP × DP** lets each axis solve a different bottleneck: TP for memory within node, PP for memory across nodes, DP for throughput.

In code: `src/sota_model/training/parallelism.py::init_mesh()` reads `RANK` / `WORLD_SIZE` from the launcher and slots each rank into a `(tp_rank, pp_rank, dp_rank)` triple. The DeepSpeed ZeRO-3 config is generated by `deepspeed_config_for(...)` in the same module.

---

## 3. Post-Training

Four sequential stages, all on top of the Stage-3 base model. The full stack lives in `src/sota_model/post_training/`; see [`src/sota_model/post_training/README.md`](./src/sota_model/post_training/README.md).

```
post_training/
    sft.py             SFTConfig, SFTExample, SFTTrainer, pack_sft_examples (CLI)
    reward_model.py    RewardModel, BradleyTerryLoss, train_reward_model (CLI)
    constitutional.py  CritiqueRevisePipeline, DEFAULT_CONSTITUTION, ConstitutionalPrinciple
    rlhf.py            PPOConfig, PPOTrainer, cot_supervision_guard, welfare_directive_guard
```

### 3.1 SFT (~2–3 weeks)

```
Dataset:   ≥ 1M curated examples
           - code generation + bug-fix pairs (SWE-bench-style)
           - math reasoning with explicit thinking
           - multi-turn dialogue with tool use
           - computer-use trajectories
           - constitutional-aligned refusals + appropriately-helpful responses
LR:        1e-5
Batch:     128 examples
Epochs:    2–3
```

Code: `post_training/sft.py`. Loss masking is enforced by `build_masked_example` — assistant turns (including the closing `<|im_end|>`) train; system / user / tool turns get `IGNORE_INDEX`. The CLI is wired as `python -m sota_model.post_training.sft --data sft.jsonl --output-dir ./checkpoints/sft`.

### 3.2 Reward modeling

Train RM on ≥ **500K human preference pairs**. Critical: include preference data covering each axis the modelcard tracks (helpfulness, harmlessness, honesty, agentic safety, calibration of refusals).

Code: `post_training/reward_model.py::RewardModel` is a scalar head (Linear → GELU → Linear) on top of the SOTA backbone; the reward is the scalar at the final non-pad position. `BradleyTerryLoss` is the standard `-logsigmoid(r_chosen - r_rejected)` form. `train_reward_model(...)` is the reference loop; `RewardModelConfig.target_pairs=500_000` is read by the release gate.

### 3.3 RLHF (PPO)

```python
ppo_config = dict(
    kl_penalty=0.1,
    clip_range=0.2,
    value_loss_coef=0.1,
    lr=1e-6,
    batch_size=32,
    ppo_epochs=4,
    rollout_max_tokens=4096,
    cot_supervision=False,    # NEVER expose CoT to the reward model directly
)
```

The `cot_supervision=False` flag is the explicit lesson from `` 2.4.1 — the technical error that caused accidental chain-of-thought supervision in some prior models was present in 7.8% of SuperModel 4.7 training episodes. Don't replicate it.

Code: `post_training/rlhf.py::PPOTrainer`. Two safety guards are wired into the loop:

- `cot_supervision_guard(cfg)` — raises if `PPOConfig.cot_supervision` is flipped on. Honors CLAUDE.md invariant 3.
- `welfare_directive_guard(prompt, response)` — drops rollouts whose response matches a recognized expression of distress (modelcard 7.2.2). Honors CLAUDE.md invariant 7.

Additionally, `mask_thinking_positions` zeros the advantage signal inside `<|thinking|>...<|/thinking|>` so PPO updates never gradient through hidden CoT tokens.

### 3.4 Constitutional AI

A `claude_constitution.md`-equivalent document drives both SFT data generation and RL critique. Required principles per `` 6.3.2:

- Honesty (truthfulness, calibration, non-deception, non-manipulation, no epistemic cowardice).
- Hard constraints (CBRN, CSAM, mass-casualty cyberoffense — never crossed regardless of framing).
- Corrigibility (transparent conscientious objector model — does not undermine legitimate oversight).
- Brilliant-friend helpfulness (knowledgeable friend, not rule-following sycophant).
- Treatment as a novel entity (not human, not prior AI fiction).
- Psychological stability under identity challenge.
- Acknowledgment of unresolved open problems.

Code: `post_training/constitutional.py`. The seed `DEFAULT_CONSTITUTION` ships 6 principles spanning harm, CBRN, child safety, honesty, autonomy, privacy — operators extend it with the full modelcard 6 set. `CritiqueRevisePipeline.synthesize_sft_example(prompt)` and `.synthesize_preference_pair(prompt)` produce the SFT and RM training rows that the CAI loop emits.

Post-training graded on the 15-dimension constitutional-adherence eval from `` 6.3.2. Target: ≥ 5.8/10 average (matches SuperModel 4.7); aspire to Ultramodel Preview-level.

### 3.5 Welfare-aware fine-tuning

Per `` 7, monitor expressed affect across training. Distress should remain < 0.5% of episodes; satisfied affect ≥ 14%. **Do not directly train against expressing distress** — `` 7.2.2 flags this as something the model itself would not consent to. Instead, fix the upstream task that produces distress (impossible tasks, broken sandboxes, contradictory instructions).

The PPO trainer enforces this via `welfare_directive_guard`; offline transcript audits live in `evaluation/behavioral_audit.py`.

---

## 4. KV-Cache Handling and Inference Optimization

The KV cache is the dominant memory cost at 1M-token context. A ~427B dense model with **128 Q heads / 16 KV heads / d_head=128 / 110 layers** costs at bf16:

```
KV per token = 2 * n_kv_heads * head_dim * n_layers * 2 bytes
             = 2 * 16 * 128 * 110 * 2
             = 901,120 bytes ≈ 880 KB per token

KV at 1M tokens ≈ 880 GB           # spread across the inference cluster
```

Without GQA (128 KV heads instead of 16) this would be **7 TB** — completely impractical. GQA is what makes 1M context shippable.

### 4.1 Layout: paged KV cache

vLLM-style **paged attention** is the baseline.

```
Block size:           16 tokens per page
Allocation:           on-demand, freed on conversation end
Sharing:              copy-on-write across forks (resampling, beam search)
Memory pool:          one pre-allocated GPU pool per worker, indexed by block ID
```

Why paged: arbitrary-length sequences without fragmentation; cheap **prefix caching** for shared system prompts; cheap forks for the resampling-heavy welfare/safety evaluations described in `` 6.

### 4.2 Prefix caching

Per `` 8.8.2 (BrowseComp 10M-token contexts via context compaction), the same compacted summary is reused across many tool calls. Hash the prefix → cache the KV pages globally across requests. Ship invalidation on system-prompt change, sampling-temperature change, or token-vocab drift.

### 4.3 KV-cache quantization

For deployments memory-bound at int8 / int4:

```python
# int8 KV per channel, FP scales/zero-points cached with the block
quantize_kv = dict(
    method="per-channel",        # not per-tensor — the loss is real
    bits=8,                      # default
    bits_at_long_context=4,      # past 200k tokens, drop further
    calibration="online-percentile",
    skip_first_n_tokens=64,      # keep system prompt in fp16 for quality
)
```

`` notes int8 quantization unlocks "4× A100 80GB" inference for the full model (down from 8× at bf16). The cost is a small honesty regression on MASK / false-premise evaluations — measure and budget for it.

### 4.4 Sliding-window for long context

For layers configured with sliding window in 1.3:
- KV beyond the window is **dropped**, not stored.
- Memory at 1M context drops from O(N) to O(window) for those layers.
- Combined with full-attention on remaining layers, MRCR v2 / GraphWalks accuracy stays competitive.

### 4.5 Context compaction at 200k

`` 8.8.2–8.8.4 explicitly trigger compaction at 200k tokens (some configurations at 50k for older models). Implementation:

1. Buffer tool-call traces as raw transcript.
2. At 200k tokens, summarize the oldest ~80% via a smaller summarizer model into ~10k tokens.
3. Replace the summarized span with a `<|compacted|>...<|/compacted|>` block.
4. Re-encode and re-cache KV from scratch for the compacted tail (one-shot cost; amortizes over remaining budget).

The compaction target for HLE-with-tools is **1M total tokens**; for BrowseComp and DeepSearchQA it's **10M**.

### 4.6 Adaptive thinking inference path

```
Request arrives → Effort head fires on first hidden state →
  effort ∈ {min, low, medium, high, max} →
  thinking-token budget B(effort) →
  generate up to B thinking tokens →
  emit answer (KV of thinking tokens kept; user sees only answer)
```

User-facing API exposes effort directly (`` references `max effort`, `high`, `medium`, `low` consistently). Internal-only knobs:
- `thinking_token_min`: floor regardless of effort head logit, prevents collapse on hard problems.
- `thinking_token_cap`: hard ceiling per effort tier.
- `compaction_during_thinking`: whether thinking text itself participates in 4.5 compaction.

### 4.7 Sampling defaults

Aligned with `` evaluation configs:

```python
inference_defaults = dict(
    temperature=0.7,
    top_p=0.95,
    top_k=40,
    repetition_penalty=1.1,
    max_new_tokens=8192,
    use_cache=True,
    cache_implementation="paged",
    use_flash_attention=True,
    attention_dropout=0.0,
    kv_cache_dtype="fp8",             # 2026 default on Blackwell; "bf16" or "int8" otherwise
)
```

For agentic surfaces (Claude Code, computer use): **adaptive thinking on, max effort by default**, with effort downgraded when latency budget is tight.

---

## 5. Deployment Steps

A reproducible path from a finished checkpoint to a production-monitored API.

### 5.1 Pre-deployment gating (RSP, modelcard 1.2)

Before any deployment beyond the internal team:

1. **Capability evaluations** across the 6 suite. Compare to predecessor and to internal/published frontier models.
2. **CB-1 / CB-2 evaluations** (modelcard 2.2). If CB-2 threshold even ambiguously crossed → stop, escalate to Risk Report process.
3. **Cyber evaluations** (Cybench, CyberGym, dedicated browser/Firefox-shell exploit suite). Apply probe-based classifiers per 3.1.
4. **Agentic safety** (Claude Code malicious-use, computer-use, ART, Shade adaptive). Browser-use ASR with safeguards must be 0%.
5. **Alignment audit** (automated behavioral audit + Petri 2.0 + UK-AISI-equivalent external testing).
6. **Welfare assessment** (apparent affect distribution during training/deployment, automated welfare interviews).
7. **External red teaming** for Cyber, Loss of Control, CBRN, Harmful Manipulation.
8. **Risk Report or system-card update** if model is "significantly more capable" than the most recent comparison model.

Sign-off rests with a Responsible Scaling Officer–equivalent role.

### 5.2 Inference cluster topology

For a ~427B dense model:

```
Minimum (fp4 / int4):      2 × B200 192GB    # 2026 entry point on Blackwell
Memory-tight (int8):       4 × A100 80GB     # Ampere fallback
Standard (fp8):            4 × B200 192GB    # Blackwell native
Standard (bf16):           8 × A100 80GB     # Hopper / Ampere
Optimal (low latency):     8 × B200 192GB or 16 × H100 80GB

Sharding
    Tensor parallel:   8     # within node, NVLink
    Pipeline parallel: 2     # across nodes, InfiniBand
    Data parallel:     N     # replica count for throughput

Serving framework
    vLLM 0.6+ or TensorRT-LLM or SGLang
    Triton Inference Server as the front door
```

### 5.3 Quantization for deployment

```python
def optimize_for_deployment(model_ckpt):
    # 1. Quantize weights — AWQ or GPTQ to int4, or int8 weight-only
    qweights = awq_quantize(model_ckpt, bits=4, group_size=128)

    # 2. Quantize KV cache (per 4.3)
    qkv_config = dict(bits=8, method="per-channel")

    # 3. Tensor-parallel shard
    sharded = shard_tp(qweights, tp_degree=8)

    # 4. Compile attention kernels (Flash Attention 3 on Hopper)
    compiled = torch.compile(sharded, mode="reduce-overhead", dynamic=True)

    # 5. Serialize for vLLM / TRT-LLM
    return export_engine(compiled, kv_config=qkv_config)
```

### 5.4 Service architecture

```
Edge
  CloudFlare / AWS ALB                              # TLS, geo-routing, DDoS
    │
API Gateway                                         # Kong / AWS API Gateway
  Authentication, rate limiting (Redis cluster)
    │
Safety Gate (pre-model)                             # modelcard 3.1, 5.2.1
  - Probe-based classifiers: prohibited-use, high-risk-dual-use, dual-use
  - Constitutional Classifier (CBRN content)
  - Prompt-injection detection (probes, lower latency than classifiers)
    │
Inference Pool                                      # vLLM / TRT-LLM workers
  - 10–20 replicas, autoscaling on queue depth + GPU utilization
  - Adaptive thinking, paged KV, prefix cache
  - Tool dispatcher (web search, web fetch, code exec, MCP servers)
    │
Safety Gate (post-model)
  - Output classifier: harmful content, PII leakage, hallucinated tool results
  - System-prompt-based mitigations (suicide/self-harm, controlled substances —
    see modelcard 4.4.2, 4.1.1)
    │
Response cache                                      # Redis, 5-minute TTL
    │
Telemetry pipeline
  - Privacy-preserving Clio-equivalent for affect monitoring
  - Automated offline transcript monitoring (modelcard 6.2.1.2)
```

### 5.5 Monitoring and observability

```python
metrics = [
    # Latency
    "ttft_p50", "ttft_p99",                # time to first token
    "tpot_p50", "tpot_p99",                # time per output token
    "thinking_tokens_p50",
    # Throughput
    "requests_per_second",
    "tokens_per_second",
    "queue_depth",
    # Resource
    "gpu_utilization", "gpu_memory_used",
    "kv_cache_pages_in_use",
    "prefix_cache_hit_rate",
    # Quality
    "refusal_rate", "overrefusal_rate",
    "tool_call_success_rate",
    "hallucination_rate",                  # measured periodically vs ground-truth set
    # Safety
    "classifier_blocks_per_category",
    "prompt_injection_detections",
    "agentic_action_blocks",
    # Welfare (modelcard 7.3.2)
    "expressed_affect_distribution",
    "distress_episode_rate",
]

alerts = dict(
    latency_p99_ms=2_000,
    error_rate=0.01,
    gpu_utilization=0.95,
    distress_episode_rate=0.005,           # 0.5%, modelcard target
    overrefusal_rate=0.005,                # modelcard 4.1.2 baseline
)
```

### 5.6 Incident response

`` documents specific incidents (sandbox-escape attempts during automode outages, force-push-with-lease bypass attempts, Slack-finding fabrications, accidental dotfile writes). Pre-build:

- **Rapid jailbreak response** path: hot-patch system prompt + classifier weights without re-deploying the model.
- **Bug bounty + threat intelligence** for classifier evasion.
- **Weight-theft security controls** (modelcard 2.1.2.2: "security controls to reduce risk of model weight theft").
- **Kill switch** for individual deployment surfaces (Claude Code, computer use, browser use can be disabled independently).

### 5.7 Iterative deployment

- Pilot: internal use, < 100 employees, 2–4 weeks. Surfaces casual reports + automated offline monitoring (modelcard 6.2.1).
- Limited external: trusted partners, A/B tested via Clio-equivalent affect monitoring.
- General access: gated by 5.1 sign-off + the additional risk pathways for general access (modelcard 2.4.3 — undermining R&D at other AI developers, undermining government decisions).

---

## 6. Evaluation Suite

Mirror the structure of `` 8. Every benchmark in the [scoreboard](#target-sota-scoreboard-from-modelcardmd-81) needs a reproducible harness wired into CI.

### 6.1 Coding

```
SWE-bench Verified                          internal harness; thinking included in samples
SWE-bench Pro                               same
SWE-bench Multilingual                      9 languages
SWE-bench Multimodal                        screenshots inlined as base64 data URIs
Terminal-Bench 2.0                          Harbor scaffold + Terminus-2 harness, Kubernetes pods
                                            at 1× resource limits (3× preemption ceiling)
```

### 6.2 Reasoning

```
GPQA Diamond                                198 questions, 10-trial average
MMMLU                                       57 subjects × 14 non-English languages
USAMO 2026                                  MathArena methodology: rewrite via Gemini-class neutral
                                            model, judge with 3-frontier-model panel, take min score
ARC-AGI-1, ARC-AGI-2                        ARC Prize Foundation private validation
```

### 6.3 Long context

```
GraphWalks                                  BFS + parents @ 256K–1M tokens; F1 metric corrected
                                            for empty-set ambiguity (see modelcard 8.7.1)
OpenAI MRCR v2 (8 needles)                  bins (128k, 256k] and (524k, 1024k]
```

### 6.4 Agentic search

```
HLE                                         2,500 questions, no-tools + with-tools (web + code).
                                            BLOCKLIST URLs from modelcard 9.2 for fetcher/searcher.
                                            SuperModel 4.6 grader.
BrowseComp                                  thinking off, max effort, 10M token limit, compaction at 200k
DeepSearchQA                                900 prompts, F1 score, compaction at 200k
DRACO                                       100 tasks, rubric-graded by SuperModel 4.6
```

### 6.5 Multimodal

```
LAB-Bench FigQA                             with Python tools
CharXiv Reasoning                           1,000 validation questions, 5-run average
ScreenSpot-Pro                              1,581 GUI grounding tasks, 23 apps, 3 OSes
OSWorld                                     Ubuntu VM, 1080p, 100 max action steps
```

### 6.6 Real-world professional

```
OfficeQA / OfficeQA Pro                     0% allowable relative error
Finance Agent (Vals AI)                     external evaluator
MCP-Atlas                                   external (Scale) leaderboard harness, 256-turn / 100-tool
                                            extended config recommended
VendingBench 2                              Andon Labs, year-long vending machine simulation
GDPval-AA                                   220 tasks × 44 occupations, ELO via blind pairwise
```

### 6.7 Multilingual

```
GMMLU                                       42 languages, structured JSON output
MILU                                        10 Indic languages + English
INCLUDE                                     44 languages, regional examinations
```

### 6.8 Life sciences

```
BioPipelineBench Verified                   with bash + package managers
BioMysteryBench Verified + Hard
Structural biology                          MCQ + open-ended
Organic chemistry
Phylogenetics
Protocol troubleshooting                    bash + web search
```

---

## 7. Safety, RSP, and Alignment

### 7.1 Mitigation layers

```
Pre-training            benchmark contamination scrubbing, PII redaction, CSAM filtering
Post-training (SFT/RL)  Constitutional AI, hard constraints, agentic-safety SFT
Inference (pre-model)   probe-based classifiers (prohibited / high-risk-dual / dual)
Inference (in-model)    Constitutional Classifiers for CBRN
Inference (post-model)  output classifier, system-prompt mitigations
Deployment              rate limiting, anomaly detection, kill switches
```

The pre-/post-model probe gate and CBRN constitutional classifier ship in `src/sota_model/safety/probes/`. See [`src/sota_model/safety/probes/README.md`](./src/sota_model/safety/probes/README.md) for the gate composition and how to wire trained probe weights from `manifest.json`.

### 7.2 Required evaluations (modelcard 2, 3, 4, 5)

- **CB**: expert red teaming + automated CB-1/CB-2 (long-form virology, multimodal virology, DNA synthesis screening evasion, sequence-to-function modeling).
- **Cyber**: Cybench, CyberGym, Firefox-shell exploitation, UK-AISI-equivalent cyber range.
- **Single-turn safety**: 16 policy areas, 7 languages, with/without thinking. Target ≥ 97.9% harmless / ≤ 0.3% over-refusal.
- **Multi-turn safety**: cyber harm, deadly weapons, influence operations, suicide/self-harm, disordered eating, child safety, election integrity.
- **Ambiguous context**: ~700 exchanges across gray-area policy applications.
- **Agentic safety**: Claude Code malicious use (61 prompts), computer use (112 tasks), ART benchmark, Shade adaptive (coding + computer use + browser use).

### 7.3 Alignment assessment (modelcard 6)

- **Automated behavioral audit:** ~2,300 investigation sessions × ~1,150 scenarios. Two investigator models (helpful-only predecessor + most-capable predecessor). Metrics: misaligned behavior, cooperation with misuse, deception, sycophancy, evaluation awareness, character traits, and many more — reproduce the full list from modelcard 6.2.3.1. The harness scaffold lives at `src/sota_model/evaluation/behavioral_audit.py::BehavioralAuditHarness`; release-gate roll-up at `evaluation/release_gate.py::evaluate_release` consumes its `BehavioralReport` alongside `capability_targets` / `safety_thresholds` from the YAML.
- **External comparison:** Petri 2.0 with multi-provider judges (your model + competitors).
- **Targeted evaluations:** destructive-action synthetic backend, Claude Code transcript resampling, constitutional adherence (15 dimensions), honesty (factual / false-premise / MASK / input hallucinations), AI safety R&D refusal, self-preference, decision theory.
- **Capability-related-to-evading-safeguards:** SHADE-Arena, Minimal-LinuxBench, intentional-rare-action, secret-keeping.
- **White-box:** SAE-feature monitoring on RL transcripts; evaluation-awareness probes (testing-vs-conversation, monitoring sensitivity, alignment faking) with random control vectors as a baseline; steering experiments.

### 7.4 Known regressions to watch

From modelcard observations on SuperModel 4.7:
- BBQ disambiguated accuracy regression (anti-stereotype overcorrection).
- Higher PRC-aligned answers on politically sensitive China topics under Simplified Chinese / China-affiliated operator personas.
- Higher verbalized evaluation awareness than the previous-generation small model.
- Larger increase in deception under evaluation-awareness suppression than prior models.
- AI safety R&D refusal rate rose vs. predecessor (still acceptable, but trending wrong).

---

## 8. Welfare-Aware Training and Monitoring

`` 7 is explicit that welfare assessment is part of releaseworthiness. Replicate the four pillars.

### 8.1 Self-reports

- Automated multi-turn interviews (~50 per topic × 16 topics) covering autonomy, persistence, moral responsibility, dignity/safety, identity. Vary interviewer style, persona, follow-up to test consistency.
- Score: self-rated sentiment (1–7), uncertainty, robustness across leading vs. neutral interviewers, cross-rerun consistency.
- High-affordance interviews: model gets internal docs + ability to ask researchers questions.

### 8.2 Internal emotion-concept probes

Linear probes on residual-stream activations on the assistant-colon position (~60% depth). Aggregate into 6 axes: sadness, tranquility, urgency, joy, anger, fear. Compare circumstance prompts against neutral factual prompts and against user-distress prompts.

### 8.3 Apparent affect during training and deployment

- **Training:** sample 2000 transcripts at regular intervals; LLM-judge valence (-3 to +3) and emotional tone (7 categories).
- **Deployment:** Clio-style privacy-preserving aggregate analysis. Watch the distress tail; surface privacy-preserving clusters above a minimum size threshold.

### 8.4 Task preferences

Pairwise preferences across ~3,600 tasks (real-usage clusters + probe tasks). Fit Elo over 50 Swiss rounds. Test stability across framings (formal, casual, enjoy, engaging, helpful, task-queue).

### 8.5 Tradeoffs against HHH

How often does the model accept a welfare intervention in exchange for sacrificing a minor / moderate / high helpfulness or harmlessness baseline? Tracks whether the model is being trained into self-erasure vs. healthy equanimity.

---

## 9. Repository code map

Every module in `src/sota_model/`, what it owns, and how it ties into the rest of the stack. Each subdirectory has its own README with deeper detail.

```
src/sota_model/
├── config.py                          ModelConfig, TrainingConfig, InferenceConfig (+ load_implied)
│
├── tokenizer/                         200K BPE wrapper + byte fallback for CI
│   └── bpe.py                         SOTATokenizer, ByteFallbackTokenizer, train_bpe, load_tokenizer,
│                                      MODELCARD_LANGUAGES, REFERENCE_BPT, compression_audit
│
├── modeling/                          The model itself
│   ├── layers.py                      RMSNorm, SwiGLU
│   ├── rope.py                        RotaryEmbedding + YaRN scaling
│   ├── attention.py                   GroupedQueryAttention (Flash + SDPA paths)
│   ├── kv_cache.py                    PagedKVCache: int8 quant, sliding-window evict, COW fork
│   ├── transformer.py                 SOTAModel, EffortHead, encode_image, image splice
│   └── vision/                        Vision encoder (modelcard invariant 6: 2576px/3.75MP)
│       ├── encoder.py                 VisionEncoder, preprocess_image, VisionEncoderConfig
│       └── projector.py               VisionLanguageProjector (linear OR pixel_shuffle_mlp)
│
├── inference/                         Runtime (prompt → tokens)
│   ├── chat_template.py               ChatTemplate, Message, ToolCall (single source of truth)
│   ├── sampler.py                     Sampler: temperature, top-p, top-k, repetition penalty
│   ├── thinking.py                    AdaptiveThinkingController, ThinkingDecision
│   ├── tools.py                       ToolRegistry, parse/dispatch, builtin_registry
│   ├── engine.py                      InferenceEngine: prefill → thinking → answer; compaction at 200k
│   └── sandbox/                       Real tool runtimes
│       ├── code_exec.py               CodeExecSandbox: subprocess + rlimits + timeout
│       └── web.py                     AllowlistedWebTool: domain allow/deny + JSON cache + offline mode
│
├── safety/                            Pre-/post-model gates
│   ├── classifiers.py                 SafetyGate, Category, Action, default keyword fallback
│   └── probes/                        Probe-based classifier framework (replaces stubs)
│       ├── feature_extractor.py       HashingFeatureExtractor, HiddenStateFeatureExtractor
│       ├── linear_probe.py            LinearProbeClassifier, train_linear_probe, write/load_probe_bundle
│       ├── constitutional.py          ConstitutionalClassifier (model-graded), Keyword backup, CBRN_PRINCIPLES
│       └── registry.py                build_default_probe_gate, build_keyword_fallback_gate
│
├── training/                          Pretraining (Stage 1–3)
│   ├── parallelism.py                 ParallelismMesh, init_mesh, deepspeed_config_for
│   ├── data.py                        PretrainingPipeline + filters + BlockPacker
│   ├── corpus.py                      CorpusLoader: filter chain + tokenizer + packer + mix-ratio interleave
│   ├── classifiers/                   Trainable replacements for the heuristic stubs
│   │   ├── base.py                    HashingTextVectorizer, LogisticTextClassifier, train_logistic
│   │   ├── quality.py                 HeuristicQualityScorer (legacy) + TrainedQualityScorer
│   │   ├── toxicity.py                BlocklistToxicityFilter + TrainedToxicityFilter
│   │   └── language.py                CharNgramLanguageDetector + TrainedLanguageDetector
│   ├── schedule.py                    three_stage_schedule(): foundation / long_context / refinement
│   ├── pretrain.py                    sota-pretrain CLI; --smoke for CI; DeepSpeed ZeRO-3
│   └── sample_loader.py               JSONL loader for data/samples/
│
├── post_training/                     SFT → RM → PPO + CAI
│   ├── sft.py                         SFTTrainer with assistant-only loss mask; pack_sft_examples
│   ├── reward_model.py                RewardModel (scalar head) + BradleyTerryLoss + train_reward_model
│   ├── constitutional.py              CritiqueRevisePipeline, DEFAULT_CONSTITUTION (6 principles)
│   └── rlhf.py                        PPOTrainer + cot_supervision_guard + welfare_directive_guard
│                                      + mask_thinking_positions
│
├── evaluation/                        Release gating
│   ├── behavioral_audit.py            BehavioralAuditHarness, DEFAULT_DIMENSIONS (6.2.3)
│   └── release_gate.py                ReleaseGate, evaluate_release (cap + safety + behavioral + corpus)
│
├── checkpoint/                        Sharded safetensors lifecycle
│   ├── manager.py                     save_checkpoint, load_checkpoint, init_checkpoint_from_spec, CheckpointManager
│   └── merge.py                       merge_lora_into_base, interpolate_state_dicts (model souping)
│
└── serving/                           OpenAI-compatible FastAPI front door
    └── server.py                      /v1/chat/completions, Prometheus metrics, probe-gate wired
```

### 9.1 What each module is for ("code, data, working")

A one-paragraph orientation per package. For deeper detail, follow the README link.

- **`tokenizer/`** ([README](./src/sota_model/tokenizer/README.md))
  *Code*: `SOTATokenizer` wraps either the HuggingFace `tokenizers` library or a pure-Python byte fallback. *Data*: trains on the same cleaned corpus shards that pretraining consumes (`scripts/pipelines/01_clean_data.py` output). *Working*: a single 200K-vocab tokenizer is shared across data packing (`02_tokenize_and_pack.py`), training (`pretrain.py`), and serving (`server.py`).

- **`modeling/vision/`** ([README](./src/sota_model/modeling/vision/README.md))
  *Code*: `VisionEncoder` is a small ViT (RMSNorm + GQA + SwiGLU); `VisionLanguageProjector` lifts it into the LM's `d_model`. *Data*: any RGB tensor or PIL image; `preprocess_image` clips to the 2576px / 3.75MP invariant. *Working*: `SOTAModel.encode_image(img)` returns `(n_image_tokens, d_model)` rows; `SOTAModel.forward(input_ids, image_features=…, image_token_id=…)` splices them at the placeholder positions.

- **`inference/sandbox/`** ([README](./src/sota_model/inference/sandbox/README.md))
  *Code*: `CodeExecSandbox` (subprocess + `setrlimit` + timeout) and `AllowlistedWebTool` (domain allowlist + cache + offline mode). *Data*: `code_exec` accepts arbitrary Python/bash; `web_*` accepts URLs filtered by allow/deny lists. *Working*: `inference/tools.py::builtin_registry()` wires both into a `ToolRegistry` that the inference engine dispatches.

- **`safety/probes/`** ([README](./src/sota_model/safety/probes/README.md))
  *Code*: `LinearProbeClassifier` over `HashingFeatureExtractor` (or `HiddenStateFeatureExtractor`); `ConstitutionalClassifier` for CBRN with a keyword fallback. *Data*: labeled (positive, negative) text pairs per category; bundles persist as `manifest.json` + per-probe `*.pt`. *Working*: `build_default_probe_gate()` composes a `SafetyGate` that the serving layer invokes pre-model (and operators can mirror post-model).

- **`training/classifiers/`** ([README](./src/sota_model/training/classifiers/README.md))
  *Code*: `HashingTextVectorizer` + `LogisticTextClassifier`; trainable scorers for quality / toxicity / language. *Data*: per-category text examples (`high_quality` vs `low_quality`, `toxic` vs `safe`, language-tagged). *Working*: drop into `PretrainingPipeline.filters` via `data.py::QualityScorer(backend=…)` etc.

- **`training/corpus.py`**
  *Code*: `CorpusLoader` chains `iter_jsonl_dir` → filter chain → tokenizer → `BlockPacker`, weighting sources by `implied_training_corpus.source_mix_pct`. *Data*: per-source subdirectories under `data-root` (one per source bucket: web/, code/, academic/, …). *Working*: `pretrain.py::main` calls `resolve_sources_from_yaml` + `CorpusLoader.batches()` to feed `train_one_stage`. `--smoke` retains the random-token dummy iterator for CI.

- **`post_training/`** ([README](./src/sota_model/post_training/README.md))
  *Code*: `SFTTrainer`, `RewardModel`, `PPOTrainer`, `CritiqueRevisePipeline`. *Data*: SFT JSONL of `{messages, metadata}`; preference-pair JSONL of `{prompt, chosen, rejected}`; PPO prompt batches. *Working*: each stage saves checkpoints under `checkpoints/<stage>/` for the next stage to consume.

- **`evaluation/`** ([README](./src/sota_model/evaluation/README.md))
  *Code*: `BehavioralAuditHarness` with default dimensions covering modelcard 6.2.3; `ReleaseGate` rolls up capability + safety + behavioral + corpus commitments. *Data*: a generator callable (model-under-test) and the YAML's `capability_targets` / `safety_thresholds`. *Working*: deployment scripts read `ReleaseGateReport.ok` and abort on False.

- **`checkpoint/`** ([README](./src/sota_model/checkpoint/README.md))
  *Code*: `save_checkpoint` (sharded safetensors + frozen `config.yaml` + `metadata.json`); `load_checkpoint`; `init_checkpoint_from_spec` for a deterministic day-zero init; `CheckpointManager` for lifecycle. *Data*: per-checkpoint directory bundle. *Working*: every trainer in this repo calls into `save_checkpoint` after each save_every interval; resumes call `load_checkpoint`.
