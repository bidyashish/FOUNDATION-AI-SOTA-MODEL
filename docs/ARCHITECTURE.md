# UltraModel 5 — Underlying Architecture

The architecture reference for the current spec ([`configs/sota_ultra_5.yaml`](../configs/sota_ultra_5.yaml), ~627B dense).
Code lives in [`src/sota_model/modeling/`](../src/sota_model/modeling/); per-file design rationale in
[`modeling/README.md`](../src/sota_model/modeling/README.md); the visual shape-by-shape forward-pass walkthrough is
[`TRANSFORMER_FLOW.md`](./TRANSFORMER_FLOW.md). Every number below is recomputable from the config —
nothing here is independent of it.

## At a glance

| | UltraModel 5 | Code |
|---|---|---|
| Type | dense decoder-only transformer (no MoE) | `transformer.py::SOTAModel` |
| Total / active params | 627.06B / 627.06B (dense — all active) | `ModelConfig.estimate_params_billions()` |
| Layers | 128 | |
| Hidden (d_model) | 18,432 | |
| Attention | GQA — 144 Q / 18 KV heads, head_dim 128 (8:1) | `attention.py::GroupedQueryAttention` |
| QK-norm | per-head RMSNorm on Q,K pre-RoPE | `ModelConfig.qk_norm = true` |
| FFN | SwiGLU, 73,728 (4.0×d) | `layers.py::SwiGLU` |
| Norm | RMSNorm, pre-norm, eps 1e-5, fp32 internals | `layers.py::RMSNorm` |
| Positions | RoPE base 1e6 + YaRN ×128 (8K → 1M) | `rope.py` |
| Context | 1,048,576 (10M agentic via compaction at 200K) | |
| Attention pattern | hybrid: every 2nd layer sliding-window 32K, rest full | `sliding_window_layer_stride: 2` |
| Vocab | 200,000 BPE, byte fallback, untied embeddings | `tokenizer/bpe.py` |
| Multimodal | ViT tower + 2×2 pixel-shuffle MLP projector | `modeling/vision/` |
| Thinking | learned effort head, budgets 0 → 131,072 tokens | `transformer.py::EffortHead` |
| Train precision | fp8 (TE E4M3 fwd / E5M2 bwd), bf16 master under DeepSpeed | |

## The block

128 identical-shaped pre-norm blocks (per-layer overrides excepted):

```
        ┌────────────────────────────────────────────────┐
  x ────┤  RMSNorm                                       │
        │     ↓                                          │
        │  GQA: q_proj (144·128)  k,v_proj (18·128 each) │
        │       QK-norm (RMSNorm over head_dim, pre-RoPE)│
        │       RoPE (base 1e6, YaRN ×128)               │
        │       FlashAttention-3 / SDPA fallback         │
        │       [full attn | 32K sliding window]         │
        │       o_proj                                   │
  x ──(+)── residual                                     │
        │  RMSNorm                                       │
        │     ↓                                          │
        │  SwiGLU: down( silu(gate(x)) · up(x) ), 73728  │
  x ──(+)── residual                                     │
        └────────────────────────────────────────────────┘
```

Embedding → 128 blocks → final RMSNorm → untied LM head (200K logits).

## Parameter accounting (sums to 627.06B)

| Component | Per layer | × | Total |
|---|---|---|---|
| Token embeddings | — | | 3.686B |
| Attention (q,k,v,o) | 764.4M | 128 | 97.84B |
| SwiGLU FFN (gate,up,down) | 4.077B | 128 | 521.84B |
| RMSNorms (incl. QK-norm) | ~37K | 128 | ~0.005B |
| LM head (untied) | — | | 3.686B |
| **Total** | | | **627.06B** |

FFN holds 83% of the parameters — that is what `tapered_ffn_overrides()` exploits to land precise
param targets without touching `d_model`/`n_layers` (per-layer `ffn_dim` flows through
`ModelConfig.layer_config(i)` into each block).

## Attention: why this shape

**GQA 8:1.** KV cache per token = `n_layers × n_kv_heads × head_dim × 2 (K+V) × 2 B (bf16)`
= 128 × 18 × 128 × 4 = **1152 KiB/token**, i.e. 1152 GiB at the full 1M context (576 GiB at the
fp8 cache default). Full MHA (144 KV heads) would be 8× that — ~9 TiB at 1M, undeployable.
18 KV heads (not 16) keeps the 8:1 ratio at 144 Q heads **and** shards at TP 9 — the GB300 NVL72
layout (`docs` in the YAML; 18 does *not* divide by TP 8).

**QK-norm.** RMSNorm over `head_dim` applied to Q and K per head, before RoPE. Bounds
attention-logit growth — the dominant fp8 instability at this width. This is the 2026 successor
to Gemma-2-style logit softcapping (dropped industry-wide because FlashAttention-3 has no
softcap path). Costs 2×128 params/layer — noise.

**Hybrid attention pattern.** `sliding_window_layer_stride: 2` puts a 32K sliding window on
layers 2, 4, …, 126 (63 layers) and full attention on the remaining 65. Windowed layers drop KV
beyond 32K, halving long-context KV growth; full layers preserve global retrieval (GraphWalks
BFS @ 1M ≥ 79 gate). Pin individual layers with `hybrid_attention_overrides(128, (0, -1))`.

**Positions.** RoPE at base 1e6 with YaRN scaling ×128 — exactly `max_position_embeddings /
rope_yarn_original_max_position` = 1,048,576 / 8,192. YaRN interpolates only low-frequency bands
and leaves high-frequency (short-range) bands untouched (`rope.py::_yarn_corrected_freqs`).

## Stability & initialization (the fp8-era kit)

| Mechanism | Setting | Where |
|---|---|---|
| QK-norm | on | `attention.py` |
| Z-loss | 1e-4 | `pretrain.py::lm_loss` — penalizes log²Z, stops logit drift |
| Residual-scaled init | normal(0, 1/√18432 ≈ 0.00737); `o_proj`/`down` × 1/√(2·128) | `transformer.py::_init_parameters` |
| fp32 norm internals | always | `layers.py::RMSNorm` |
| fp8 training | TE E4M3 fwd / E5M2 bwd; DeepSpeed runs bf16 master + comms | `parallelism.py::deepspeed_config_for` |
| Grad clip / β₂ | 1.0 / 0.95 | YAML `training:` |

## Adaptive thinking

A small MLP (`EffortHead`: d_model → 1024 → scalar) pools the last 8 prompt positions and emits
an effort logit, thresholded into {min, low, medium, high, max} → token budgets
{0, 1024, 8192, 32768, 131072}. Thinking tokens stream into `<|thinking|>…<|/thinking|>`,
hidden from the user but kept in KV; RL post-training never grades the hidden channel
(`rlhf.py::mask_thinking_positions`).

## Multimodal path

Images resize to ≤2576 px long edge / ≤3.75 MP, patchify at 14×14 into a ViT (same RMSNorm +
GQA + SwiGLU blocks), then a 2×2 pixel-shuffle + MLP projector lifts patches into LM `d_model`
rows — max ⌊3.75M / 14²⌋ / 4 = **4783 image tokens** (4738 at exactly 2576 px long edge).
`SOTAModel.forward(image_features=…)` splices rows 1-to-1 over `<|image_start|>…<|image_end|>`
placeholders.

## What it deliberately is NOT

- **Not MoE** — predictable latency under agentic/tool loads, clean white-box attribution, no
  expert-balancing pathologies; the cost is FLOPs/inference (see README §1.2).
- **No MLA** (DeepSeek-style latent KV) — would require redesigning the uniform-shape paged KV
  cache; GQA at 8:1 already clears the 1M memory bar.
- **No attention softcapping** — superseded by QK-norm (FA3 compatibility).
- **No biases** on any Linear; **no learned positions**; **no embedding tying** (200K vocab head
  earns its 3.7B at this scale).
- **Per-layer `n_kv_heads`/`head_dim` heterogeneity** — unsupported by design; the paged KV cache
  assumes a uniform `(n_layers, n_kv_heads, head_dim)` shape.

## How it shards (GB300 NVL72 reference layout)

One replica = one rack: **TP 9 × PP 8 = 72 B300s**; DP 64 → 4608 GPUs.

| Per-GPU shard | Value |
|---|---|
| Pipeline stage | 16 layers (128 / 8) |
| Hidden slice | 2,048 (18,432 / 9) |
| Heads | 16 Q + 2 KV (144 / 9, 18 / 9) |
| FFN slice | 8,192 (73,728 / 9) |
| Params | ≈8.71B (627B / 72) |

`parallelism.py::validate_parallel_layout` enforces TP divisibility at launch (vocab 200,000 pads
to 200,007 at TP 9 — warned, Megatron-standard).
