# UltraModel 5 — Transformer Flow, Visually

Companion to [`ARCHITECTURE.md`](./ARCHITECTURE.md): that doc says *what* the model is; this one
traces *how* a token flows through it — every tensor shape, at the real UltraModel 5 dimensions,
matching the code in [`src/sota_model/modeling/`](../src/sota_model/modeling/) line for line.

**Shape legend** (from `configs/sota_ultra_5.yaml`):

| Symbol | Value | Meaning |
|---|---|---|
| `B` | batch (1 on the cached inference path) | |
| `T` | tokens in this forward call | prompt length at prefill, 1 at decode |
| `d` | 18,432 | hidden width (`d_model`) |
| `Hq` / `Hkv` | 144 / 18 | query / KV heads (8:1 GQA) |
| `Dh` | 128 | head dim (`Hq × Dh = d`) |
| `L` | 128 | layers |
| `V` | 200,000 | vocab |
| `N` | tokens already in the KV cache | grows by T each call |

## 1. Request lifecycle (inference, 10,000 ft)

```
 prompt text ──► tokenizer (200K BPE) ──► input_ids (1, T)
                                              │
                                              ▼  PREFILL — one forward over the whole prompt
                              ┌────────────────────────────────┐
                              │ SOTAModel.forward(...)         │──► KV cache filled with N=T tokens
                              │ compute_effort=True            │──► effort logit ──► tier ∈ {min…max}
                              └────────────────────────────────┘        │
                                              ┌─────────────────────────┘
                                              ▼  budget B(tier) ∈ {0, 1K, 8K, 32K, 131K}
                               <|thinking|> … hidden tokens … <|/thinking|>     (kept in KV,
                                              │                                  never shown)
                                              ▼  DECODE LOOP — T=1 per step
                          ┌──► forward(last_token, kv_cache) ──► logits (1, 1, 200000)
                          │                                          │
                          │    sampler: ÷ temperature 0.7 ──► top_p 0.95 ──► multinomial
                          │    (top_k=0 and repetition_penalty=1.0 — both no-ops)
                          └────────── next token ◄───────────────────┘
                                              │ until <|im_end|> / max_new_tokens
                                              ▼
                          (compaction at 200K ctx: oldest ~80% → summary → re-prefill)
```

## 2. One forward pass, shape by shape

```
input_ids (B, T) ─ int64
   │
   ▼  nn.Embedding(200000, 18432)                            transformer.py::SOTAModel.forward
hidden (B, T, 18432)
   │  [multimodal: rows at <|image|> placeholder positions are overwritten 1:1
   │   with projector output — up to 4783 rows per image]
   ▼
╔═════════════════ × 128 blocks (16 per pipeline stage) ════════════════════╗
║                                                                           ║
║  x ──► RMSNorm ──► ATTENTION (§3) ──► +x   (residual)                     ║
║              (B,T,18432)      (B,T,18432)                                 ║
║                                                                           ║
║  x ──► RMSNorm ──► SwiGLU (§4) ──────► +x  (residual)                     ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
   │
   ▼  final RMSNorm
   ▼  lm_head: Linear(18432 → 200000), untied
logits (B, T, 200000)        effort_logit (B,) when compute_effort=True
```

## 3. Inside attention (`attention.py::GroupedQueryAttention`)

```
x (B, T, 18432)
   │
   ├─ q_proj ──► (B, T, 18432) ──view──► (B, T, 144, 128)  ┐
   ├─ k_proj ──► (B, T,  2304) ──view──► (B, T,  18, 128)  ├─ QK-norm: RMSNorm over
   └─ v_proj ──► (B, T,  2304) ──view──► (B, T,  18, 128)  ┘  Dh on q,k (pre-RoPE)
   │
   ▼  transpose → (B, H, T, 128);  RoPE rotates q,k at positions
   │  [past_len, past_len+T) where past_len = kv_cache.logical_position
   │  (NOT n_tokens — numbering survives sliding-window eviction)
   ▼
   KV CACHE (B=1 path):  append(layer_idx, k, v) ──► gather ──► full_k, full_v
                                                       (1, 18, N+T, 128)
   ▼
   GQA: 144 q heads attend over 18 KV heads — each KV head serves a
   group of 8 q heads (repeat_interleave on the portable path; the
   FlashAttention-3 kernel groups natively):

        q heads   0..7   8..15  16..23 ... 136..143
                   └──┬──┘ └──┬──┘ └──┬──┘     └──┬──┘
        kv head      0       1       2    ...    17
   ▼
   softmax(q·kᵀ/√128 + mask) · v          mask = causal  AND, on windowed
   (B, 144, T, 128)                       layers, |i−j| < 32768
   ▼
   transpose/reshape → (B, T, 18432) ──► o_proj ──► (B, T, 18432)
```

**Which layers see what** (`sliding_window_layer_stride: 2`):

```
layer:    0    1    2    3    4    5   ...  125  126  127
attn:   FULL FULL  SWA FULL  SWA FULL  ...  FULL  SWA FULL
          └─ 65 full layers: global retrieval (GraphWalks @1M)
             63 SWA layers: KV beyond 32K dropped, not stored
```

## 4. Inside the FFN (`layers.py::SwiGLU`)

```
x (B, T, 18432)
   ├─ gate: Linear(18432 → 73728) ─► silu ─┐
   ├─ up:   Linear(18432 → 73728) ─────────┤  elementwise ·
   │                                       ▼
   └─ down: Linear(73728 → 18432) ◄── (B, T, 73728)
```

Three projections, no biases — 4.077B params/layer, 83% of the whole model. Per-layer
`ffn_dim` overrides (`tapered_ffn_overrides`) land here via `cfg.layer_config(i)`.

## 5. KV-cache flow (`kv_cache.py::PagedKVCache`)

Storage is paged: per layer, a pool of 16-token blocks indexed by a block table.

```
 append(k,v for T new tokens)          gather(layer) → contiguous (N, 18, 128)
        │                                       ▲
        ▼                                       │
 block table: [b₃][b₇][b₂][b₉]...  ──reads──────┘
   each block: (16, 18, 128) ×2 (K,V)     1152 KiB per token at bf16
                                          (576 KiB at the fp8 default)
 sliding eviction (SWA layers): drop whole leading blocks;
   _position_offset += 16·dropped  →  logical_position stays monotonic,
   so RoPE numbering never shifts under eviction
 fork(): copy-on-write clone — block table copied, data shared
   (cheap resampling / best-of-n; safety evals lean on this)
 int8 mode: per-(token, head) scale stored beside each block
```

Decode-step cost asymmetry: **prefill** runs T tokens through all weights once
(compute-bound); **decode** reads the entire cache (N × 1152 KiB across layers) to
produce each single token (memory-bandwidth-bound). That asymmetry is why GQA (8× cache
reduction) and the fp8 cache default exist.

## 6. Training flow (`training/pretrain.py`)

```
corpus batch (micro_batch=1, seq_len=8192)  ◄─ CorpusLoader: mix-weighted interleave,
   │                                            filter chain, BlockPacker, 2048-block
   ▼                                            shuffle reservoir (data_seed)
 forward — no KV cache, full causal mask, gradient checkpointing on
   │        (activations recomputed in backward, layer by layer)
   ▼
 lm_loss: shift logits/labels by 1 → cross-entropy (ignore −100)
          + 1e-4 · mean(log²Z)            ◄─ z-loss: logit-drift guard under fp8
   │
   ▼  backward; accumulate ×16 micro-steps
 clip grad-norm 1.0 ──► AdamW(β₂=0.95, wd=0.1) ──► cosine LR
                                                   (3000 warmup → floor 0.1×peak)

 one optimizer step = 8192 × 1 × 16 × 64 DP replicas = 8,388,608 tokens
```

Where parallelism cuts this flow (GB300 NVL72, one replica = one rack):

```
 TP 9   slices every matmul 9-way: q_proj computes (B,T,16·128) per rank,
        SwiGLU computes a 8192-wide slice; all-reduce after attn + FFN
 PP 8   layers 0-15 on stage 0 … 112-127 on stage 7; activations hop stages
 DP 64  64 racks run replicas; ZeRO-3 shards params/grads/optimizer across DP
```

## 7. File map for this flow

| Step | Code |
|---|---|
| Embedding, block stack, image splice, effort head | `modeling/transformer.py` |
| Q/K/V/O, QK-norm, GQA, Flash/SDPA, window mask | `modeling/attention.py` |
| RMSNorm, SwiGLU | `modeling/layers.py` |
| RoPE + YaRN band correction | `modeling/rope.py` |
| Paged cache, eviction, fork, int8 | `modeling/kv_cache.py` |
| Sampler (temperature → top-p) | `inference/sampler.py` |
| Prefill/decode/compaction orchestration | `inference/engine.py` |
| Loss, accumulation, LR schedule | `training/pretrain.py` |
