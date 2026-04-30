# `sota_model/modeling/`

The model itself. Five files + the vision tower, each one design-decision per file.

```
layers.py         RMSNorm + SwiGLU
rope.py           Rotary embeddings + YaRN scaling
attention.py      Grouped-Query Attention with Flash-Attn fast path
kv_cache.py       Paged KV cache with int8 quantization + sliding eviction + COW fork
transformer.py    Top-level model: stacks blocks, adds EffortHead, image-feature splice
vision/           Vision encoder + projector (modelcard invariant 6: 2576 px / 3.75 MP).
                  See vision/README.md.
```

---

## `layers.py`

### RMSNorm — why, not LayerNorm

LayerNorm subtracts the mean and divides by the std-dev. RMSNorm only divides by the root-mean-square. At LLM scale:

- **Same loss curve.** RMSNorm and LayerNorm converge to indistinguishable validation loss.
- **~10% fewer FLOPs** per norm — the mean computation and subtraction are gone.
- **Better numerical behavior** at bf16: variance computed without first centering avoids one round of fp16 catastrophic cancellation.

This is now the default for every frontier dense model — Llama, Mistral, Qwen, the SuperModel-class architecture in ``.

### SwiGLU — why, not plain GeLU

SwiGLU computes `down(silu(gate(x)) * up(x))` — three matrices instead of two, but the gating doubles effective capacity per parameter:

- **+1–2 perplexity points** at fixed parameter count vs. GeLU + plain MLP.
- The `silu(gate(x)) * up(x)` form lets the model learn *what to compute* and *whether to compute it* simultaneously.
- `ffn_dim=65536 = 4×d_model` is sized so that with the gate+up split, total FFN params land near "vanilla 4×" — the comparable-capacity baseline.

---

## `rope.py`

### RoPE — why, not learned position embeddings

- **Generalizes beyond training length.** YaRN scaling extends an 8K-trained model out to 1M without retraining. Learned absolute positions cannot do this.
- **Encodes relative position naturally.** A dot product of two RoPE-rotated vectors depends only on the difference of their angles — exactly what attention needs.
- **No extra parameters.** Embedding tables for 1M positions would dwarf the LM head.

### YaRN scaling — why, not naive linear interpolation

Linear interpolation of RoPE frequencies (NTK-aware scaling) blurs short-range signal. YaRN keeps high-frequency dimensions un-scaled (preserving short-range information) and only interpolates low-frequency dimensions (for long-range generalization). The cutoff is determined by the original max position seen in training (`rope_yarn_original_max_position=8192`).

Concretely in `_yarn_corrected_freqs()`:

```
for each frequency band:
    if wavelength < high_freq_cutoff:    keep extrapolated (no scaling)
    if wavelength > low_freq_cutoff:     fully interpolated by 1/scale
    otherwise:                           smooth interpolation between the two
```

`rope_base=1e6` (vs. classic 1e4) raises the wavelength of every dimension by 100×, so even high-frequency dims don't wrap around at 1M.

---

## `attention.py`

### GQA — why 128:16, not 128:128 (MHA) or 1:1 (MQA)

The KV cache is the dominant memory cost at long context. Per-token KV size:

```
KV per token  =  2 (K + V) × n_kv_heads × head_dim × n_layers × bytes_per_element
              =  2 × 16 × 128 × 110 × 2  bytes  (bf16)
              =  ~880 KB per token

At 1M tokens                   =  ~880 GB     (fits across cluster)
At 1M tokens with 128 KV heads =  ~7 TB       (does not fit, full stop)
At 1M tokens with 1 KV head    =  ~55 GB      (fits, but quality regression)
```

128:16 is the sweet spot:
- 8× memory reduction over MHA — the difference between "ships" and "doesn't ship."
- Quality regression vs. MHA is within noise on every benchmark in `` 8.
- MQA (1 KV head) loses measurable quality on long-context tasks like MRCR v2.

### Flash Attention path

The fast path uses `flash_attn_func` (Flash Attention 2 on Ampere, Flash Attention 3 on Hopper):
- Tiles the softmax along the sequence axis.
- Keeps the working set in on-chip SRAM, avoiding the O(N²) HBM materialization of standard attention.
- Same numerical result, ~2–4× faster for long sequences.

The SDPA fallback is functionally identical and runs anywhere PyTorch runs — used for CPU-only test runs and CUDA installs without `flash-attn`.

### Sliding window per layer

Every other layer (controlled by `sliding_window_layer_stride=2`) uses a sliding window of `sliding_window_size=32768` tokens. The remaining layers stay full-attention so global-retrieval evals (MRCR v2 at 1M, GraphWalks BFS) still work.

This hybrid is what keeps long-context KV memory bounded while preserving needle-in-haystack capability — pure-sliding models fail GraphWalks; pure-full-attention models OOM.

---

## `kv_cache.py`

### Paged layout — why, not a single contiguous tensor

A naive `(n_layers, max_seq, n_kv_heads, head_dim)` tensor allocated up-front:
- Wastes memory on every conversation shorter than max_seq (most of them).
- Fragments under arbitrary-length agentic sessions.
- Can't share a system prompt across requests.

vLLM-style **paged attention** allocates fixed-size blocks (`block_size=16` tokens) on demand, indexed by a per-sequence block table. Three benefits the design depends on:

1. **No fragmentation.** Every block is the same size; freed blocks go back to a free list.
2. **Prefix caching.** Two requests with the same system prompt share the prefix blocks. The block table for one points at the same memory.
3. **Copy-on-write fork.** Resampling, beam search, and the welfare/safety evals in `` 6 fork conversations cheaply — `fork()` shares storage and copies only the index.

### int8 quantization

For deployments memory-bound at int8:
- Per-channel scales (one float per (layer, block, head)) recover most of the precision loss.
- First N tokens (`quantize_skip_first_n_tokens=64`) stay in fp — the system prompt's KV is high-leverage and worth full precision.
- Quality cost on `` benchmarks: ≤1pp on MASK / false-premise; negligible elsewhere.

### Sliding-window eviction

When `sliding_window` is set on the cache, blocks past the window are returned to the free list. This bounds per-sequence memory at `sliding_window / block_size` blocks — enabling sliding-attention layers (attention) to actually save the memory the math promises.

---

## `transformer.py`

### Block ordering: pre-norm + residual

```
x  ─►  RMSNorm  ─►  GQA       ─►  +x  ─►  RMSNorm  ─►  SwiGLU  ─►  +x  ─►
```

Pre-norm (norm before the sublayer) is more stable at depth than the original post-norm. Every frontier dense model uses pre-norm.

### EffortHead — why a separate small MLP

Adaptive thinking needs to decide effort *before* answer generation begins. The cheapest way:

1. Run prefill over the prompt as normal.
2. Pool the last 8 hidden states of the final layer.
3. Pass the pool through a 2-layer GeLU MLP → single scalar logit.
4. Threshold the logit to pick `min/low/medium/high/max` — see `inference/thinking.py`.

Cost: ~`d_model × hidden + hidden` parameters, ≈8M params on a 200B model — invisible.

### Gradient checkpointing

`enable_gradient_checkpointing()` swaps every block's forward pass for `torch.utils.checkpoint.checkpoint(...)`. Activation memory drops ~6×; forward FLOPs increase ~30%. The deal is paid for at 110 layers × 8K seq.

### Tied embeddings

`tie_embeddings=False` at this scale. The LM head is a 200K × 16384 matrix (~3.3B params); tying it to the embedding table constrains the output distribution unhelpfully. Smaller models tie to save params — not relevant here.

### Image-feature splice

`SOTAModel.encode_image(img)` runs the vision encoder + projector and returns `(n_image_tokens, d_model)`. `SOTAModel.forward(input_ids, image_features=…, image_token_id=…)` finds the `image_token_id` slots in `input_ids` and replaces those embedding rows with `image_features` — 1-to-1, so the rest of the LM stack treats text and image tokens uniformly. See [`vision/README.md`](./vision/README.md) for the full splice contract.
