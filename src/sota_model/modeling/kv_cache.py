"""Paged KV cache with prefix sharing, sliding-window eviction, and int8 quantization.

Sized for SuperModel 4.7-class GQA workloads (16 KV heads × 110 layers × head_dim 128
≈ 880 KB/token at bf16). The paged layout matches vLLM's design so we can plug into
its kernels later if useful, but the core path here is plain PyTorch and works with
any attention backend.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch


KVDtype = Literal["bf16", "fp16", "int8"]


@dataclass
class KVCacheConfig:
    n_layers: int
    n_kv_heads: int
    head_dim: int
    block_size: int = 16
    max_blocks_per_seq: int = 65_536  # 1M tokens / 16 = 65,536 blocks
    dtype: KVDtype = "bf16"
    enable_prefix_cache: bool = True
    sliding_window: int | None = None  # tokens; None = unbounded
    quantize_skip_first_n_tokens: int = 64  # keep system-prompt KV in fp


def _torch_dtype(name: KVDtype) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "int8": torch.int8}[name]


class PagedKVCache:
    """Paged KV cache for a single sequence (or a small batch sharing layout).

    Memory layout: per layer, two tensors of shape
        (max_blocks_per_seq, block_size, n_kv_heads, head_dim)
    plus an index table mapping logical token positions → block IDs.

    int8 quantization stores per-channel scales/zeros alongside each block.
    """

    def __init__(self, cfg: KVCacheConfig, device: torch.device | str = "cuda"):
        self.cfg = cfg
        self.device = torch.device(device)
        self._kv_dtype = _torch_dtype(cfg.dtype)

        shape = (
            cfg.n_layers,
            cfg.max_blocks_per_seq,
            cfg.block_size,
            cfg.n_kv_heads,
            cfg.head_dim,
        )
        # We keep K and V in separate buffers for clarity; identical layout.
        self.k_storage = torch.zeros(shape, dtype=self._kv_dtype, device=self.device)
        self.v_storage = torch.zeros(shape, dtype=self._kv_dtype, device=self.device)

        # int8 quantization is per-token, per-head — one scale per
        # (layer, block, in_block, head). A coarser per-block layout would be
        # overwritten on every token write inside the block.
        if cfg.dtype == "int8":
            scale_shape = (
                cfg.n_layers, cfg.max_blocks_per_seq, cfg.block_size, cfg.n_kv_heads,
            )
            self.k_scale = torch.ones(scale_shape, dtype=torch.float32, device=self.device)
            self.v_scale = torch.ones(scale_shape, dtype=torch.float32, device=self.device)
        else:
            self.k_scale = self.v_scale = None

        # Logical position → block_id, intra-block offset.
        self._next_block_id = 0
        self._free_blocks: list[int] = []
        self._block_table: list[int] = []  # one entry per logical block of this seq
        self._n_tokens = 0
        self._position_offset = 0   # tokens evicted from the front (for RoPE consistency)

    # --- allocation ---

    def _alloc_block(self) -> int:
        if self._free_blocks:
            return self._free_blocks.pop()
        if self._next_block_id >= self.cfg.max_blocks_per_seq:
            raise RuntimeError(
                f"KV cache exhausted: {self.cfg.max_blocks_per_seq} blocks allocated "
                f"({self.cfg.max_blocks_per_seq * self.cfg.block_size} tokens). "
                "Trigger context compaction (see  4.5)."
            )
        block_id = self._next_block_id
        self._next_block_id += 1
        return block_id

    def _ensure_capacity(self, n_new_tokens: int) -> None:
        needed_blocks = (self._n_tokens + n_new_tokens + self.cfg.block_size - 1) // self.cfg.block_size
        while len(self._block_table) < needed_blocks:
            self._block_table.append(self._alloc_block())

    # --- update ---

    def append(
        self,
        layer_idx: int,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> None:
        """Append k,v of shape (T, n_kv_heads, head_dim) for one layer.

        Caller passes one layer at a time; this avoids materializing the full
        (n_layers, T, ...) tensor in tight memory regimes.
        """
        # Invariant: within one forward pass, `_n_tokens` is constant across
        # layers — it only increments after the LAST layer. So all layers
        # write to the same logical range `[_n_tokens, _n_tokens+T)` of their
        # own per-layer slice of storage.
        T = k.shape[0]
        if layer_idx == 0:
            self._ensure_capacity(T)

        bs = self.cfg.block_size
        for offset in range(T):
            global_pos = self._n_tokens + offset
            block_idx = global_pos // bs
            in_block = global_pos % bs
            block_id = self._block_table[block_idx]

            self._write(layer_idx, block_id, in_block, k[offset], v[offset])

        if layer_idx == self.cfg.n_layers - 1:
            self._n_tokens += T
            self._evict_sliding()

    def _write(
        self,
        layer_idx: int,
        block_id: int,
        in_block: int,
        k_tok: torch.Tensor,
        v_tok: torch.Tensor,
    ) -> None:
        skip_quant = self._n_tokens < self.cfg.quantize_skip_first_n_tokens
        if self.cfg.dtype == "int8" and not skip_quant:
            k_q, k_s = _quantize_int8_per_head(k_tok)
            v_q, v_s = _quantize_int8_per_head(v_tok)
            self.k_storage[layer_idx, block_id, in_block] = k_q
            self.v_storage[layer_idx, block_id, in_block] = v_q
            self.k_scale[layer_idx, block_id, in_block] = k_s
            self.v_scale[layer_idx, block_id, in_block] = v_s
        else:
            self.k_storage[layer_idx, block_id, in_block] = k_tok.to(self._kv_dtype)
            self.v_storage[layer_idx, block_id, in_block] = v_tok.to(self._kv_dtype)

    def _evict_sliding(self) -> None:
        if self.cfg.sliding_window is None:
            return
        max_tokens = self.cfg.sliding_window
        if self._n_tokens <= max_tokens:
            return
        # Drop oldest blocks; keep at most ceil(window/bs) blocks.
        keep_blocks = (max_tokens + self.cfg.block_size - 1) // self.cfg.block_size
        while len(self._block_table) > keep_blocks:
            evicted = self._block_table.pop(0)
            self._free_blocks.append(evicted)
            self._n_tokens -= self.cfg.block_size
            self._position_offset += self.cfg.block_size

    # --- read ---

    def gather(self, layer_idx: int, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        """Materialize the full (n_tokens, n_kv_heads, head_dim) K,V for one layer.

        For long-context attention prefer specialized paged-attention kernels;
        this method is the simple correctness path.
        """
        if not self._block_table:
            shape = (0, self.cfg.n_kv_heads, self.cfg.head_dim)
            return (torch.empty(shape, dtype=dtype, device=self.device),) * 2

        bs = self.cfg.block_size
        n_full_blocks = self._n_tokens // bs
        tail_tokens = self._n_tokens % bs

        ks: list[torch.Tensor] = []
        vs: list[torch.Tensor] = []
        for i, block_id in enumerate(self._block_table):
            block_k = self.k_storage[layer_idx, block_id]
            block_v = self.v_storage[layer_idx, block_id]
            if self.cfg.dtype == "int8" and self.k_scale is not None:
                block_k = _dequantize_int8(block_k, self.k_scale[layer_idx, block_id])
                block_v = _dequantize_int8(block_v, self.v_scale[layer_idx, block_id])
            if i < n_full_blocks:
                ks.append(block_k)
                vs.append(block_v)
            else:
                ks.append(block_k[:tail_tokens])
                vs.append(block_v[:tail_tokens])
                break
        return torch.cat(ks, dim=0).to(dtype), torch.cat(vs, dim=0).to(dtype)

    @property
    def logical_position(self) -> int:
        """Total tokens seen since cache creation (= n_tokens + evicted).

        Use this — not `n_tokens` — for RoPE position of the next write:
        stored K were rotated at their original logical position, so
        post-eviction writes must continue the same global counter.
        """
        return self._n_tokens + self._position_offset

    @property
    def n_tokens(self) -> int:
        return self._n_tokens

    def reset(self) -> None:
        self._free_blocks = list(range(self._next_block_id))
        self._block_table = []
        self._n_tokens = 0
        self._position_offset = 0

    # --- prefix caching helper ---

    def fork(self) -> "PagedKVCache":
        """Create a copy-on-write fork. Used for resampling / beam search.

        The forked cache shares the storage tensors (memory-cheap) and copies
        only the index table; subsequent writes go to newly allocated blocks.
        """
        forked = PagedKVCache.__new__(PagedKVCache)
        forked.cfg = self.cfg
        forked.device = self.device
        forked._kv_dtype = self._kv_dtype
        forked.k_storage = self.k_storage  # shared
        forked.v_storage = self.v_storage  # shared
        forked.k_scale = self.k_scale
        forked.v_scale = self.v_scale
        forked._next_block_id = self._next_block_id
        forked._free_blocks = []
        forked._block_table = list(self._block_table)
        forked._n_tokens = self._n_tokens
        forked._position_offset = self._position_offset
        return forked


def _quantize_int8_per_head(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # x: (n_kv_heads, head_dim)
    absmax = x.abs().amax(dim=-1).clamp_min(1e-6)
    scale = absmax / 127.0
    q = (x / scale.unsqueeze(-1)).round().clamp(-127, 127).to(torch.int8)
    return q, scale.to(torch.float32)


def _dequantize_int8(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    # q:     (block_size, n_kv_heads, head_dim) int8
    # scale: (block_size, n_kv_heads)            per (token, head) — required
    #                                             when block_size > 1.
    return q.to(torch.float32) * scale.unsqueeze(-1)
