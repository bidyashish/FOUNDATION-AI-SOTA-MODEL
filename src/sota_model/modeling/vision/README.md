# `sota_model/modeling/vision/`

The multimodal vision tower that lifts modelcard invariant 6 (2576 px long edge / 3.75 MP cap) into the LM. Without this module, ScreenSpot-Pro and LAB-Bench FigQA targets in 8.9 are unreachable.

```
__init__.py     re-exports
encoder.py      VisionEncoder, VisionEncoderConfig, ImageInput, VisionFeatures, preprocess_image
projector.py    VisionLanguageProjector (linear OR pixel_shuffle_mlp)
```

## Code

- **`preprocess_image(image, cfg)`** — resize → 2576 px long edge → 3.75 MP cap → patch-aligned crop → ImageNet normalize. Accepts PIL or `(C,H,W)` torch.Tensor in `[0,1]`.
- **`VisionEncoder`** — 32-layer ViT (default config), `RMSNorm + GroupedQueryAttention + SwiGLU` — same building blocks as the LM so kernel paths are shared. Returns `VisionFeatures(features=(N,D_v), grid=(Hp,Wp))`.
- **`VisionLanguageProjector`** — two modes:
  - `linear`: a single `nn.Linear(D_v, D_lm)`. Smallest, used for ablation.
  - `pixel_shuffle_mlp` *(default)*: tile 2×2 patches into one merged token, then RMSNorm + 2-layer MLP. Cuts the image-token count by 4× while preserving OCR-relevant detail.
- **`VisionEncoderConfig.from_model_config(mc)`** — derives caps from `ModelConfig.vision_*` fields so the encoder is locked to the modelcard invariant by construction.

## Data

- Input: any RGB image. Worst-case shapes after preprocess (patch_size=14):
  - `2576 × 1456` → `184 × 104` patches → 19,136 tokens.
  - `2576 × 2576` (clipped to 3.75 MP) → ~`182 × 105` → ~19,110 tokens.
  Both fit comfortably inside the 1M-token context window.
- Reference normalization: ImageNet mean `(0.485, 0.456, 0.406)`, std `(0.229, 0.224, 0.225)`. Operators retraining vision should re-export the matching mean/std.
- No vision corpus shipped in this repo. Production datasets are operator-supplied; format is `(image_bytes, caption_text)` pairs (SigLIP-style contrastive) plus image-conditioned captioning.

## Working

The splice into the LM is deterministic:

1. The chat template emits `<|image_start|><|image|>×N<|image_end|>` where `N` is the per-image token count chosen by the projector.
2. Inference calls `feats = SOTAModel.encode_image(img)`; `feats.shape == (N, d_model)`.
3. The forward call passes both `image_features=feats` and `image_token_id=<id of <|image|>>`.
4. `SOTAModel.forward` finds the `image_token_id` positions in `input_ids` and replaces those embedding rows with `feats`.
5. The merged sequence flows through the standard transformer stack — RoPE, GQA, KV cache, the lot.

This 1-to-1 splice is the contract that lets the same LM forward pass handle text-only AND multimodal turns without branching the model code.

### Training-time wiring

- Stage-3 refinement adds joint vision-LM training (modelcard 1.4): freeze nothing, fine-tune both vision and LM under the standard next-token loss with image-conditioned captions.
- Stage-2 long-context still trains text-only; the vision encoder is initialized but receives no gradient.
- For LoRA-style adapter merges into the projector or attention, `checkpoint/merge.py::merge_lora_into_base` handles the math.

### Operator checklist

- [ ] `vision_enabled=True` in `ModelConfig` (or YAML).
- [ ] Confirm `preprocess_image(huge, cfg).pixels.shape` clips to `<= 2576` long edge AND `<= 3_750_000` total pixels.
- [ ] Confirm `(Hp * Wp)` matches `len(VisionEncoder(...).features)` → no off-by-one in patch counting.
- [ ] After projector, `n_image_tokens` divides exactly into the `<|image|>` placeholder count emitted by the chat template.
- [ ] Image-conditioned captions in the SFT corpus pass the same contamination filter as text shards.
