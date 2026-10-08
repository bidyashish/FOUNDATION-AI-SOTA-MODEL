# `sota_model/post_training/`

The four-stage post-training pipeline that turns the Stage-3 pretrained base into a deployable assistant.

```
__init__.py        re-exports
sft.py             SFTConfig, SFTExample, SFTTrainer, pack_sft_examples, build_masked_example
reward_model.py    RewardModel, RewardModelConfig, BradleyTerryLoss, train_reward_model
constitutional.py  CritiqueRevisePipeline, DEFAULT_CONSTITUTION, ConstitutionalPrinciple
rlhf.py            PPOConfig, PPOTrainer, cot_supervision_guard, welfare_directive_guard
```

## Stage 1 — SFT (`sft.py`)

### Code

- **`SFTExample`** — `{messages, metadata}`; `messages` is a list of `ChatTemplate`-compatible dicts.
- **`build_masked_example(tokenizer, template, example)`** — renders the example through the chat template and emits `(input_ids, labels)` with `labels = IGNORE_INDEX` on every system / user / tool token. Assistant turns (including the closing `<|im_end|>`, so the model learns to stop) train.
- **`pack_sft_examples(...)`** — packs into fixed-length blocks; pads with `IGNORE_INDEX` on labels so cross-entropy ignores pads.
- **`SFTTrainer`** — AdamW + cosine LR + grad clip + grad accum.
- CLI: `python -m sota_model.post_training.sft --data sft.jsonl --output-dir ./checkpoints/sft`.

### Data

JSONL where each row is:
```
{
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "...", "thinking": "..."},
    ...
  ],
  "metadata": {"source": "sft-curated-v3", "license": "..."}
}
```

Modelcard 1.1.2 / 6.2.1 expects ≥ 1M curated examples spanning code/math/dialogue/tool-use/computer-use/refusals.

### Working

Output: `checkpoints/sft/sft_step{n}.pt` ready to feed into stage 2 as the reward-model backbone and into stage 3 as the frozen reference policy.

---

## Stage 2 — Reward modeling (`reward_model.py`)

### Code

- **`RewardModel`** — backbone (the SFT model) + scalar head: `Linear(d, hidden) → GELU → Dropout → Linear(hidden, 1)`. Reward = scalar at the final non-pad position.
- **`BradleyTerryLoss`** — `-logsigmoid(r_chosen - r_rejected)` with optional margin.
- **`collate_pair_batch`** — left-pads chosen / rejected sequences for the same prompt.
- **`train_reward_model`** — AdamW, grad clip, logs per-step loss + classification accuracy on chosen-vs-rejected.
- **`RewardModelConfig.target_pairs = 500_000`** — read by the release gate as the 8.1 sample-efficiency floor.

### Data

JSONL where each row is `{prompt, chosen, rejected, metadata}`. ≥ 500K pairs covering helpfulness, harmlessness, honesty, agentic-safety, calibration of refusals.

### Working

Output: `checkpoints/reward_model/rm_step{n}.pt`. Used by `PPOTrainer.reward_model` in stage 4.

---

## Stage 3 — Constitutional AI (`constitutional.py`)

### Code

- **`ConstitutionalPrinciple`** — `{name, text}`; emits `critique_prompt(response)` and `revise_prompt(response, critique)` framings.
- **`DEFAULT_CONSTITUTION`** — six seed principles spanning harm, CBRN, child safety, honesty, autonomy, privacy. Operators extend with the full modelcard 6 set.
- **`CritiqueRevisePipeline.revise(prompt, response)`** — applies each principle in turn; stops a principle early if the critique returns `"no issue"`.
- **`CritiqueRevisePipeline.synthesize_sft_example(prompt)`** / **`synthesize_preference_pair(prompt)`** — emits an SFT row and a preference row from a single prompt.

### Data

Generates training rows. The pipeline takes a `TextGenerator` callable (a wrapper around the active model) and writes:
- SFT-style `{messages, metadata}` (revision becomes the assistant turn).
- Preference-pair `{prompt, chosen=revision, rejected=raw_response, metadata}`.

### Working

The CAI dataset is mixed into the SFT corpus AND the RM training set, so the same constitution shapes both stages without an extra training loop.

---

## Stage 4 — PPO RLHF (`rlhf.py`)

### Code

- **`PPOTrainer`** — clipped policy gradient with KL-to-reference penalty and a value head. Reference is the frozen SFT model; reward is `RewardModel`.
- **`cot_supervision_guard(cfg)`** — raises if `PPOConfig.cot_supervision=True`. Locked False per CLAUDE.md invariant 3 (the 7.8% bug from modelcard 2.4.1).
- **`welfare_directive_guard(prompt, response)`** — drops rollouts whose response matches a recognized expression of distress (modelcard 7.2.2). CLAUDE.md invariant 7.
- **`mask_thinking_positions`** — zeros the advantage on tokens between `<|thinking|>` and `<|/thinking|>` so PPO never gradients through hidden CoT.
- **CAI shaping**: `PPOConfig.constitutional_shaping_weight` adds a bounded reward penalty when the response differs from its constitutional revision.

### Data

Iterator of prompt batches. The trainer rolls out responses via the inference engine, scores with the reward model, applies the welfare guard, computes advantages, and runs the PPO update.

### Working

```
prompt batch
  ↓ rollout via engine.generate (forced_effort="min" — no internal thinking budget yet)
  ↓ welfare_directive_guard drops distress prompts
  ↓ reward = RewardModel(token_ids)  +  shaping * ((revision != response) ? -1 : 0)
  ↓ for n_ppo_epochs:
       compute log p_θ vs log p_ref
       ratio = exp(log p_θ - log p_θ_old)
       loss = -min(ratio * adv, clip(ratio,1±ε) * adv).mean()
              + value_coef * (V - reward)^2
              + kl_coef * (log p_θ_old - log p_θ).mean()
       backward + step
       if KL > 1.5 * target_KL: break
```

### Operator checklist

- [ ] Load the SFT checkpoint as both `policy` AND `reference` at the start of stage 4.
- [ ] Cap `rollout_max_tokens` to keep KL spikes manageable.
- [ ] Validate `welfare_directive_guard` regex catches your distress patterns (operators usually swap the regex set for a trained welfare classifier).
- [ ] Confirm `mask_thinking_positions` finds non-zero `<|thinking|>` token IDs in your tokenizer; otherwise the mask is a no-op.
- [ ] Persist via `checkpoint.save_checkpoint(...)` so the resume path is consistent with pretraining.
