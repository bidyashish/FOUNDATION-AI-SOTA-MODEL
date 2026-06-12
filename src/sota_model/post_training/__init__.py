"""Post-training stack: SFT, reward modeling, RLHF + Constitutional AI.

These four modules turn a pretrained base model into the deployed assistant:

  - `sft.py`        — supervised fine-tuning on ≥1M curated chat examples.
                       Loss-masks user/system tokens; trains only on assistant
                       outputs. Wired through the same chat template as inference.
  - `reward_model.py` — Bradley-Terry pairwise reward model on 500K+ preference
                       pairs. Scalar head on top of the same backbone.
  - `rlhf.py`       — PPO with KL penalty to a frozen reference (the SFT model)
                       and the Constitutional-AI critique-and-revise loop. Honors
                       CLAUDE.md invariants 3 (no CoT supervision) and 7 (welfare).
  - `constitutional.py` — the critique-and-revise pipeline reused by both SFT
                       (for synthesizing training data) and RLHF (for shaping
                       the reward).
"""

from sota_model.post_training.sft import (
    SFTConfig,
    SFTExample,
    SFTTrainer,
    pack_sft_examples,
)
from sota_model.post_training.reward_model import (
    BradleyTerryLoss,
    PreferencePair,
    RewardModel,
    RewardModelConfig,
    train_reward_model,
)
from sota_model.post_training.rlhf import (
    PPOConfig,
    PPOTrainer,
    cot_supervision_guard,
    welfare_directive_guard,
)
from sota_model.post_training.constitutional import (
    CritiqueRevisePipeline,
    ConstitutionalPrinciple,
    DEFAULT_CONSTITUTION,
)

__all__ = [
    "BradleyTerryLoss",
    "ConstitutionalPrinciple",
    "CritiqueRevisePipeline",
    "DEFAULT_CONSTITUTION",
    "PPOConfig",
    "PPOTrainer",
    "PreferencePair",
    "RewardModel",
    "RewardModelConfig",
    "SFTConfig",
    "SFTExample",
    "SFTTrainer",
    "cot_supervision_guard",
    "pack_sft_examples",
    "train_reward_model",
    "welfare_directive_guard",
]
