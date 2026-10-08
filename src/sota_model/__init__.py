"""SOTA dense foundation model targeting UltraModel 5-class capabilities.

See the UltraModel 5 System Card (mirrored as `capability_targets` /
`safety_thresholds` in configs/sota_ultra_5.yaml) for the targets this
implementation aims at.
"""

from sota_model.config import (
    InferenceConfig,
    ModelConfig,
    TrainingConfig,
    load_implied,
)

__version__ = "0.1.0"
__all__ = ["ModelConfig", "TrainingConfig", "InferenceConfig", "load_implied"]
