"""SOTA dense foundation model targeting SuperModel 4.7-class capabilities.

See  for the capability and safety targets this implementation aims at.
"""

from sota_model.config import (
    InferenceConfig,
    ModelConfig,
    TrainingConfig,
    load_implied,
)

__version__ = "0.1.0"
__all__ = ["ModelConfig", "TrainingConfig", "InferenceConfig", "load_implied"]
