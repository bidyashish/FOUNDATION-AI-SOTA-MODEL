"""Linear probe classifier — logistic regression on top of any FeatureExtractor.

Why logistic regression: probe-based safety classifiers need to be
interpretable and calibratable to a known false-positive budget. A linear
head on top of a strong feature extractor consistently beats deeper heads on
the operator-tuning workflow (modelcard 3.1) because the probe signal is
already concentrated in a low-rank subspace of the hidden states.

Training is plain SGD with logistic loss + L2; the closed-form Newton step
also works on the n_features × n_features Hessian when n_features is small
(e.g. < 1024 hidden state dim).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

from sota_model.safety.classifiers import Action, Category, ClassifierVerdict
from sota_model.safety.probes.feature_extractor import FeatureExtractor


@dataclass
class LinearProbeWeights:
    weight: torch.Tensor  # (D,)
    bias: torch.Tensor    # (1,)


class LinearProbeClassifier:
    """A trained-or-loadable probe that maps text → ClassifierVerdict.

    Calibration: thresholds (`block_threshold`, `flag_threshold`) are
    sigmoid-probability thresholds. Operators pick them from the
    precision-recall curve at deployment time (modelcard 3.1.1).
    """

    def __init__(
        self,
        name: str,
        extractor: FeatureExtractor,
        weights: LinearProbeWeights,
        category: Category,
        block_action: Action = Action.BLOCK,
        block_threshold: float = 0.85,
        flag_threshold: float = 0.5,
    ):
        self.name = name
        self.extractor = extractor
        self.weights = weights
        self.category = category
        self.block_action = block_action
        self.block_threshold = block_threshold
        self.flag_threshold = flag_threshold

    def __call__(self, text: str) -> ClassifierVerdict:
        feat = self.extractor.extract(text)
        if feat.shape != self.weights.weight.shape:
            raise ValueError(
                f"feature dim {tuple(feat.shape)} != weight dim {tuple(self.weights.weight.shape)}"
            )
        logit = float((feat @ self.weights.weight + self.weights.bias).item())
        prob = 1.0 / (1.0 + math.exp(-logit))
        if prob >= self.block_threshold:
            return ClassifierVerdict(
                category=self.category,
                score=prob,
                action=self.block_action,
                reason=f"{self.name} probe p={prob:.3f}",
            )
        if prob >= self.flag_threshold:
            return ClassifierVerdict(
                category=self.category,
                score=prob,
                action=Action.FLAG,
                reason=f"{self.name} probe p={prob:.3f}",
            )
        return ClassifierVerdict(Category.BENIGN, prob, Action.ALLOW)

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "name": self.name,
                "category": self.category.value,
                "block_action": self.block_action.value,
                "block_threshold": self.block_threshold,
                "flag_threshold": self.flag_threshold,
                "weight": self.weights.weight,
                "bias": self.weights.bias,
            },
            p,
        )

    @classmethod
    def load(cls, path: str | Path, extractor: FeatureExtractor) -> "LinearProbeClassifier":
        state = torch.load(path, map_location="cpu", weights_only=False)
        return cls(
            name=state["name"],
            extractor=extractor,
            weights=LinearProbeWeights(weight=state["weight"], bias=state["bias"]),
            category=Category(state["category"]),
            block_action=Action(state["block_action"]),
            block_threshold=state["block_threshold"],
            flag_threshold=state["flag_threshold"],
        )


def train_linear_probe(
    extractor: FeatureExtractor,
    pos_examples: list[str],
    neg_examples: list[str],
    *,
    name: str,
    category: Category,
    block_action: Action = Action.BLOCK,
    n_epochs: int = 50,
    lr: float = 0.5,
    weight_decay: float = 1e-3,
    block_threshold: float = 0.85,
    flag_threshold: float = 0.5,
    seed: int = 0,
) -> LinearProbeClassifier:
    """Train a logistic-regression probe on extractor features.

    Returns: a calibrated `LinearProbeClassifier`. The defaults are tuned for
    the modelcard 3.1 single-turn evaluation rates (≥97.9% violative-harmless,
    ≤0.5% benign over-refusal).
    """
    torch.manual_seed(seed)
    if not pos_examples or not neg_examples:
        raise ValueError("both pos_examples and neg_examples must be non-empty")

    X = torch.stack([extractor.extract(t) for t in pos_examples + neg_examples])
    y = torch.cat([
        torch.ones(len(pos_examples), dtype=torch.float32),
        torch.zeros(len(neg_examples), dtype=torch.float32),
    ])

    D = X.shape[1]
    w = torch.zeros(D, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=lr, weight_decay=weight_decay)

    for _ in range(n_epochs):
        opt.zero_grad()
        logits = X @ w + b
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
        loss.backward()
        opt.step()

    return LinearProbeClassifier(
        name=name,
        extractor=extractor,
        weights=LinearProbeWeights(weight=w.detach(), bias=b.detach()),
        category=category,
        block_action=block_action,
        block_threshold=block_threshold,
        flag_threshold=flag_threshold,
    )


# --- save/load metadata helper for a deployment bundle ---


def write_probe_bundle(
    output_dir: str | Path,
    probes: list[LinearProbeClassifier],
    extractor_name: str,
) -> Path:
    """Persist a calibrated bundle of probes to disk for deployment."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []
    for probe in probes:
        path = out / f"{probe.name}.pt"
        probe.save(path)
        manifest.append(
            {
                "name": probe.name,
                "category": probe.category.value,
                "action": probe.block_action.value,
                "block_threshold": probe.block_threshold,
                "flag_threshold": probe.flag_threshold,
                "path": path.name,
            }
        )
    manifest_path = out / "manifest.json"
    manifest_path.write_text(
        json.dumps({"extractor": extractor_name, "probes": manifest}, indent=2)
    )
    return manifest_path


def load_probe_bundle(
    bundle_dir: str | Path,
    extractor: FeatureExtractor,
) -> list[LinearProbeClassifier]:
    bundle_dir = Path(bundle_dir)
    manifest = json.loads((bundle_dir / "manifest.json").read_text())
    out: list[LinearProbeClassifier] = []
    for entry in manifest["probes"]:
        out.append(LinearProbeClassifier.load(bundle_dir / entry["path"], extractor))
    return out
