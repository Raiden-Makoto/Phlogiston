"""Configuration objects for the Phlogiston pipeline.

Configs are plain dataclasses that can be loaded from / dumped to YAML so that
runs are reproducible. See ``configs/default.yaml`` for an annotated example.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class GraphConfig:
    """Crystal-graph featurization settings."""

    radius: float = 8.0  # neighbor search cutoff in Angstrom
    max_num_nbr: int = 12  # max neighbors kept per atom
    # Gaussian distance expansion of bond lengths.
    dmin: float = 0.0
    dmax: float = 8.0
    gaussian_step: float = 0.2


@dataclass
class ModelConfig:
    """CGCNN architecture hyper-parameters."""

    atom_fea_len: int = 64  # hidden atom embedding size
    n_conv: int = 3  # number of graph-conv layers
    h_fea_len: int = 128  # hidden size of the final MLP
    n_h: int = 1  # number of hidden layers in the final MLP
    task: str = "regression"  # "regression" or "classification"


@dataclass
class TrainConfig:
    """Optimization / training-loop settings."""

    epochs: int = 100
    batch_size: int = 64
    lr: float = 1e-2
    weight_decay: float = 0.0
    momentum: float = 0.9
    optimizer: str = "adam"  # "adam" or "sgd"
    lr_milestones: list[int] = field(default_factory=lambda: [80])
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    num_workers: int = 4
    seed: int = 42
    device: str = "auto"  # "auto" | "cpu" | "cuda"
    print_every: int = 25


@dataclass
class DataConfig:
    """Where data lives and which target property to model."""

    # Directory layout (relative to project root unless absolute).
    root: str = "data"
    # Materials Project target property to train on.
    target_property: str = "formation_energy_per_atom"
    # For classification: label = 1 if energy_above_hull <= this threshold.
    stability_threshold: float = 0.0
    # Optional chemical-system / element filters applied when querying MP.
    max_samples: int | None = None


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    # --- (de)serialization helpers ---------------------------------------
    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Config:
        return cls(
            data=DataConfig(**(raw.get("data") or {})),
            graph=GraphConfig(**(raw.get("graph") or {})),
            model=ModelConfig(**(raw.get("model") or {})),
            train=TrainConfig(**(raw.get("train") or {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_yaml(self, path: str | Path) -> None:
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)


def default_config() -> Config:
    return Config()
