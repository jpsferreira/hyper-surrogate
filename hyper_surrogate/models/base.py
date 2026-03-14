from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
import torch.nn as nn


@dataclass
class LayerInfo:
    weights: str
    bias: str
    activation: str


@dataclass
class BranchInfo:
    """Describes one branch of a multi-branch model (e.g. PolyconvexICNN)."""

    name: str
    input_indices: list[int]
    layers: list[LayerInfo] = field(default_factory=list)


class SurrogateModel(nn.Module, ABC):
    @property
    @abstractmethod
    def input_dim(self) -> int: ...

    @property
    @abstractmethod
    def output_dim(self) -> int: ...

    @abstractmethod
    def layer_sequence(self) -> list[LayerInfo]: ...

    def branch_sequence(self) -> list[BranchInfo] | None:
        """Return branch info for multi-branch models. None for single-branch."""
        return None

    def export_weights(self) -> dict[str, np.ndarray]:
        return {k: v.detach().cpu().numpy() for k, v in self.state_dict().items()}
