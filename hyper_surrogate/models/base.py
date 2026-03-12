from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import torch.nn as nn


@dataclass
class LayerInfo:
    weights: str
    bias: str
    activation: str


class SurrogateModel(nn.Module, ABC):
    @property
    @abstractmethod
    def input_dim(self) -> int: ...

    @property
    @abstractmethod
    def output_dim(self) -> int: ...

    @abstractmethod
    def layer_sequence(self) -> list[LayerInfo]: ...

    def export_weights(self) -> dict[str, np.ndarray]:
        return {k: v.detach().cpu().numpy() for k, v in self.state_dict().items()}
