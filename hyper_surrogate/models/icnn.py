from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from hyper_surrogate.models.base import LayerInfo, SurrogateModel


class ICNN(SurrogateModel):
    """Input-Convex Neural Network (Amos+ 2017).

    Guarantees convexity of output w.r.t. input via:
    - Non-negative weights on z-path (enforced via softplus)
    - Skip connections from input to every layer
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] | None = None,
        activation: str = "softplus",
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 64]
        self._input_dim = input_dim
        self._output_dim = 1
        self._activation_name = activation
        self._hidden_dims = hidden_dims

        # First layer: only x-path
        self.wx_layers = nn.ModuleList()
        self.wz_layers = nn.ModuleList()

        # wx_0: input -> hidden_0
        self.wx_layers.append(nn.Linear(input_dim, hidden_dims[0]))

        # Hidden layers: wz (non-negative) + wx (skip)
        for i in range(1, len(hidden_dims)):
            self.wz_layers.append(nn.Linear(hidden_dims[i - 1], hidden_dims[i], bias=False))
            self.wx_layers.append(nn.Linear(input_dim, hidden_dims[i]))

        # Output layer
        self.wz_final = nn.Linear(hidden_dims[-1], 1, bias=False)
        self.wx_final = nn.Linear(input_dim, 1)

        # Activation
        act_map = {"softplus": nn.Softplus(), "relu": nn.ReLU(), "tanh": nn.Tanh()}
        self._activation = act_map[activation]

    @property
    def input_dim(self) -> int:
        return self._input_dim

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # First hidden layer (x-path only)
        z = self._activation(self.wx_layers[0](x))

        # Subsequent hidden layers (z-path with non-neg weights + x skip)
        for wz, wx in zip(self.wz_layers, self.wx_layers[1:], strict=False):
            z = self._activation(F.linear(z, F.softplus(wz.weight)) + wx(x))

        # Output layer
        return F.linear(z, F.softplus(self.wz_final.weight)) + self.wx_final(x)  # type: ignore[no-any-return]

    def layer_sequence(self) -> list[LayerInfo]:
        result = []
        # First wx layer
        result.append(
            LayerInfo(
                weights="wx_layers.0.weight",
                bias="wx_layers.0.bias",
                activation=self._activation_name,
            )
        )
        # Hidden wz + wx pairs
        for i in range(len(self.wz_layers)):
            result.append(
                LayerInfo(
                    weights=f"wz_layers.{i}.weight",
                    bias=f"wx_layers.{i + 1}.bias",
                    activation=self._activation_name,
                )
            )
        # Output
        result.append(
            LayerInfo(
                weights="wz_final.weight",
                bias="wx_final.bias",
                activation="identity",
            )
        )
        return result
