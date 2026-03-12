from __future__ import annotations

import torch
import torch.nn as nn

from hyper_surrogate.models.base import LayerInfo, SurrogateModel

ACTIVATIONS = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "softplus": nn.Softplus,
}


class MLP(SurrogateModel):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: list[int] | None = None,
        activation: str = "tanh",
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 64]
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._activation_name = activation
        act_cls = ACTIVATIONS[activation]

        dims = [input_dim, *hidden_dims, output_dim]
        layer_list: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layer_list.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:  # no activation on output
                layer_list.append(act_cls())
        self.layers = nn.Sequential(*layer_list)

    @property
    def input_dim(self) -> int:
        return self._input_dim

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)  # type: ignore[no-any-return]

    def layer_sequence(self) -> list[LayerInfo]:
        result = []
        linear_idx = 0
        for module in self.layers:
            if isinstance(module, nn.Linear):
                is_last = linear_idx == len([m for m in self.layers if isinstance(m, nn.Linear)]) - 1
                act = "identity" if is_last else self._activation_name
                prefix = f"layers.{list(self.layers).index(module)}"
                result.append(
                    LayerInfo(
                        weights=f"{prefix}.weight",
                        bias=f"{prefix}.bias",
                        activation=act,
                    )
                )
                linear_idx += 1
        return result
