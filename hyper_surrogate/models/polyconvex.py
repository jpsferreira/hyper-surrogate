"""Polyconvex ICNN: multi-branch input-convex network preserving polyconvexity.

Each branch is an independent ICNN operating on a group of invariants.
The total energy is the sum of branch outputs — convex superposition
preserves convexity per group.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from hyper_surrogate.models.base import BranchInfo, LayerInfo, SurrogateModel
from hyper_surrogate.models.icnn import ICNN


class PolyconvexICNN(SurrogateModel):
    """Multi-branch ICNN for polyconvex strain energy.

    Args:
        groups: List of input index lists, e.g. ``[[0], [1], [2]]`` for
            isotropic or ``[[0], [1], [2], [3, 4]]`` for anisotropic.
            Each group feeds a separate ICNN branch.
        hidden_dims: Hidden layer sizes for each branch ICNN.
        activation: Activation function for all branches.
    """

    def __init__(
        self,
        groups: list[list[int]],
        hidden_dims: list[int] | None = None,
        activation: str = "softplus",
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [32, 32]

        all_indices = [idx for g in groups for idx in g]
        if len(all_indices) != len(set(all_indices)):
            msg = "Input indices must not overlap across groups"
            raise ValueError(msg)

        self._groups = groups
        self._input_dim = max(all_indices) + 1
        self._hidden_dims = hidden_dims
        self._activation_name = activation

        self.branches = nn.ModuleList([
            ICNN(input_dim=len(g), hidden_dims=hidden_dims, activation=activation) for g in groups
        ])

    @property
    def input_dim(self) -> int:
        return self._input_dim

    @property
    def output_dim(self) -> int:
        return 1

    @property
    def groups(self) -> list[list[int]]:
        return self._groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        total = torch.zeros(x.shape[0], 1, device=x.device, dtype=x.dtype)
        for branch, group in zip(self.branches, self._groups, strict=True):
            x_branch = x[:, group]
            total = total + branch(x_branch)
        return total

    def layer_sequence(self) -> list[LayerInfo]:
        """Return layers from the first branch (backward compat)."""
        return self._branch_layers(0)

    def branch_sequence(self) -> list[BranchInfo]:
        """Return BranchInfo per branch with prefixed weight keys."""
        result = []
        for i, group in enumerate(self._groups):
            result.append(
                BranchInfo(
                    name=f"branch_{i}",
                    input_indices=group,
                    layers=self._branch_layers(i),
                )
            )
        return result

    def _branch_layers(self, branch_idx: int) -> list[LayerInfo]:
        """Get LayerInfo for a specific branch with prefixed keys."""
        prefix = f"branches.{branch_idx}."
        branch: ICNN = self.branches[branch_idx]  # type: ignore[assignment]
        raw_layers = branch.layer_sequence()
        return [
            LayerInfo(
                weights=prefix + layer.weights,
                bias=prefix + layer.bias,
                activation=layer.activation,
            )
            for layer in raw_layers
        ]
