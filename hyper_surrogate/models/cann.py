"""Constitutive Artificial Neural Network (CANN).

Based on Linka et al. (2023): a physics-constrained architecture where the output is
a weighted sum of known strain energy basis functions with learnable non-negative weights.

W = sum_i w_i * psi_i(invariants)

Basis functions include polynomial, exponential, and logarithmic forms.
Non-negative weights are enforced via softplus (same as ICNN).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from hyper_surrogate.models.base import LayerInfo, SurrogateModel


class CANN(SurrogateModel):
    """Constitutive Artificial Neural Network with interpretable basis functions.

    The model computes: W = sum_i softplus(w_i) * psi_i(inputs)

    where psi_i are predefined basis functions and w_i are learnable weights.

    Args:
        input_dim: Number of input features (invariants).
        n_polynomial: Number of polynomial basis terms per invariant: (I-3)^1, (I-3)^2, ...
        n_exponential: Number of exponential basis terms per invariant.
        use_logarithmic: Include logarithmic basis terms.
        learnable_exponents: If True, exponent/stiffness parameters are also learnable.
    """

    def __init__(
        self,
        input_dim: int,
        n_polynomial: int = 3,
        n_exponential: int = 2,
        use_logarithmic: bool = True,
        learnable_exponents: bool = False,
    ) -> None:
        super().__init__()
        self._input_dim = input_dim
        self._n_polynomial = n_polynomial
        self._n_exponential = n_exponential
        self._use_logarithmic = use_logarithmic
        self._learnable_exponents = learnable_exponents

        # Count total basis functions
        # For each invariant: n_polynomial + n_exponential + (1 if logarithmic)
        n_basis_per_inv = n_polynomial + n_exponential + (1 if use_logarithmic else 0)
        self._n_basis_per_inv = n_basis_per_inv
        self._n_basis = n_basis_per_inv * input_dim

        # Learnable non-negative weights (via softplus)
        self.raw_weights = nn.Parameter(torch.zeros(self._n_basis))

        # Learnable exponents for exponential terms (if enabled)
        if learnable_exponents and n_exponential > 0:
            # Initialize with small positive values
            self.raw_exp_params = nn.Parameter(torch.linspace(0.5, 2.0, n_exponential * input_dim))
        else:
            self.register_buffer(
                "raw_exp_params", torch.linspace(0.5, 2.0, max(n_exponential * input_dim, 1))
            )

    @property
    def input_dim(self) -> int:
        return self._input_dim

    @property
    def output_dim(self) -> int:
        return 1

    @property
    def weights(self) -> torch.Tensor:
        """Non-negative weights via softplus."""
        return F.softplus(self.raw_weights)

    def _compute_basis(self, x: torch.Tensor) -> torch.Tensor:
        """Compute all basis function values.

        Args:
            x: Input tensor (N, input_dim), typically normalized invariants.

        Returns:
            Basis function values (N, n_basis).
        """
        basis_list: list[torch.Tensor] = []

        for d in range(self._input_dim):
            xi = x[:, d : d + 1]  # (N, 1)

            # Polynomial: (xi)^1, (xi)^2, ..., (xi)^n
            for p in range(1, self._n_polynomial + 1):
                basis_list.append(xi**p)

            # Exponential: exp(b_k * xi^2) - 1
            for k in range(self._n_exponential):
                idx = d * self._n_exponential + k
                b_k = F.softplus(self.raw_exp_params[idx])  # type: ignore[index]
                basis_list.append(torch.exp(b_k * xi**2) - 1)

            # Logarithmic: log(1 + xi^2)
            if self._use_logarithmic:
                basis_list.append(torch.log1p(xi**2))

        return torch.cat(basis_list, dim=1)  # (N, n_basis)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute W = sum_i w_i * psi_i(x)."""
        basis = self._compute_basis(x)  # (N, n_basis)
        w = self.weights  # (n_basis,)
        return (basis * w).sum(dim=1, keepdim=True)  # (N, 1)

    def get_active_terms(self, threshold: float = 1e-3) -> list[dict]:
        """Identify basis functions with significant weights (for model discovery).

        Args:
            threshold: Minimum weight to consider active.

        Returns:
            List of dicts describing active terms.
        """
        w = self.weights.detach().cpu().numpy()
        terms = []
        idx = 0
        for d in range(self._input_dim):
            for p in range(1, self._n_polynomial + 1):
                if w[idx] > threshold:
                    terms.append({"type": "polynomial", "invariant": d, "power": p, "weight": float(w[idx])})
                idx += 1
            for k in range(self._n_exponential):
                if w[idx] > threshold:
                    b_idx = d * self._n_exponential + k
                    b_val = float(F.softplus(self.raw_exp_params[b_idx]).item())  # type: ignore[index]
                    terms.append({
                        "type": "exponential",
                        "invariant": d,
                        "stiffness": b_val,
                        "weight": float(w[idx]),
                    })
                idx += 1
            if self._use_logarithmic:
                if w[idx] > threshold:
                    terms.append({"type": "logarithmic", "invariant": d, "weight": float(w[idx])})
                idx += 1
        return terms

    def layer_sequence(self) -> list[LayerInfo]:
        """CANN does not have a standard layer sequence; return description."""
        return [LayerInfo(weights="raw_weights", bias="", activation="softplus_weighted_basis")]

    def export_weights(self) -> dict[str, Any]:
        import numpy as np

        return {
            "weights": self.weights.detach().cpu().numpy(),
            "exp_params": F.softplus(self.raw_exp_params).detach().cpu().numpy(),  # type: ignore[arg-type]
            "config": np.array([
                self._input_dim,
                self._n_polynomial,
                self._n_exponential,
                int(self._use_logarithmic),
            ]),
        }
