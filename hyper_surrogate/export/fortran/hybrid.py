"""Hybrid UMAT generator: NN-based SEF with analytical continuum mechanics.

The NN learns W(invariants). Everything else — kinematics,
stress, and tangent — is computed analytically in Fortran:

    DFGRD1 → C → invariants → NN(W) → backprop(dW/dI, d²W/dI²)
           → PK2 → Cauchy stress → spatial tangent + Jaumann correction

Supports:
  - Isotropic: W(I1_bar, I2_bar, J)        — input_dim=3
  - Anisotropic: W(I1_bar, I2_bar, J, I4, I5) — input_dim=5
"""

from __future__ import annotations

import datetime
from pathlib import Path

import numpy as np

from hyper_surrogate.export.weights import ExportedModel


class HybridUMATEmitter:
    """Emit a complete Abaqus UMAT subroutine with NN-based strain energy."""

    SUPPORTED_ARCHITECTURES = ("mlp", "polyconvexicnn")

    def __init__(self, exported: ExportedModel) -> None:
        arch = exported.metadata.get("architecture")
        if arch not in self.SUPPORTED_ARCHITECTURES:
            msg = f"HybridUMATEmitter supports {self.SUPPORTED_ARCHITECTURES}, got '{arch}'"
            raise ValueError(msg)
        if exported.metadata.get("output_dim") != 1:
            msg = "HybridUMATEmitter requires a scalar (output_dim=1) energy model"
            raise ValueError(msg)
        self.exported = exported

    # ------------------------------------------------------------------
    # Fortran formatting helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _fmt_1d(arr: np.ndarray, name: str) -> str:
        n = arr.shape[0]
        vals = ", &\n    ".join(", ".join(f"{v:.15e}d0" for v in arr[i : i + 4]) for i in range(0, n, 4))
        return f"DOUBLE PRECISION, PARAMETER :: {name}({n}) = (/ &\n    {vals} /)"

    @staticmethod
    def _fmt_2d(arr: np.ndarray, name: str) -> str:
        rows, cols = arr.shape
        # Fortran is column-major
        vals = ", &\n    ".join(
            ", ".join(f"{v:.15e}d0" for v in arr.T.flat[i : i + 4]) for i in range(0, rows * cols, 4)
        )
        return (
            f"DOUBLE PRECISION, PARAMETER :: {name}({rows},{cols}) = RESHAPE((/ &\n    {vals} /), (/ {rows}, {cols} /))"
        )

    # ------------------------------------------------------------------
    # Code generation
    # ------------------------------------------------------------------

    def _emit_nn_parameters(self) -> str:
        """Emit weight/bias arrays and normalizer constants."""
        arch = self.exported.metadata.get("architecture")
        if arch == "polyconvexicnn":
            return self._emit_poly_nn_parameters()
        return self._emit_mlp_nn_parameters()

    def _emit_mlp_nn_parameters(self) -> str:
        """Emit MLP weight/bias arrays."""
        lines: list[str] = []
        layers = self.exported.layers
        weights = self.exported.weights

        for i, layer in enumerate(layers):
            w = weights[layer.weights]
            b = weights[layer.bias]
            lines.append(self._fmt_2d(w, f"w{i}"))
            lines.append(self._fmt_1d(b, f"b{i}"))

        if self.exported.input_normalizer:
            lines.append(self._fmt_1d(self.exported.input_normalizer["mean"], "in_mean"))
            lines.append(self._fmt_1d(self.exported.input_normalizer["std"], "in_std"))

        return "\n".join(lines)

    def _emit_poly_nn_parameters(self) -> str:
        """Emit per-branch ICNN weight/bias arrays."""
        lines: list[str] = []
        weights = self.exported.weights
        branches = self.exported.metadata["branches"]

        for bi, branch in enumerate(branches):
            branch_layers = branch["layers"]
            for li, layer in enumerate(branch_layers):
                w = weights[layer["weights"]]
                b = weights[layer["bias"]]
                # Pre-apply softplus to wz weights (non-negative constraint)
                if "wz" in layer["weights"]:
                    w = np.log(1.0 + np.exp(w))
                lines.append(self._fmt_2d(w, f"w_b{bi}_{li}"))
                lines.append(self._fmt_1d(b, f"b_b{bi}_{li}"))

            # wx skip-connection weights (layers 1..L-1 have wx_layers)
            prefix = f"branches.{bi}."
            wx_keys = sorted([
                k
                for k in weights
                if k.startswith(prefix + "wx_layers") and "weight" in k and k != branch_layers[0]["weights"]
            ])
            for wi, wk in enumerate(wx_keys):
                lines.append(self._fmt_2d(weights[wk], f"wx_b{bi}_{wi + 1}"))
            # wx_final skip
            wx_final_key = prefix + "wx_final.weight"
            if wx_final_key in weights:
                lines.append(self._fmt_2d(weights[wx_final_key], f"wxf_b{bi}"))

        if self.exported.input_normalizer:
            lines.append(self._fmt_1d(self.exported.input_normalizer["mean"], "in_mean"))
            lines.append(self._fmt_1d(self.exported.input_normalizer["std"], "in_std"))

        return "\n".join(lines)

    def _emit_nn_forward_and_backward(self) -> str:  # noqa: C901
        """Emit NN forward pass + analytical backward pass for dW/dI and d²W/dI²."""
        arch = self.exported.metadata.get("architecture")
        if arch == "polyconvexicnn":
            return self._emit_poly_nn_forward_and_backward()
        layers = self.exported.layers
        weights = self.exported.weights
        n_layers = len(layers)
        hidden_dims = [weights[layer.weights].shape[0] for layer in layers]
        in_dim = self.exported.metadata["input_dim"]

        lines: list[str] = []

        # Local variables for forward pass
        for i in range(n_layers):
            lines.append(f"DOUBLE PRECISION :: z{i}({hidden_dims[i]})")
            lines.append(f"DOUBLE PRECISION :: a{i}({hidden_dims[i]})")  # pre-activation
            lines.append(f"DOUBLE PRECISION :: dact{i}({hidden_dims[i]})")  # activation derivative
            lines.append(f"DOUBLE PRECISION :: d2act{i}({hidden_dims[i]})")  # second derivative
        lines.append(f"DOUBLE PRECISION :: x_norm({in_dim})")
        lines.append("")

        # Local variables for backward pass
        for i in range(n_layers - 1, -1, -1):
            lines.append(f"DOUBLE PRECISION :: delta{i}({hidden_dims[i]})")
        lines.append(f"DOUBLE PRECISION :: grad_x({in_dim})")
        lines.append("")

        # For Jacobian propagation and Hessian
        for i in range(n_layers):
            lines.append(f"DOUBLE PRECISION :: P{i}({hidden_dims[i]},{in_dim})")
            lines.append(f"DOUBLE PRECISION :: J{i}({hidden_dims[i]},{in_dim})")
        lines.append(f"DOUBLE PRECISION :: d2W_dx2({in_dim},{in_dim})")
        lines.append("DOUBLE PRECISION :: coeff")
        lines.append("")

        # --- Normalize input ---
        lines.append("! Normalize invariants")
        lines.append("x_norm = (nn_input - in_mean) / in_std")
        lines.append("")

        # --- Forward pass with d2act ---
        lines.append("! Forward pass")
        for i, layer in enumerate(layers):
            input_var = "x_norm" if i == 0 else f"z{i - 1}"
            lines.append(f"a{i} = MATMUL(w{i}, {input_var}) + b{i}")
            act = layer.activation
            if act == "identity":
                lines.append(f"z{i} = a{i}")
                lines.append(f"dact{i} = 1.0d0")
                lines.append(f"d2act{i} = 0.0d0")
            elif act == "tanh":
                lines.append(f"z{i} = tanh(a{i})")
                lines.append(f"dact{i} = 1.0d0 - z{i}**2")
                lines.append(f"d2act{i} = -2.0d0 * z{i} * dact{i}")
            elif act == "softplus":
                lines.append(f"z{i} = log(1.0d0 + exp(a{i}))")
                lines.append(f"dact{i} = 1.0d0 / (1.0d0 + exp(-a{i}))")
                lines.append(f"d2act{i} = dact{i} * (1.0d0 - dact{i})")
            elif act == "relu":
                lines.append(f"DO ii = 1, {hidden_dims[i]}")
                lines.append(f"  z{i}(ii) = max(0.0d0, a{i}(ii))")
                lines.append(f"  IF (a{i}(ii) > 0.0d0) THEN")
                lines.append(f"    dact{i}(ii) = 1.0d0")
                lines.append("  ELSE")
                lines.append(f"    dact{i}(ii) = 0.0d0")
                lines.append("  END IF")
                lines.append(f"  d2act{i}(ii) = 0.0d0")
                lines.append("END DO")
            elif act == "sigmoid":
                lines.append(f"z{i} = 1.0d0 / (1.0d0 + exp(-a{i}))")
                lines.append(f"dact{i} = z{i} * (1.0d0 - z{i})")
                lines.append(f"d2act{i} = dact{i} * (1.0d0 - 2.0d0 * z{i})")
            lines.append("")

        # W = z_{last}(1) (scalar output)
        last = n_layers - 1
        lines.append(f"W_nn = z{last}(1)")
        lines.append("")

        # --- Backward pass: dW/dx_norm ---
        lines.append("! Backward pass: dW/d(x_norm)")
        # delta for output layer
        lines.append(f"delta{last}(1) = dact{last}(1)")
        # Backpropagate
        for i in range(last - 1, -1, -1):
            lines.append(f"DO ii = 1, {hidden_dims[i]}")
            lines.append(f"  delta{i}(ii) = 0.0d0")
            lines.append(f"  DO jj = 1, {hidden_dims[i + 1]}")
            lines.append(f"    delta{i}(ii) = delta{i}(ii) + w{i + 1}(jj, ii) * delta{i + 1}(jj)")
            lines.append("  END DO")
            lines.append(f"  delta{i}(ii) = delta{i}(ii) * dact{i}(ii)")
            lines.append("END DO")

        # grad_x = W_0^T * delta_0
        lines.append(f"DO ii = 1, {in_dim}")
        lines.append("  grad_x(ii) = 0.0d0")
        lines.append(f"  DO jj = 1, {hidden_dims[0]}")
        lines.append("    grad_x(ii) = grad_x(ii) + w0(jj, ii) * delta0(jj)")
        lines.append("  END DO")
        lines.append("END DO")
        lines.append("")

        # dW/dI = dW/dx_norm / std (chain rule for normalization)
        lines.append("! Convert gradient to raw invariant space: dW/dI = dW/dx_norm / std")
        for k in range(in_dim):
            lines.append(f"dW_dI({k + 1}) = grad_x({k + 1}) / in_std({k + 1})")
        lines.append("")

        # --- Jacobian propagation: P_i = W_i * J_{i-1}, J_i = diag(dact_i) * P_i ---
        lines.append("! Jacobian propagation (forward mode)")
        # P0 = W0, J0 = diag(dact0) * W0
        lines.append(f"DO ii = 1, {hidden_dims[0]}")
        lines.append(f"  DO jj = 1, {in_dim}")
        lines.append("    P0(ii, jj) = w0(ii, jj)")
        lines.append("    J0(ii, jj) = dact0(ii) * w0(ii, jj)")
        lines.append("  END DO")
        lines.append("END DO")

        for i in range(1, n_layers):
            # P_i = W_i * J_{i-1}  (pre-activation Jacobian)
            # J_i = diag(dact_i) * P_i
            lines.append(f"DO ii = 1, {hidden_dims[i]}")
            lines.append(f"  DO jj = 1, {in_dim}")
            lines.append(f"    P{i}(ii, jj) = 0.0d0")
            lines.append(f"    DO kk = 1, {hidden_dims[i - 1]}")
            lines.append(f"      P{i}(ii, jj) = P{i}(ii, jj) + w{i}(ii, kk) * J{i - 1}(kk, jj)")
            lines.append("    END DO")
            lines.append(f"    J{i}(ii, jj) = dact{i}(ii) * P{i}(ii, jj)")
            lines.append("  END DO")
            lines.append("END DO")
        lines.append("")

        # --- Exact analytical Hessian: d²W/dx² = Σ_i P_i^T diag(beta_i * d2act_i) P_i ---
        # beta_i = delta_i / dact_i = dW/dz_i (sensitivity to post-activation)
        lines.append("! Exact analytical Hessian: d²W/dx_norm²")
        lines.append("d2W_dx2 = 0.0d0")
        for i in range(n_layers):
            act = layers[i].activation
            # Skip layers with zero d2act (relu, identity) — they contribute nothing
            if act in ("relu", "identity"):
                lines.append(f"! Layer {i} ({act}): d2act=0, no Hessian contribution")
                continue
            lines.append(f"! Layer {i} ({act})")
            lines.append(f"DO ii = 1, {hidden_dims[i]}")
            # coeff = beta_i(ii) * d2act_i(ii) = (delta_i(ii) / dact_i(ii)) * d2act_i(ii)
            lines.append(f"  coeff = delta{i}(ii) / dact{i}(ii) * d2act{i}(ii)")
            lines.append(f"  DO jj = 1, {in_dim}")
            lines.append(f"    DO kk = 1, {in_dim}")
            lines.append(f"      d2W_dx2(jj, kk) = d2W_dx2(jj, kk) + coeff * P{i}(ii, jj) * P{i}(ii, kk)")
            lines.append("    END DO")
            lines.append("  END DO")
            lines.append("END DO")
        lines.append("")

        # Convert to raw invariant space: d2W/dI2(k,l) = d2W/dx2(k,l) / (std(k) * std(l))
        lines.append("! Convert Hessian to raw invariant space: d2W/dI2(k,l) = d2W/dx2(k,l) / (std(k)*std(l))")
        lines.append(f"DO ii = 1, {in_dim}")
        lines.append(f"  DO jj = 1, {in_dim}")
        lines.append("    d2W_dI2(ii, jj) = d2W_dx2(ii, jj) / (in_std(ii) * in_std(jj))")
        lines.append("  END DO")
        lines.append("END DO")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Anisotropic Fortran snippets (generalized for N fiber families)
    # ------------------------------------------------------------------

    @staticmethod
    def _emit_aniso_declarations(num_fibers: int) -> str:
        """Extra local variable declarations for anisotropic UMAT."""
        lines = ["  ! Fiber invariants (anisotropic)"]
        for k in range(num_fibers):
            lines.append(f"  DOUBLE PRECISION :: a0_{k + 1}(3), Ca0_{k + 1}(3), I4_{k + 1}, I5_{k + 1}")
            lines.append(f"  DOUBLE PRECISION :: dI4_{k + 1}_dC(3,3), dI5_{k + 1}_dC(3,3)")
        return "\n".join(lines) + "\n"

    @staticmethod
    def _emit_aniso_invariants(num_fibers: int) -> str:
        """Compute fiber invariants I4, I5 from C and a0 for each fiber family."""
        lines: list[str] = []
        for k in range(num_fibers):
            prop_start = k * 3 + 1
            lines.append(f"""
  ! Fiber family {k + 1}: direction from props({prop_start}:{prop_start + 2})
  a0_{k + 1}(1) = props({prop_start})
  a0_{k + 1}(2) = props({prop_start + 1})
  a0_{k + 1}(3) = props({prop_start + 2})

  ! Ca0_{k + 1} = C * a0_{k + 1}
  DO ii = 1, 3
    Ca0_{k + 1}(ii) = 0.0d0
    DO jj = 1, 3
      Ca0_{k + 1}(ii) = Ca0_{k + 1}(ii) + C(ii,jj) * a0_{k + 1}(jj)
    END DO
  END DO

  ! I4_{k + 1} = a0_{k + 1} . C . a0_{k + 1}
  I4_{k + 1} = 0.0d0
  DO ii = 1, 3
    I4_{k + 1} = I4_{k + 1} + a0_{k + 1}(ii) * Ca0_{k + 1}(ii)
  END DO

  ! I5_{k + 1} = a0_{k + 1} . C^2 . a0_{k + 1} = Ca0_{k + 1} . Ca0_{k + 1}
  I5_{k + 1} = 0.0d0
  DO ii = 1, 3
    I5_{k + 1} = I5_{k + 1} + Ca0_{k + 1}(ii) * Ca0_{k + 1}(ii)
  END DO""")
        return "\n".join(lines)

    @staticmethod
    def _emit_aniso_nn_input(num_fibers: int) -> str:
        """Set fiber invariant entries in nn_input."""
        lines: list[str] = []
        for k in range(num_fibers):
            idx_i4 = 3 + 2 * k + 1  # Fortran 1-indexed
            idx_i5 = 3 + 2 * k + 2
            lines.append(f"  nn_input({idx_i4}) = I4_{k + 1}")
            lines.append(f"  nn_input({idx_i5}) = I5_{k + 1}")
        return "\n".join(lines) + "\n"

    @staticmethod
    def _emit_aniso_didC(num_fibers: int) -> str:
        """Compute dI4/dC and dI5/dC for each fiber family."""
        lines: list[str] = []
        for k in range(num_fibers):
            lines.append(f"""
  ! dI4_{k + 1}/dC = a0_{k + 1} (x) a0_{k + 1}
  DO ii = 1, 3
    DO jj = 1, 3
      dI4_{k + 1}_dC(ii,jj) = a0_{k + 1}(ii) * a0_{k + 1}(jj)
    END DO
  END DO

  ! dI5_{k + 1}/dC = a0_{k + 1} (x) Ca0_{k + 1} + Ca0_{k + 1} (x) a0_{k + 1}
  DO ii = 1, 3
    DO jj = 1, 3
      dI5_{k + 1}_dC(ii,jj) = a0_{k + 1}(ii) * Ca0_{k + 1}(jj) + Ca0_{k + 1}(ii) * a0_{k + 1}(jj)
    END DO
  END DO""")
        return "\n".join(lines)

    @staticmethod
    def _emit_aniso_pk2_terms(num_fibers: int) -> str:
        """Additional PK2 terms for fiber invariants."""
        lines: list[str] = []
        for k in range(num_fibers):
            idx_i4 = 3 + 2 * k + 1  # Fortran 1-indexed
            idx_i5 = 3 + 2 * k + 2
            lines.append(f" &\n        + dW_dI({idx_i4}) * dI4_{k + 1}_dC(ii,jj)")
            lines.append(f" &\n        + dW_dI({idx_i5}) * dI5_{k + 1}_dC(ii,jj)")
        return "".join(lines)

    @staticmethod
    def _emit_aniso_dIdC_pack(num_fibers: int) -> str:
        """Pack fiber dI/dC into dIdC array."""
        lines = ["    DO ii = 1, 3", "      DO jj = 1, 3"]
        for k in range(num_fibers):
            idx_i4 = 3 + 2 * k + 1
            idx_i5 = 3 + 2 * k + 2
            lines.append(f"        dIdC(ii, jj, {idx_i4}) = dI4_{k + 1}_dC(ii, jj)")
            lines.append(f"        dIdC(ii, jj, {idx_i5}) = dI5_{k + 1}_dC(ii, jj)")
        lines.extend(["      END DO", "    END DO"])
        return "\n".join(lines) + "\n"

    @staticmethod
    def _emit_aniso_d2IdC2(num_fibers: int) -> str:
        """Second derivatives d²I4/dCdC=0, d²I5/dCdC for tangent."""
        lines: list[str] = []
        for k in range(num_fibers):
            idx_i5 = 3 + 2 * k + 2  # Fortran 1-indexed
            lines.append(f"""
    ! d²I4_{k + 1}/dCdC = 0 (dI4/dC = a0 (x) a0 is constant w.r.t. C)

    ! d²I5_{k + 1}/(dC_AB dC_CD)
    DO ii = 1, 3
      DO jj = 1, 3
        DO kk = 1, 3
          DO ll = 1, 3
            val = 0.5d0 * ( &
                a0_{k + 1}(ii)*a0_{k + 1}(ll)*eye3(jj,kk) + a0_{k + 1}(ii)*a0_{k + 1}(kk)*eye3(jj,ll) &
              + a0_{k + 1}(jj)*a0_{k + 1}(ll)*eye3(ii,kk) + a0_{k + 1}(jj)*a0_{k + 1}(kk)*eye3(ii,ll))
            dPK2_dC(ii,jj,kk,ll) = dPK2_dC(ii,jj,kk,ll) + 4.0d0 * dW_dI({idx_i5}) * val
          END DO
        END DO
      END DO
    END DO""")
        return "\n".join(lines)

    def _emit_poly_nn_forward_and_backward(self) -> str:  # noqa: C901
        """Emit polyconvex ICNN forward/backward with per-branch Hessian."""
        weights = self.exported.weights
        branches = self.exported.metadata["branches"]
        in_dim = self.exported.metadata["input_dim"]

        lines: list[str] = []

        # Local variables
        lines.append(f"DOUBLE PRECISION :: x_norm({in_dim})")
        for bi, branch in enumerate(branches):
            b_layers = branch["layers"]
            b_in = len(branch["input_indices"])
            n_hidden = len(b_layers) - 1
            lines.append(f"DOUBLE PRECISION :: xb{bi}({b_in})")
            for li in range(n_hidden):
                hd = weights[b_layers[li]["weights"]].shape[0]
                lines.append(f"DOUBLE PRECISION :: z_b{bi}_{li}({hd})")
                lines.append(f"DOUBLE PRECISION :: a_b{bi}_{li}({hd})")
                lines.append(f"DOUBLE PRECISION :: dact_b{bi}_{li}({hd})")
                lines.append(f"DOUBLE PRECISION :: d2act_b{bi}_{li}({hd})")
                lines.append(f"DOUBLE PRECISION :: delta_b{bi}_{li}({hd})")
                lines.append(f"DOUBLE PRECISION :: P_b{bi}_{li}({hd},{b_in})")
                lines.append(f"DOUBLE PRECISION :: J_b{bi}_{li}({hd},{b_in})")
            lines.append(f"DOUBLE PRECISION :: grad_b{bi}({b_in})")
            lines.append(f"DOUBLE PRECISION :: d2W_b{bi}({b_in},{b_in})")
        lines.append("DOUBLE PRECISION :: coeff, branch_W")
        lines.append("")

        # Normalize input
        lines.append("! Normalize invariants")
        lines.append("x_norm = (nn_input - in_mean) / in_std")
        lines.append("")

        # Initialize outputs
        lines.append("W_nn = 0.0d0")
        lines.append("dW_dI = 0.0d0")
        lines.append("d2W_dI2 = 0.0d0")
        lines.append("")

        # Per-branch forward + backward + Hessian
        for bi, branch in enumerate(branches):
            b_layers = branch["layers"]
            indices = branch["input_indices"]
            b_in = len(indices)
            n_hidden = len(b_layers) - 1

            lines.append(f"! ---- Branch {bi}: inputs [{", ".join(str(i + 1) for i in indices)}] ----")

            # Slice input
            for si, idx in enumerate(indices):
                lines.append(f"xb{bi}({si + 1}) = x_norm({idx + 1})")
            lines.append("")

            # Forward pass
            lines.append(f"! Forward pass branch {bi}")
            # Layer 0: x-path only
            lines.append(f"a_b{bi}_0 = MATMUL(w_b{bi}_0, xb{bi}) + b_b{bi}_0")
            act0 = b_layers[0]["activation"]
            hd0 = weights[b_layers[0]["weights"]].shape[0]
            self._emit_icnn_act(lines, bi, 0, act0, hd0)
            lines.append("")

            # Hidden layers with wz + wx skip
            for li in range(1, n_hidden):
                layer = b_layers[li]
                hd = weights[layer["weights"]].shape[0]
                act = layer["activation"]
                lines.append(
                    f"a_b{bi}_{li} = MATMUL(w_b{bi}_{li}, z_b{bi}_{li - 1}) + MATMUL(wx_b{bi}_{li}, xb{bi}) + b_b{bi}_{li}"
                )
                self._emit_icnn_act(lines, bi, li, act, hd)
                lines.append("")

            # Output: identity activation
            last_li = len(b_layers) - 1
            last_hidden = n_hidden - 1
            lines.append(
                f"branch_W = DOT_PRODUCT(w_b{bi}_{last_li}(1,:), z_b{bi}_{last_hidden}) + DOT_PRODUCT(wxf_b{bi}(1,:), xb{bi}) + b_b{bi}_{last_li}(1)"
            )
            lines.append("W_nn = W_nn + branch_W")
            lines.append("")

            # Backward pass
            lines.append(f"! Backward pass branch {bi}")
            # delta for last hidden layer
            hd_last = weights[b_layers[last_hidden]["weights"]].shape[0]
            lines.append(f"DO ii = 1, {hd_last}")
            lines.append(f"  delta_b{bi}_{last_hidden}(ii) = w_b{bi}_{last_li}(1, ii) * dact_b{bi}_{last_hidden}(ii)")
            lines.append("END DO")

            # Backprop through hidden layers
            for li in range(last_hidden - 1, -1, -1):
                hd = weights[b_layers[li]["weights"]].shape[0]
                hd_next = weights[b_layers[li + 1]["weights"]].shape[0]
                lines.append(f"DO ii = 1, {hd}")
                lines.append(f"  delta_b{bi}_{li}(ii) = 0.0d0")
                lines.append(f"  DO jj = 1, {hd_next}")
                lines.append(
                    f"    delta_b{bi}_{li}(ii) = delta_b{bi}_{li}(ii) + w_b{bi}_{li + 1}(jj, ii) * delta_b{bi}_{li + 1}(jj)"
                )
                lines.append("  END DO")
                lines.append(f"  delta_b{bi}_{li}(ii) = delta_b{bi}_{li}(ii) * dact_b{bi}_{li}(ii)")
                lines.append("END DO")

            # grad_b = W0^T * delta_0 + wx_final^T
            lines.append(f"DO ii = 1, {b_in}")
            lines.append(f"  grad_b{bi}(ii) = wxf_b{bi}(1, ii)")
            lines.append(f"  DO jj = 1, {hd0}")
            lines.append(f"    grad_b{bi}(ii) = grad_b{bi}(ii) + w_b{bi}_0(jj, ii) * delta_b{bi}_0(jj)")
            lines.append("  END DO")
            lines.append("END DO")
            lines.append("")

            # Scatter gradient
            for si, idx in enumerate(indices):
                lines.append(f"dW_dI({idx + 1}) = dW_dI({idx + 1}) + grad_b{bi}({si + 1}) / in_std({idx + 1})")
            lines.append("")

            # Jacobian propagation for Hessian
            lines.append(f"! Jacobian propagation branch {bi}")
            # P0 = W0, J0 = diag(dact0) * W0
            lines.append(f"DO ii = 1, {hd0}")
            lines.append(f"  DO jj = 1, {b_in}")
            lines.append(f"    P_b{bi}_0(ii, jj) = w_b{bi}_0(ii, jj)")
            lines.append(f"    J_b{bi}_0(ii, jj) = dact_b{bi}_0(ii) * w_b{bi}_0(ii, jj)")
            lines.append("  END DO")
            lines.append("END DO")

            for li in range(1, n_hidden):
                hd = weights[b_layers[li]["weights"]].shape[0]
                hd_prev = weights[b_layers[li - 1]["weights"]].shape[0]
                # P_i = Wz_i * J_{i-1} + Wx_i  (ICNN skip connection)
                lines.append(f"DO ii = 1, {hd}")
                lines.append(f"  DO jj = 1, {b_in}")
                lines.append(f"    P_b{bi}_{li}(ii, jj) = wx_b{bi}_{li}(ii, jj)")
                lines.append(f"    DO kk = 1, {hd_prev}")
                lines.append(
                    f"      P_b{bi}_{li}(ii, jj) = P_b{bi}_{li}(ii, jj) + w_b{bi}_{li}(ii, kk) * J_b{bi}_{li - 1}(kk, jj)"
                )
                lines.append("    END DO")
                lines.append(f"    J_b{bi}_{li}(ii, jj) = dact_b{bi}_{li}(ii) * P_b{bi}_{li}(ii, jj)")
                lines.append("  END DO")
                lines.append("END DO")
            lines.append("")

            # Hessian: d2W_b = sum_i P_i^T diag(beta_i * d2act_i) P_i
            lines.append(f"! Hessian branch {bi}")
            lines.append(f"d2W_b{bi} = 0.0d0")
            for li in range(n_hidden):
                act = b_layers[li]["activation"]
                if act in ("relu", "identity"):
                    lines.append(f"! Layer {li} ({act}): d2act=0, skip")
                    continue
                hd = weights[b_layers[li]["weights"]].shape[0]
                lines.append(f"DO ii = 1, {hd}")
                lines.append(f"  coeff = delta_b{bi}_{li}(ii) / dact_b{bi}_{li}(ii) * d2act_b{bi}_{li}(ii)")
                lines.append(f"  DO jj = 1, {b_in}")
                lines.append(f"    DO kk = 1, {b_in}")
                lines.append(
                    f"      d2W_b{bi}(jj, kk) = d2W_b{bi}(jj, kk) + coeff * P_b{bi}_{li}(ii, jj) * P_b{bi}_{li}(ii, kk)"
                )
                lines.append("    END DO")
                lines.append("  END DO")
                lines.append("END DO")
            lines.append("")

            # Scatter Hessian (block-diagonal)
            for si, idx_i in enumerate(indices):
                for sj, idx_j in enumerate(indices):
                    lines.append(
                        f"d2W_dI2({idx_i + 1},{idx_j + 1}) = d2W_dI2({idx_i + 1},{idx_j + 1})"
                        f" + d2W_b{bi}({si + 1},{sj + 1}) / (in_std({idx_i + 1}) * in_std({idx_j + 1}))"
                    )
            lines.append("")

        return "\n".join(lines)

    @staticmethod
    def _emit_icnn_act(lines: list[str], bi: int, li: int, act: str, hd: int) -> None:
        """Emit activation + dact + d2act for ICNN branch layer."""
        if act == "identity":
            lines.append(f"z_b{bi}_{li} = a_b{bi}_{li}")
            lines.append(f"dact_b{bi}_{li} = 1.0d0")
            lines.append(f"d2act_b{bi}_{li} = 0.0d0")
        elif act == "softplus":
            lines.append(f"z_b{bi}_{li} = log(1.0d0 + exp(a_b{bi}_{li}))")
            lines.append(f"dact_b{bi}_{li} = 1.0d0 / (1.0d0 + exp(-a_b{bi}_{li}))")
            lines.append(f"d2act_b{bi}_{li} = dact_b{bi}_{li} * (1.0d0 - dact_b{bi}_{li})")
        elif act == "tanh":
            lines.append(f"z_b{bi}_{li} = tanh(a_b{bi}_{li})")
            lines.append(f"dact_b{bi}_{li} = 1.0d0 - z_b{bi}_{li}**2")
            lines.append(f"d2act_b{bi}_{li} = -2.0d0 * z_b{bi}_{li} * dact_b{bi}_{li}")
        elif act == "relu":
            lines.append(f"DO ii = 1, {hd}")
            lines.append(f"  z_b{bi}_{li}(ii) = max(0.0d0, a_b{bi}_{li}(ii))")
            lines.append(f"  IF (a_b{bi}_{li}(ii) > 0.0d0) THEN")
            lines.append(f"    dact_b{bi}_{li}(ii) = 1.0d0")
            lines.append("  ELSE")
            lines.append(f"    dact_b{bi}_{li}(ii) = 0.0d0")
            lines.append("  END IF")
            lines.append(f"  d2act_b{bi}_{li}(ii) = 0.0d0")
            lines.append("END DO")

    def emit(self) -> str:
        """Generate the complete hybrid UMAT Fortran code."""
        today = datetime.datetime.now().strftime("%Y-%m-%d")
        nn_params = self._emit_nn_parameters()
        layers = self.exported.layers
        weights = self.exported.weights
        hidden_dims = [weights[layer.weights].shape[0] for layer in layers]
        in_dim = self.exported.metadata["input_dim"]
        num_fibers = (in_dim - 3) // 2
        aniso = num_fibers > 0
        n_inv = in_dim

        if aniso:
            inv_parts = ["I1_bar", "I2_bar", "J"]
            for k in range(num_fibers):
                inv_parts.extend([f"I4_{k + 1}", f"I5_{k + 1}"])
            input_desc = ", ".join(inv_parts)
        else:
            input_desc = "I1_bar, I2_bar, J"

        code = f"""\
!>********************************************************************
!> Hybrid UMAT: NN-based strain energy with analytical mechanics
!>
!> Generated: {today}
!> Architecture: MLP {" x ".join(str(d) for d in hidden_dims)}
!> Input: {input_desc}  ->  Output: W (strain energy)
!>
!> The neural network provides W({input_desc}).
!> Stress and tangent are derived analytically via chain rule.
!>   Cauchy = (1/J) * F * PK2 * F^T
!>   Tangent = push-forward of material tangent + Jaumann correction
{"!> Fiber directions: " + ", ".join(f"a0_{k + 1} from props({k * 3 + 1}:{k * 3 + 3})" for k in range(num_fibers)) if aniso else ""}
!>********************************************************************

MODULE nn_sef
  IMPLICIT NONE

  ! NN weights and biases
  {nn_params}

CONTAINS

  SUBROUTINE nn_eval(nn_input, W_nn, dW_dI, d2W_dI2)
    !> Evaluate NN: W({input_desc}), dW/dI, and d²W/dI² via backprop
    DOUBLE PRECISION, INTENT(IN)  :: nn_input({n_inv})
    DOUBLE PRECISION, INTENT(OUT) :: W_nn
    DOUBLE PRECISION, INTENT(OUT) :: dW_dI({n_inv})
    DOUBLE PRECISION, INTENT(OUT) :: d2W_dI2({n_inv},{n_inv})

    ! Local variables
    INTEGER :: ii, jj, kk
    {self._emit_nn_forward_and_backward()}

  END SUBROUTINE nn_eval

END MODULE nn_sef


SUBROUTINE umat(stress, statev, ddsdde, sse, spd, scd, rpl, &
    ddsddt, drplde, drpldt, stran, dstran, time, dtime, temp, &
    dtemp, predef, dpred, cmname, ndi, nshr, ntens, nstatev, &
    props, nprops, coords, drot, pnewdt, celent, dfgrd0, &
    dfgrd1, noel, npt, layer, kspt, kstep, kinc)

  USE nn_sef
  IMPLICIT NONE

  ! Standard UMAT interface
  INTEGER, INTENT(IN OUT) :: noel, npt, layer, kspt, kstep, kinc
  INTEGER, INTENT(IN OUT) :: ndi, nshr, ntens, nstatev, nprops
  DOUBLE PRECISION, INTENT(IN OUT) :: sse, spd, scd, rpl, dtime
  DOUBLE PRECISION, INTENT(IN OUT) :: drpldt, temp, dtemp, pnewdt, celent
  CHARACTER(LEN=8), INTENT(IN OUT) :: cmname
  DOUBLE PRECISION, INTENT(IN OUT) :: stress(ntens), statev(nstatev)
  DOUBLE PRECISION, INTENT(IN OUT) :: ddsdde(ntens, ntens)
  DOUBLE PRECISION, INTENT(IN OUT) :: ddsddt(ntens), drplde(ntens)
  DOUBLE PRECISION, INTENT(IN OUT) :: stran(ntens), dstran(ntens)
  DOUBLE PRECISION, INTENT(IN OUT) :: time(2), predef(1), dpred(1)
  DOUBLE PRECISION, INTENT(IN)     :: props(nprops)
  DOUBLE PRECISION, INTENT(IN OUT) :: coords(3), drot(3, 3)
  DOUBLE PRECISION, INTENT(IN OUT) :: dfgrd0(3, 3), dfgrd1(3, 3)

  ! Local variables
  DOUBLE PRECISION :: C(3,3), invC(3,3), detC, detF
  DOUBLE PRECISION :: I1, I2, I1_bar, I2_bar, Jac
  DOUBLE PRECISION :: trC, trC2
  DOUBLE PRECISION :: nn_input({n_inv}), W_nn, dW_dI({n_inv})
  DOUBLE PRECISION :: PK2(3,3), sigma_full(3,3)
  DOUBLE PRECISION :: dI1_dC(3,3), dI2_dC(3,3), dJ_dC(3,3)
  DOUBLE PRECISION :: eye3(3,3)
  DOUBLE PRECISION :: Jm23, Jm43
  INTEGER :: ii, jj, kk, ll
{self._emit_aniso_declarations(num_fibers) if aniso else ""}
  ! d²W/dI² from analytical NN Hessian
  DOUBLE PRECISION :: d2W_dI2({n_inv},{n_inv})

  ! For tangent: material tangent in reference config
  DOUBLE PRECISION :: dPK2_dC(3,3,3,3)
  ! Spatial tangent + Jaumann
  DOUBLE PRECISION :: cmat_spatial(3,3,3,3)

  ! Voigt indices
  INTEGER :: v1(6), v2(6)
  DATA v1 /1, 2, 3, 1, 1, 2/
  DATA v2 /1, 2, 3, 2, 3, 3/

  ! Identity tensor
  eye3 = 0.0d0
  eye3(1,1) = 1.0d0
  eye3(2,2) = 1.0d0
  eye3(3,3) = 1.0d0

  ! ================================================================
  ! 1. Kinematics: F -> C -> invariants
  ! ================================================================
  ! Right Cauchy-Green: C = F^T F
  DO ii = 1, 3
    DO jj = 1, 3
      C(ii,jj) = 0.0d0
      DO kk = 1, 3
        C(ii,jj) = C(ii,jj) + dfgrd1(kk,ii) * dfgrd1(kk,jj)
      END DO
    END DO
  END DO

  ! Determinants
  detC = C(1,1)*(C(2,2)*C(3,3) - C(2,3)*C(3,2)) &
       - C(1,2)*(C(2,1)*C(3,3) - C(2,3)*C(3,1)) &
       + C(1,3)*(C(2,1)*C(3,2) - C(2,2)*C(3,1))
  Jac = sqrt(detC)
  detF = Jac

  ! Invariants
  trC = C(1,1) + C(2,2) + C(3,3)
  trC2 = 0.0d0
  DO ii = 1, 3
    DO jj = 1, 3
      trC2 = trC2 + C(ii,jj) * C(jj,ii)
    END DO
  END DO

  Jm23 = detC**(-1.0d0/3.0d0)
  Jm43 = detC**(-2.0d0/3.0d0)

  I1_bar = trC * Jm23
  I2_bar = 0.5d0 * (trC**2 - trC2) * Jm43
{self._emit_aniso_invariants(num_fibers) if aniso else ""}
  ! ================================================================
  ! 2. NN evaluation: W({input_desc}) and dW/dI
  ! ================================================================
  nn_input(1) = I1_bar
  nn_input(2) = I2_bar
  nn_input(3) = Jac
{self._emit_aniso_nn_input(num_fibers) if aniso else ""}  CALL nn_eval(nn_input, W_nn, dW_dI, d2W_dI2)

  ! Store strain energy
  sse = W_nn

  ! ================================================================
  ! 3. Analytical derivatives: dI/dC
  ! ================================================================
  ! Inverse of C (for dJ/dC)
  invC(1,1) = (C(2,2)*C(3,3) - C(2,3)*C(3,2)) / detC
  invC(2,2) = (C(1,1)*C(3,3) - C(1,3)*C(3,1)) / detC
  invC(3,3) = (C(1,1)*C(2,2) - C(1,2)*C(2,1)) / detC
  invC(1,2) = (C(1,3)*C(3,2) - C(1,2)*C(3,3)) / detC
  invC(2,1) = invC(1,2)
  invC(1,3) = (C(1,2)*C(2,3) - C(1,3)*C(2,2)) / detC
  invC(3,1) = invC(1,3)
  invC(2,3) = (C(1,3)*C(2,1) - C(1,1)*C(2,3)) / detC
  invC(3,2) = invC(2,3)

  ! dI1_bar/dC = J^(-2/3) * (I - (1/3)*I1_bar * C^-1)
  DO ii = 1, 3
    DO jj = 1, 3
      dI1_dC(ii,jj) = Jm23 * (eye3(ii,jj) - (1.0d0/3.0d0) * trC * invC(ii,jj))
    END DO
  END DO

  ! dI2_bar/dC = J^(-4/3) * (trC*I - C - (2/3)*I2_bar*J^(4/3) * C^-1)
  DO ii = 1, 3
    DO jj = 1, 3
      dI2_dC(ii,jj) = Jm43 * (trC * eye3(ii,jj) - C(ii,jj)) &
                     - (2.0d0/3.0d0) * I2_bar * invC(ii,jj)
    END DO
  END DO

  ! dJ/dC = (1/2) * J * C^-1
  DO ii = 1, 3
    DO jj = 1, 3
      dJ_dC(ii,jj) = 0.5d0 * Jac * invC(ii,jj)
    END DO
  END DO
{self._emit_aniso_didC(num_fibers) if aniso else ""}
  ! ================================================================
  ! 4. PK2 stress: S = 2 * dW/dC = 2 * sum_k (dW/dIk * dIk/dC)
  ! ================================================================
  DO ii = 1, 3
    DO jj = 1, 3
      PK2(ii,jj) = 2.0d0 * ( &
          dW_dI(1) * dI1_dC(ii,jj) &
        + dW_dI(2) * dI2_dC(ii,jj) &
        + dW_dI(3) * dJ_dC(ii,jj){self._emit_aniso_pk2_terms(num_fibers) if aniso else ""} )
    END DO
  END DO

  ! ================================================================
  ! 5. Cauchy stress: sigma = (1/J) * F * S * F^T
  ! ================================================================
  sigma_full = 0.0d0
  DO ii = 1, 3
    DO jj = 1, 3
      DO kk = 1, 3
        DO ll = 1, 3
          sigma_full(ii,jj) = sigma_full(ii,jj) &
            + dfgrd1(ii,kk) * PK2(kk,ll) * dfgrd1(jj,ll)
        END DO
      END DO
    END DO
  END DO
  sigma_full = sigma_full / detF

  ! To Voigt: stress(1..6) = [s11, s22, s33, s12, s13, s23]
  DO ii = 1, ntens
    stress(ii) = sigma_full(v1(ii), v2(ii))
  END DO

  ! ================================================================
  ! 6. Material tangent: C_ABCD = 4 * d²W/dC²
  !    d²W/dCdC = sum_k sum_l (d²W/dIk*dIl) * (dIk/dC) x (dIl/dC)
  !             + sum_k (dW/dIk) * (d²Ik/dCdC)
  ! ================================================================
  dPK2_dC = 0.0d0

  BLOCK
    DOUBLE PRECISION :: dIdC(3, 3, {n_inv})  ! dIdC(:,:,k) = dIk/dC
    DOUBLE PRECISION :: val

    DO ii = 1, 3
      DO jj = 1, 3
        dIdC(ii, jj, 1) = dI1_dC(ii, jj)
        dIdC(ii, jj, 2) = dI2_dC(ii, jj)
        dIdC(ii, jj, 3) = dJ_dC(ii, jj)
      END DO
    END DO
{self._emit_aniso_dIdC_pack(num_fibers) if aniso else ""}
    ! C_ABCD += 4 * sum_k sum_l d²W/dIk*dIl * (dIk/dC)_AB * (dIl/dC)_CD
    DO ii = 1, 3
      DO jj = 1, 3
        DO kk = 1, 3
          DO ll = 1, 3
            val = 0.0d0
            DO mm = 1, {n_inv}
              DO nn = 1, {n_inv}
                val = val + d2W_dI2(mm, nn) * dIdC(ii, jj, mm) * dIdC(kk, ll, nn)
              END DO
            END DO
            dPK2_dC(ii, jj, kk, ll) = 4.0d0 * val
          END DO
        END DO
      END DO
    END DO

    ! Term 2: 4 * sum_k dW/dIk * d²Ik/dCdC

    ! d²J/(dC_AB dC_CD) = J/4 * (invC_AB*invC_CD - invC_AC*invC_BD - invC_AD*invC_BC)
    DO ii = 1, 3
      DO jj = 1, 3
        DO kk = 1, 3
          DO ll = 1, 3
            val = 0.25d0 * Jac * ( &
                invC(ii,jj) * invC(kk,ll) &
              - 0.5d0 * (invC(ii,kk)*invC(jj,ll) + invC(ii,ll)*invC(jj,kk)))
            dPK2_dC(ii,jj,kk,ll) = dPK2_dC(ii,jj,kk,ll) + 4.0d0 * dW_dI(3) * val
          END DO
        END DO
      END DO
    END DO

    ! d²I1_bar/dCdC
    DO ii = 1, 3
      DO jj = 1, 3
        DO kk = 1, 3
          DO ll = 1, 3
            val = Jm23 * ( &
              (-1.0d0/3.0d0) * (eye3(ii,jj)*invC(kk,ll) + invC(ii,jj)*eye3(kk,ll)) &
              + (1.0d0/9.0d0) * trC * invC(ii,jj)*invC(kk,ll) &
              + (1.0d0/3.0d0) * trC * 0.5d0 * &
                  (invC(ii,kk)*invC(jj,ll) + invC(ii,ll)*invC(jj,kk)))
            dPK2_dC(ii,jj,kk,ll) = dPK2_dC(ii,jj,kk,ll) + 4.0d0 * dW_dI(1) * val
          END DO
        END DO
      END DO
    END DO

    ! d²I2_bar/dCdC
    DO ii = 1, 3
      DO jj = 1, 3
        DO kk = 1, 3
          DO ll = 1, 3
            val = Jm43 * ( &
              eye3(ii,jj)*eye3(kk,ll) &
              - 0.5d0*(eye3(ii,kk)*eye3(jj,ll) + eye3(ii,ll)*eye3(jj,kk)) &
              + (-2.0d0/3.0d0) * (trC*eye3(ii,jj) - C(ii,jj)) * invC(kk,ll) &
              + (-2.0d0/3.0d0) * invC(ii,jj) * (trC*eye3(kk,ll) - C(kk,ll)) &
              + (4.0d0/9.0d0) * I2_bar / Jm43 * invC(ii,jj) * invC(kk,ll) &
              + (2.0d0/3.0d0) * I2_bar / Jm43 * 0.5d0 * &
                  (invC(ii,kk)*invC(jj,ll) + invC(ii,ll)*invC(jj,kk)))
            dPK2_dC(ii,jj,kk,ll) = dPK2_dC(ii,jj,kk,ll) + 4.0d0 * dW_dI(2) * val
          END DO
        END DO
      END DO
    END DO
{self._emit_aniso_d2IdC2(num_fibers) if aniso else ""}
  END BLOCK

  ! ================================================================
  ! 7. Push forward to spatial tangent: c_ijkl = (1/J) F_iA F_jB F_kC F_lD C_ABCD
  ! ================================================================
  cmat_spatial = 0.0d0
  DO ii = 1, 3
    DO jj = 1, 3
      DO kk = 1, 3
        DO ll = 1, 3
          val = 0.0d0
          DO mm = 1, 3
            DO nn = 1, 3
              DO pp = 1, 3
                DO qq = 1, 3
                  val = val + dfgrd1(ii,mm)*dfgrd1(jj,nn)*dfgrd1(kk,pp)*dfgrd1(ll,qq) &
                            * dPK2_dC(mm,nn,pp,qq)
                END DO
              END DO
            END DO
          END DO
          cmat_spatial(ii,jj,kk,ll) = val / detF
        END DO
      END DO
    END DO
  END DO

  ! Jaumann correction: c_ijkl += 0.5*(delta_ik*sigma_jl + sigma_ik*delta_jl
  !                                   + delta_il*sigma_jk + sigma_il*delta_jk)
  DO ii = 1, 3
    DO jj = 1, 3
      DO kk = 1, 3
        DO ll = 1, 3
          cmat_spatial(ii,jj,kk,ll) = cmat_spatial(ii,jj,kk,ll) + 0.5d0 * ( &
            eye3(ii,kk)*sigma_full(jj,ll) + sigma_full(ii,kk)*eye3(jj,ll) &
          + eye3(ii,ll)*sigma_full(jj,kk) + sigma_full(ii,ll)*eye3(jj,kk))
        END DO
      END DO
    END DO
  END DO

  ! To Voigt: ddsdde(6,6)
  DO ii = 1, ntens
    DO jj = 1, ntens
      ddsdde(ii, jj) = cmat_spatial(v1(ii), v2(ii), v1(jj), v2(jj))
    END DO
  END DO

RETURN
END SUBROUTINE umat
"""
        return code

    def write(self, path: str | Path) -> None:
        """Write the hybrid UMAT to a file."""
        Path(path).write_text(self.emit())
