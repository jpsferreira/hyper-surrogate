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

    def __init__(self, exported: ExportedModel) -> None:
        if exported.metadata.get("architecture") != "mlp":
            msg = "HybridUMATEmitter only supports MLP architecture"
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

    def _emit_nn_forward_and_backward(self) -> str:  # noqa: C901
        """Emit NN forward pass + analytical backward pass for dW/dI and d²W/dI²."""
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
    # Anisotropic (in_dim=5) Fortran snippets
    # ------------------------------------------------------------------

    @staticmethod
    def _emit_aniso_declarations() -> str:
        """Extra local variable declarations for anisotropic UMAT."""
        return """\
  ! Fiber invariants (anisotropic)
  DOUBLE PRECISION :: a0(3), Ca0(3), I4, I5
  DOUBLE PRECISION :: dI4_dC(3,3), dI5_dC(3,3)
"""

    @staticmethod
    def _emit_aniso_invariants() -> str:
        """Compute fiber invariants I4, I5 from C and a0."""
        return """
  ! Fiber direction from material properties
  a0(1) = props(1)
  a0(2) = props(2)
  a0(3) = props(3)

  ! Ca0 = C * a0
  DO ii = 1, 3
    Ca0(ii) = 0.0d0
    DO jj = 1, 3
      Ca0(ii) = Ca0(ii) + C(ii,jj) * a0(jj)
    END DO
  END DO

  ! I4 = a0 . C . a0
  I4 = 0.0d0
  DO ii = 1, 3
    I4 = I4 + a0(ii) * Ca0(ii)
  END DO

  ! I5 = a0 . C^2 . a0 = Ca0 . Ca0
  I5 = 0.0d0
  DO ii = 1, 3
    I5 = I5 + Ca0(ii) * Ca0(ii)
  END DO
"""

    @staticmethod
    def _emit_aniso_nn_input() -> str:
        """Set fiber invariant entries in nn_input."""
        return """\
  nn_input(4) = I4
  nn_input(5) = I5
"""

    @staticmethod
    def _emit_aniso_didC() -> str:
        """Compute dI4/dC and dI5/dC."""
        return """
  ! dI4/dC = a0 (x) a0
  DO ii = 1, 3
    DO jj = 1, 3
      dI4_dC(ii,jj) = a0(ii) * a0(jj)
    END DO
  END DO

  ! dI5/dC = a0 (x) Ca0 + Ca0 (x) a0
  DO ii = 1, 3
    DO jj = 1, 3
      dI5_dC(ii,jj) = a0(ii) * Ca0(jj) + Ca0(ii) * a0(jj)
    END DO
  END DO
"""

    @staticmethod
    def _emit_aniso_pk2_terms() -> str:
        """Additional PK2 terms for I4, I5."""
        return """ &
        + dW_dI(4) * dI4_dC(ii,jj) &
        + dW_dI(5) * dI5_dC(ii,jj)"""

    @staticmethod
    def _emit_aniso_dIdC_pack() -> str:
        """Pack fiber dI/dC into dIdC array."""
        return """\
    DO ii = 1, 3
      DO jj = 1, 3
        dIdC(ii, jj, 4) = dI4_dC(ii, jj)
        dIdC(ii, jj, 5) = dI5_dC(ii, jj)
      END DO
    END DO
"""

    @staticmethod
    def _emit_aniso_d2IdC2() -> str:
        """Second derivatives d²I4/dCdC=0, d²I5/dCdC for tangent."""
        return """
    ! d²I4/dCdC = 0 (dI4/dC = a0 (x) a0 is constant w.r.t. C)
    ! No contribution needed for I4.

    ! d²I5/(dC_AB dC_CD) = 0.5*(a0_A*a0_D*delta_BC + a0_A*a0_C*delta_BD
    !                          + a0_B*a0_D*delta_AC + a0_B*a0_C*delta_AD)
    DO ii = 1, 3
      DO jj = 1, 3
        DO kk = 1, 3
          DO ll = 1, 3
            val = 0.5d0 * ( &
                a0(ii)*a0(ll)*eye3(jj,kk) + a0(ii)*a0(kk)*eye3(jj,ll) &
              + a0(jj)*a0(ll)*eye3(ii,kk) + a0(jj)*a0(kk)*eye3(ii,ll))
            dPK2_dC(ii,jj,kk,ll) = dPK2_dC(ii,jj,kk,ll) + 4.0d0 * dW_dI(5) * val
          END DO
        END DO
      END DO
    END DO
"""

    def emit(self) -> str:
        """Generate the complete hybrid UMAT Fortran code."""
        today = datetime.datetime.now().strftime("%Y-%m-%d")
        nn_params = self._emit_nn_parameters()
        layers = self.exported.layers
        weights = self.exported.weights
        hidden_dims = [weights[layer.weights].shape[0] for layer in layers]
        in_dim = self.exported.metadata["input_dim"]
        aniso = in_dim == 5

        if aniso:
            input_desc = "I1_bar, I2_bar, J, I4, I5"
            n_inv = 5
        else:
            input_desc = "I1_bar, I2_bar, J"
            n_inv = 3

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
{"!> Fiber direction a0 is read from props(1:3)." if aniso else ""}
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
{self._emit_aniso_declarations() if aniso else ""}
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
{self._emit_aniso_invariants() if aniso else ""}
  ! ================================================================
  ! 2. NN evaluation: W({input_desc}) and dW/dI
  ! ================================================================
  nn_input(1) = I1_bar
  nn_input(2) = I2_bar
  nn_input(3) = Jac
{self._emit_aniso_nn_input() if aniso else ""}  CALL nn_eval(nn_input, W_nn, dW_dI, d2W_dI2)

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
{self._emit_aniso_didC() if aniso else ""}
  ! ================================================================
  ! 4. PK2 stress: S = 2 * dW/dC = 2 * sum_k (dW/dIk * dIk/dC)
  ! ================================================================
  DO ii = 1, 3
    DO jj = 1, 3
      PK2(ii,jj) = 2.0d0 * ( &
          dW_dI(1) * dI1_dC(ii,jj) &
        + dW_dI(2) * dI2_dC(ii,jj) &
        + dW_dI(3) * dJ_dC(ii,jj){self._emit_aniso_pk2_terms() if aniso else ""} )
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
{self._emit_aniso_dIdC_pack() if aniso else ""}
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
{self._emit_aniso_d2IdC2() if aniso else ""}
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
