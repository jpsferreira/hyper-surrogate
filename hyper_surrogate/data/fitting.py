"""Parameter fitting for constitutive models to experimental data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class FitResult:
    """Result of a material parameter fitting procedure."""

    parameters: dict[str, float]
    r_squared: float
    residual_norm: float
    n_evaluations: int
    success: bool
    message: str


def _compute_uniaxial_stress(
    material_cls: type,
    params_dict: dict[str, float],
    c_exp: np.ndarray,
    stretch: np.ndarray,
) -> np.ndarray:
    """Compute Cauchy stress for uniaxial test from a material model."""
    from sympy import Symbol
    from sympy import lambdify as sym_lambdify

    from hyper_surrogate.mechanics.kinematics import Kinematics

    mat = material_cls(parameters=params_dict)
    i1s, i2s, js = Symbol("I1b"), Symbol("I2b"), Symbol("J")
    W = mat.sef_from_invariants(i1s, i2s, js)
    dW_dI1 = W.diff(i1s)

    param_syms = list(mat._symbols.values())
    fn = sym_lambdify((i1s, i2s, js, *param_syms), dW_dI1, modules="numpy")

    i1 = Kinematics.isochoric_invariant1(c_exp)
    i2 = Kinematics.isochoric_invariant2(c_exp)
    j = np.sqrt(Kinematics.det_invariant(c_exp))
    param_vals = list(mat._params.values())

    dw_di1 = np.array([float(fn(i1[n], i2[n], j[n], *param_vals)) for n in range(len(c_exp))])
    lam = stretch[:, 0]
    return (2.0 * dw_di1 * (lam**2 - 1.0 / lam)).reshape(-1, 1)


def _compute_biaxial_stress(
    material_cls: type,
    params_dict: dict[str, float],
    c_exp: np.ndarray,
    stretch: np.ndarray,
) -> np.ndarray:
    """Compute Cauchy stress for biaxial test from a material model."""
    mat = material_cls(parameters=params_dict)
    pk2 = mat.evaluate_pk2(c_exp)
    lam1 = stretch[:, 0]
    lam2 = stretch[:, 1]
    sigma_11 = pk2[:, 0, 0] * lam1**2
    sigma_22 = pk2[:, 1, 1] * lam2**2
    return np.column_stack([sigma_11, sigma_22])


def _resolve_defaults(
    material_cls: type,
    initial_guess: dict[str, float] | None,
    fixed_params: dict[str, float] | None,
) -> tuple[dict[str, float], dict[str, float]]:
    """Resolve initial guess and fixed params from material defaults."""
    if hasattr(material_cls, "DEFAULT_PARAMS"):
        default_params = dict(material_cls.DEFAULT_PARAMS)
    else:
        msg = f"{material_cls.__name__} has no DEFAULT_PARAMS"
        raise ValueError(msg)

    if fixed_params is None:
        fixed_params = {"KBULK": default_params.get("KBULK", 1000.0)}
    if initial_guess is None:
        initial_guess = {k: v for k, v in default_params.items() if k not in fixed_params}
    return initial_guess, fixed_params


def _compute_stress_for_params(
    material_cls: type,
    params_dict: dict[str, float],
    c_exp: np.ndarray,
    stretch: np.ndarray,
    test_type: str,
) -> np.ndarray:
    """Dispatch stress computation based on test type."""
    if test_type == "uniaxial":
        return _compute_uniaxial_stress(material_cls, params_dict, c_exp, stretch)
    if test_type == "biaxial":
        return _compute_biaxial_stress(material_cls, params_dict, c_exp, stretch)
    msg = f"Unsupported test type: {test_type}"
    raise ValueError(msg)


def fit_material(
    material_cls: type,
    data: Any,
    initial_guess: dict[str, float] | None = None,
    bounds: dict[str, tuple[float, float]] | None = None,
    fixed_params: dict[str, float] | None = None,
) -> tuple[Any, FitResult]:
    """Fit material parameters to experimental data.

    Uses scipy.optimize.minimize to minimize the stress residual between
    the analytical model and experimental data.

    Args:
        material_cls: Material class to instantiate (e.g. NeoHooke, Yeoh).
        data: ExperimentalData instance with stretch/stress pairs.
        initial_guess: Initial parameter values. Defaults to material's DEFAULT_PARAMS.
        bounds: Parameter bounds as {name: (lower, upper)}.
        fixed_params: Parameters to hold fixed (e.g. {"KBULK": 1000.0}).

    Returns:
        Tuple of (fitted Material instance, FitResult).
    """
    try:
        from scipy.optimize import minimize
    except ImportError:
        msg = "scipy is required for parameter fitting. Install with: pip install scipy"
        raise ImportError(msg) from None

    from hyper_surrogate.mechanics.kinematics import Kinematics

    initial_guess, fixed_params = _resolve_defaults(material_cls, initial_guess, fixed_params)

    free_names = list(initial_guess.keys())
    x0 = np.array([initial_guess[k] for k in free_names])
    scipy_bounds: list[tuple[float | None, float | None]] = (
        [bounds.get(k, (None, None)) for k in free_names]
        if bounds is not None
        else [(1e-10, None) if k != "KBULK" else (None, None) for k in free_names]
    )

    F_exp = data.to_deformation_gradients()
    C_exp = Kinematics.right_cauchy_green(F_exp)
    stress_exp = data.stress

    def objective(x: np.ndarray) -> float:
        params_dict = {**fixed_params}
        for i, name in enumerate(free_names):
            params_dict[name] = float(x[i])
        try:
            sigma = _compute_stress_for_params(material_cls, params_dict, C_exp, data.stretch, data.test_type)
            return float(np.sum((sigma - stress_exp) ** 2))
        except Exception:
            return 1e20

    result = minimize(objective, x0, method="L-BFGS-B", bounds=scipy_bounds)

    final_params = {**fixed_params}
    for i, name in enumerate(free_names):
        final_params[name] = float(result.x[i])

    sigma_model = _compute_stress_for_params(material_cls, final_params, C_exp, data.stretch, data.test_type)
    ss_res = np.sum((stress_exp - sigma_model) ** 2)
    ss_tot = np.sum((stress_exp - np.mean(stress_exp)) ** 2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return material_cls(parameters=final_params), FitResult(
        parameters=final_params,
        r_squared=r_squared,
        residual_norm=float(np.sqrt(ss_res)),
        n_evaluations=result.nfev,
        success=result.success,
        message=result.message,
    )
