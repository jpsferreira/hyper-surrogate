"""Tests for parameter fitting to experimental data."""

import numpy as np
import pytest

from hyper_surrogate.data.experimental import ExperimentalData
from hyper_surrogate.mechanics.materials import NeoHooke

scipy = pytest.importorskip("scipy")


def _generate_synthetic_uniaxial(material_cls, params, n=20):
    """Generate synthetic uniaxial stress-stretch data from a known model."""
    from sympy import Symbol, lambdify

    mat = material_cls(parameters=params)
    stretches = np.linspace(1.01, 1.5, n)

    i1s, i2s, js = Symbol("I1b"), Symbol("I2b"), Symbol("J")
    W = mat.sef_from_invariants(i1s, i2s, js)
    dW_dI1 = W.diff(i1s)
    param_syms = list(mat._symbols.values())
    fn = lambdify((i1s, *param_syms), dW_dI1, modules="numpy")
    param_vals = list(mat._params.values())

    stresses = []
    for lam in stretches:
        # For incompressible uniaxial: I1_bar ~ lam^2 + 2/lam
        i1_bar = lam**2 + 2.0 / lam
        dw = float(fn(i1_bar, *param_vals))
        sigma = 2.0 * dw * (lam**2 - 1.0 / lam)
        stresses.append(sigma)

    return ExperimentalData.from_uniaxial(np.array(stretches), np.array(stresses))


def test_fit_neohooke():
    """Fit NeoHooke to synthetic NeoHooke data — should recover parameters."""
    from hyper_surrogate.data.fitting import fit_material

    true_params = {"C10": 0.8, "KBULK": 1000.0}
    data = _generate_synthetic_uniaxial(NeoHooke, true_params)

    _fitted_mat, result = fit_material(
        NeoHooke,
        data,
        initial_guess={"C10": 0.5},
        fixed_params={"KBULK": 1000.0},
    )

    assert result.success
    assert result.r_squared > 0.99
    np.testing.assert_allclose(result.parameters["C10"], 0.8, rtol=0.05)


def test_fit_result_fields():
    """Verify all FitResult fields are populated."""
    from hyper_surrogate.data.fitting import fit_material

    data = _generate_synthetic_uniaxial(NeoHooke, {"C10": 0.5, "KBULK": 1000.0})
    _, result = fit_material(NeoHooke, data)

    assert isinstance(result.parameters, dict)
    assert isinstance(result.r_squared, float)
    assert isinstance(result.residual_norm, float)
    assert result.n_evaluations > 0
    assert isinstance(result.message, str)
