"""Tests for GasserOgdenHolzapfel model with fiber dispersion."""

import numpy as np
from sympy import Symbol

from hyper_surrogate.mechanics.kinematics import Kinematics
from hyper_surrogate.mechanics.materials import GasserOgdenHolzapfel, HolzapfelOgden


def _identity_batch(n=1):
    return np.tile(np.eye(3), (n, 1, 1))


def _uniaxial_F(stretch, axis=0, n=1):
    F = np.tile(np.eye(3), (n, 1, 1))
    lateral = 1.0 / np.sqrt(stretch)
    F[:, 0, 0] = lateral
    F[:, 1, 1] = lateral
    F[:, 2, 2] = lateral
    F[:, axis, axis] = stretch
    return F


def _evaluate_energy_from_invariants(mat, i1_bar, i2_bar, j, i4=None, i5=None):
    from sympy import lambdify

    syms_inv = [Symbol("I1b"), Symbol("I2b"), Symbol("J")]
    if i4 is not None:
        syms_inv += [Symbol("I4"), Symbol("I5")]
    W_expr = mat.sef_from_invariants(*syms_inv[:3], *syms_inv[3:] if len(syms_inv) > 3 else [])
    param_syms = list(mat._symbols.values())
    fn = lambdify((*syms_inv, *param_syms), W_expr, modules="numpy")
    param_vals = list(mat._params.values())
    if i4 is not None:
        return fn(i1_bar, i2_bar, j, i4, i5, *param_vals)
    return fn(i1_bar, i2_bar, j, *param_vals)


def test_is_anisotropic():
    mat = GasserOgdenHolzapfel()
    assert mat.is_anisotropic
    assert mat.fiber_direction is not None


def test_default_kappa():
    mat = GasserOgdenHolzapfel()
    assert mat._params["kappa"] == 0.226


def test_energy_at_identity_is_zero():
    mat = GasserOgdenHolzapfel()
    W = _evaluate_energy_from_invariants(mat, 3.0, 3.0, 1.0, 1.0, 1.0)
    np.testing.assert_allclose(W, 0.0, atol=1e-14)


def test_energy_positive_under_stretch():
    mat = GasserOgdenHolzapfel()
    F = _uniaxial_F(1.1, axis=0)
    C = Kinematics.right_cauchy_green(F)
    a0 = mat.fiber_direction
    i1 = Kinematics.isochoric_invariant1(C)[0]
    i2 = Kinematics.isochoric_invariant2(C)[0]
    j = np.sqrt(Kinematics.det_invariant(C))[0]
    i4 = Kinematics.fiber_invariant4(C, a0)[0]
    i5 = Kinematics.fiber_invariant5(C, a0)[0]
    W = float(_evaluate_energy_from_invariants(mat, i1, i2, j, i4, i5))
    assert W > 0


def test_kappa_zero_matches_holzapfel():
    """kappa=0 should give same result as HolzapfelOgden."""
    shared = {"a": 0.059, "b": 8.023, "af": 18.472, "bf": 16.026, "KBULK": 1000.0}
    goh = GasserOgdenHolzapfel(parameters={**shared, "kappa": 0.0})
    ho = HolzapfelOgden(parameters=shared)

    F = _uniaxial_F(1.15, axis=0)
    C = Kinematics.right_cauchy_green(F)
    a0 = np.array([1.0, 0.0, 0.0])
    i1 = float(Kinematics.isochoric_invariant1(C)[0])
    i2 = float(Kinematics.isochoric_invariant2(C)[0])
    j = float(np.sqrt(Kinematics.det_invariant(C)[0]))
    i4 = float(Kinematics.fiber_invariant4(C, a0)[0])
    i5 = float(Kinematics.fiber_invariant5(C, a0)[0])

    W_goh = float(_evaluate_energy_from_invariants(goh, i1, i2, j, i4, i5))
    W_ho = float(_evaluate_energy_from_invariants(ho, i1, i2, j, i4, i5))
    np.testing.assert_allclose(W_goh, W_ho, rtol=1e-10)


def test_kappa_third_isotropic_fiber():
    """kappa=1/3: E_bar = (1/3)(I1_bar-3), no I4 dependence in E_bar."""
    mat = GasserOgdenHolzapfel(parameters={
        "a": 0.0, "b": 1.0, "af": 1.0, "bf": 1.0,
        "kappa": 1.0 / 3.0, "KBULK": 0.0,
    })
    # With kappa=1/3, E_bar = (1/3)(I1_bar-3) regardless of I4
    W_a = float(_evaluate_energy_from_invariants(mat, 3.3, 3.0, 1.0, 1.5, 1.0))
    W_b = float(_evaluate_energy_from_invariants(mat, 3.3, 3.0, 1.0, 0.8, 1.0))
    np.testing.assert_allclose(W_a, W_b, rtol=1e-10)


def test_grad_finite_difference():
    mat = GasserOgdenHolzapfel()
    F = _uniaxial_F(1.1, axis=0)
    C = Kinematics.right_cauchy_green(F)
    a0 = mat.fiber_direction

    i1 = float(Kinematics.isochoric_invariant1(C)[0])
    i2 = float(Kinematics.isochoric_invariant2(C)[0])
    j = float(np.sqrt(Kinematics.det_invariant(C)[0]))
    i4 = float(Kinematics.fiber_invariant4(C, a0)[0])
    i5 = float(Kinematics.fiber_invariant5(C, a0)[0])

    inv_vals = [i1, i2, j, i4, i5]
    grad_analytical = mat.evaluate_energy_grad_invariants(C)[0]

    eps = 1e-7
    grad_fd = np.zeros(5)
    for k in range(5):
        inv_p = list(inv_vals)
        inv_m = list(inv_vals)
        inv_p[k] += eps
        inv_m[k] -= eps
        W_p = float(_evaluate_energy_from_invariants(mat, *inv_p))
        W_m = float(_evaluate_energy_from_invariants(mat, *inv_m))
        grad_fd[k] = (W_p - W_m) / (2 * eps)

    np.testing.assert_allclose(grad_analytical, grad_fd, rtol=1e-5, atol=1e-10)


def test_sef_symbolic():
    mat = GasserOgdenHolzapfel()
    i1, i2, j, i4, i5 = (Symbol(s) for s in ["I1b", "I2b", "J", "I4", "I5"])
    W = mat.sef_from_invariants(i1, i2, j, i4, i5)
    free = W.free_symbols
    assert Symbol("kappa") in free
    assert Symbol("af") in free
