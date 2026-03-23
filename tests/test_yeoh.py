"""Tests for Yeoh hyperelastic model."""

import numpy as np
from sympy import Symbol

from hyper_surrogate.mechanics.kinematics import Kinematics
from hyper_surrogate.mechanics.materials import Yeoh


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


def _evaluate_energy_from_invariants(mat, i1_bar, i2_bar, j):
    from sympy import lambdify

    syms_inv = [Symbol("I1b"), Symbol("I2b"), Symbol("J")]
    W_expr = mat.sef_from_invariants(*syms_inv)
    param_syms = list(mat._symbols.values())
    fn = lambdify((*syms_inv, *param_syms), W_expr, modules="numpy")
    param_vals = list(mat._params.values())
    return fn(i1_bar, i2_bar, j, *param_vals)


def test_is_isotropic():
    mat = Yeoh()
    assert not mat.is_anisotropic


def test_energy_at_identity_is_zero():
    mat = Yeoh()
    W = _evaluate_energy_from_invariants(mat, 3.0, 3.0, 1.0)
    np.testing.assert_allclose(W, 0.0, atol=1e-14)


def test_energy_positive_under_stretch():
    mat = Yeoh()
    F = _uniaxial_F(1.2, axis=0)
    C = Kinematics.right_cauchy_green(F)
    i1 = float(Kinematics.isochoric_invariant1(C)[0])
    i2 = float(Kinematics.isochoric_invariant2(C)[0])
    j = float(np.sqrt(Kinematics.det_invariant(C)[0]))
    W = float(_evaluate_energy_from_invariants(mat, i1, i2, j))
    assert W > 0


def test_manual_energy():
    """Verify energy against manual computation."""
    mat = Yeoh(parameters={"C10": 1.0, "C20": 0.5, "C30": 0.1, "KBULK": 0.0})
    i1_dev = 0.5  # I1_bar - 3
    i1_bar = 3.5
    expected = 1.0 * i1_dev + 0.5 * i1_dev**2 + 0.1 * i1_dev**3
    W = float(_evaluate_energy_from_invariants(mat, i1_bar, 3.0, 1.0))
    np.testing.assert_allclose(W, expected, rtol=1e-10)


def test_grad_shape():
    mat = Yeoh()
    C = _identity_batch(5) * 1.01
    grad = mat.evaluate_energy_grad_invariants(C)
    assert grad.shape == (5, 3)


def test_grad_finite_difference():
    mat = Yeoh()
    F = _uniaxial_F(1.1, axis=0)
    C = Kinematics.right_cauchy_green(F)

    i1 = float(Kinematics.isochoric_invariant1(C)[0])
    i2 = float(Kinematics.isochoric_invariant2(C)[0])
    j = float(np.sqrt(Kinematics.det_invariant(C)[0]))

    inv_vals = [i1, i2, j]
    grad_analytical = mat.evaluate_energy_grad_invariants(C)[0]

    eps = 1e-7
    grad_fd = np.zeros(3)
    for k in range(3):
        inv_p = list(inv_vals)
        inv_m = list(inv_vals)
        inv_p[k] += eps
        inv_m[k] -= eps
        W_p = float(_evaluate_energy_from_invariants(mat, *inv_p))
        W_m = float(_evaluate_energy_from_invariants(mat, *inv_m))
        grad_fd[k] = (W_p - W_m) / (2 * eps)

    np.testing.assert_allclose(grad_analytical, grad_fd, rtol=1e-5, atol=1e-10)


def test_reduces_to_neohooke():
    """With C20=C30=0, Yeoh reduces to NeoHooke form."""
    mat = Yeoh(parameters={"C10": 0.5, "C20": 0.0, "C30": 0.0, "KBULK": 1000.0})
    from hyper_surrogate.mechanics.materials import NeoHooke

    nh = NeoHooke(parameters={"C10": 0.5, "KBULK": 1000.0})

    F = _uniaxial_F(1.1, axis=0)
    C = Kinematics.right_cauchy_green(F)
    i1 = float(Kinematics.isochoric_invariant1(C)[0])
    i2 = float(Kinematics.isochoric_invariant2(C)[0])
    j = float(np.sqrt(Kinematics.det_invariant(C)[0]))

    W_yeoh = float(_evaluate_energy_from_invariants(mat, i1, i2, j))

    syms = [Symbol("I1b"), Symbol("I2b"), Symbol("J")]
    from sympy import lambdify

    W_expr = nh.sef_from_invariants(*syms)
    fn = lambdify((*syms, *nh._symbols.values()), W_expr, modules="numpy")
    W_nh = float(fn(i1, i2, j, *nh._params.values()))

    np.testing.assert_allclose(W_yeoh, W_nh, rtol=1e-10)
