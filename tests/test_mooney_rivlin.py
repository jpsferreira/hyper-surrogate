"""Comprehensive tests for MooneyRivlin material model.

Mirrors NeoHooke test coverage: symbolic SEF, energy evaluation,
PK2 stress, CMAT tangent, energy gradients, and finite-difference validation.
"""

import numpy as np
import pytest
import sympy as sym

from hyper_surrogate.mechanics.kinematics import Kinematics as K
from hyper_surrogate.mechanics.materials import MooneyRivlin

# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture
def mat():
    return MooneyRivlin()


@pytest.fixture
def f0():
    return np.eye(3)


@pytest.fixture
def f_uni():
    # Uniaxial stretch 3 along x (incompressible)
    return np.diag([3.0, 1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)])


@pytest.fixture
def f_arbitrary():
    return np.array([[3.0, 0.4, 0.1], [0.4, 1.0 / np.sqrt(3.0), 0.1], [0.9, -0.2, 0.6]])


@pytest.fixture
def f_arbitrary2():
    return np.array([[3.0, 0.4, 0.1], [0.4, 0.45, 0.1], [0.9, -0.2, 0.9]])


@pytest.fixture
def f_batch(f0, f_uni, f_arbitrary, f_arbitrary2):
    return np.array([f0, f_uni, f_arbitrary, f_arbitrary2])


@pytest.fixture
def c_batch(f_batch):
    return K.right_cauchy_green(f_batch)


# ── Properties ────────────────────────────────────────────────────


def test_default_params():
    mr = MooneyRivlin()
    assert mr._params["C10"] == 0.3
    assert mr._params["C01"] == 0.2
    assert mr._params["KBULK"] == 1000.0


def test_custom_params():
    mr = MooneyRivlin({"C10": 1.0, "C01": 0.5, "KBULK": 500.0})
    assert mr._params["C10"] == 1.0
    assert mr._params["C01"] == 0.5
    assert mr._params["KBULK"] == 500.0


def test_not_anisotropic():
    assert not MooneyRivlin().is_anisotropic
    assert MooneyRivlin().fiber_direction is None


# ── SEF symbolic ──────────────────────────────────────────────────


def test_sef_expression():
    mr = MooneyRivlin()
    h = mr.handler
    C10, C01, KBULK = sym.Symbol("C10"), sym.Symbol("C01"), sym.Symbol("KBULK")
    I1 = h.isochoric_invariant1
    I2 = h.isochoric_invariant2
    I3 = h.invariant3
    expected = (I1 - 3) * C10 + (I2 - 3) * C01 + 0.25 * KBULK * (I3 - 1 - 2 * sym.log(I3**0.5))
    assert sym.simplify(mr.sef - expected) == 0


def test_pk2_expr_shape():
    mr = MooneyRivlin()
    assert mr.pk2_expr.shape == (3, 3)
    assert isinstance(mr.pk2_expr, sym.Matrix)


def test_cmat_expr_shape():
    mr = MooneyRivlin()
    assert mr.cmat_expr.shape == (3, 3, 3, 3)


# ── Energy evaluation ─────────────────────────────────────────────


def test_energy_at_identity(c_batch):
    mr = MooneyRivlin({"C10": 1.0, "C01": 0.5, "KBULK": 1.0})
    energy = mr.evaluate_energy(c_batch)
    assert energy.shape == (4,)
    assert energy[0] == pytest.approx(0.0, abs=1e-12)


def test_energy_positive_under_deformation(c_batch):
    mr = MooneyRivlin({"C10": 1.0, "C01": 0.5, "KBULK": 1.0})
    energy = mr.evaluate_energy(c_batch)
    assert np.all(energy[1:] > 0)


def test_energy_reduces_to_neohooke_when_c01_zero(c_batch):
    """With C01=0, MooneyRivlin should match NeoHooke."""
    from hyper_surrogate.mechanics.materials import NeoHooke

    mr = MooneyRivlin({"C10": 0.5, "C01": 0.0, "KBULK": 1000.0})
    nh = NeoHooke({"C10": 0.5, "KBULK": 1000.0})
    np.testing.assert_allclose(mr.evaluate_energy(c_batch), nh.evaluate_energy(c_batch), rtol=1e-10)


def test_energy_i2_contribution():
    """C01 term should increase energy beyond NeoHooke."""
    from hyper_surrogate.mechanics.materials import NeoHooke

    F = np.diag([1.1, 1.0 / np.sqrt(1.1), 1.0 / np.sqrt(1.1)]).reshape(1, 3, 3)
    C = K.right_cauchy_green(F)
    mr = MooneyRivlin({"C10": 0.5, "C01": 0.2, "KBULK": 1000.0})
    nh = NeoHooke({"C10": 0.5, "KBULK": 1000.0})
    # MooneyRivlin has extra I2 term, so energy should differ
    assert mr.evaluate_energy(C)[0] > nh.evaluate_energy(C)[0]


# ── PK2 stress ────────────────────────────────────────────────────


def test_pk2_at_identity():
    """MooneyRivlin PK2 at identity should be zero (stress-free reference)."""
    mr = MooneyRivlin()
    C = np.eye(3).reshape(1, 3, 3)
    pk2 = mr.evaluate_pk2(C)
    assert pk2.shape == (1, 3, 3)
    np.testing.assert_allclose(pk2[0], np.zeros((3, 3)), atol=1e-10)


def test_pk2_symmetry(c_batch):
    """PK2 should be symmetric."""
    mr = MooneyRivlin()
    pk2 = mr.evaluate_pk2(c_batch)
    for i in range(len(c_batch)):
        np.testing.assert_allclose(pk2[i], pk2[i].T, atol=1e-10)


def test_pk2_shape(c_batch):
    mr = MooneyRivlin()
    pk2 = mr.evaluate_pk2(c_batch)
    assert pk2.shape == (4, 3, 3)


@pytest.mark.parametrize(
    "params",
    [
        {"C10": 1.0, "C01": 0.0, "KBULK": 0.0},  # isochoric I1 only
        {"C10": 0.0, "C01": 1.0, "KBULK": 0.0},  # isochoric I2 only
        {"C10": 0.0, "C01": 0.0, "KBULK": 1000.0},  # volumetric only
        {"C10": 0.3, "C01": 0.2, "KBULK": 1000.0},  # full
    ],
)
def test_pk2_superposition(params, c_batch):
    """PK2 with combined params == sum of individual PK2 contributions."""
    mr = MooneyRivlin(params)
    pk2_combined = mr.evaluate_pk2(c_batch)

    pk2_sum = np.zeros_like(pk2_combined)
    for key, val in params.items():
        if val != 0.0:
            single_params = {k: 0.0 for k in params}
            single_params[key] = val
            pk2_sum += MooneyRivlin(single_params).evaluate_pk2(c_batch)

    np.testing.assert_allclose(pk2_combined, pk2_sum, atol=1e-10)


def test_pk2_reduces_to_neohooke_when_c01_zero(c_batch):
    from hyper_surrogate.mechanics.materials import NeoHooke

    mr = MooneyRivlin({"C10": 0.5, "C01": 0.0, "KBULK": 1000.0})
    nh = NeoHooke({"C10": 0.5, "KBULK": 1000.0})
    np.testing.assert_allclose(mr.evaluate_pk2(c_batch), nh.evaluate_pk2(c_batch), atol=1e-10)


# ── CMAT tangent ──────────────────────────────────────────────────


def test_cmat_shape(c_batch):
    mr = MooneyRivlin()
    cmat = mr.evaluate_cmat(c_batch)
    assert cmat.shape == (4, 3, 3, 3, 3)


def test_cmat_at_identity():
    mr = MooneyRivlin({"C10": 1.0, "C01": 0.0, "KBULK": 0.0})
    C = np.eye(3).reshape(1, 3, 3)
    cmat = mr.evaluate_cmat(C)
    assert cmat.shape == (1, 3, 3, 3, 3)
    assert np.all(np.isfinite(cmat))


def test_cmat_major_symmetry(c_batch):
    """CMAT should have major symmetry: C_ijkl = C_klij."""
    mr = MooneyRivlin()
    cmat = mr.evaluate_cmat(c_batch)
    for n in range(len(c_batch)):
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):  # noqa: E741
                        assert cmat[n, i, j, k, l] == pytest.approx(cmat[n, k, l, i, j], abs=1e-10)


def test_cmat_superposition(c_batch):
    """CMAT with combined params == sum of individual CMAT contributions."""
    params = {"C10": 0.3, "C01": 0.2, "KBULK": 1000.0}
    mr = MooneyRivlin(params)
    cmat_combined = mr.evaluate_cmat(c_batch)

    cmat_sum = np.zeros_like(cmat_combined)
    for key, val in params.items():
        if val != 0.0:
            single_params = {k: 0.0 for k in params}
            single_params[key] = val
            cmat_sum += MooneyRivlin(single_params).evaluate_cmat(c_batch)

    np.testing.assert_allclose(cmat_combined, cmat_sum, atol=1e-10)


# ── Energy gradients ──────────────────────────────────────────────


def test_grad_shape():
    mr = MooneyRivlin()
    C = np.eye(3).reshape(1, 3, 3) * 1.01
    C = (C + np.transpose(C, (0, 2, 1))) / 2
    grad = mr.evaluate_energy_grad_invariants(C)
    assert grad.shape == (1, 3)


def test_grad_at_identity():
    """At identity: dW/dI1_bar = C10, dW/dI2_bar = C01, dW/dJ = 0."""
    mr = MooneyRivlin({"C10": 0.3, "C01": 0.2, "KBULK": 1000.0})
    C = np.eye(3).reshape(1, 3, 3)
    grad = mr.evaluate_energy_grad_invariants(C)
    np.testing.assert_allclose(grad[0, 0], 0.3, rtol=1e-10)  # dW/dI1_bar = C10
    np.testing.assert_allclose(grad[0, 1], 0.2, rtol=1e-10)  # dW/dI2_bar = C01
    np.testing.assert_allclose(grad[0, 2], 0.0, atol=1e-14)  # dW/dJ = 0 at J=1


def test_grad_finite_difference():
    """Verify gradients against finite differences of sef_from_invariants."""
    from sympy import Symbol, lambdify

    mr = MooneyRivlin()
    F = np.diag([1.15, 0.95, 1.0 / (1.15 * 0.95)]).reshape(1, 3, 3)
    C = K.right_cauchy_green(F)

    i1 = float(K.isochoric_invariant1(C)[0])
    i2 = float(K.isochoric_invariant2(C)[0])
    j = float(np.sqrt(K.det_invariant(C)[0]))

    # Symbolic energy function
    I1s, I2s, Js = Symbol("I1b"), Symbol("I2b"), Symbol("J")
    W_expr = mr.sef_from_invariants(I1s, I2s, Js)
    param_syms = list(mr._symbols.values())
    fn = lambdify((I1s, I2s, Js, *param_syms), W_expr, modules="numpy")
    param_vals = list(mr._params.values())

    grad_analytical = mr.evaluate_energy_grad_invariants(C)[0]

    eps = 1e-7
    inv_vals = [i1, i2, j]
    grad_fd = np.zeros(3)
    for k in range(3):
        inv_p = list(inv_vals)
        inv_m = list(inv_vals)
        inv_p[k] += eps
        inv_m[k] -= eps
        W_p = float(fn(*inv_p, *param_vals))
        W_m = float(fn(*inv_m, *param_vals))
        grad_fd[k] = (W_p - W_m) / (2 * eps)

    np.testing.assert_allclose(grad_analytical, grad_fd, rtol=1e-5, atol=1e-10)


# ── PK2 finite-difference validation ─────────────────────────────


def test_pk2_consistent_with_energy():
    """PK2 = 2 * dW/dC: verify numerically via finite differences on energy."""
    mr = MooneyRivlin({"C10": 0.5, "C01": 0.3, "KBULK": 100.0})
    F = np.diag([1.1, 0.95, 1.0 / (1.1 * 0.95)])
    C = F.T @ F

    eps = 1e-7
    pk2_fd = np.zeros((3, 3))
    for i in range(3):
        for j in range(i, 3):
            C_p = C.copy()
            C_m = C.copy()
            # Symmetric perturbation: perturb E_ij = (C_ij + C_ji)/2
            C_p[i, j] += eps
            C_p[j, i] += eps
            C_m[i, j] -= eps
            C_m[j, i] -= eps
            W_p = mr.evaluate_energy(C_p.reshape(1, 3, 3))[0]
            W_m = mr.evaluate_energy(C_m.reshape(1, 3, 3))[0]
            # S_ij = dW/dE_ij = (W_p - W_m) / (2*eps) for the symmetric part
            val = (W_p - W_m) / (2 * eps)
            pk2_fd[i, j] = val
            pk2_fd[j, i] = val

    pk2_analytical = mr.evaluate_pk2(C.reshape(1, 3, 3))[0]
    np.testing.assert_allclose(pk2_analytical, pk2_fd, rtol=1e-4)


# ── Batch consistency ─────────────────────────────────────────────


def test_batch_consistency():
    """Batch of identical deformations gives identical results."""
    mr = MooneyRivlin()
    F_single = np.diag([1.1, 0.95, 1.0 / (1.1 * 0.95)])
    F_batch = np.tile(F_single, (5, 1, 1))
    C = K.right_cauchy_green(F_batch)
    pk2 = mr.evaluate_pk2(C)
    for i in range(1, 5):
        np.testing.assert_allclose(pk2[i], pk2[0], rtol=1e-12)
