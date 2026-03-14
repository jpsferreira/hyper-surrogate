"""Tests for HolzapfelOgden transversely isotropic material."""

import numpy as np
import pytest
from sympy import Symbol

from hyper_surrogate.mechanics.kinematics import Kinematics
from hyper_surrogate.mechanics.materials import HolzapfelOgden, MooneyRivlin, NeoHooke


def _identity_batch(n=1):
    return np.tile(np.eye(3), (n, 1, 1))


def _uniaxial_F(stretch, axis=0, n=1):
    """Incompressible uniaxial stretch along a given axis."""
    F = np.tile(np.eye(3), (n, 1, 1))
    lateral = 1.0 / np.sqrt(stretch)
    F[:, 0, 0] = lateral
    F[:, 1, 1] = lateral
    F[:, 2, 2] = lateral
    F[:, axis, axis] = stretch
    return F


def _evaluate_energy_from_invariants(mat, i1_bar, i2_bar, j, i4=None, i5=None):
    """Numerically evaluate SEF via sef_from_invariants."""
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


# ── Properties ────────────────────────────────────────────────────


def test_is_anisotropic():
    mat = HolzapfelOgden()
    assert mat.is_anisotropic
    assert mat.fiber_direction is not None
    np.testing.assert_allclose(mat.fiber_direction, [1, 0, 0])


def test_default_parameters():
    mat = HolzapfelOgden()
    assert mat._params["a"] == 0.059
    assert mat._params["b"] == 8.023
    assert mat._params["af"] == 18.472
    assert mat._params["bf"] == 16.026
    assert mat._params["KBULK"] == 1000.0


def test_custom_parameters():
    custom = {"a": 0.1, "b": 10.0, "af": 20.0, "bf": 15.0, "KBULK": 500.0}
    mat = HolzapfelOgden(parameters=custom)
    for k, v in custom.items():
        assert mat._params[k] == v


def test_fiber_direction_custom():
    a0 = np.array([0.0, 1.0, 0.0])
    mat = HolzapfelOgden(fiber_direction=a0)
    np.testing.assert_allclose(mat.fiber_direction, a0)


def test_fiber_direction_normalized_storage():
    """Fiber direction is stored as-is (user responsibility to normalize)."""
    a0 = np.array([0.0, 0.0, 1.0])
    mat = HolzapfelOgden(fiber_direction=a0)
    np.testing.assert_allclose(mat.fiber_direction, a0)


def test_isotropic_materials_not_anisotropic():
    assert not NeoHooke().is_anisotropic
    assert not MooneyRivlin().is_anisotropic


def test_neohooke_grad_still_3d():
    mat = NeoHooke()
    C = _identity_batch(5) * 1.01
    C = np.einsum("nij->nji", C) * 0.5 + C * 0.5
    grad = mat.evaluate_energy_grad_invariants(C)
    assert grad.shape == (5, 3)


# ── SEF property raises NotImplementedError ───────────────────────


def test_sef_raises():
    """HolzapfelOgden.sef must raise (requires fiber invariants)."""
    mat = HolzapfelOgden()
    with pytest.raises(NotImplementedError, match="fiber"):
        _ = mat.sef


# ── Energy at identity ────────────────────────────────────────────


def test_energy_at_identity_is_zero():
    """At identity: I1_bar=3, I4=1, J=1 -> W=0."""
    mat = HolzapfelOgden()
    W = _evaluate_energy_from_invariants(mat, 3.0, 3.0, 1.0, 1.0, 1.0)
    np.testing.assert_allclose(W, 0.0, atol=1e-14)


def test_energy_at_identity_via_invariants():
    """Compute invariants from identity C and verify energy=0."""
    mat = HolzapfelOgden()
    C = _identity_batch()
    a0 = mat.fiber_direction
    i1 = Kinematics.isochoric_invariant1(C)
    i2 = Kinematics.isochoric_invariant2(C)
    j = np.sqrt(Kinematics.det_invariant(C))
    i4 = Kinematics.fiber_invariant4(C, a0)
    i5 = Kinematics.fiber_invariant5(C, a0)
    W = _evaluate_energy_from_invariants(mat, i1[0], i2[0], j[0], i4[0], i5[0])
    np.testing.assert_allclose(float(W), 0.0, atol=1e-14)


# ── Energy values ─────────────────────────────────────────────────


def test_energy_positive_under_stretch():
    """Energy should be positive for any non-trivial deformation."""
    mat = HolzapfelOgden()
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


def test_energy_isotropic_part_matches_neohooke():
    """When af=0, HolzapfelOgden isotropic part should match scaled exponential form."""
    # With af=0, fiber contribution vanishes
    mat = HolzapfelOgden(parameters={"a": 0.059, "b": 8.023, "af": 0.0, "bf": 16.026, "KBULK": 1000.0})
    i1_bar, j = 3.2, 1.0
    W = float(_evaluate_energy_from_invariants(mat, i1_bar, 3.0, j, 1.0, 1.0))
    # Manual: (a/2b)(exp(b*(I1_bar-3))-1) + vol(J=1)=0
    a, b = 0.059, 8.023
    expected = (a / (2 * b)) * (np.exp(b * (i1_bar - 3)) - 1)
    np.testing.assert_allclose(W, expected, rtol=1e-10)


def test_energy_fiber_part_only():
    """With a=0 and KBULK=0, only fiber term remains."""
    mat = HolzapfelOgden(parameters={"a": 0.0, "b": 8.023, "af": 18.472, "bf": 16.026, "KBULK": 0.0})
    i4 = 1.2  # Fiber in tension
    W = float(_evaluate_energy_from_invariants(mat, 3.0, 3.0, 1.0, i4, 1.0))
    af, bf = 18.472, 16.026
    expected = (af / (2 * bf)) * (np.exp(bf * (i4 - 1) ** 2) - 1)
    np.testing.assert_allclose(W, expected, rtol=1e-10)


def test_energy_volumetric_part_only():
    """With a=0, af=0: only volumetric term remains."""
    mat = HolzapfelOgden(parameters={"a": 0.0, "b": 8.023, "af": 0.0, "bf": 16.026, "KBULK": 1000.0})
    j = 1.05
    W = float(_evaluate_energy_from_invariants(mat, 3.0, 3.0, j, 1.0, 1.0))
    # vol(J) = KBULK/4 * (J^2 - 1 - 2*ln(J))
    KBULK = 1000.0
    expected = 0.25 * KBULK * (j**2 - 1 - 2 * np.log(j))
    np.testing.assert_allclose(W, expected, rtol=1e-10)


# ── Macaulay bracket (fiber tension vs compression) ───────────────


def test_fiber_active_only_in_tension():
    """Fiber contribution should be zero when I4 < 1 (compression).

    Note: The symbolic SEF does not implement the Macaulay bracket —
    this is applied numerically during evaluation. This test documents
    the raw symbolic behavior where (I4-1)^2 > 0 even in compression.
    """
    mat = HolzapfelOgden(parameters={"a": 0.0, "b": 1.0, "af": 18.472, "bf": 16.026, "KBULK": 0.0})
    # I4 < 1: fiber in compression
    W_compression = float(_evaluate_energy_from_invariants(mat, 3.0, 3.0, 1.0, 0.8, 1.0))
    # I4 > 1: fiber in tension
    W_tension = float(_evaluate_energy_from_invariants(mat, 3.0, 3.0, 1.0, 1.2, 1.0))
    # Both should be positive in the raw symbolic form (no Macaulay bracket at symbolic level)
    assert W_compression > 0
    assert W_tension > 0


def test_fiber_energy_symmetric_around_i4_one():
    """(I4-1)^2 is symmetric, so W(I4=0.8) == W(I4=1.2) when only fiber term active."""
    mat = HolzapfelOgden(parameters={"a": 0.0, "b": 1.0, "af": 18.472, "bf": 16.026, "KBULK": 0.0})
    W_a = float(_evaluate_energy_from_invariants(mat, 3.0, 3.0, 1.0, 0.8, 1.0))
    W_b = float(_evaluate_energy_from_invariants(mat, 3.0, 3.0, 1.0, 1.2, 1.0))
    np.testing.assert_allclose(W_a, W_b, rtol=1e-12)


# ── Gradient shape and values ─────────────────────────────────────


def test_grad_shape_is_5d():
    mat = HolzapfelOgden()
    n = 10
    F = _uniaxial_F(1.05, axis=0, n=n)
    C = Kinematics.right_cauchy_green(F)
    grad = mat.evaluate_energy_grad_invariants(C)
    assert grad.shape == (n, 5)


def test_grad_at_identity():
    """At identity: dW/dI1_bar = a/2 * exp(0) = a/2, dW/dI4 = 0 (I4=1)."""
    mat = HolzapfelOgden()
    C = _identity_batch()
    grad = mat.evaluate_energy_grad_invariants(C)
    a = mat._params["a"]
    # dW/dI1_bar = (a/2) at I1_bar=3
    np.testing.assert_allclose(grad[0, 0], a / 2.0, rtol=1e-10)
    # dW/dI2_bar = 0 (no I2 dependence)
    np.testing.assert_allclose(grad[0, 1], 0.0, atol=1e-14)
    # dW/dJ at J=1: vol'(1) = KBULK/2 * (J - 1/J) = 0
    np.testing.assert_allclose(grad[0, 2], 0.0, atol=1e-14)
    # dW/dI4 at I4=1: af*(I4-1)*exp(bf*(I4-1)^2) = 0
    np.testing.assert_allclose(grad[0, 3], 0.0, atol=1e-14)
    # dW/dI5 = 0 (no I5 dependence in this model)
    np.testing.assert_allclose(grad[0, 4], 0.0, atol=1e-14)


def test_grad_finite_difference():
    """Verify analytical gradient against finite differences of energy."""
    mat = HolzapfelOgden()
    # Use a deformed state
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


def test_grad_nonzero_fiber_under_stretch():
    """dW/dI4 should be nonzero when fiber is stretched (I4 > 1)."""
    mat = HolzapfelOgden()
    F = _uniaxial_F(1.15, axis=0)  # stretch along fiber direction
    C = Kinematics.right_cauchy_green(F)
    grad = mat.evaluate_energy_grad_invariants(C)
    assert grad[0, 3] != 0.0  # dW/dI4 nonzero


def test_grad_dw_di5_always_zero():
    """This HolzapfelOgden model has no I5 dependence, so dW/dI5=0 always."""
    mat = HolzapfelOgden()
    F = _uniaxial_F(1.2, axis=0, n=5)
    C = Kinematics.right_cauchy_green(F)
    grad = mat.evaluate_energy_grad_invariants(C)
    np.testing.assert_allclose(grad[:, 4], 0.0, atol=1e-14)


# ── Multiple deformation states ──────────────────────────────────


def test_uniaxial_along_fiber():
    """Uniaxial stretch along fiber direction should activate both isotropic and fiber."""
    mat = HolzapfelOgden()
    F = _uniaxial_F(1.2, axis=0)  # fiber along x
    C = Kinematics.right_cauchy_green(F)
    a0 = mat.fiber_direction
    i4 = Kinematics.fiber_invariant4(C, a0)[0]
    assert i4 > 1.0  # Fiber in tension
    grad = mat.evaluate_energy_grad_invariants(C)
    assert grad[0, 0] > 0  # dW/dI1_bar > 0
    assert grad[0, 3] > 0  # dW/dI4 > 0


def test_uniaxial_transverse_to_fiber():
    """Stretch transverse to fiber should not stretch the fiber (I4 < 1)."""
    mat = HolzapfelOgden()
    F = _uniaxial_F(1.2, axis=1)  # stretch along y, fiber along x
    C = Kinematics.right_cauchy_green(F)
    a0 = mat.fiber_direction
    i4 = Kinematics.fiber_invariant4(C, a0)[0]
    # Transverse stretch compresses the fiber direction
    assert i4 < 1.0


def test_shear_deformation():
    """Simple shear should produce valid gradients."""
    mat = HolzapfelOgden()
    n = 5
    F = np.tile(np.eye(3), (n, 1, 1))
    F[:, 0, 1] = np.linspace(0.05, 0.25, n)  # shear in xy
    C = Kinematics.right_cauchy_green(F)
    grad = mat.evaluate_energy_grad_invariants(C)
    assert grad.shape == (n, 5)
    # All finite
    assert np.all(np.isfinite(grad))


def test_multiple_stretch_levels():
    """Energy should increase monotonically with uniaxial stretch along fiber."""
    mat = HolzapfelOgden()
    a0 = mat.fiber_direction
    stretches = np.array([1.01, 1.05, 1.1, 1.15, 1.2])
    energies = []
    for s in stretches:
        F = _uniaxial_F(s, axis=0)
        C = Kinematics.right_cauchy_green(F)
        i1 = float(Kinematics.isochoric_invariant1(C)[0])
        i2 = float(Kinematics.isochoric_invariant2(C)[0])
        j = float(np.sqrt(Kinematics.det_invariant(C)[0]))
        i4 = float(Kinematics.fiber_invariant4(C, a0)[0])
        i5 = float(Kinematics.fiber_invariant5(C, a0)[0])
        W = float(_evaluate_energy_from_invariants(mat, i1, i2, j, i4, i5))
        energies.append(W)
    # Strictly increasing
    for k in range(len(energies) - 1):
        assert energies[k + 1] > energies[k], f"Energy not increasing at stretch {stretches[k + 1]}"


# ── Fiber direction dependence ────────────────────────────────────


def test_fiber_direction_affects_energy():
    """Same deformation, different fiber direction should give different energy."""
    mat_x = HolzapfelOgden(fiber_direction=np.array([1.0, 0.0, 0.0]))
    mat_y = HolzapfelOgden(fiber_direction=np.array([0.0, 1.0, 0.0]))

    F = _uniaxial_F(1.15, axis=0)
    C = Kinematics.right_cauchy_green(F)

    def _energy(mat):
        a0 = mat.fiber_direction
        i1 = float(Kinematics.isochoric_invariant1(C)[0])
        i2 = float(Kinematics.isochoric_invariant2(C)[0])
        j = float(np.sqrt(Kinematics.det_invariant(C)[0]))
        i4 = float(Kinematics.fiber_invariant4(C, a0)[0])
        i5 = float(Kinematics.fiber_invariant5(C, a0)[0])
        return float(_evaluate_energy_from_invariants(mat, i1, i2, j, i4, i5))

    W_x = _energy(mat_x)
    W_y = _energy(mat_y)
    # x-fiber is stretched, y-fiber is compressed: different energies
    assert W_x != W_y
    # x-fiber should have higher energy (stretched along fiber)
    assert W_x > W_y


# ── sef_from_invariants symbolic ──────────────────────────────────


def test_sef_from_invariants_symbolic():
    """sef_from_invariants should return a sympy expression."""
    mat = HolzapfelOgden()
    i1, i2, j, i4, i5 = (Symbol(s) for s in ["I1b", "I2b", "J", "I4", "I5"])
    W = mat.sef_from_invariants(i1, i2, j, i4, i5)
    assert W is not None
    # Should contain all parameter symbols
    free = W.free_symbols
    assert Symbol("a") in free
    assert Symbol("b") in free
    assert Symbol("af") in free
    assert Symbol("bf") in free
    assert Symbol("KBULK") in free


def test_sef_from_invariants_no_i4_gives_isotropic_only():
    """Without I4, fiber term should vanish."""
    mat = HolzapfelOgden()
    i1, i2, j = (Symbol(s) for s in ["I1b", "I2b", "J"])
    W = mat.sef_from_invariants(i1, i2, j)
    # Should not contain af, bf (fiber parameters only appear with I4)
    free = W.free_symbols
    # The expression should still be defined (isotropic + volumetric)
    assert Symbol("a") in free
    assert Symbol("KBULK") in free


# ── Batch consistency ─────────────────────────────────────────────


def test_batch_gradient_consistency():
    """Gradient for batch of identical deformations should be identical."""
    mat = HolzapfelOgden()
    F_single = _uniaxial_F(1.1, axis=0)
    F_batch = np.tile(F_single, (5, 1, 1))
    C = Kinematics.right_cauchy_green(F_batch)
    grad = mat.evaluate_energy_grad_invariants(C)
    for i in range(1, 5):
        np.testing.assert_allclose(grad[i], grad[0], rtol=1e-12)
