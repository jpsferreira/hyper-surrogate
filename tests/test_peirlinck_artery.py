"""Tests for the PeirlinckArtery two-fiber arterial SEF.

Mirrors the testing pattern from test_holzapfel_ogden.py; covers the
public API (instantiation, classmethods, the two NotImplementedError
surfaces, custom parameters, fiber-direction validation) and the
energy evaluation at the reference configuration and a simple
biaxial state.
"""

import numpy as np
import pytest
from sympy import Symbol

from hyper_surrogate.mechanics.materials import PeirlinckArtery


def _identity_batch(n=1):
    return np.tile(np.eye(3), (n, 1, 1))


def _equibiaxial_C(stretch: float) -> np.ndarray:
    """C for incompressible equibiaxial in the (1,2) plane: F = diag(λ, λ, 1/λ²)."""
    F = np.diag([stretch, stretch, stretch**-2])
    return (F.T @ F).reshape(1, 3, 3)


# ── Construction ─────────────────────────────────────────────────────


def test_default_parameters_are_media_layer():
    """DEFAULT_PARAMS matches the media-layer parameter set from Peirlinck 2024."""
    mat = PeirlinckArtery()
    assert mat._params["mu1"] == 33.45
    assert mat._params["a"] == 3.74
    assert mat._params["b"] == 6.66
    assert mat._params["mu5"] == 2.17
    assert mat._params["KBULK"] == 1000.0


def test_default_fiber_directions_are_symmetric_at_7deg():
    """No-arg construction yields two fibers at ±7° from circumferential."""
    mat = PeirlinckArtery()
    a0, a1 = mat._fiber_directions
    theta = np.radians(7.0)
    np.testing.assert_allclose(a0, [np.cos(theta), np.sin(theta), 0.0], atol=1e-12)
    np.testing.assert_allclose(a1, [np.cos(theta), -np.sin(theta), 0.0], atol=1e-12)


def test_is_anisotropic():
    assert PeirlinckArtery().is_anisotropic
    assert PeirlinckArtery().num_fiber_families == 2


# ── Classmethod factories ────────────────────────────────────────────


def test_media_factory_matches_default():
    """`PeirlinckArtery.media()` reproduces the default parameter set."""
    media = PeirlinckArtery.media()
    default = PeirlinckArtery()
    assert media._params == default._params
    np.testing.assert_allclose(media._fiber_directions, default._fiber_directions, atol=1e-12)


def test_adventitia_factory_overrides_parameters_and_angle():
    """`PeirlinckArtery.adventitia()` shifts mu1/a/b/mu5 and fiber angle to 66.78°."""
    adv = PeirlinckArtery.adventitia()
    assert adv._params["mu1"] == 8.30
    assert adv._params["a"] == 1.42
    assert adv._params["b"] == 6.34
    assert adv._params["mu5"] == 0.49
    a0, a1 = adv._fiber_directions
    theta = np.radians(66.78)
    np.testing.assert_allclose(a0, [np.cos(theta), np.sin(theta), 0.0], atol=1e-10)
    np.testing.assert_allclose(a1, [np.cos(theta), -np.sin(theta), 0.0], atol=1e-10)


# ── Construction-time validation ─────────────────────────────────────


def test_construction_rejects_wrong_number_of_fibers():
    """Exactly two fiber directions are required."""
    one_fiber = [np.array([1.0, 0.0, 0.0])]
    with pytest.raises(ValueError, match="exactly 2 fiber directions"):
        PeirlinckArtery(fiber_directions=one_fiber)

    three_fibers = [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
    ]
    with pytest.raises(ValueError, match="exactly 2 fiber directions"):
        PeirlinckArtery(fiber_directions=three_fibers)


def test_custom_parameters_override_defaults():
    mat = PeirlinckArtery(parameters={"mu1": 50.0, "b": 10.0})
    assert mat._params["mu1"] == 50.0
    assert mat._params["b"] == 10.0
    # Untouched parameters fall back to defaults
    assert mat._params["a"] == 3.74
    assert mat._params["mu5"] == 2.17
    assert mat._params["KBULK"] == 1000.0


def test_custom_fiber_angle_overrides_default():
    """A user-supplied `fiber_angle_deg` builds symmetric fibers at ±angle."""
    mat = PeirlinckArtery(fiber_angle_deg=45.0)
    a0, a1 = mat._fiber_directions
    theta = np.radians(45.0)
    np.testing.assert_allclose(a0, [np.cos(theta), np.sin(theta), 0.0], atol=1e-12)
    np.testing.assert_allclose(a1, [np.cos(theta), -np.sin(theta), 0.0], atol=1e-12)


# ── SEF surfaces (isotropic API explicitly disallowed) ───────────────


def test_isotropic_sef_property_is_not_implemented():
    """The 2-fiber SEF cannot be expressed without fiber invariants."""
    with pytest.raises(NotImplementedError, match="sef_from_all_invariants"):
        _ = PeirlinckArtery().sef


def test_isotropic_sef_from_invariants_is_not_implemented():
    """sef_from_invariants is the single-fiber convenience API; not valid here."""
    mat = PeirlinckArtery()
    with pytest.raises(NotImplementedError, match="sef_from_all_invariants"):
        mat.sef_from_invariants(Symbol("I1b"), Symbol("I2b"), Symbol("J"))


def test_sef_from_all_invariants_at_reference_equals_zero():
    """W(C=I) with both fiber I5 = 1 should be zero."""
    mat = PeirlinckArtery()
    i1, i2, j = Symbol("I1b"), Symbol("I2b"), Symbol("J")
    i4_0, i5_0 = Symbol("I4_0"), Symbol("I5_0")
    i4_1, i5_1 = Symbol("I4_1"), Symbol("I5_1")
    W_expr = mat.sef_from_all_invariants(i1, i2, j, [(i4_0, i5_0), (i4_1, i5_1)])

    # At C=I: I1=3, I2=3, J=1, I4=I5=1. W must collapse to zero.
    subs = {i1: 3.0, i2: 3.0, j: 1.0, i4_0: 1.0, i5_0: 1.0, i4_1: 1.0, i5_1: 1.0}
    for s, v in mat._symbols.items():
        subs[v] = float(mat._params[s])
    assert float(W_expr.subs(subs)) == pytest.approx(0.0, abs=1e-12)


# ── Numerical energy evaluation ──────────────────────────────────────


def test_evaluate_energy_at_identity_is_zero():
    """No deformation -> zero strain energy."""
    mat = PeirlinckArtery()
    W = mat.evaluate_energy(_identity_batch())
    np.testing.assert_allclose(W, 0.0, atol=1e-10)


def test_evaluate_energy_positive_under_biaxial_stretch():
    """A non-trivial equibiaxial stretch produces strictly positive energy."""
    mat = PeirlinckArtery()
    W = mat.evaluate_energy(_equibiaxial_C(1.1))
    assert W[0] > 0.0


def test_evaluate_energy_adventitia_is_softer_than_media_at_same_strain():
    """Adventitia parameters are softer than media; energy reflects that."""
    media = PeirlinckArtery.media()
    adventitia = PeirlinckArtery.adventitia()
    C = _equibiaxial_C(1.1)
    # Note that the materials have different fiber directions, so the
    # comparison fairness isn't perfect, but the deviatoric-isotropic
    # part (mu1) is the dominant driver and mu1_media >> mu1_adv.
    W_media = media.evaluate_energy(C)
    W_adventitia = adventitia.evaluate_energy(C)
    assert W_media[0] > W_adventitia[0]


def test_evaluate_energy_batch_consistency():
    """Batched evaluation matches independent calls."""
    mat = PeirlinckArtery()
    C_batch = np.concatenate([_equibiaxial_C(1.05), _equibiaxial_C(1.1)])
    W_batch = mat.evaluate_energy(C_batch)
    W_a = mat.evaluate_energy(_equibiaxial_C(1.05))
    W_b = mat.evaluate_energy(_equibiaxial_C(1.1))
    np.testing.assert_allclose([W_batch[0], W_batch[1]], [W_a[0], W_b[0]], atol=1e-12)
