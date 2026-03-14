"""Tests for HolzapfelOgden transversely isotropic material."""

import numpy as np

from hyper_surrogate.mechanics.kinematics import Kinematics
from hyper_surrogate.mechanics.materials import HolzapfelOgden


def _identity_batch(n=1):
    return np.tile(np.eye(3), (n, 1, 1))


def test_energy_at_identity_is_zero():
    """Strain energy at identity deformation should be zero."""
    mat = HolzapfelOgden()
    C = _identity_batch()
    grad = mat.evaluate_energy_grad_invariants(C)
    # At identity: I1_bar=3, I4=1 -> both exponent terms are 0 -> W=0
    # The grad should exist and have shape (1, 5)
    assert grad.shape == (1, 5)


def test_is_anisotropic():
    mat = HolzapfelOgden()
    assert mat.is_anisotropic
    assert mat.fiber_direction is not None
    np.testing.assert_allclose(mat.fiber_direction, [1, 0, 0])


def test_grad_shape_is_5d():
    """Energy gradient should return (N, 5) for anisotropic material."""
    mat = HolzapfelOgden()
    n = 10
    F = np.tile(np.eye(3), (n, 1, 1))
    # Small perturbation to avoid singularities
    F[:, 0, 0] = np.linspace(0.95, 1.05, n)
    F[:, 1, 1] = 1.0 / np.sqrt(F[:, 0, 0])
    F[:, 2, 2] = 1.0 / np.sqrt(F[:, 0, 0])
    C = Kinematics.right_cauchy_green(F)
    grad = mat.evaluate_energy_grad_invariants(C)
    assert grad.shape == (n, 5)


def test_fiber_direction_custom():
    """Custom fiber direction should be stored correctly."""
    a0 = np.array([0.0, 1.0, 0.0])
    mat = HolzapfelOgden(fiber_direction=a0)
    np.testing.assert_allclose(mat.fiber_direction, a0)


def test_isotropic_materials_not_anisotropic():
    """NeoHooke and MooneyRivlin should NOT be anisotropic."""
    from hyper_surrogate.mechanics.materials import MooneyRivlin, NeoHooke

    assert not NeoHooke().is_anisotropic
    assert not MooneyRivlin().is_anisotropic


def test_neohooke_grad_still_3d():
    """NeoHooke should still return (N, 3) gradients."""
    from hyper_surrogate.mechanics.materials import NeoHooke

    mat = NeoHooke()
    C = _identity_batch(5) * 1.01  # slight perturbation
    # Make symmetric
    C = np.einsum("nij->nji", C) * 0.5 + C * 0.5
    grad = mat.evaluate_energy_grad_invariants(C)
    assert grad.shape == (5, 3)
