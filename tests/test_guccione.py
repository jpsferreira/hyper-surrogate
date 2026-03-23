"""Tests for Guccione transversely isotropic cardiac model."""

import numpy as np
import pytest

from hyper_surrogate.mechanics.kinematics import Kinematics
from hyper_surrogate.mechanics.materials import Guccione


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


def test_is_anisotropic():
    mat = Guccione()
    assert mat.is_anisotropic


def test_energy_at_identity_is_zero():
    mat = Guccione()
    C = _identity_batch()
    W = mat.evaluate_energy(C)
    np.testing.assert_allclose(W, 0.0, atol=1e-14)


def test_energy_positive_under_stretch():
    mat = Guccione()
    F = _uniaxial_F(1.1, axis=0)
    C = Kinematics.right_cauchy_green(F)
    W = mat.evaluate_energy(C)
    assert W[0] > 0


def test_fiber_direction_matters():
    """Different fiber directions should give different energies for asymmetric deformation."""
    mat_x = Guccione(fiber_direction=np.array([1.0, 0.0, 0.0]))
    mat_y = Guccione(fiber_direction=np.array([0.0, 1.0, 0.0]), sheet_direction=np.array([0.0, 0.0, 1.0]))

    F = _uniaxial_F(1.15, axis=0)
    C = Kinematics.right_cauchy_green(F)

    W_x = mat_x.evaluate_energy(C)[0]
    W_y = mat_y.evaluate_energy(C)[0]
    assert W_x != W_y


def test_monotonic_energy():
    mat = Guccione()
    stretches = [1.01, 1.05, 1.1, 1.15, 1.2]
    energies = []
    for s in stretches:
        F = _uniaxial_F(s, axis=0)
        C = Kinematics.right_cauchy_green(F)
        energies.append(mat.evaluate_energy(C)[0])
    for i in range(len(energies) - 1):
        assert energies[i + 1] > energies[i]


def test_sef_raises():
    mat = Guccione()
    with pytest.raises(NotImplementedError):
        _ = mat.sef


def test_grad_shape():
    mat = Guccione()
    F = _uniaxial_F(1.1, axis=0, n=3)
    C = Kinematics.right_cauchy_green(F)
    grad = mat.evaluate_energy_grad_voigt(C)
    assert grad.shape == (3, 6)
