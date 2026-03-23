"""Tests for Fung-type exponential hyperelastic model."""

import numpy as np
import pytest

from hyper_surrogate.mechanics.kinematics import Kinematics
from hyper_surrogate.mechanics.materials import Fung


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


def test_is_isotropic():
    assert not Fung().is_anisotropic


def test_energy_at_identity_is_zero():
    mat = Fung()
    C = _identity_batch()
    W = mat.evaluate_energy(C)
    np.testing.assert_allclose(W, 0.0, atol=1e-14)


def test_energy_positive_under_stretch():
    mat = Fung()
    F = _uniaxial_F(1.2, axis=0)
    C = Kinematics.right_cauchy_green(F)
    W = mat.evaluate_energy(C)
    assert W[0] > 0


def test_manual_energy():
    """Verify against manual computation for uniaxial stretch."""
    mat = Fung(parameters={"c": 1.0, "b1": 10.0, "b2": 0.0, "KBULK": 0.0})
    stretch = 1.1
    F = _uniaxial_F(stretch, axis=0)
    C = Kinematics.right_cauchy_green(F)
    E = 0.5 * (C - np.eye(3))

    Q = 10.0 * (E[0, 0, 0] ** 2 + E[0, 1, 1] ** 2 + E[0, 2, 2] ** 2)
    expected = 0.5 * (np.exp(Q) - 1)

    W = mat.evaluate_energy(C)
    np.testing.assert_allclose(W[0], expected, rtol=1e-10)


def test_monotonic_energy():
    mat = Fung()
    stretches = [1.01, 1.05, 1.1, 1.15, 1.2]
    energies = []
    for s in stretches:
        F = _uniaxial_F(s, axis=0)
        C = Kinematics.right_cauchy_green(F)
        energies.append(mat.evaluate_energy(C)[0])
    for i in range(len(energies) - 1):
        assert energies[i + 1] > energies[i]


def test_sef_raises():
    mat = Fung()
    with pytest.raises(NotImplementedError):
        _ = mat.sef


def test_grad_shape():
    mat = Fung()
    F = _uniaxial_F(1.1, axis=0, n=3)
    C = Kinematics.right_cauchy_green(F)
    grad = mat.evaluate_energy_grad_invariants(C)
    assert grad.shape == (3, 6)
