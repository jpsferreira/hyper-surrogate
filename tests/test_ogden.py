"""Tests for Ogden hyperelastic model (principal stretch formulation)."""

import numpy as np
import pytest

from hyper_surrogate.mechanics.kinematics import Kinematics
from hyper_surrogate.mechanics.materials import Ogden


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
    assert not Ogden().is_anisotropic


def test_energy_at_identity_is_zero():
    mat = Ogden()
    C = _identity_batch()
    W = mat.evaluate_energy(C)
    np.testing.assert_allclose(W, 0.0, atol=1e-14)


def test_energy_positive_under_stretch():
    mat = Ogden()
    F = _uniaxial_F(1.2, axis=0)
    C = Kinematics.right_cauchy_green(F)
    W = mat.evaluate_energy(C)
    assert W[0] > 0


def test_alpha2_neohooke_equivalence():
    """Ogden with alpha=2, mu=2*C10 should match NeoHooke isochoric part."""
    C10 = 0.5
    mat = Ogden(parameters={"mu1": 2 * C10, "alpha1": 2.0, "KBULK": 0.0})
    F = _uniaxial_F(1.15, axis=0)
    C = Kinematics.right_cauchy_green(F)
    W_ogden = mat.evaluate_energy(C)[0]

    # NeoHooke isochoric: C10 * (I1_bar - 3)
    i1_bar = float(Kinematics.isochoric_invariant1(C)[0])
    W_nh = C10 * (i1_bar - 3)
    np.testing.assert_allclose(W_ogden, W_nh, rtol=1e-10)


def test_multi_term():
    """Test 2-term Ogden model."""
    mat = Ogden(parameters={
        "mu1": 1.0, "alpha1": 2.0,
        "mu2": 0.5, "alpha2": -2.0,
        "KBULK": 1000.0,
    })
    assert mat._n_terms == 2
    F = _uniaxial_F(1.1, axis=0)
    C = Kinematics.right_cauchy_green(F)
    W = mat.evaluate_energy(C)
    assert W[0] > 0


def test_monotonic_energy():
    mat = Ogden()
    stretches = [1.01, 1.05, 1.1, 1.15, 1.2]
    energies = []
    for s in stretches:
        F = _uniaxial_F(s, axis=0)
        C = Kinematics.right_cauchy_green(F)
        energies.append(mat.evaluate_energy(C)[0])
    for i in range(len(energies) - 1):
        assert energies[i + 1] > energies[i]


def test_sef_raises():
    mat = Ogden()
    with pytest.raises(NotImplementedError):
        _ = mat.sef


def test_sef_from_invariants_raises():
    from sympy import Symbol

    mat = Ogden()
    with pytest.raises(NotImplementedError):
        mat.sef_from_invariants(Symbol("I1"), Symbol("I2"), Symbol("J"))


def test_grad_shape():
    mat = Ogden()
    F = _uniaxial_F(1.1, axis=0, n=3)
    C = Kinematics.right_cauchy_green(F)
    grad = mat.evaluate_energy_grad_invariants(C)
    assert grad.shape == (3, 6)
