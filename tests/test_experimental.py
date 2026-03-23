"""Tests for experimental data loading and preprocessing."""

import numpy as np
import pytest

from hyper_surrogate.data.experimental import ExperimentalData


def test_from_uniaxial():
    stretch = np.array([1.0, 1.1, 1.2, 1.3])
    stress = np.array([0.0, 0.1, 0.3, 0.6])
    data = ExperimentalData.from_uniaxial(stretch, stress)
    assert data.test_type == "uniaxial"
    assert data.stretch.shape == (4, 1)
    assert data.stress.shape == (4, 1)


def test_from_biaxial():
    n = 10
    data = ExperimentalData.from_biaxial(
        stretch_11=np.linspace(1.0, 1.3, n),
        stretch_22=np.linspace(1.0, 1.2, n),
        stress_11=np.linspace(0, 1.0, n),
        stress_22=np.linspace(0, 0.5, n),
    )
    assert data.test_type == "biaxial"
    assert data.stretch.shape == (n, 2)


def test_to_deformation_gradients_uniaxial():
    stretch = np.array([1.0, 1.1, 1.2])
    stress = np.array([0.0, 0.1, 0.3])
    data = ExperimentalData.from_uniaxial(stretch, stress)
    F = data.to_deformation_gradients()
    assert F.shape == (3, 3, 3)
    # Identity at stretch=1
    np.testing.assert_allclose(F[0], np.eye(3), atol=1e-14)
    # F[1] should have stretch 1.1 along axis 0
    np.testing.assert_allclose(F[1, 0, 0], 1.1)
    np.testing.assert_allclose(F[1, 1, 1], 1.0 / np.sqrt(1.1), rtol=1e-10)


def test_to_deformation_gradients_biaxial():
    data = ExperimentalData.from_biaxial(
        stretch_11=np.array([1.0, 1.1]),
        stretch_22=np.array([1.0, 1.2]),
        stress_11=np.array([0.0, 0.5]),
        stress_22=np.array([0.0, 0.3]),
    )
    F = data.to_deformation_gradients()
    # Check incompressibility: det(F) ~ 1
    np.testing.assert_allclose(F[0], np.eye(3), atol=1e-14)
    # At point 1: F33 = 1/(1.1*1.2)
    np.testing.assert_allclose(F[1, 2, 2], 1.0 / (1.1 * 1.2), rtol=1e-10)


def test_to_invariants():
    stretch = np.array([1.0, 1.1, 1.2])
    stress = np.array([0.0, 0.1, 0.3])
    data = ExperimentalData.from_uniaxial(stretch, stress)
    inv = data.to_invariants()
    assert inv.shape == (3, 3)
    # At identity: I1_bar=3, I2_bar=3, J=1
    np.testing.assert_allclose(inv[0], [3.0, 3.0, 1.0], atol=1e-10)


def test_load_reference_treloar():
    data = ExperimentalData.load_reference("treloar")
    assert data.test_type == "uniaxial"
    assert len(data.stretch) > 10
    assert data.stretch[0, 0] == 1.0


def test_load_reference_unknown():
    with pytest.raises(ValueError, match="Unknown reference"):
        ExperimentalData.load_reference("nonexistent_dataset")
