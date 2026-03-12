import numpy as np

from hyper_surrogate.data.deformation import DeformationGenerator


def test_uniaxial_shape():
    gen = DeformationGenerator(seed=42)
    F = gen.uniaxial(100)
    assert F.shape == (100, 3, 3)


def test_uniaxial_incompressible():
    gen = DeformationGenerator(seed=42)
    F = gen.uniaxial(100)
    dets = np.linalg.det(F)
    np.testing.assert_allclose(dets, np.ones(100), atol=1e-10)


def test_biaxial_shape():
    gen = DeformationGenerator(seed=42)
    F = gen.biaxial(50)
    assert F.shape == (50, 3, 3)


def test_shear_shape():
    gen = DeformationGenerator(seed=42)
    F = gen.shear(50)
    assert F.shape == (50, 3, 3)


def test_combined_shape():
    gen = DeformationGenerator(seed=42)
    F = gen.combined(200)
    assert F.shape == (200, 3, 3)


def test_random_rotation_orthogonal():
    gen = DeformationGenerator(seed=42)
    R = gen.random_rotation(50)
    for i in range(50):
        np.testing.assert_allclose(R[i] @ R[i].T, np.eye(3), atol=1e-10)


def test_seed_reproducibility():
    gen1 = DeformationGenerator(seed=42)
    gen2 = DeformationGenerator(seed=42)
    F1 = gen1.combined(100)
    F2 = gen2.combined(100)
    np.testing.assert_array_equal(F1, F2)
