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


def test_volumetric_dilation_J_in_range():
    gen = DeformationGenerator(seed=0)
    F = gen.volumetric_dilation(1000, j_range=(0.85, 1.15))
    J = np.linalg.det(F)
    assert J.min() >= 0.85 - 1e-12
    assert J.max() <= 1.15 + 1e-12
    # Pure isotropic dilation: F should be a scalar multiple of I
    np.testing.assert_allclose(F[:, 0, 1], 0.0, atol=1e-12)
    np.testing.assert_allclose(F[:, 0, 0], F[:, 1, 1], atol=1e-12)
    np.testing.assert_allclose(F[:, 1, 1], F[:, 2, 2], atol=1e-12)


def test_combined_compressible_J_in_range():
    gen = DeformationGenerator(seed=0)
    F = gen.combined_compressible(2000, j_range=(0.85, 1.15))
    J = np.linalg.det(F)
    # det(A B) = det(A) det(B); det(combined) = 1, det(Fv) ∈ [0.85, 1.15]
    assert J.min() >= 0.85 - 1e-9
    assert J.max() <= 1.15 + 1e-9
    # Sanity: shape, and that J is not concentrated at 1
    assert F.shape == (2000, 3, 3)
    assert J.std() > 0.05  # really spans the range


def test_combined_compressible_reduces_to_combined_at_zero_dilation():
    # j_range = (1.0, 1.0) -> Fv = I -> combined_compressible has det F = 1
    gen = DeformationGenerator(seed=7)
    F_cc = gen.combined_compressible(100, j_range=(1.0, 1.0))
    np.testing.assert_allclose(np.linalg.det(F_cc), 1.0, atol=1e-12)
