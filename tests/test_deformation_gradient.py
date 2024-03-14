# Generated by CodiumAI
import numpy as np
import pytest

from hyper_surrogate.deformation_gradient import DeformationGradient, DeformationGradientGenerator


class TestDeformationGradient:
    @pytest.fixture
    def deformation(self):
        return DeformationGradient()

    @pytest.fixture
    def deformation_generator(self):
        return DeformationGradientGenerator()

    # DeformationGradient.uniaxial returns a 3x3 numpy array with the correct values for a given stretch factor
    def test_uniaxial_returns_correct_values(self, deformation):
        stretch = 4.0
        expected_result = np.array([[[4.0, 0, 0], [0, 0.5, 0], [0, 0, 0.5]]])
        assert np.array_equal(deformation.uniaxial(stretch), expected_result)

    # DeformationGradient.shear returns a 3x3 numpy array with the correct values for a given shear factor
    def test_shear_returns_correct_values(self, deformation):
        shear = 0.5
        expected_result = np.array([[[1, 0.5, 0], [0, 1, 0], [0, 0, 1]]])
        assert np.array_equal(deformation.shear(shear), expected_result)

    # DeformationGradient.biaxial returns a 3x3 numpy array with the correct values for given stretch factors
    def test_biaxial_returns_correct_values(self, deformation):
        stretch1 = 2.0
        stretch2 = 2.0
        expected_result = np.array([[[2.0, 0, 0], [0, 2.0, 0], [0, 0, 0.25]]])
        assert np.array_equal(deformation.biaxial(stretch1, stretch2), expected_result)

    # DeformationGradient.angle_axis_rotation returns a 3x3 numpy array with the correct values for a given axis and angle
    def test_angle_axis_rotation_returns_correct_values(self, deformation):
        axis = 0
        angle = np.pi / 4
        expected_result = np.array([[[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]]])
        assert np.array_equal(deformation.rotation(axis, angle), expected_result)

    # seed=None generates different deformation gradient

    # can call to_radians method without errors
    def test_to_radians_method(self):
        rdg = DeformationGradient()
        degree = 45
        radians = rdg.to_radians(degree)
        assert isinstance(radians, float)

    # can call axis method without errors
    def test_can_call_axis_method_without_errors(self, deformation_generator):
        try:
            deformation_generator.axis()
        except Exception:
            pytest.fail("Calling axis method raised an exception")

    # can call angle method without errors
    def test_can_call_angle_method_without_errors(self, deformation_generator):
        try:
            deformation_generator.angle()
        except Exception:
            pytest.fail("Calling angle method raised an exception")

    # can call random_rotation method without errors
    def test_random_rotation_method(self, deformation_generator):
        try:
            deformation_generator.rotate()
        except Exception:
            pytest.fail("random_rotation method raised an exception unexpectedly")

    # can generate a random deformation gradient without errors
    def test_generate_random_deformation_gradient(self, deformation_generator):
        deformation_gradient = deformation_generator.rotate()
        assert isinstance(deformation_gradient, np.ndarray)

    # n_axis=1 generates deformation gradients with zeros
    def test_n_axis_1_generates_eye(self, deformation_generator):
        np.random.seed(0)
        result = deformation_generator.rotate(n_axis=1)
        expected = np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]])
        assert np.array_equal(result, expected)

    # n_axis=4 generates deformation gradients with zeros
    def test_n_axis_4_generates_eye(self, deformation_generator):
        np.random.seed(0)
        deformation_gradient = deformation_generator.rotate(n_axis=4)
        assert np.all(deformation_gradient == np.eye(3))

    # can generate multiple random deformation gradients without errors
    def test_generate_multiple_random_deformation_gradients(self, deformation_generator):
        for _ in range(10):
            deformation_gradient = deformation_generator.rotate()
            assert isinstance(deformation_gradient, np.ndarray)
        for _ in range(10):
            deformation_gradient = deformation_generator.uniaxial(2.0)
            assert isinstance(deformation_gradient, np.ndarray)
        for _ in range(10):
            deformation_gradient = deformation_generator.shear(0.5)
            assert isinstance(deformation_gradient, np.ndarray)
        for _ in range(10):
            deformation_gradient = deformation_generator.biaxial(2.0, 3.0)
            assert isinstance(deformation_gradient, np.ndarray)
        for _ in range(10):
            deformation_gradient = deformation_generator.rotation(0, np.pi / 4)
            assert isinstance(deformation_gradient, np.ndarray)

    # min_interval=180 generates deformation gradients with eye matrix
    def test_min_interval_180_generates_eye(self, deformation_generator):
        deformation_generator.seed = 12345
        deformation_gradient = deformation_generator.rotate(n_axis=1, min_interval=180)
        assert np.all(deformation_gradient - np.eye(3) < 1e-10)

    # min_interval=0 generates deformation gradients with eye matrix
    def test_min_interval_zero_generates_eye(self, deformation_generator):
        deformation_generator.seed = 0
        deformation_gradient = deformation_generator.rotate(n_axis=1, min_interval=0)
        assert np.all(deformation_gradient == np.eye(3))

    # can generate deformation gradients with negative values
    def test_generate_deformation_gradients_with_negative_values(self, deformation_generator):
        deformation_gradient = deformation_generator.rotate(
            n_axis=1,
        )
        assert np.any(deformation_gradient < 0)
