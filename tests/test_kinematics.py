import numpy as np
import pytest

from hyper_surrogate.data.deformation import DeformationGenerator
from hyper_surrogate.mechanics.kinematics import Kinematics

SIZE = 2


@pytest.fixture
def K():
    return Kinematics()


@pytest.fixture
def def_gradients():
    return DeformationGenerator(seed=42).combined(SIZE)


# kinematics testing
@pytest.fixture
def right_cauchys(def_gradients):
    return np.array([np.matmul(f, f.T) for f in def_gradients])


@pytest.fixture
def left_cauchys(def_gradients):
    return np.array([np.matmul(f.T, f) for f in def_gradients])


def test_right_cauchys(def_gradients, K):
    right_cauchys = K.right_cauchy_green(def_gradients)
    assert np.allclose(right_cauchys, np.array([np.matmul(f.T, f) for f in def_gradients]))


def test_left_cauchys(def_gradients, K):
    left_cauchys = K.left_cauchy_green(def_gradients)
    assert np.allclose(left_cauchys, np.array([np.matmul(f, f.T) for f in def_gradients]))


def test_jacobian(def_gradients, K):
    jacobians = K.jacobian(def_gradients)
    assert np.allclose(jacobians, np.array([np.linalg.det(f) for f in def_gradients]))


def test_trace_invariant(def_gradients, K):
    result = K.trace_invariant(def_gradients)
    assert np.allclose(result, np.array([np.trace(f) for f in def_gradients]))


def test_quadratic_invariant(def_gradients, K):
    result = K.quadratic_invariant(def_gradients)
    assert np.allclose(
        result,
        np.array([0.5 * (np.trace(f) ** 2 - np.trace(np.matmul(f, f))) for f in def_gradients]),
    )


def test_det_invariant(def_gradients, K):
    result = K.det_invariant(def_gradients)
    assert np.allclose(result, np.array([np.linalg.det(f) for f in def_gradients]))


def test_isochoric_invariant1(def_gradients, K):
    c = K.right_cauchy_green(def_gradients)
    result = K.isochoric_invariant1(c)
    # Manual computation: tr(C) * det(C)^(-1/3)
    expected = np.array([np.trace(c[i]) * np.linalg.det(c[i]) ** (-1.0 / 3.0) for i in range(SIZE)])
    assert np.allclose(result, expected)


def test_isochoric_invariant2(def_gradients, K):
    c = K.right_cauchy_green(def_gradients)
    result = K.isochoric_invariant2(c)
    # Manual computation: 0.5*(I1^2 - tr(C^2)) * det(C)^(-2/3)
    expected = np.array([
        0.5 * (np.trace(c[i]) ** 2 - np.trace(c[i] @ c[i])) * np.linalg.det(c[i]) ** (-2.0 / 3.0) for i in range(SIZE)
    ])
    assert np.allclose(result, expected)


def test_pushforward(def_gradients, K):
    # test pushforward operation on unit tensors  (F * I * F^T)
    tensor_3x3 = np.array([np.eye(3) for _ in range(SIZE)])
    forwards = K.pushforward(def_gradients, tensor_3x3)
    assert np.allclose(forwards, np.array([f @ np.eye(3) @ f.T for f in def_gradients]))


def test_principal_stretches(def_gradients, K):
    principal_stretches = K.principal_stretches(def_gradients)
    assert np.allclose(
        principal_stretches,
        np.array([np.sqrt(np.linalg.eigvals(f.T @ f)) for f in def_gradients]),
    )


def test_principal_directions(def_gradients, K):
    principal_directions = K.principal_directions(def_gradients)
    assert np.allclose(
        principal_directions,
        np.array([np.linalg.eig(f.T @ f)[1] for f in def_gradients]),
    )


def test_right_stretch_tensor(def_gradients, K):
    right_stretch_tensor = K.right_stretch_tensor(def_gradients)
    assert right_stretch_tensor.shape == (SIZE, 3, 3)


def test_left_stretch_tensor(def_gradients, K):
    left_stretch_tensor = K.left_stretch_tensor(def_gradients)
    assert left_stretch_tensor.shape == (SIZE, 3, 3)


def test_rotation_tensor(def_gradients, K):
    rotation_tensor = K.rotation_tensor(def_gradients)
    assert rotation_tensor.shape == (SIZE, 3, 3)
