import numpy as np
import pytest

from hyper_surrogate.deformation_gradient import DeformationGradientGenerator as FGen
from hyper_surrogate.kinematics import Kinematics

SIZE = 2


@pytest.fixture
def K():
    return Kinematics()


@pytest.fixture
def def_gradients():
    return FGen(seed=42, size=SIZE).generate()


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

def test_invariant1(def_gradients, K):
    invariant1 = K.invariant1(def_gradients)
    assert np.allclose(invariant1, np.array([np.trace(f) for f in def_gradients]))

def test_invariant2(def_gradients, K):
    invariant2 = K.invariant2(def_gradients)
    assert np.allclose(invariant2, np.array([0.5 * (np.trace(f) ** 2 - np.trace(np.matmul(f, f))) for f in def_gradients]))

def test_invariant3(def_gradients, K):
    invariant3 = K.invariant3(def_gradients)
    assert np.allclose(invariant3, np.array([np.linalg.det(f) for f in def_gradients]))
        
def test_pushforward(def_gradients, K):
    # test pushforward operation on unit tensors  (F * I * F^T)
    tensor_3x3 = np.array([np.eye(3) for _ in range(SIZE)])
    forwards = K.pushforward(def_gradients, tensor_3x3)
    assert np.allclose(forwards, np.array([f @ np.eye(3) @ f.T for f in def_gradients]))


def test_rotation_tensor(def_gradients, K):
    rotation_tensors = K.rotation_tensor(def_gradients)
    assert np.allclose(rotation_tensors, np.array([f @ np.linalg.inv(f) for f in def_gradients]))


def test_principal_stretches(def_gradients, K):
    principal_stretches = K.principal_stretches(def_gradients)
    assert np.allclose(principal_stretches, np.array([np.sqrt(np.linalg.eigvals(f.T @ f)) for f in def_gradients]))


def test_principal_directions(def_gradients, K):
    principal_directions = K.principal_directions(def_gradients)
    assert np.allclose(principal_directions, np.array([np.linalg.eig(f.T @ f)[1] for f in def_gradients]))
