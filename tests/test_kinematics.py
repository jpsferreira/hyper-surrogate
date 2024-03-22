import numpy as np
import pytest

from hyper_surrogate.deformation_gradient import DeformationGradientGenerator as FGen
from hyper_surrogate.kinematics import Kinematics

SIZE = 1


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


def test_strain_tensor(def_gradients, K):
    strain_tensors = K.strain_tensor(def_gradients)
    assert np.allclose(strain_tensors, 0.5 * (np.array([f.T @ f for f in def_gradients]) - np.eye(3)))


# def test_stretch_tensor(def_gradients, K):
#     stretch_tensors = K.stretch_tensor(def_gradients)
#     logging.info(stretch_tensors)
# assert np.allclose(stretch_tensors, np.sqrt(np.array([f.T @ f for f in def_gradients])))
