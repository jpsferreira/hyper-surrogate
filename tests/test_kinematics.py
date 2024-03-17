import numpy as np
import pytest

from hyper_surrogate.deformation_gradient import DeformationGradientGenerator as FGen
from hyper_surrogate.kinematics import Kinematics as K

SIZE = 2


@pytest.fixture
def def_gradients():
    return FGen(seed=42, size=SIZE).generate()


# kinematics testing
@pytest.fixture
def right_cauchys(def_gradients):
    return np.array([np.matmul(f, f.T) for f in def_gradients])


def test_right_cauchys(def_gradients):
    right_cauchys = K.right_cauchy_green(def_gradients)
    assert np.allclose(right_cauchys, np.array([np.matmul(f.T, f) for f in def_gradients]))
