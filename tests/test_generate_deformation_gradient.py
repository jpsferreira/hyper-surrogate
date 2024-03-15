import numpy as np
import pytest

from hyper_surrogate.deformation_gradient import DeformationGradientGenerator


@pytest.fixture
def def_gradient():
    return DeformationGradientGenerator(seed=42, size=10)


def test_generate_def_grad(def_gradient):
    stretch = def_gradient.generator.uniform(0, 1)
    assert len(stretch) == 10
    assert all(isinstance(x, float) for x in stretch)
    assert all(0 <= x <= 1 for x in stretch)
    assert isinstance(stretch, np.ndarray)
    assert isinstance(stretch[0], float)
    assert isinstance(stretch[1], float)


def test_generate_size_def_grad(def_gradient):
    f = def_gradient.generate()
    assert f.shape == (10, 3, 3)
    assert isinstance(f, np.ndarray)
    assert isinstance(f[0, 0, 0], float)
