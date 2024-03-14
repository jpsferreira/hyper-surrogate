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


# def test_generate_10_def_grad(def_gradient):
#     logging.info("Testing 10 deformation gradients")
#     logging.info(def_gradient.generate())
