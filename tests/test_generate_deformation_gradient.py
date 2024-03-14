import pytest

from hyper_surrogate.deformation_gradient import DeformationGradientGenerator
from hyper_surrogate.generator import Generator


@pytest.fixture
def def_gradient():
    return DeformationGradientGenerator(seed=42, size=10)


@pytest.fixture
def generator():
    return Generator(seed=42, size=10)


# def test_generate_def_grad(def_gradient):
#     stretch = def_gradient.generate.uniform(1, 2)
#     logging.info(f"Stretch: {def_gradient.shear([2,1])}")
