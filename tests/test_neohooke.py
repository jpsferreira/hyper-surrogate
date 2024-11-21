import numpy as np
import pytest
import sympy as sym

from hyper_surrogate.kinematics import Kinematics as K
from hyper_surrogate.materials import NeoHooke

# seed random number generator
np.random.seed(0)


@pytest.fixture
def neohooke():
    return NeoHooke()


@pytest.fixture
def f0():
    return np.array(np.eye(3))


@pytest.fixture
def f1():
    return np.array(np.eye(3) + 0.1 * np.random.rand(3, 3))


@pytest.fixture
def f(f0, f1):
    return np.array([f0, f1])


@pytest.fixture
def c_tensor(f):
    return K.right_cauchy_green(f)


def test_sef(neohooke):
    assert neohooke.sef == (neohooke.invariant1 - 3) * sym.Symbol("C10")


def test_lambdify_sef(neohooke, c_tensor):
    sef_func = neohooke.substitute_iterator(neohooke.sef, c_tensor, {"C10": 1})
    sef_values = np.array(list(sef_func))
    assert sef_values.shape == (2,)
    assert sef_values[0] == 0
    assert sef_values[1] > 0


def test_lambdify_pk2(neohooke, c_tensor):
    # logging.info(neohooke.pk2_symb)
    pk2_func = neohooke.evaluate_iterator(neohooke.pk2(), c_tensor, 1)
    pk2_values = np.array(list(pk2_func))
    assert pk2_values.shape == (2, 3, 3)


def test_lambdify_cmat(neohooke, c_tensor):
    cmat_func = neohooke.evaluate_iterator(neohooke.cmat(), c_tensor, 2)
    cmat_values = np.array(list(cmat_func))
    assert cmat_values.shape == (2, 3, 3, 3, 3)
