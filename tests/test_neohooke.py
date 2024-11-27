import numpy as np
import pytest
import sympy as sym

from hyper_surrogate.kinematics import Kinematics as K
from hyper_surrogate.materials import NeoHooke
from tests.mock_data import (
    CMAT_ISO0,
    CMAT_ISO_ARBITRARY,
    CMAT_ISO_ARBITRARY2,
    CMAT_ISO_UNI,
    CMAT_VOL0,
    CMAT_VOL_ARBITRARY,
    CMAT_VOL_ARBITRARY2,
    CMAT_VOL_UNI,
    PK2_ISO0,
    PK2_ISO_ARBITRARY,
    PK2_ISO_ARBITRARY2,
    PK2_ISO_UNI,
    PK2_VOL0,
    PK2_VOL_ARBITRARY,
    PK2_VOL_ARBITRARY2,
    PK2_VOL_UNI,
)

# seed random number generator
np.random.seed(0)


@pytest.fixture
def neohooke():
    return NeoHooke()


@pytest.fixture
def f0():
    # no deformation
    return np.array(np.eye(3))


@pytest.fixture
def f_uni():
    # uniaxial stretch 3 at x-axis. 3x3 matrix. 3,0,0; 0,1/sqrt(3),0; 0,0,1/sqrt(3)
    return np.array(np.diag([3, 1 / np.sqrt(3), 1 / np.sqrt(3)]))


@pytest.fixture
def f_arbitrary():
    return np.array([[3.0, 0.4, 0.1], [0.4, 1 / np.sqrt(3.0), 0.1], [0.9, -0.2, 0.6]])


@pytest.fixture
def f_arbitrary2():
    return np.array([[3.0, 0.4, 0.1], [0.4, 0.45, 0.1], [0.9, -0.2, 0.9]])


@pytest.fixture
def f(f0, f_uni, f_arbitrary, f_arbitrary2):
    return np.array([f0, f_uni, f_arbitrary, f_arbitrary2])


@pytest.fixture
def c_tensor(f):
    return K.right_cauchy_green(f)


def test_sef(neohooke):
    assert neohooke.sef == (neohooke.invariant1 - 3) * sym.Symbol("C10") + +0.25 * sym.Symbol("KBULK") * (
        neohooke.invariant3 - 1 - 2 * sym.log(neohooke.invariant3**0.5)
    )


def test_lambdify_sef(neohooke, c_tensor):
    sef_func = neohooke.substitute_iterator(neohooke.sef, c_tensor, {"C10": 1, "KBULK": 1})
    sef_values = np.array(list(sef_func))
    assert sef_values.shape == (4,)
    assert sef_values[0] == 0
    # all remaining values of SEF should be positive
    assert sef_values[1:].all() > 0


@pytest.mark.parametrize(
    "params, expected_values",
    [
        (
            {"C10": 1, "KBULK": 0},
            [PK2_ISO0, PK2_ISO_UNI, PK2_ISO_ARBITRARY, PK2_ISO_ARBITRARY2],
        ),
        (
            {"C10": 0, "KBULK": 1000},
            [PK2_VOL0, PK2_VOL_UNI, PK2_VOL_ARBITRARY, PK2_VOL_ARBITRARY2],
        ),
    ],
)
def test_lambdify_pk2(neohooke, c_tensor, params, expected_values):
    pk2_func = neohooke.evaluate_iterator(neohooke.pk2(), c_tensor, **params)
    pk2_values = np.array(list(pk2_func))
    assert pk2_values.shape == (4, 3, 3)
    # Verify the expected values for each deformation case
    for i, expected in enumerate(expected_values):
        assert np.allclose(pk2_values[i], expected)


@pytest.mark.parametrize(
    "params, expected_values",
    [
        (
            {"C10": 1, "KBULK": 0},
            [CMAT_ISO0, CMAT_ISO_UNI, CMAT_ISO_ARBITRARY, CMAT_ISO_ARBITRARY2],
        ),
        (
            {"C10": 0, "KBULK": 1000},
            [CMAT_VOL0, CMAT_VOL_UNI, CMAT_VOL_ARBITRARY, CMAT_VOL_ARBITRARY2],
        ),
    ],
)
def test_lambdify_cmat(neohooke, c_tensor, params, expected_values):
    cmat_func = neohooke.evaluate_iterator(neohooke.cmat(), c_tensor, **params)
    cmat_values = np.array(list(cmat_func))
    assert cmat_values.shape == (4, 3, 3, 3, 3)
    for i, expected in enumerate(expected_values):
        assert np.allclose(cmat_values[i], expected)
