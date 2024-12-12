import logging

import numpy as np
import pytest
import sympy as sym

from hyper_surrogate.kinematics import Kinematics as K
from hyper_surrogate.materials import Material, MooneyRivlin


@pytest.fixture
def material():
    return Material(["param1", "param2"])


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


def test_material_dummy_sef(material):
    assert material.sef == sym.Symbol("sef")


def test_lambdify_invariant1(material, c_tensor):
    # Fetch the symbolic invariant1 expression
    inv1_func = material.substitute_iterator(material.invariant1, c_tensor)
    inv1_results = np.array(list(inv1_func))
    assert inv1_results.shape == (2,)
    assert inv1_results[0] == 3.0
    assert inv1_results[1] > 3.0


def test_lambdify_invariant2(material, c_tensor):
    # Fetch the symbolic invariant2 expression
    inv2_func = material.substitute_iterator(material.invariant2, c_tensor)
    inv2_results = np.array(list(inv2_func))
    assert inv2_results.shape == (2,)
    assert inv2_results[0] == 0
    assert inv2_results[1] == 0


def test_lambdify_invariant3(material, c_tensor):
    # Fetch the symbolic invariant3 expression
    inv3_func = material.substitute_iterator(material.invariant3, c_tensor)
    inv3_results = np.array(list(inv3_func))
    assert inv3_results.shape == (2,)
    assert inv3_results[0] == 1.0
    assert inv3_results[1] > 1.0


def test_pk2_symbol(material):
    logging.info(material.pk2_symb)
    assert material.pk2_symb == material.pk2_tensor(material.sef)
    assert material.pk2_symb == sym.Matrix([[0, 0, 0], [0, 0, 0], [0, 0, 0]])


def test_cmat_symbol(material):
    # shape
    assert material.cmat_symb.shape == (3, 3, 3, 3)
    assert material.cmat_symb.shape == material.cmat_tensor(material.pk2_symb).shape
    assert material.cmat_symb == material.cmat_tensor(material.pk2_symb)
    assert material.cmat_symb == sym.ImmutableDenseNDimArray(np.zeros((3, 3, 3, 3), dtype=int))


def test_mooneyrivlin_sef():
    mooneyrivlin = MooneyRivlin()
    assert mooneyrivlin.sef == (mooneyrivlin.invariant1 - 3) * sym.Symbol("C10") + (
        mooneyrivlin.invariant2 - 3
    ) * sym.Symbol("C01") + 0.25 * sym.Symbol("KBULK") * (
        mooneyrivlin.invariant3 - 1 - 2 * sym.log(mooneyrivlin.invariant3 ** 0.5)
    )
    
