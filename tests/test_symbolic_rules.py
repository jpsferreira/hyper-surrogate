import numpy as np
import pytest
import sympy as sym

from hyper_surrogate.deformation_gradient import DeformationGradientGenerator as FGen
from hyper_surrogate.kinematics import Kinematics as K
from hyper_surrogate.symbolic import SymbolicHandler

SIZE = 2


# numeric fixtures
@pytest.fixture
def def_gradients() -> np.ndarray:
    return FGen(seed=42, size=SIZE).generate()


@pytest.fixture
def right_cauchys(def_gradients) -> np.ndarray:
    return K.right_cauchy_green(def_gradients)


# symbolic fixtures
@pytest.fixture
def handler() -> SymbolicHandler:
    return SymbolicHandler()


# mooney_rivlin
@pytest.fixture
def sef(handler) -> sym.Expr:
    return (handler.invariant1 - 3) * sym.Symbol("C10") + (handler.invariant2 - 3) * sym.Symbol("C01")


@pytest.fixture
def sef_args() -> dict:
    return {"C10": 1, "C01": 1}


@pytest.fixture
def pk2(handler, sef) -> sym.Matrix:
    return handler.pk2_tensor(sef)


@pytest.fixture
def cmat(handler, pk2) -> sym.ImmutableDenseNDimArray:
    return handler.cmat_tensor(pk2)


# testing


def test_symbolic_pk2_cmat(pk2, cmat):
    # derivative of sef in order to c_tensor

    # assert instance
    assert isinstance(pk2, sym.Matrix)
    assert isinstance(cmat, sym.ImmutableDenseNDimArray)
    # assert shape
    assert pk2.shape == (3, 3)
    assert cmat.shape == (3, 3, 3, 3)


def test_symbolic_subs_in_c(handler):
    # substitute the c_tensor with a 3x3 matrix of ones
    c_tensor = handler.c_tensor
    # dummy c values
    c = np.ones((3, 3))
    # subs c_tensor with c values
    # c_tensor_subs = handler.c_tensor.subs({c_tensor[i, j]: c[i, j] for i in range(3) for j in range(3)})
    c_tensor_subs = handler.substitute(c_tensor, c)
    assert all(c_tensor_subs[i, j] == 1 for i in range(3) for j in range(3))


def test_symbolic_subs_in_pk2(handler, pk2, right_cauchys, sef_args):
    # right_cauchys # (N, 3, 3)
    # for each c_tensor in c, substitute the pk2 tensor with c values and material parameters values.
    assert all(
        isinstance(subs, sym.Matrix)
        and subs.shape == (3, 3)
        and all(isinstance(subs[i, j], sym.Expr) for i in range(3) for j in range(3))
        for subs in handler.substitute_iterator(pk2, right_cauchys, sef_args)
    )


def test_symbolic_subs_in_cmat(handler, cmat, right_cauchys, sef_args):
    # right_cauchys # (N, 3, 3)
    # for each c_tensor in c, substitute the cmat tensor with c values and material parameters values.
    assert all(
        isinstance(subs, sym.ImmutableDenseNDimArray)
        and subs.shape == (3, 3, 3, 3)
        and all(
            isinstance(subs[i, j, k, ll], sym.Expr)
            for i in range(3)
            for j in range(3)
            for k in range(3)
            for ll in range(3)
        )
        for subs in handler.substitute_iterator(cmat, right_cauchys, sef_args)
    )
