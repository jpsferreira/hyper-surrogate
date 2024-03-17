import logging

import numpy as np
import pytest
import sympy as sym

from hyper_surrogate.deformation_gradient import DeformationGradientGenerator as FGen
from hyper_surrogate.kinematics import Kinematics as K
from hyper_surrogate.symbolic import SymbolicHandler

SIZE = 2


@pytest.fixture
def def_gradients():
    return FGen(seed=42, size=SIZE).generate()


@pytest.fixture
def handler():
    return SymbolicHandler()


@pytest.fixture
def sef():
    return (SymbolicHandler().invariant1 - 3) * sym.Symbol("C10")


# kinematics testing
@pytest.fixture
def right_cauchys(def_gradients):
    return np.array([np.matmul(f, f.T) for f in def_gradients])


def test_symbolic_pk2_cmat(handler, sef):
    # derivative of sef in order to c_tensor
    pk2 = handler.pk2_tensor(sef)
    cmat = handler.cmat_tensor(pk2)
    # assert instance
    assert isinstance(pk2, sym.MutableDenseNDimArray)
    assert isinstance(cmat, sym.MutableDenseNDimArray)
    # assert shape
    assert pk2.shape == (3, 3)
    assert cmat.shape == (3, 3, 3, 3)


def test_symbolic_subs(handler):
    # substitute the c_tensor with a 3x3 matrix of ones
    c_tensor = handler.c_tensor
    # dummy c values
    c = np.ones((3, 3))
    # subs c_tensor with c values
    # c_tensor_subs = handler.c_tensor.subs({c_tensor[i, j]: c[i, j] for i in range(3) for j in range(3)})
    c_tensor_subs = handler.substitute(c_tensor, c)
    assert all(c_tensor_subs[i, j] == 1 for i in range(3) for j in range(3))


def test_symbolic_subs_in_batch(handler, def_gradients):
    #
    c_tensor = handler.c_tensor
    # get first c from def_gradients
    c = K.right_cauchy_green(def_gradients)
    # subs c_tensor with c values
    c_all = [handler.substitute(c_tensor, cc) for cc in c]
    logging.info(c_all)


# voigt notation
# def test_voigt_notation(handler):
#     c_tensor = handler.c_tensor
#     c_voigt = sym.Matrix([c_tensor[0, 0], c_tensor[1, 1], c_tensor[2, 2], c_tensor[0, 1], c_tensor[0, 2], c_tensor[1, 2]])

# invariant3 = symbolic_handler.invariant3_symbolic()


# def test_derivative(def_gradients,c_tensor_symbolic):
#     logging.info(def_gradients)

#     C_10_sym=sym.Symbol("C_10_sym")
#     #get symbols from c_tensor_symbolic
#     C11,C12,C13,C21,C22,C23,C31,C32,C33=c_tensor_symbolic.args
#     I3=C11*C22*C33 - C11*C23*C32 - C12*C21*C33 + C12*C23*C31 + C13*C21*C32 - C13*C22*C31
#     trace=C11+C22+C33
#     I1=trace*(I3)**(-1/3)

#     #Symbolic Strain Energy Function
#     SEF=(I1-3)*C_10_sym
#     logging.info(SEF)
