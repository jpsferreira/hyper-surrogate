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


@pytest.fixture
def f() -> sym.Matrix:
    return sym.Matrix(3, 3, lambda i, j: sym.Symbol(f"F_{i + 1}{j + 1})"))


@pytest.fixture
def sigma(handler, sef, f) -> sym.Matrix:
    return handler.sigma_tensor(sef, f)


@pytest.fixture
def smat(handler, sef, f) -> sym.Matrix:
    return handler.smat_tensor(sef, f)


# testing


def test_f_symbols(handler):
    # assert f_symbols
    f_symbols = handler.f_symbols()
    assert isinstance(f_symbols, list)
    assert all(isinstance(f, sym.Symbol) for f in f_symbols)


def test_c_symbols(handler):
    # assert c_symbols
    c_symbols = handler.c_symbols()
    assert isinstance(c_symbols, list)
    assert all(isinstance(c, sym.Symbol) for c in c_symbols)


def test_symbolic_pk2_cmat(pk2, cmat):
    # derivative of sef in order to c_tensor

    # assert instance
    assert isinstance(pk2, sym.Matrix)
    assert isinstance(cmat, sym.ImmutableDenseNDimArray)
    # assert shape
    assert pk2.shape == (3, 3)
    assert cmat.shape == (3, 3, 3, 3)


def test_substitute_with_wrong_shape(handler):
    # subs c_tensor with c values
    with pytest.raises(ValueError):
        handler.substitute(handler.c_tensor, np.ones((3, 4)))


def test_symbolic_subs_in_c(handler):
    # substitute the c_tensor with a 3x3 matrix of ones
    c_tensor = handler.c_tensor
    # dummy c values
    c = np.ones((3, 3), dtype=int)
    # subs c_tensor with c values
    # c_tensor_subs = handler.c_tensor.subs({c_tensor[i, j]: c[i, j] for i in range(3) for j in range(3)})
    c_tensor_subs = handler.substitute(c_tensor, c)
    assert all(c_tensor_subs[i, j] == 1 for i in range(3) for j in range(3))


def test_symbolic_subs_in_pk2(handler, pk2, right_cauchys, sef_args):
    # right_cauchys # (N, 3, 3)
    # for each c_tensor in pk2, substitute the pk2 tensor with c values and material parameters values.
    assert all(
        isinstance(subs, sym.Matrix)
        and subs.shape == (3, 3)
        and all(isinstance(subs[i, j], sym.Expr) for i in range(3) for j in range(3))
        for subs in handler.substitute_iterator(pk2, right_cauchys, sef_args)
    )


def test_symbolic_subs_in_cmat(handler, cmat, right_cauchys, sef_args):
    # right_cauchys # (N, 3, 3)
    # for each c_tensor in cmat, substitute the cmat tensor with c values and material parameters values.
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


# skip this test for now. implement the functionality in the handler and test it.
@pytest.mark.skip(reason="Feature not implemented yet")
def test_symbolic_subs_in_sigma(handler, sigma, right_cauchys, sef_args):
    # right_cauchys # (N, 3, 3)
    #
    # for each c_tensor in sigma, substitute the sigma tensor with c values and material parameters values.
    assert all(
        isinstance(subs, sym.Matrix)
        and subs.shape == (3, 3)
        and all(isinstance(subs[i, j], sym.Expr) for i in range(3) for j in range(3))
        for subs in handler.substitute_iterator(sigma, right_cauchys, sef_args)
        # TODO: incorporate iterator over multiple matrices: right cauchys and def_gradients
    )


@pytest.mark.skip(reason="Feature not implemented yet")
def test_symbolic_subs_in_smat(handler, smat, right_cauchys, sef_args):
    # right_cauchys # (N, 3, 3)
    # for each c_tensor in cmat, substitute the cmat tensor with c values and material parameters values.
    assert all(
        isinstance(subs, sym.MutableDenseNDimArray)
        and subs.shape == (3, 3, 3, 3)
        and all(
            isinstance(subs[i, j, k, ll], sym.Expr)
            for i in range(3)
            for j in range(3)
            for k in range(3)
            for ll in range(3)
        )
        for subs in handler.substitute_iterator(smat, right_cauchys, sef_args)
    )


def test_pk2_lambdify_iterator(handler, sef_args, right_cauchys, pk2):
    # pk2 function
    pk2_func = handler.lambda_tensor(pk2, *sef_args.keys())
    pk2_values = np.array(list(handler.evaluate_iterator(pk2_func, right_cauchys, *sef_args.values())))
    assert all(isinstance(pk2_value, np.ndarray) for pk2_value in pk2_values)
    assert all(pk2_value.shape == (3, 3) for pk2_value in pk2_values)
    assert all(isinstance(pk2_value[i, j], float) for pk2_value in pk2_values for i in range(3) for j in range(3))


def test_cmat_lambdify_iterator(handler, sef_args, right_cauchys, cmat):
    # cmat function
    cmat_func = handler.lambda_tensor(cmat, *sef_args.keys())
    cmat_values = np.array(list(handler.evaluate_iterator(cmat_func, right_cauchys, *sef_args.values())))
    assert cmat_values.shape == (SIZE, 3, 3, 3, 3)
    assert all(isinstance(cmat_value, np.ndarray) for cmat_value in cmat_values)
    assert all(cmat_value.shape == (3, 3, 3, 3) for cmat_value in cmat_values)


def test_reduce_2nd_order(handler, pk2):
    # reduce order of pk2. assert shape
    assert handler.reduce_2nd_order(pk2).shape == (6, 1)


def test_reduce_2nd_order_with_wrong_shape(handler):
    # reduce order of pk2. assert shape
    with pytest.raises(ValueError):
        handler.reduce_2nd_order(np.ones((3, 4)))


def test_reduce_4th_order(handler, cmat):
    # reduce order of cmat. assert shape
    assert handler.reduce_4th_order(cmat).shape == (6, 6)


def test_reduce_4th_order_with_wrong_shape(handler):
    # reduce order of cmat. assert shape
    with pytest.raises(ValueError):
        handler.reduce_4th_order(np.ones((3, 4, 3, 4)))


@pytest.mark.skip(reason="Slow test")
def test_pushforward_2nd_order(handler, pk2, f):
    # pushforward pk2 to cauchy stress tensor. assert shape
    assert handler.pushforward_2nd_order(pk2, f).shape == (3, 3)


@pytest.mark.skip(reason="Slow test")
def test_pushforward_4th_order(handler, cmat, f):
    # pushforward cmat to spatial stiffness tensor. assert shape
    assert handler.pushforward_4th_order(cmat, f).shape == (3, 3, 3, 3)


def test_jr(handler, sigma):
    # jaumann rate tensor. assert shape
    assert handler.jr(sigma).shape == (3, 3, 3, 3)


def test_jr_with_wrong_shape(handler):
    # jaumann rate tensor. assert shape
    with pytest.raises(ValueError):
        handler.jr(np.ones((3, 4)))
