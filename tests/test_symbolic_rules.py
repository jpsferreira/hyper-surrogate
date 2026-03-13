import numpy as np
import pytest
import sympy as sym

from hyper_surrogate.data.deformation import DeformationGenerator
from hyper_surrogate.mechanics.kinematics import Kinematics as K
from hyper_surrogate.mechanics.symbolic import SymbolicHandler

SIZE = 2


# numeric fixtures
@pytest.fixture
def def_gradients() -> np.ndarray:
    return DeformationGenerator(seed=42).combined(SIZE)


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
    return (handler.isochoric_invariant1 - 3) * sym.Symbol("C10") + (handler.isochoric_invariant2 - 3) * sym.Symbol(
        "C01"
    )


@pytest.fixture
def sef_args() -> dict:
    return {"C10": 1, "C01": 1}


@pytest.fixture
def pk2(handler, sef) -> sym.Matrix:
    return handler.pk2(sef)


@pytest.fixture
def cmat(handler, pk2) -> sym.ImmutableDenseNDimArray:
    return handler.cmat(pk2)


@pytest.fixture
def f() -> sym.Matrix:
    return sym.Matrix(3, 3, lambda i, j: sym.Symbol(f"F_{i + 1}{j + 1})"))


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


def test_symbolic_invariant1(handler):
    invariant1 = handler.isochoric_invariant1
    assert isinstance(invariant1, sym.Expr)


def test_symbolic_invariant2(handler):
    invariant2 = handler.isochoric_invariant2
    assert isinstance(invariant2, sym.Expr)


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


@pytest.mark.slow
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


def test_pk2_lambdify_iterator(handler, sef_args, right_cauchys, pk2):
    # pk2 function
    pk2_func = handler.lambdify(pk2, *sef_args.keys())
    pk2_values = np.array(list(handler.evaluate_iterator(pk2_func, right_cauchys, *sef_args.values())))
    assert all(isinstance(pk2_value, np.ndarray) for pk2_value in pk2_values)
    assert all(pk2_value.shape == (3, 3) for pk2_value in pk2_values)
    assert all(isinstance(pk2_value[i, j], float) for pk2_value in pk2_values for i in range(3) for j in range(3))


@pytest.mark.slow
def test_cmat_lambdify_iterator(handler, sef_args, right_cauchys, cmat):
    # cmat function
    cmat_func = handler.lambdify(cmat, *sef_args.keys())
    cmat_values = np.array(list(handler.evaluate_iterator(cmat_func, right_cauchys, *sef_args.values())))
    assert cmat_values.shape == (SIZE, 3, 3, 3, 3)
    assert all(isinstance(cmat_value, np.ndarray) for cmat_value in cmat_values)
    assert all(cmat_value.shape == (3, 3, 3, 3) for cmat_value in cmat_values)


def test_to_voigt_2(handler, pk2):
    # reduce order of pk2. assert shape
    assert handler.to_voigt_2(pk2).shape == (6, 1)


def test_to_voigt_2_with_wrong_shape(handler):
    # reduce order of pk2. assert shape
    with pytest.raises(ValueError):
        handler.to_voigt_2(np.ones((3, 4)))


def test_to_voigt_4(handler):
    # Use a lightweight symbolic (3,3,3,3) tensor instead of deriving cmat
    x = sym.Symbol("x")
    lightweight_cmat = sym.ImmutableDenseNDimArray(
        [x if i == j == k == ll else 0 for i in range(3) for j in range(3) for k in range(3) for ll in range(3)],
        (3, 3, 3, 3),
    )
    assert handler.to_voigt_4(lightweight_cmat).shape == (6, 6)


def test_to_voigt_4_with_wrong_shape(handler):
    # reduce order of cmat. assert shape
    with pytest.raises(ValueError):
        handler.to_voigt_4(np.ones((3, 4, 3, 4)))


def test_pushforward_2nd_order(handler, f):
    # Use a lightweight symbolic (3,3) matrix instead of deriving pk2
    lightweight_pk2 = sym.Matrix(3, 3, lambda i, j: sym.Symbol(f"S_{i}{j}"))
    assert handler.pushforward_2nd_order(lightweight_pk2, f).shape == (3, 3)


def test_pushforward_4th_order(handler, f):
    # Use a sparse symbolic (3,3,3,3) tensor — only diagonal entries nonzero
    x = sym.Symbol("x")
    lightweight_cmat = sym.ImmutableDenseNDimArray(
        [x if i == j == k == ll else 0 for i in range(3) for j in range(3) for k in range(3) for ll in range(3)],
        (3, 3, 3, 3),
    )
    assert handler.pushforward_4th_order(lightweight_cmat, f).shape == (3, 3, 3, 3)


def test_jr(handler):
    # Use a lightweight symbolic sigma instead of deriving through cauchy
    lightweight_sigma = sym.Matrix(3, 3, lambda i, j: sym.Symbol(f"sig_{i}{j}"))
    assert handler.jr(lightweight_sigma).shape == (3, 3, 3, 3)


def test_jr_with_wrong_shape(handler):
    # jaumann rate tensor. assert shape
    with pytest.raises(ValueError):
        handler.jr(np.ones((3, 4)))
