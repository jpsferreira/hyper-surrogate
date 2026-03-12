import numpy as np
import pytest
import sympy as sym

from hyper_surrogate.mechanics.kinematics import Kinematics as K
from hyper_surrogate.mechanics.materials import MooneyRivlin, NeoHooke


@pytest.fixture
def material():
    return NeoHooke()  # Use NeoHooke as a concrete implementation


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


def test_material_sef(material):
    # Test that the sef property returns a SymPy expression
    assert isinstance(material.sef, sym.Expr)
    # For NeoHooke specifically
    C10 = sym.Symbol("C10")
    KBULK = sym.Symbol("KBULK")
    h = material.handler
    I1 = h.isochoric_invariant1
    I3 = h.invariant3
    expected_sef = (I1 - 3) * C10 + 0.25 * KBULK * (I3 - 1 - 2 * sym.log(I3**0.5))
    assert sym.simplify(material.sef - expected_sef) == 0


def test_pk2_expr_shape(material):
    # pk2_expr should be a 3x3 symbolic matrix
    assert material.pk2_expr.shape == (3, 3)
    assert isinstance(material.pk2_expr, sym.Matrix)


def test_cmat_expr_shape(material):
    # cmat_expr should be a 3x3x3x3 array
    assert material.cmat_expr.shape == (3, 3, 3, 3)


def test_mooneyrivlin_sef():
    mooneyrivlin = MooneyRivlin()
    h = mooneyrivlin.handler
    assert mooneyrivlin.sef == (mooneyrivlin.handler.isochoric_invariant1 - 3) * sym.Symbol("C10") + (
        mooneyrivlin.handler.isochoric_invariant2 - 3
    ) * sym.Symbol("C01") + 0.25 * sym.Symbol("KBULK") * (h.invariant3 - 1 - 2 * sym.log(h.invariant3**0.5))


def test_neohooke_default_params():
    nh = NeoHooke()
    assert nh._params["C10"] == 0.5
    assert nh._params["KBULK"] == 1000.0


def test_mooneyrivlin_default_params():
    mr = MooneyRivlin()
    assert mr._params["C10"] == 0.3
    assert mr._params["C01"] == 0.2
    assert mr._params["KBULK"] == 1000.0


def test_neohooke_custom_params():
    nh = NeoHooke({"C10": 1.0, "KBULK": 2000.0})
    assert nh._params["C10"] == 1.0
    assert nh._params["KBULK"] == 2000.0


def test_evaluate_pk2_identity(material):
    c_identity = np.eye(3).reshape(1, 3, 3)
    result = material.evaluate_pk2(c_identity)
    assert result.shape == (1, 3, 3)
    np.testing.assert_allclose(result[0], np.zeros((3, 3)), atol=1e-10)
