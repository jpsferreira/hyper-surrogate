import numpy as np
import pytest
import sympy as sym

from hyper_surrogate.kinematics import Kinematics as K
from hyper_surrogate.materials import MooneyRivlin, NeoHooke


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
    I1 = material.invariant1
    I3 = material.invariant3
    expected_sef = (I1 - 3) * C10 + 0.25 * KBULK * (I3 - 1 - 2 * sym.log(I3**0.5))
    assert sym.simplify(material.sef - expected_sef) == 0


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
    # Test that pk2_symb is derived from sef
    assert material.pk2_symb == material.pk2_tensor(material.sef)
    # Test shape and symmetry
    assert material.pk2_symb.shape == (3, 3)
    # Check if it's symmetric
    assert (material.pk2_symb - material.pk2_symb.T).norm() < 1e-10


def test_cmat_symbol(material):
    # Test shape
    assert material.cmat_symb.shape == (3, 3, 3, 3)
    assert material.cmat_symb.shape == material.cmat_tensor(material.pk2_symb).shape
    # Verify it's calculated correctly from pk2_symb
    assert material.cmat_symb == material.cmat_tensor(material.pk2_symb)


def test_mooneyrivlin_sef():
    mooneyrivlin = MooneyRivlin()
    assert mooneyrivlin.sef == (mooneyrivlin.invariant1 - 3) * sym.Symbol("C10") + (
        mooneyrivlin.invariant2 - 3
    ) * sym.Symbol("C01") + 0.25 * sym.Symbol("KBULK") * (
        mooneyrivlin.invariant3 - 1 - 2 * sym.log(mooneyrivlin.invariant3**0.5)
    )


def test_get_default_parameters():
    # Test for NeoHooke
    nh = NeoHooke()
    nh_defaults = nh.get_default_parameters()
    assert "C10" in nh_defaults
    assert "KBULK" in nh_defaults
    assert nh_defaults["C10"] == 0.5
    assert nh_defaults["KBULK"] == 1000.0

    # Test for MooneyRivlin
    mr = MooneyRivlin()
    mr_defaults = mr.get_default_parameters()
    assert "C10" in mr_defaults
    assert "C01" in mr_defaults
    assert "KBULK" in mr_defaults
    assert mr_defaults["C10"] == 0.3
    assert mr_defaults["C01"] == 0.2
    assert mr_defaults["KBULK"] == 1000.0


def test_validate_parameters():
    # Test for NeoHooke
    nh = NeoHooke()

    # Test with valid parameters
    params = {"C10": 1.0, "KBULK": 2000.0}
    validated = nh.validate_parameters(params)
    assert validated["C10"] == 1.0
    assert validated["KBULK"] == 2000.0

    # Test with missing parameters (should use defaults)
    params = {"C10": 1.0}
    validated = nh.validate_parameters(params)
    assert validated["C10"] == 1.0
    assert validated["KBULK"] == 1000.0  # Default value

    # Test with unknown parameters (should raise ValueError)
    params = {"C10": 1.0, "UNKNOWN": 5.0}
    with pytest.raises(ValueError) as excinfo:
        nh.validate_parameters(params)
    assert "UNKNOWN" in str(excinfo.value)
