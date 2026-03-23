import logging

import pytest
import sympy as sym

from hyper_surrogate.export.fortran.analytical import UMATHandler
from hyper_surrogate.mechanics.materials import Material, NeoHooke

# Configure logging
logging.basicConfig(level=logging.INFO)


@pytest.fixture
def material():
    """Fixture using NeoHooke material model."""
    return NeoHooke({"C10": 0.5, "KBULK": 1000.0})


@pytest.fixture
def umat_handler(material):
    """Fixture creating UMATHandler with the base material fixture."""
    return UMATHandler(material_model=material)


def test_initialize_with_valid_material(umat_handler, material):
    """Test initialization with a valid material model."""
    assert umat_handler.material == material
    assert umat_handler.sigma_code is None
    assert umat_handler.smat_code is None
    assert isinstance(umat_handler.material, Material)


@pytest.mark.slow
def test_generate_umat_code(tmp_path):
    """Test generating the UMAT code to a file."""
    material_model = NeoHooke({"C10": 0.5, "KBULK": 1000.0})
    umat_handler = UMATHandler(material_model)

    filename = tmp_path / "umat.f"
    # Exercise
    umat_handler.generate(filename)

    assert filename.is_file()
    assert filename.stat().st_size > 0
    assert filename.suffix == ".f"


def test_common_subexpressions_with_empty_tensor(umat_handler):
    """Test common subexpression elimination with an empty tensor (original version)."""
    empty_tensor = sym.Matrix(0, 0, [])
    assert umat_handler.common_subexpressions(empty_tensor, "var") == []


def test_f_property(umat_handler):
    """Test the deformation gradient property."""
    f = umat_handler.f
    assert f.shape == (3, 3)
    # Check it's made of DFGRD1 symbols
    assert "DFGRD1(1,1)" in str(f[0, 0])


def test_sub_exp_property(umat_handler):
    """Test the substitution expressions property."""
    sub = umat_handler.sub_exp
    assert isinstance(sub, dict)
    assert len(sub) == 9  # 3x3 components


def test_generate_props_code(umat_handler):
    """Test Fortran properties code generation."""
    code = umat_handler.generate_props_code()
    assert any("C10" in line for line in code)
    assert any("KBULK" in line for line in code)
    assert any("PROPS(" in line for line in code)
    assert any("DOUBLE PRECISION" in line for line in code)


def test_code_as_string():
    """Test code_as_string static method."""
    lines = ["line1", "line2", "line3"]
    result = UMATHandler.code_as_string(lines)
    assert result == "line1\nline2\nline3"


def test_code_as_string_empty():
    assert UMATHandler.code_as_string([]) == ""


def test_common_subexpressions_vector(umat_handler):
    """Test CSE with a column vector (vector branch)."""
    x = sym.Symbol("x")
    vec = sym.Matrix([x + 1, x * 2, x**2])
    result = umat_handler.common_subexpressions(vec, "v")
    assert len(result) > 0
    code = "\n".join(result)
    assert "v(" in code


@pytest.mark.slow
def test_cauchy_property(umat_handler):
    """Test Cauchy stress property generates symbolic expression."""
    cauchy = umat_handler.cauchy
    assert isinstance(cauchy, sym.Matrix)
    assert cauchy.shape == (6, 1)


@pytest.mark.slow
def test_tangent_property(umat_handler):
    """Test tangent property generates symbolic expression."""
    tangent = umat_handler.tangent
    assert isinstance(tangent, sym.Matrix)
    assert tangent.shape == (6, 6)


def test_generate_expression(umat_handler):
    """Test generate_expression wraps common_subexpressions."""
    x = sym.Symbol("x")
    mat = sym.Matrix([[x + 1, x], [x * 2, x**2]])
    code = umat_handler.generate_expression(mat, "test")
    assert len(code) > 0
    code_str = "\n".join(code)
    assert "test(" in code_str


def test_write_umat_code(umat_handler, tmp_path):
    """Test writing UMAT code to file."""
    filename = tmp_path / "test_umat.f90"
    umat_handler.write_umat_code("! props", "! stress", "! tangent", filename)
    assert filename.is_file()
    content = filename.read_text()
    assert "SUBROUTINE umat" in content
    assert "! props" in content
    assert "! stress" in content
    assert "! tangent" in content
    assert "NeoHooke" in content


@pytest.mark.slow
def test_cauchy_no_deformation(material):
    """Test Cauchy stress for no deformation."""
    # Setup
    f = sym.eye(3)
    # Exercise
    cauchy = material.cauchy_voigt(f)
    # Verify shape is (6, 1)
    assert cauchy.shape == (6, 1)
