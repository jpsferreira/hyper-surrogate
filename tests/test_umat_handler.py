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


def test_cauchy_no_deformation(material):
    """Test Cauchy stress for no deformation."""
    # Setup
    f = sym.eye(3)
    # Exercise
    cauchy = material.cauchy_voigt(f)
    # Verify shape is (6, 1)
    assert cauchy.shape == (6, 1)
