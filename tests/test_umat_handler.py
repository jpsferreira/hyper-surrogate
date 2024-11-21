from pathlib import Path

import pytest
import sympy as sym

from hyper_surrogate.materials import Material
from hyper_surrogate.umat_handler import UMATHandler


@pytest.fixture
def material():
    return Material(["param1", "param2"])


@pytest.fixture
def umat_handler(material):
    return UMATHandler(material_model=material)


def test_initialize_with_valid_material(umat_handler, material):
    # Assert that the material attribute is set correctly
    assert umat_handler.material == material
    # Assert that the sigma_code attribute is None
    assert umat_handler.sigma_code is None
    # Assert that the smat_code attribute is None
    assert umat_handler.smat_code is None
    # Assert that the material attribute is an instance of Material
    assert isinstance(umat_handler.material, Material)


def test_generate_umat_code(tmp_path):
    # Setup
    material_model = Material(["param1", "param2"])
    umat_handler = UMATHandler(material_model)
    
    # Create a temporary file in the temporary directory
    filename = tmp_path / "umat.f"
    # Exercise
    umat_handler.generate(filename)
    # Verify
    assert filename.is_file()
    # assert that the file is not empty
    assert filename.stat().st_size > 0
    # assert that the file is a Fortran file
    assert filename.suffix == ".f"


def test_common_subexpressions_with_empty_tensor(umat_handler):
    # Create an empty tensor (0x0 matrix)
    empty_tensor = sym.Matrix(0, 0, [])
    # Assert that the result is an empty list, as there are no expressions to process
    assert umat_handler.common_subexpressions(empty_tensor, "var") == []


def test_cauchy_no_deformation(material):
    # Setup
    f = sym.eye(3)
    # Exercise
    cauchy = material.cauchy(f)
    print(cauchy)
    # Verify
    assert cauchy == sym.Matrix([[0], [0], [0], [0], [0], [0]])
