import logging

import numpy as np
import pytest
import sympy as sym

from hyper_surrogate.materials import Material


@pytest.fixture
def material():
    return Material([])


def test_material_dummy_sef(material):
    assert material.sef == sym.Symbol("sef")


def test_pk2_symbol(material):
    logging.info(material.pk2_symb)
    assert material.pk2_symb == material.pk2_tensor(material.sef)
    assert material.pk2_symb == sym.Matrix([[0, 0, 0], [0, 0, 0], [0, 0, 0]])


def test_cmat_symbol(material):
    assert material.cmat_symb == material.cmat_tensor(material.pk2_symb)
    assert material.cmat_symb == sym.ImmutableDenseNDimArray(np.zeros((3, 3, 3, 3)))
