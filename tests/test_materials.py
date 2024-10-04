import logging

import numpy as np
import pytest
import sympy as sym

from hyper_surrogate.materials import Material, MooneyRivlin, NeoHooke


@pytest.fixture
def material():
    return Material(["param1", "param2"])


def test_material_dummy_sef(material):
    assert material.sef == sym.Symbol("sef")


def test_pk2_symbol(material):
    logging.info(material.pk2_symb)
    assert material.pk2_symb == material.pk2_tensor(material.sef)
    assert material.pk2_symb == sym.Matrix([[0, 0, 0], [0, 0, 0], [0, 0, 0]])


def test_cmat_symbol(material):
    # shape
    assert material.cmat_symb.shape == (3, 3, 3, 3)
    assert material.cmat_symb.shape == material.cmat_tensor(material.pk2_symb).shape
    assert material.cmat_symb == material.cmat_tensor(material.pk2_symb)
    assert material.cmat_symb == sym.MutableDenseNDimArray(np.zeros((3, 3, 3, 3), dtype=int))


def test_neohooke_sef():
    neohooke = NeoHooke()
    assert neohooke.sef == (neohooke.invariant1 - 3) * sym.Symbol("C10")


def test_mooneyrivlin_sef():
    mooneyrivlin = MooneyRivlin()
    assert mooneyrivlin.sef == (mooneyrivlin.invariant1 - 3) * sym.Symbol("C10") + (
        mooneyrivlin.invariant2 - 3
    ) * sym.Symbol("C01")
