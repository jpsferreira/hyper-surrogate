import numpy as np

from hyper_surrogate.mechanics.materials import MooneyRivlin, NeoHooke


def test_neohooke_has_handler():
    mat = NeoHooke({"C10": 0.5, "KBULK": 1000.0})
    assert hasattr(mat, "_handler")


def test_neohooke_sef_is_expr():
    from sympy import Expr

    mat = NeoHooke({"C10": 0.5, "KBULK": 1000.0})
    assert isinstance(mat.sef, Expr)


def test_neohooke_pk2_func_callable():
    mat = NeoHooke({"C10": 0.5, "KBULK": 1000.0})
    func = mat.pk2_func
    assert callable(func)


def test_neohooke_evaluate_pk2_identity():
    mat = NeoHooke({"C10": 0.5, "KBULK": 1000.0})
    c_identity = np.eye(3).reshape(1, 3, 3)
    result = mat.evaluate_pk2(c_identity)
    assert result.shape == (1, 3, 3)
    # At identity, PK2 should be zero (no deformation)
    np.testing.assert_allclose(result[0], np.zeros((3, 3)), atol=1e-10)


def test_mooneyrivlin_params():
    mat = MooneyRivlin({"C10": 0.3, "C01": 0.2, "KBULK": 1000.0})
    assert mat._params == {"C10": 0.3, "C01": 0.2, "KBULK": 1000.0}
