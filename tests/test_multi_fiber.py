"""Tests for multi-fiber family support across the pipeline."""

from __future__ import annotations

import numpy as np
import pytest
from sympy import Symbol

from hyper_surrogate.mechanics.kinematics import Kinematics
from hyper_surrogate.mechanics.materials import (
    HolzapfelOgden,
    HolzapfelOgdenBiaxial,
    NeoHooke,
)


def _identity_batch(n: int = 1) -> np.ndarray:
    return np.tile(np.eye(3), (n, 1, 1))


def _uniaxial_F(stretch: float, axis: int = 0, n: int = 1) -> np.ndarray:
    F = np.tile(np.eye(3), (n, 1, 1))
    lateral = 1.0 / np.sqrt(stretch)
    F[:, 0, 0] = lateral
    F[:, 1, 1] = lateral
    F[:, 2, 2] = lateral
    F[:, axis, axis] = stretch
    return F


# ── Material properties ──────────────────────────────────────────


class TestHolzapfelOgdenBiaxialProperties:
    def test_default_fiber_directions(self):
        mat = HolzapfelOgdenBiaxial()
        assert mat.num_fiber_families == 2
        assert mat.is_anisotropic
        assert mat.input_dim == 7
        assert len(mat.fiber_directions) == 2
        # First fiber direction is +theta
        theta = np.radians(39.0)
        np.testing.assert_allclose(mat.fiber_directions[0], [np.cos(theta), np.sin(theta), 0.0])

    def test_custom_fiber_directions(self):
        dirs = [np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])]
        mat = HolzapfelOgdenBiaxial(fiber_directions=dirs)
        np.testing.assert_allclose(mat.fiber_directions[0], [1, 0, 0])
        np.testing.assert_allclose(mat.fiber_directions[1], [0, 1, 0])

    def test_rejects_wrong_count(self):
        with pytest.raises(ValueError, match="exactly 2"):
            HolzapfelOgdenBiaxial(fiber_directions=[np.array([1.0, 0.0, 0.0])])

    def test_backward_compat_fiber_direction(self):
        """fiber_direction returns first fiber for backward compat."""
        mat = HolzapfelOgdenBiaxial()
        assert mat.fiber_direction is not None
        np.testing.assert_allclose(mat.fiber_direction, mat.fiber_directions[0])

    def test_input_dim_isotropic(self):
        mat = NeoHooke()
        assert mat.input_dim == 3
        assert mat.num_fiber_families == 0

    def test_input_dim_single_fiber(self):
        mat = HolzapfelOgden()
        assert mat.input_dim == 5
        assert mat.num_fiber_families == 1


# ── Symbolic energy ──────────────────────────────────────────────


class TestBiaxialEnergy:
    def test_energy_at_identity_zero(self):
        mat = HolzapfelOgdenBiaxial()
        i1s, i2s, js = Symbol("I1b"), Symbol("I2b"), Symbol("J")
        fiber_invs = [(Symbol("I4_0"), Symbol("I5_0")), (Symbol("I4_1"), Symbol("I5_1"))]
        W = mat.sef_from_all_invariants(i1s, i2s, js, fiber_invs)

        from sympy import lambdify

        param_syms = list(mat._symbols.values())
        all_syms = [i1s, i2s, js, *[s for pair in fiber_invs for s in pair], *param_syms]
        fn = lambdify(all_syms, W, modules="numpy")
        # At identity: I1_bar=3, I2_bar=3, J=1, I4=1, I5=1
        val = fn(3.0, 3.0, 1.0, 1.0, 1.0, 1.0, 1.0, *list(mat._params.values()))
        np.testing.assert_allclose(val, 0.0, atol=1e-12)

    def test_energy_monotonic_under_stretch(self):
        """Energy should increase as fiber stretch increases."""
        mat = HolzapfelOgdenBiaxial()
        i1s, i2s, js = Symbol("I1b"), Symbol("I2b"), Symbol("J")
        fiber_invs = [(Symbol("I4_0"), Symbol("I5_0")), (Symbol("I4_1"), Symbol("I5_1"))]
        W = mat.sef_from_all_invariants(i1s, i2s, js, fiber_invs)

        from sympy import lambdify

        param_syms = list(mat._symbols.values())
        all_syms = [i1s, i2s, js, *[s for pair in fiber_invs for s in pair], *param_syms]
        fn = lambdify(all_syms, W, modules="numpy")
        pvals = list(mat._params.values())

        W_prev = 0.0
        for lam in [1.01, 1.05, 1.1, 1.2]:
            i4_val = lam**2  # fiber stretch squared
            W_val = float(fn(3.0, 3.0, 1.0, i4_val, i4_val, i4_val, i4_val, *pvals))
            assert W_val > W_prev
            W_prev = W_val

    def test_two_fiber_equals_manual_composition(self):
        """Biaxial model should match manual composition of two single-fiber models."""
        theta = np.radians(39.0)
        fiber1 = np.array([np.cos(theta), np.sin(theta), 0.0])
        fiber2 = np.array([np.cos(theta), -np.sin(theta), 0.0])
        params = {"a": 0.059, "b": 8.023, "af": 18.472, "bf": 16.026, "KBULK": 1000.0}

        mat_bi = HolzapfelOgdenBiaxial(parameters=params, fiber_directions=[fiber1, fiber2])
        mat1 = HolzapfelOgden(parameters=params, fiber_direction=fiber1)

        from sympy import lambdify

        i1s, i2s, js = Symbol("I1b"), Symbol("I2b"), Symbol("J")
        i4s_0, i5s_0 = Symbol("I4_0"), Symbol("I5_0")
        i4s_1, i5s_1 = Symbol("I4_1"), Symbol("I5_1")

        # Biaxial energy
        W_bi = mat_bi.sef_from_all_invariants(i1s, i2s, js, [(i4s_0, i5s_0), (i4s_1, i5s_1)])

        # Manual: shared iso+vol + fiber1 + fiber2
        W1 = mat1.sef_from_invariants(i1s, i2s, js, i4s_0, i5s_0)
        W2_full = mat1.sef_from_invariants(i1s, i2s, js, i4s_1, i5s_1)
        W2_iso = mat1.sef_from_invariants(i1s, i2s, js)
        W_manual = W1 + (W2_full - W2_iso)

        param_syms = list(mat_bi._symbols.values())
        all_syms = [i1s, i2s, js, i4s_0, i5s_0, i4s_1, i5s_1, *param_syms]
        fn_bi = lambdify(all_syms, W_bi, modules="numpy")
        fn_manual = lambdify(all_syms, W_manual, modules="numpy")

        pvals = list(mat_bi._params.values())
        test_vals = (3.5, 3.2, 1.02, 1.3, 1.5, 1.1, 1.2, *pvals)
        np.testing.assert_allclose(fn_bi(*test_vals), fn_manual(*test_vals), rtol=1e-10)


# ── Kinematics ───────────────────────────────────────────────────


class TestMultiFiberKinematics:
    def test_fiber_invariants_multi(self):
        C = _identity_batch(5)
        dirs = [np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])]
        result = Kinematics.fiber_invariants_multi(C, dirs)
        assert result.shape == (5, 4)  # 2 fibers * 2 invariants
        # At identity: I4=1, I5=1 for all fibers
        np.testing.assert_allclose(result, 1.0, atol=1e-12)

    def test_fiber_invariants_multi_under_stretch(self):
        F = _uniaxial_F(1.5, axis=0)
        C = Kinematics.right_cauchy_green(F)
        dirs = [np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])]
        result = Kinematics.fiber_invariants_multi(C, dirs)
        # Fiber 1 along stretch: I4 = 1.5^2 = 2.25
        assert result[0, 0] > 1.0
        # Fiber 2 transverse: I4 = 1/1.5 < 1
        assert result[0, 2] < 1.0


# ── Data pipeline ────────────────────────────────────────────────


class TestMultiFiberDataPipeline:
    def test_evaluate_energy_grad_invariants(self):
        mat = HolzapfelOgdenBiaxial()
        C = _identity_batch(3) * 1.01  # slightly perturbed
        grad = mat.evaluate_energy_grad_invariants(C)
        assert grad.shape == (3, 7)  # 3 isotropic + 2*2 fiber

    def test_create_datasets_biaxial(self):
        from hyper_surrogate.data.dataset import create_datasets

        mat = HolzapfelOgdenBiaxial()
        train_ds, val_ds, _in_norm, _out_norm = create_datasets(
            mat, n_samples=50, target_type="energy", val_fraction=0.2
        )
        # Inputs should be 7-dimensional
        assert train_ds.inputs.shape[1] == 7
        assert val_ds.inputs.shape[1] == 7


# ── Hybrid UMAT emitter ─────────────────────────────────────────


class TestMultiFiberHybridUMAT:
    @pytest.fixture()
    def exported_7d(self):
        torch = pytest.importorskip("torch")  # noqa: F841
        from hyper_surrogate.data.dataset import Normalizer
        from hyper_surrogate.export.weights import extract_weights
        from hyper_surrogate.models.mlp import MLP

        model = MLP(input_dim=7, output_dim=1, hidden_dims=[8, 8], activation="softplus")
        in_norm = Normalizer().fit(np.random.randn(50, 7))
        energy_norm = Normalizer().fit(np.random.randn(50, 1))
        return extract_weights(model, in_norm, energy_norm)

    def test_emit_two_fiber_produces_umat(self, exported_7d):
        from hyper_surrogate.export.fortran.hybrid import HybridUMATEmitter

        code = HybridUMATEmitter(exported_7d).emit()
        assert "MODULE nn_sef" in code
        assert "SUBROUTINE umat" in code

    def test_emit_two_fiber_invariant_desc(self, exported_7d):
        from hyper_surrogate.export.fortran.hybrid import HybridUMATEmitter

        code = HybridUMATEmitter(exported_7d).emit()
        assert "I1_bar, I2_bar, J, I4_1, I5_1, I4_2, I5_2" in code

    def test_emit_two_fiber_directions_from_props(self, exported_7d):
        from hyper_surrogate.export.fortran.hybrid import HybridUMATEmitter

        code = HybridUMATEmitter(exported_7d).emit()
        # Fiber 1 from props(1:3)
        assert "a0_1(1) = props(1)" in code
        # Fiber 2 from props(4:6)
        assert "a0_2(1) = props(4)" in code

    def test_emit_two_fiber_invariant_computation(self, exported_7d):
        from hyper_surrogate.export.fortran.hybrid import HybridUMATEmitter

        code = HybridUMATEmitter(exported_7d).emit()
        assert "I4_1 = I4_1 + a0_1(ii) * Ca0_1(ii)" in code
        assert "I4_2 = I4_2 + a0_2(ii) * Ca0_2(ii)" in code

    def test_emit_two_fiber_nn_input(self, exported_7d):
        from hyper_surrogate.export.fortran.hybrid import HybridUMATEmitter

        code = HybridUMATEmitter(exported_7d).emit()
        assert "nn_input(4) = I4_1" in code
        assert "nn_input(5) = I5_1" in code
        assert "nn_input(6) = I4_2" in code
        assert "nn_input(7) = I5_2" in code

    def test_emit_two_fiber_tangent(self, exported_7d):
        from hyper_surrogate.export.fortran.hybrid import HybridUMATEmitter

        code = HybridUMATEmitter(exported_7d).emit()
        assert "dIdC(3, 3, 7)" in code
        assert "DO mm = 1, 7" in code
        assert "DO nn = 1, 7" in code

    def test_emit_two_fiber_no_fd(self, exported_7d):
        from hyper_surrogate.export.fortran.hybrid import HybridUMATEmitter

        code = HybridUMATEmitter(exported_7d).emit()
        assert "eps_fd" not in code
        assert "d2W_dI2" in code

    def test_emit_single_fiber_still_works(self):
        """Single fiber (input_dim=5) should still work with generalized code."""
        torch = pytest.importorskip("torch")  # noqa: F841
        from hyper_surrogate.data.dataset import Normalizer
        from hyper_surrogate.export.fortran.hybrid import HybridUMATEmitter
        from hyper_surrogate.export.weights import extract_weights
        from hyper_surrogate.models.mlp import MLP

        model = MLP(input_dim=5, output_dim=1, hidden_dims=[8, 8], activation="softplus")
        in_norm = Normalizer().fit(np.random.randn(50, 5))
        energy_norm = Normalizer().fit(np.random.randn(50, 1))
        exported = extract_weights(model, in_norm, energy_norm)
        code = HybridUMATEmitter(exported).emit()
        assert "a0_1(1) = props(1)" in code
        assert "I4_1" in code
        assert "I5_1" in code
        assert "a0_2" not in code
