from unittest.mock import patch

import numpy as np

from hyper_surrogate.data.dataset import MaterialDataset, Normalizer, create_datasets


class TestNormalizer:
    def test_fit_transform(self):
        data = np.random.randn(100, 3)
        norm = Normalizer().fit(data)
        transformed = norm.transform(data)
        np.testing.assert_allclose(transformed.mean(axis=0), 0.0, atol=1e-10)
        np.testing.assert_allclose(transformed.std(axis=0), 1.0, atol=1e-1)

    def test_inverse_transform(self):
        data = np.random.randn(100, 3) * 5 + 10
        norm = Normalizer().fit(data)
        roundtrip = norm.inverse_transform(norm.transform(data))
        np.testing.assert_allclose(roundtrip, data, atol=1e-10)

    def test_params(self):
        data = np.random.randn(50, 6)
        norm = Normalizer().fit(data)
        params = norm.params
        assert "mean" in params
        assert "std" in params
        assert params["mean"].shape == (6,)
        assert params["std"].shape == (6,)


class TestMaterialDataset:
    def test_len(self):
        inputs = np.random.randn(100, 3)
        targets = np.random.randn(100, 6)
        ds = MaterialDataset(inputs, targets)
        assert len(ds) == 100

    def test_getitem(self):
        inputs = np.random.randn(100, 3)
        targets = np.random.randn(100, 6)
        ds = MaterialDataset(inputs, targets)
        x, y = ds[0]
        assert x.shape == (3,)
        assert y.shape == (6,)


class TestCreateDatasets:
    def test_create_invariants_pk2(self):
        from hyper_surrogate.mechanics.materials import NeoHooke

        material = NeoHooke({"C10": 0.5, "KBULK": 1000.0})
        train_ds, val_ds, in_norm, out_norm = create_datasets(
            material,
            n_samples=100,
            input_type="invariants",
            target_type="pk2_voigt",
        )
        assert len(train_ds) + len(val_ds) == 100
        x, y = train_ds[0]
        assert x.shape == (3,)  # I1_bar, I2_bar, J
        assert y.shape == (6,)  # PK2 Voigt

    def test_create_energy(self):
        from hyper_surrogate.mechanics.materials import NeoHooke

        material = NeoHooke({"C10": 0.5, "KBULK": 1000.0})

        # Mock the slow SymPy-based energy gradient computation
        def fake_energy_grad(c_batch: np.ndarray) -> np.ndarray:
            n = len(c_batch)
            return np.random.default_rng(0).standard_normal((n, 3))

        with patch.object(material, "evaluate_energy_grad_invariants", side_effect=fake_energy_grad):
            train_ds, val_ds, in_norm, out_norm = create_datasets(
                material,
                n_samples=100,
                input_type="invariants",
                target_type="energy",
            )
        x, y = train_ds[0]
        assert x.shape == (3,)
        # energy target is (energy_scalar, dW_dI_3) = tuple of 2
        assert isinstance(y, tuple)
        assert y[1].shape == (3,)  # dW/d(invariants) matches input dim

    def test_combined_compressible_inputs_have_J_variation(self):
        """`deformation_mode='combined_compressible'` produces invariant
        inputs whose J component genuinely varies (closes the J=1
        training-data hole)."""
        from hyper_surrogate.mechanics.materials import NeoHooke

        material = NeoHooke({"C10": 0.5, "KBULK": 1000.0})
        train_ds, _val, in_norm, _out = create_datasets(
            material,
            n_samples=500,
            input_type="invariants",
            target_type="pk2_voigt",
            deformation_mode="combined_compressible",
            j_range=(0.85, 1.15),
            seed=0,
        )

        # Inputs are normalised in storage; undo to inspect raw J.
        all_inputs = np.array([train_ds[i][0] for i in range(len(train_ds))])
        raw = all_inputs * in_norm.params["std"] + in_norm.params["mean"]
        J = raw[:, 2]
        # The combined deformation product means actual det F values can
        # land outside [0.85, 1.15] (volumetric_dilation alone respects
        # the range; combined() carries no det-F variation, so the
        # product is bounded by the dilation factor).
        assert J.min() >= 0.85 - 1e-6
        assert J.max() <= 1.15 + 1e-6
        assert J.std() > 0.04, "J variance must be substantially > 0 (not a J=1 grid)"

    def test_combined_compressible_defaults_when_j_range_omitted(self):
        """Omitting j_range falls back to the generator's default
        (currently `[0.85, 1.15]`); test by exercising the code path
        with no kwargs."""
        from hyper_surrogate.mechanics.materials import NeoHooke

        material = NeoHooke({"C10": 0.5, "KBULK": 1000.0})
        # Just exercise the no-j_range branch; no assertion on the
        # specific default beyond "the call succeeds and produces J
        # variation".
        train_ds, _val, in_norm, _out = create_datasets(
            material,
            n_samples=200,
            input_type="invariants",
            target_type="pk2_voigt",
            deformation_mode="combined_compressible",
            seed=0,
        )
        raw = np.array([train_ds[i][0] for i in range(len(train_ds))]) * in_norm.params["std"] + in_norm.params["mean"]
        assert raw[:, 2].std() > 0.0
