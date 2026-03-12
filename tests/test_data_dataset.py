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
        train_ds, val_ds, in_norm, out_norm = create_datasets(
            material,
            n_samples=100,
            input_type="invariants",
            target_type="energy",
        )
        x, y = train_ds[0]
        assert x.shape == (3,)
        # energy target is (energy_scalar, pk2_voigt_6) = tuple of 2
        assert isinstance(y, tuple)
        assert y[1].shape == (6,)
