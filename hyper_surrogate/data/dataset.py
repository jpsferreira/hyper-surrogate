from __future__ import annotations

from typing import Any, Literal

import numpy as np

try:
    from torch.utils.data import Dataset

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    Dataset = object  # type: ignore[misc,assignment]


class Normalizer:
    """Standard (zero-mean, unit-variance) normalization with export support."""

    def __init__(self) -> None:
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None

    def fit(self, data: np.ndarray) -> Normalizer:
        self._mean = data.mean(axis=0)
        self._std = data.std(axis=0)
        self._std[self._std < 1e-12] = 1.0  # type: ignore[index,operator]  # avoid division by zero
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        return (data - self._mean) / self._std  # type: ignore[no-any-return]

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        return data * self._std + self._mean  # type: ignore[no-any-return]

    @property
    def params(self) -> dict[str, np.ndarray]:
        return {"mean": self._mean, "std": self._std}  # type: ignore[dict-item]


class MaterialDataset(Dataset):
    """Wraps (input, target) pairs for training."""

    def __init__(self, inputs: np.ndarray, targets: Any) -> None:
        self.inputs = inputs
        self.targets = targets

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int) -> tuple:
        x = self.inputs[idx]
        y = tuple(t[idx] for t in self.targets) if isinstance(self.targets, tuple) else self.targets[idx]
        return x, y


def _pk2_voigt(material: Any, c_batch: np.ndarray) -> np.ndarray:
    """Compute PK2 stress in Voigt notation."""
    pk2_batch = material.evaluate_pk2(c_batch)  # (N, 3, 3)
    return np.column_stack([
        pk2_batch[:, 0, 0],
        pk2_batch[:, 1, 1],
        pk2_batch[:, 2, 2],
        pk2_batch[:, 0, 1],
        pk2_batch[:, 0, 2],
        pk2_batch[:, 1, 2],
    ])


def create_datasets(  # noqa: C901  -- branchy by design; one branch per supported deformation_mode
    material: Any,
    n_samples: int,
    input_type: Literal["invariants", "cauchy_green"] = "invariants",
    target_type: Literal["energy", "pk2_voigt", "pk2_voigt+cmat_voigt"] = "pk2_voigt",
    val_fraction: float = 0.15,
    seed: int = 42,
    deformation_mode: Literal["combined", "combined_compressible", "uniaxial", "biaxial", "shear"] = "combined",
    stretch_range: tuple[float, float] | None = None,
    shear_range: tuple[float, float] | None = None,
    j_range: tuple[float, float] | None = None,
) -> tuple[MaterialDataset, MaterialDataset, Normalizer, Normalizer]:
    """Generate data, normalize, split, and wrap in datasets.

    deformation_mode selects the loading subspace. "uniaxial" mirrors the
    canonical experimental tensile-test setup and is recommended when a
    constrained architecture (ICNN, Polyconvex) struggles to fit the full
    combined deformation space.  "combined_compressible" extends "combined"
    with a random isotropic dilation so the resulting F has det F ≠ 1,
    supplying training signal for the J-direction of compressible
    hyperelastic surrogates.
    """
    from hyper_surrogate.data.deformation import DeformationGenerator
    from hyper_surrogate.mechanics.kinematics import Kinematics

    # Generate deformations
    gen = DeformationGenerator(seed=seed)
    sr = stretch_range
    hr = shear_range
    jr = j_range
    if deformation_mode == "uniaxial":
        F = gen.uniaxial(n_samples) if sr is None else gen.uniaxial(n_samples, stretch_range=sr)
    elif deformation_mode == "biaxial":
        F = gen.biaxial(n_samples) if sr is None else gen.biaxial(n_samples, stretch_range=sr)
    elif deformation_mode == "shear":
        F = gen.shear(n_samples) if hr is None else gen.shear(n_samples, shear_range=hr)
    elif deformation_mode == "combined_compressible":
        kw: dict[str, Any] = {}
        if sr is not None:
            kw["stretch_range"] = sr
        if hr is not None:
            kw["shear_range"] = hr
        if jr is not None:
            kw["j_range"] = jr
        F = gen.combined_compressible(n_samples, **kw)
    else:  # "combined"
        if sr is None and hr is None:
            F = gen.combined(n_samples)
        else:
            kw = {}
            if sr is not None:
                kw["stretch_range"] = sr
            if hr is not None:
                kw["shear_range"] = hr
            F = gen.combined(n_samples, **kw)
    C = Kinematics.right_cauchy_green(F)

    # Compute inputs
    if input_type == "invariants":
        i1 = Kinematics.isochoric_invariant1(C)
        i2 = Kinematics.isochoric_invariant2(C)
        j = np.sqrt(Kinematics.det_invariant(C))  # J = sqrt(det(C))
        if hasattr(material, "is_anisotropic") and material.is_anisotropic:
            num_fibers = getattr(material, "num_fiber_families", 1)
            if num_fibers > 1:
                fiber_invs = Kinematics.fiber_invariants_multi(C, material.fiber_directions)
                inputs = np.column_stack([i1, i2, j, fiber_invs])
            else:
                i4 = Kinematics.fiber_invariant4(C, material.fiber_direction)
                i5 = Kinematics.fiber_invariant5(C, material.fiber_direction)
                inputs = np.column_stack([i1, i2, j, i4, i5])
        else:
            inputs = np.column_stack([i1, i2, j])
    else:  # cauchy_green
        # 6 unique Voigt components: C11, C22, C33, C12, C13, C23
        inputs = np.column_stack([
            C[:, 0, 0],
            C[:, 1, 1],
            C[:, 2, 2],
            C[:, 0, 1],
            C[:, 0, 2],
            C[:, 1, 2],
        ])

    if target_type == "energy":
        energy = material.evaluate_energy(C)  # (N,)
        targets_raw = energy.reshape(-1, 1)

        # Stress target must match input dimensionality for EnergyStressLoss.
        # Invariant inputs (3D+) → dW/d(invariants); cauchy_green (6D) → PK2 Voigt.
        if input_type == "invariants":
            stress_target = material.evaluate_energy_grad_invariants(C)
        else:
            stress_target = _pk2_voigt(material, C)

        in_norm = Normalizer().fit(inputs)
        inputs_normed = in_norm.transform(inputs)

        # EnergyStressLoss computes dW/d(x_norm) via autograd. Since
        # x_norm = (I - mean) / std, the chain rule gives
        # dW/d(x_norm) = dW/dI * std. Scale the stress targets to match.
        stress_target_scaled = stress_target * in_norm.params["std"]

        # Split
        n_val = int(n_samples * val_fraction)
        rng = np.random.default_rng(seed)
        idx = rng.permutation(n_samples)
        train_idx, val_idx = idx[n_val:], idx[:n_val]

        # Energy normalizer is still returned for Fortran export (denormalization),
        # but targets stored raw. Trainer/loss handles raw values.
        energy_norm = Normalizer().fit(targets_raw)

        train_ds = MaterialDataset(
            inputs_normed[train_idx].astype(np.float32),
            (targets_raw[train_idx].astype(np.float32), stress_target_scaled[train_idx].astype(np.float32)),
        )
        val_ds = MaterialDataset(
            inputs_normed[val_idx].astype(np.float32),
            (targets_raw[val_idx].astype(np.float32), stress_target_scaled[val_idx].astype(np.float32)),
        )
        return train_ds, val_ds, in_norm, energy_norm

    elif target_type == "pk2_voigt":
        targets_raw = _pk2_voigt(material, C)
    elif target_type == "pk2_voigt+cmat_voigt":
        cmat_batch = material.evaluate_cmat(C)  # (N, 3, 3, 3, 3)
        # Extract 21 unique Voigt components (upper triangle of 6x6)
        ii1 = [0, 1, 2, 0, 0, 1]
        ii2 = [0, 1, 2, 1, 2, 2]
        cmat_voigt = np.zeros((n_samples, 21))
        k = 0
        for i in range(6):
            for j in range(i, 6):
                cmat_voigt[:, k] = 0.5 * (
                    cmat_batch[:, ii1[i], ii2[i], ii1[j], ii2[j]] + cmat_batch[:, ii1[i], ii2[i], ii2[j], ii1[j]]
                )
                k += 1
        targets_raw = np.column_stack([_pk2_voigt(material, C), cmat_voigt])
    else:
        msg = f"Unknown target_type: {target_type}"
        raise ValueError(msg)

    # Normalize
    in_norm = Normalizer().fit(inputs)
    out_norm = Normalizer().fit(targets_raw)
    inputs_normed = in_norm.transform(inputs)
    targets_normed = out_norm.transform(targets_raw)

    # Split
    n_val = int(n_samples * val_fraction)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n_samples)
    train_idx, val_idx = idx[n_val:], idx[:n_val]

    train_ds = MaterialDataset(
        inputs_normed[train_idx].astype(np.float32), targets_normed[train_idx].astype(np.float32)
    )
    val_ds = MaterialDataset(inputs_normed[val_idx].astype(np.float32), targets_normed[val_idx].astype(np.float32))

    return train_ds, val_ds, in_norm, out_norm
