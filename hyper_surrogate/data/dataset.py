from __future__ import annotations

from typing import Any, Literal

import numpy as np

try:
    from torch.utils.data import Dataset

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    Dataset = object  # type: ignore[assignment,misc]


class Normalizer:
    """Standard (zero-mean, unit-variance) normalization with export support."""

    def __init__(self) -> None:
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None

    def fit(self, data: np.ndarray) -> Normalizer:
        self._mean = data.mean(axis=0)
        self._std = data.std(axis=0)
        self._std[self._std < 1e-12] = 1.0  # avoid division by zero
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        return (data - self._mean) / self._std

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        return data * self._std + self._mean

    @property
    def params(self) -> dict[str, np.ndarray]:
        return {"mean": self._mean, "std": self._std}


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


def create_datasets(
    material: Any,
    n_samples: int,
    input_type: Literal["invariants", "cauchy_green"] = "invariants",
    target_type: Literal["energy", "pk2_voigt", "pk2_voigt+cmat_voigt"] = "pk2_voigt",
    val_fraction: float = 0.15,
    seed: int = 42,
) -> tuple[MaterialDataset, MaterialDataset, Normalizer, Normalizer]:
    """Generate data, normalize, split, and wrap in datasets."""
    from hyper_surrogate.data.deformation import DeformationGenerator
    from hyper_surrogate.mechanics.kinematics import Kinematics

    # Generate deformations
    gen = DeformationGenerator(seed=seed)
    F = gen.combined(n_samples)
    C = Kinematics.right_cauchy_green(F)

    # Compute inputs
    if input_type == "invariants":
        i1 = Kinematics.isochoric_invariant1(C)
        i2 = Kinematics.isochoric_invariant2(C)
        j = np.sqrt(Kinematics.det_invariant(C))  # J = sqrt(det(C))
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

    # Compute targets
    pk2_batch = material.evaluate_pk2(C)  # (N, 3, 3)
    pk2_voigt = np.column_stack([
        pk2_batch[:, 0, 0],
        pk2_batch[:, 1, 1],
        pk2_batch[:, 2, 2],
        pk2_batch[:, 0, 1],
        pk2_batch[:, 0, 2],
        pk2_batch[:, 1, 2],
    ])

    if target_type == "energy":
        energy = material.evaluate_energy(C)  # (N,)
        targets_raw = energy.reshape(-1, 1)
        # For energy loss, we also need stress — store as tuple.
        # IMPORTANT: Neither energy nor stress is normalized here.
        # EnergyStressLoss operates entirely in raw physical space.
        # Only inputs are normalized (the NN sees normalized invariants).
        in_norm = Normalizer().fit(inputs)
        inputs_normed = in_norm.transform(inputs)

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
            (targets_raw[train_idx].astype(np.float32), pk2_voigt[train_idx].astype(np.float32)),
        )
        val_ds = MaterialDataset(
            inputs_normed[val_idx].astype(np.float32),
            (targets_raw[val_idx].astype(np.float32), pk2_voigt[val_idx].astype(np.float32)),
        )
        return train_ds, val_ds, in_norm, energy_norm

    elif target_type == "pk2_voigt":
        targets_raw = pk2_voigt
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
        targets_raw = np.column_stack([pk2_voigt, cmat_voigt])
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
