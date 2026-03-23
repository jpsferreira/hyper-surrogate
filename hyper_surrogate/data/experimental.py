"""Experimental data loading and preprocessing for biomechanical testing data."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np


@dataclass
class ExperimentalData:
    """Load and preprocess experimental tissue testing data.

    Stores stretch/stress pairs from biomechanical experiments (uniaxial, biaxial).
    Can convert to deformation gradients and integrate with the surrogate pipeline.
    """

    stretch: np.ndarray  # (N, n_components)
    stress: np.ndarray  # (N, n_components)
    test_type: Literal["uniaxial", "biaxial"] = "uniaxial"
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        test_type: Literal["uniaxial", "biaxial"] = "uniaxial",
        stretch_cols: list[str] | None = None,
        stress_cols: list[str] | None = None,
        delimiter: str = ",",
    ) -> ExperimentalData:
        """Load experimental data from a CSV file.

        Args:
            path: Path to CSV file.
            test_type: Type of test ("uniaxial" or "biaxial").
            stretch_cols: Column names for stretch data. Defaults to auto-detect.
            stress_cols: Column names for stress data. Defaults to auto-detect.
            delimiter: CSV delimiter.

        Returns:
            ExperimentalData instance.
        """
        import csv

        path = Path(path)
        with path.open() as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            rows = list(reader)

        if not rows:
            msg = f"Empty CSV file: {path}"
            raise ValueError(msg)

        headers = list(rows[0].keys())

        if stretch_cols is None:
            stretch_cols = [h for h in headers if "stretch" in h.lower() or "lambda" in h.lower()]
        if stress_cols is None:
            stress_cols = [h for h in headers if "stress" in h.lower() or "sigma" in h.lower()]

        if not stretch_cols or not stress_cols:
            msg = f"Could not auto-detect stretch/stress columns from headers: {headers}"
            raise ValueError(msg)

        stretch = np.array([[float(row[c]) for c in stretch_cols] for row in rows])
        stress = np.array([[float(row[c]) for c in stress_cols] for row in rows])

        return cls(
            stretch=stretch,
            stress=stress,
            test_type=test_type,
            metadata={"path": str(path), "stretch_cols": stretch_cols, "stress_cols": stress_cols},
        )

    @classmethod
    def from_uniaxial(cls, stretch: np.ndarray, stress: np.ndarray) -> ExperimentalData:
        """Create from uniaxial test arrays.

        Args:
            stretch: 1D array of stretch ratios (N,).
            stress: 1D array of Cauchy stress values (N,).
        """
        stretch = np.asarray(stretch).reshape(-1, 1)
        stress = np.asarray(stress).reshape(-1, 1)
        return cls(stretch=stretch, stress=stress, test_type="uniaxial")

    @classmethod
    def from_biaxial(
        cls,
        stretch_11: np.ndarray,
        stretch_22: np.ndarray,
        stress_11: np.ndarray,
        stress_22: np.ndarray,
    ) -> ExperimentalData:
        """Create from planar biaxial test data.

        Args:
            stretch_11: Stretch in 1-direction (N,).
            stretch_22: Stretch in 2-direction (N,).
            stress_11: Cauchy stress in 1-direction (N,).
            stress_22: Cauchy stress in 2-direction (N,).
        """
        stretch = np.column_stack([np.asarray(stretch_11), np.asarray(stretch_22)])
        stress = np.column_stack([np.asarray(stress_11), np.asarray(stress_22)])
        return cls(stretch=stretch, stress=stress, test_type="biaxial")

    @classmethod
    def load_reference(cls, name: str) -> ExperimentalData:
        """Load a built-in reference dataset.

        Args:
            name: Dataset name (e.g. "treloar").
        """
        ref_dir = Path(__file__).parent / "reference"
        path = ref_dir / f"{name}.csv"
        if not path.exists():
            available = [p.stem for p in ref_dir.glob("*.csv")] if ref_dir.exists() else []
            msg = f"Unknown reference dataset: {name!r}. Available: {available}"
            raise ValueError(msg)
        return cls.from_csv(path, test_type="uniaxial")

    def to_deformation_gradients(self) -> np.ndarray:
        """Convert stretch data to deformation gradients F (N, 3, 3).

        For uniaxial: F = diag(lambda, 1/sqrt(lambda), 1/sqrt(lambda)).
        For biaxial: F = diag(lambda1, lambda2, 1/(lambda1*lambda2)).
        """
        n = len(self.stretch)
        F = np.zeros((n, 3, 3))

        if self.test_type == "uniaxial":
            lam = self.stretch[:, 0]
            lat = 1.0 / np.sqrt(lam)
            F[:, 0, 0] = lam
            F[:, 1, 1] = lat
            F[:, 2, 2] = lat
        elif self.test_type == "biaxial":
            lam1 = self.stretch[:, 0]
            lam2 = self.stretch[:, 1]
            F[:, 0, 0] = lam1
            F[:, 1, 1] = lam2
            F[:, 2, 2] = 1.0 / (lam1 * lam2)
        else:
            msg = f"Unsupported test type: {self.test_type}"
            raise ValueError(msg)

        return F

    def to_invariants(self) -> np.ndarray:
        """Compute invariants (I1_bar, I2_bar, J) from stretch data. Returns (N, 3)."""
        from hyper_surrogate.mechanics.kinematics import Kinematics

        F = self.to_deformation_gradients()
        C = Kinematics.right_cauchy_green(F)
        i1 = Kinematics.isochoric_invariant1(C)
        i2 = Kinematics.isochoric_invariant2(C)
        j = np.sqrt(Kinematics.det_invariant(C))
        return np.column_stack([i1, i2, j])
