from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import ClassVar

import matplotlib
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from hyper_surrogate.mechanics.kinematics import Kinematics

matplotlib.use("Agg")
plt.set_loglevel("error")
plt.rcParams["figure.max_open_warning"] = -1


class Reporter:
    """Generate a PDF report with deformation diagnostics.

    Accepts a batch of (N, 3, 3) tensors — either deformation gradients **F**
    or right Cauchy-Green tensors **C** — and produces histograms of key
    continuum-mechanics quantities (eigenvalues, determinants, invariants,
    principal stretches).

    Args:
        tensor: Array of shape ``(N, 3, 3)``.
        tensor_type: ``"C"`` (right Cauchy-Green, default) or ``"F"``
            (deformation gradient).

    Example::

        from hyper_surrogate.reporting.reporter import Reporter

        reporter = Reporter(C)  # (N, 3, 3)
        reporter.fig_eigenvalues()
        reporter.fig_determinants()
        reporter.fig_invariants()
        reporter.fig_principal_stretches()
        reporter.generate_report("report/")
    """

    LAYOUT: ClassVar[list[str]] = ["standalone", "combined"]

    FIG_SIZE: ClassVar[tuple[float, float]] = (8.27, 11.69)

    REPORT_FIGURES: ClassVar[list[str]] = [
        "fig_eigenvalues",
        "fig_determinants",
        "fig_invariants",
        "fig_principal_stretches",
        "fig_volume_change",
    ]

    def __init__(
        self,
        tensor: np.ndarray,
        tensor_type: str = "C",
    ) -> None:
        if tensor.ndim != 3 or tensor.shape[1:] != (3, 3):
            msg = f"Expected tensor of shape (N, 3, 3), got {tensor.shape}"
            raise ValueError(msg)

        self.tensor_type = tensor_type.upper()
        if self.tensor_type == "F":
            self.F = tensor
            self.C = Kinematics.right_cauchy_green(tensor)
        elif self.tensor_type == "C":
            self.C = tensor
            self.F = None
        else:
            msg = f"tensor_type must be 'C' or 'F', got {tensor_type!r}"
            raise ValueError(msg)

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def n_samples(self) -> int:
        """Number of samples in the batch."""
        return int(self.C.shape[0])

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def basic_statistics(self) -> dict[str, dict[str, float]]:
        """Per-quantity summary statistics.

        Returns a dict keyed by quantity name, each containing
        ``mean``, ``std``, ``min``, ``max``.
        """
        quantities: dict[str, np.ndarray] = {
            "det(C)": Kinematics.det_invariant(self.C),
            "I1_bar": Kinematics.isochoric_invariant1(self.C),
            "I2_bar": Kinematics.isochoric_invariant2(self.C),
            "J": np.sqrt(Kinematics.det_invariant(self.C)),
        }
        stats: dict[str, dict[str, float]] = {}
        for name, values in quantities.items():
            stats[name] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
        return stats

    # ------------------------------------------------------------------
    # Individual figure methods
    # ------------------------------------------------------------------

    def fig_eigenvalues(self) -> list[matplotlib.figure.Figure]:
        """Histogram of eigenvalues of C."""
        eigenvalues = np.linalg.eigvalsh(self.C).real
        fig, axes = plt.subplots(1, 3, figsize=self.FIG_SIZE)
        labels = [r"$\lambda_1^2$", r"$\lambda_2^2$", r"$\lambda_3^2$"]
        for i, ax in enumerate(axes):
            ax.hist(eigenvalues[:, i], bins="auto", alpha=0.75, edgecolor="black")
            ax.set_title(f"Eigenvalue {labels[i]}")
            ax.set_xlabel("Value")
            ax.set_ylabel("Frequency")
        fig.suptitle("Eigenvalues of C", fontsize=14)
        fig.tight_layout()
        return [fig]

    def fig_determinants(self) -> list[matplotlib.figure.Figure]:
        """Histogram of det(C)."""
        determinants = Kinematics.det_invariant(self.C)
        fig, ax = plt.subplots(1, 1, figsize=self.FIG_SIZE)
        ax.hist(determinants, bins="auto", alpha=0.75, edgecolor="black")
        ax.set_title("Determinant of C")
        ax.set_xlabel("det(C)")
        ax.set_ylabel("Frequency")
        fig.tight_layout()
        return [fig]

    def fig_invariants(self) -> list[matplotlib.figure.Figure]:
        """Histograms of isochoric invariants I1_bar, I2_bar, and J."""
        i1 = Kinematics.isochoric_invariant1(self.C)
        i2 = Kinematics.isochoric_invariant2(self.C)
        j = np.sqrt(Kinematics.det_invariant(self.C))

        fig, axes = plt.subplots(1, 3, figsize=self.FIG_SIZE)
        for ax, data, label in zip(
            axes,
            [i1, i2, j],
            [r"$\bar{I}_1$", r"$\bar{I}_2$", r"$J$"],
            strict=False,
        ):
            ax.hist(data, bins="auto", alpha=0.75, edgecolor="black")
            ax.set_title(label)
            ax.set_xlabel("Value")
            ax.set_ylabel("Frequency")
        fig.suptitle("Isochoric Invariants", fontsize=14)
        fig.tight_layout()
        return [fig]

    def fig_principal_stretches(self) -> list[matplotlib.figure.Figure]:
        """Histogram of principal stretches (sorted)."""
        eigenvalues = np.sort(np.linalg.eigvalsh(self.C).real, axis=1)
        stretches = np.sqrt(np.maximum(eigenvalues, 0.0))

        fig, axes = plt.subplots(1, 3, figsize=self.FIG_SIZE)
        for i, ax in enumerate(axes):
            ax.hist(stretches[:, i], bins="auto", alpha=0.75, edgecolor="black")
            ax.set_title(rf"$\lambda_{{{i + 1}}}$")
            ax.set_xlabel("Stretch")
            ax.set_ylabel("Frequency")
        fig.suptitle("Principal Stretches", fontsize=14)
        fig.tight_layout()
        return [fig]

    def fig_volume_change(self) -> list[matplotlib.figure.Figure]:
        """Histogram of volume ratio J = sqrt(det(C))."""
        j = np.sqrt(Kinematics.det_invariant(self.C))
        fig, ax = plt.subplots(1, 1, figsize=self.FIG_SIZE)
        ax.hist(j, bins="auto", alpha=0.75, edgecolor="black")
        ax.axvline(1.0, color="red", linestyle="--", linewidth=1, label="J = 1")
        ax.set_title("Volume Ratio J")
        ax.set_xlabel("J")
        ax.set_ylabel("Frequency")
        ax.legend()
        fig.tight_layout()
        return [fig]

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------

    def generate_figures(self) -> list[matplotlib.figure.Figure]:
        """Generate all report figures."""
        fig_list: list[matplotlib.figure.Figure] = []
        for name in self.REPORT_FIGURES:
            func = getattr(self, name)
            fig_list.extend(func())
        return fig_list

    def generate_report(self, save_dir: str | Path, layout: str = "combined") -> None:
        """Create a PDF report.

        Args:
            save_dir: Directory to write the report into (created if needed).
            layout: ``"combined"`` (single PDF) or ``"standalone"`` (one PDF
                per figure).
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        metadata = {
            "Title": "Deformation Report",
            "Subject": f"Batch of {self.n_samples} tensors",
            "CreationDate": datetime.today(),
        }

        fig_list = self.generate_figures()

        if layout == "combined":
            with PdfPages(save_path / "report.pdf", metadata=metadata) as pp:
                for fig in fig_list:
                    fig.savefig(pp, format="pdf", bbox_inches="tight")
        else:
            for fig in fig_list:
                title = fig.axes[0].get_title() or "figure"
                safe_name = title.replace(" ", "_").replace("/", "_")
                with PdfPages(save_path / f"{safe_name}.pdf", metadata=metadata) as pp:
                    fig.savefig(pp, format="pdf", bbox_inches="tight")
        plt.close("all")

    # Keep backward compat alias
    create_report = generate_report
