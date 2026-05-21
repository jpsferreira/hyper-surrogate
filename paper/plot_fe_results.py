"""Plot FE validation results: analytical (closed-form) vs Abaqus UMATs.

Reads:
    results/fe_validation/validation_reference.json
        — analytical closed-form reference, three modes (uniaxial, biaxial, shear).
    results/fe_validation/abaqus_bundle/abaqus_<mode>_<umat>.csv
        — Abaqus output, 5 modes (compression + extension per uni/biaxial, plus shear)
          x 2 UMATs (analytical, hybrid_polyconvex).

Each subplot carries a small 3D wireframe inset that shows the
representative deformed cube for that loading mode.

Usage:
    uv run python paper/plot_fe_results.py
    uv run python paper/plot_fe_results.py --format png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Line3DCollection

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
})

ROOT = Path(__file__).resolve().parent
FE_DIR = ROOT / "results" / "fe_validation"
BUNDLE = FE_DIR / "abaqus_bundle"
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)

PANELS = ["uniaxial", "biaxial", "shear"]
UMATS = [
    ("analytical", "UMAT (analytical)", "#0072B2", "o"),
    ("hybrid_polyconvex", "UMAT (polyconvex)", "#D55E00", "s"),
]

# Representative deformation states used for the 3-D cube insets.
# Each mode also picks its own view angle so the relevant motion is most
# visible: the default 3/4 angle works for stretch modes, but shear is
# best seen with the y axis pointing into the page.
INSET_F = {
    "uniaxial": np.diag([1.5, 1.5**-0.5, 1.5**-0.5]),
    "biaxial": np.diag([1.4, 1.4, 1.4**-2]),
    "shear": np.array([[1.0, 0.5, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
}
INSET_VIEW = {
    "uniaxial": (18, -58),
    "biaxial": (18, -58),
    "shear": (20, -75),
}


def _load_csv(path: Path) -> dict[str, np.ndarray] | None:
    if not path.exists():
        return None
    data = np.genfromtxt(path, delimiter=",", names=True)
    if data.size < 2:
        return None
    return {name: np.atleast_1d(data[name]) for name in data.dtype.names}


def _stitch(panel: str, umat: str) -> dict[str, np.ndarray] | None:
    """Concatenate the compression and extension halves into a single trajectory."""
    if panel == "shear":
        return _load_csv(BUNDLE / f"abaqus_shear_{umat}.csv")

    pieces_dict = []
    for half in ("compression", "extension"):
        csv = _load_csv(BUNDLE / f"abaqus_{panel}_{half}_{umat}.csv")
        if csv is None:
            continue
        pieces_dict.append(csv)
    if not pieces_dict:
        return None
    columns = pieces_dict[0].keys()
    merged: dict[str, np.ndarray] = {}
    for col in columns:
        merged[col] = np.concatenate([p[col] for p in pieces_dict])
    order = np.argsort(merged["stretch"])
    for col in columns:
        merged[col] = merged[col][order]
    return merged


def _cube_edges() -> np.ndarray:
    """Return the 12 unit-cube edges as (12, 2, 3) line segments."""
    v = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1],
        ],
        dtype=float,
    )
    idx = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]
    return np.array([[v[i], v[j]] for i, j in idx])


def _add_cube_inset(
    ax: plt.Axes,
    F: np.ndarray,
    view: tuple[float, float],
    *,
    loc: tuple[float, float] = (0.04, 0.58),
) -> None:
    """Add a 3D wireframe inset showing the deformed unit cube.

    The reference (undeformed) cube is drawn dashed; the deformed cube is
    drawn solid. ``view`` is ``(elev, azim)`` so each loading mode can
    pick the angle that best surfaces its kinematic motion.
    """
    bbox = (loc[0], loc[1], 0.34, 0.40)
    ax_in = ax.inset_axes(bbox, projection="3d")
    ax_in.set_box_aspect((1, 1, 1))

    ref_edges = _cube_edges() - 0.5
    def_edges = np.einsum("ij,nej->nei", F, ref_edges)

    ax_in.add_collection3d(Line3DCollection(ref_edges, colors="#9aa0a6", linewidths=0.8, linestyles="--"))
    ax_in.add_collection3d(Line3DCollection(def_edges, colors="#222222", linewidths=1.2))

    all_pts = np.vstack([ref_edges.reshape(-1, 3), def_edges.reshape(-1, 3)])
    lo, hi = all_pts.min(0), all_pts.max(0)
    pad = 0.15 * (hi - lo).max()
    lo -= pad
    hi += pad
    ax_in.set_xlim(lo[0], hi[0])
    ax_in.set_ylim(lo[1], hi[1])
    ax_in.set_zlim(lo[2], hi[2])
    ax_in.view_init(elev=view[0], azim=view[1])
    ax_in.set_xticks([])
    ax_in.set_yticks([])
    ax_in.set_zticks([])
    ax_in.set_axis_off()
    ax_in.set_facecolor((1, 1, 1, 0.0))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--format", default="pdf", choices=["pdf", "svg", "png"])
    args = parser.parse_args()

    ref_path = FE_DIR / "validation_reference.json"
    if not ref_path.exists():
        print("No reference data found. Run `paper/fe_validation.py` first.")
        return
    with open(ref_path) as f:
        ref = json.load(f)

    mat_name = ref["material"]
    print(f"Plotting FE validation for {mat_name}")

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2))

    for ax, panel in zip(axes, PANELS, strict=False):
        test_ref = ref["tests"][panel]
        stretch_ref = np.array(test_ref["stretch"])
        cauchy_ref = np.array(test_ref["cauchy_11"])

        ax.plot(
            stretch_ref,
            cauchy_ref,
            color="black",
            linewidth=2.0,
            zorder=3,
            label="Closed-form",
        )

        for umat, label, color, marker in UMATS:
            data = _stitch(panel, umat)
            if data is None:
                continue
            ax.scatter(
                data["stretch"],
                data["sigma11"],
                s=22,
                facecolors="none",
                edgecolors=color,
                marker=marker,
                linewidths=1.0,
                zorder=4,
                label=label,
            )

        ax.set_xlabel("Stretch $\\lambda$" if panel != "shear" else "Shear $\\gamma$")
        ax.set_ylabel("Cauchy stress $\\sigma_{11}$ (MPa)")
        ax.set_title(panel.capitalize())
        _add_cube_inset(ax, INSET_F[panel], INSET_VIEW[panel])

    # Single shared legend at the bottom; remove per-axes legends to cut clutter.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, -0.02),
        frameon=False,
        fontsize=10,
    )

    fig.tight_layout(rect=(0, 0.06, 1, 1))
    out = FIGURES / f"fe_validation.{args.format}"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


if __name__ == "__main__":
    main()
