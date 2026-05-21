"""Render the hyper-surrogate pipeline block diagram (Figure 1a).

A hyperelastic material defined in Python is routed through one of three
paths -- a built-in SEF, a custom SymPy SEF, or a trained surrogate --
to a Fortran UMAT (.f90) that any Fortran-based finite element solver
can consume.

Output:
    paper/figures/fe_pipeline.pdf
    paper/figures/fe_pipeline.png  (with --format png)

Usage:
    uv run python paper/plot_flow_diagram.py
    uv run python paper/plot_flow_diagram.py --format png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.size": 9.5,
    "savefig.bbox": "tight",
    "savefig.dpi": 300,
    "figure.dpi": 300,
})


# Visual palette
TOP_FILL = "#E8EEF7"  # light blue: source material
PATH_FILL_A = "#D7E8D7"  # light green: built-in
PATH_FILL_B = "#FCE6CC"  # light orange: custom
PATH_FILL_C = "#E5D7F2"  # light purple: trained surrogate
OUTPUT_FILL = "#F8E0E0"  # light red: UMAT
SOLVER_FILL = "#F0F0F0"  # neutral grey: FE solver
EDGE = "#222222"
ARROW = "#444444"


def _box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    *,
    fill: str,
    fontweight: str = "normal",
) -> tuple[float, float]:
    """Draw a rounded rectangle with centred text. Returns (cx, cy)."""
    x, y = xy
    rect = mpatches.FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        linewidth=1.2,
        edgecolor=EDGE,
        facecolor=fill,
    )
    ax.add_patch(rect)
    cx, cy = x + width / 2, y + height / 2
    ax.text(
        cx,
        cy,
        text,
        ha="center",
        va="center",
        fontsize=9.5,
        fontweight=fontweight,
        wrap=True,
    )
    return cx, cy


def _arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
) -> None:
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={
            "arrowstyle": "-|>",
            "color": ARROW,
            "linewidth": 1.4,
            "shrinkA": 1,
            "shrinkB": 1,
        },
    )


def render() -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7.2)
    ax.set_aspect("equal")
    ax.axis("off")

    # ── Top: source ────────────────────────────────────────────────
    top_w, top_h = 4.8, 0.95
    top_x, top_y = (14 - top_w) / 2, 6.0
    cx_top, cy_top = _box(
        ax,
        (top_x, top_y),
        top_w,
        top_h,
        "Material defined in Python",
        fill=TOP_FILL,
        fontweight="bold",
    )

    # ── Middle: three branches ─────────────────────────────────────
    branch_w, branch_h = 4.0, 1.25
    branch_y = 3.3
    branches = [
        (
            0.3,
            "Built-in SEF\n(NeoHooke, HolzapfelOgden, ...)",
            PATH_FILL_A,
        ),
        (
            5.0,
            "Custom SymPy SEF\n(your own W(I1, I2, J, ...))",
            PATH_FILL_B,
        ),
        (
            9.7,
            "Trained surrogate\n(MLP / ICNN / PolyconvexICNN)",
            PATH_FILL_C,
        ),
    ]
    branch_centres = []
    for x0, label, fill in branches:
        cx, cy = _box(ax, (x0, branch_y), branch_w, branch_h, label, fill=fill)
        branch_centres.append((cx, cy))
        # Top → branch
        _arrow(
            ax,
            (cx_top + (cx - cx_top) * 0.18, top_y),
            (cx, branch_y + branch_h),
        )

    # ── Bottom: output UMAT ───────────────────────────────────────
    out_w, out_h = 6.6, 0.95
    out_x, out_y = (14 - out_w) / 2, 1.45
    cx_out, cy_out = _box(
        ax,
        (out_x, out_y),
        out_w,
        out_h,
        "UMAT (.f90)  —  Cauchy stress + consistent tangent",
        fill=OUTPUT_FILL,
        fontweight="bold",
    )

    for cx, _cy in branch_centres:
        _arrow(ax, (cx, branch_y), (cx_out + (cx - cx_out) * 0.18, out_y + out_h))

    # ── Bottom-bottom: solver ─────────────────────────────────────
    solver_w, solver_h = 6.0, 0.7
    solver_x, solver_y = (14 - solver_w) / 2, 0.15
    _box(
        ax,
        (solver_x, solver_y),
        solver_w,
        solver_h,
        "Any Fortran-based FE solver (Abaqus, FEAP, ...)",
        fill=SOLVER_FILL,
    )
    _arrow(ax, (cx_out, out_y), (cx_out, solver_y + solver_h))

    fig.tight_layout(pad=0.4)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--format", default="pdf", choices=["pdf", "svg", "png"])
    args = parser.parse_args()

    fig = render()
    out = FIGURES / f"fe_pipeline.{args.format}"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


if __name__ == "__main__":
    main()
