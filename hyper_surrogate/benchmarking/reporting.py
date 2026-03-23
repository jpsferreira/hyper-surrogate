"""Reporting utilities for benchmark results."""

from __future__ import annotations

from hyper_surrogate.benchmarking.metrics import BenchmarkResult


def results_to_latex(
    results: list[BenchmarkResult],
    metrics: list[str] | None = None,
    caption: str = "Surrogate model benchmark results",
    label: str = "tab:benchmark",
) -> str:
    """Generate a LaTeX table from benchmark results.

    Args:
        results: List of BenchmarkResult instances.
        metrics: Which metrics to include. Defaults to all found.
        caption: Table caption.
        label: Table label.

    Returns:
        LaTeX table string.
    """
    if not results:
        return ""

    if metrics is None:
        metrics = sorted({k for r in results for k in r.metrics})

    # Header
    cols = ["Material", "Model", "Params", "Time (s)"] + [m.upper() for m in metrics]
    header = " & ".join(cols)
    col_spec = "ll" + "r" * (len(cols) - 2)

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        header + r" \\",
        r"\midrule",
    ]

    for r in results:
        row_vals = [
            r.material_name,
            r.model_name,
            str(r.n_parameters),
            f"{r.training_time:.1f}",
        ]
        for m in metrics:
            val = r.metrics.get(m, float("nan"))
            row_vals.append(f"{val:.4f}")
        lines.append(" & ".join(row_vals) + r" \\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return "\n".join(lines)


def results_to_markdown(
    results: list[BenchmarkResult],
    metrics: list[str] | None = None,
) -> str:
    """Generate a Markdown table from benchmark results."""
    if not results:
        return ""

    if metrics is None:
        metrics = sorted({k for r in results for k in r.metrics})

    cols = ["Material", "Model", "Params", "Time (s)"] + [m.upper() for m in metrics]
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"

    lines = [header, sep]
    for r in results:
        row_vals = [
            r.material_name,
            r.model_name,
            str(r.n_parameters),
            f"{r.training_time:.1f}",
        ]
        for m in metrics:
            val = r.metrics.get(m, float("nan"))
            row_vals.append(f"{val:.4f}")
        lines.append("| " + " | ".join(row_vals) + " |")

    return "\n".join(lines)
