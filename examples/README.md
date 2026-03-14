# Examples

Runnable scripts demonstrating the hyper-surrogate pipeline.

| Script                     | Description                                                        | Run command                                       |
| -------------------------- | ------------------------------------------------------------------ | ------------------------------------------------- |
| `train_neohooke_sef.py`    | Train MLP on NeoHooke strain energy function with hybrid inference | `uv run python examples/train_neohooke_sef.py`    |
| `train_neohooke_stress.py` | Train MLP on PK2 stress with `StressLoss`                          | `uv run python examples/train_neohooke_stress.py` |
| `train_icnn_energy.py`     | Train ICNN with `EnergyStressLoss` for convex energy               | `uv run python examples/train_icnn_energy.py`     |
| `export_hybrid_umat.py`    | End-to-end train + `HybridUMATEmitter` export                      | `uv run python examples/export_hybrid_umat.py`    |
| `train_holzapfel_ogden.py` | Anisotropic HolzapfelOgden with 5D invariants + hybrid UMAT        | `uv run python examples/train_holzapfel_ogden.py` |
| `train_polyconvex.py`      | PolyconvexICNN with per-invariant branches + hybrid UMAT           | `uv run python examples/train_polyconvex.py`      |
| `analytical_umat.py`       | Symbolic material to Fortran UMAT via `UMATHandler`                | `uv run python examples/analytical_umat.py`       |

All examples follow the same structure: docstring, numbered sections, print statements for progress.
