# hyper-surrogate

Data-driven surrogates for hyperelastic constitutive models in finite element analysis.

## Tech Stack

- **Python 3.12+** with `from __future__ import annotations` everywhere
- **uv** for dependency management (`uv sync --all-groups --extra ml`)
- **ruff** for linting/formatting (120-char line length)
- **mypy** in strict mode on `hyper_surrogate/`
- **pytest** for testing (`@pytest.mark.slow` for expensive tests)
- **mkdocs-material** for documentation

## Key Commands

```bash
make install    # Create venv and install pre-commit hooks
make check      # Run ruff, mypy, deptry
make test       # Run pytest with coverage (excludes slow tests)
make docs       # Build and serve docs locally
make docs-test  # Verify docs build without errors
```

## Architecture

```
Material -> DeformationGenerator -> Dataset -> MLP/ICNN -> FortranEmitter -> .f90
```

### Core (always available — NumPy, SymPy)

- `mechanics/` — `SymbolicHandler` (tensor algebra), `Kinematics` (batch operations), `Material` ABC + `NeoHooke`, `MooneyRivlin`
- `data/` — `DeformationGenerator` (deformation gradients), `create_datasets`, `Normalizer`
- `reporting/` — `Reporter` for deformation statistics and PDF reports

### ML (requires `torch`)

- `models/` — `SurrogateModel` ABC, `MLP`, `ICNN`
- `training/` — `Trainer`, `EnergyStressLoss`, `StressLoss`, `StressTangentLoss`

### Export (requires `torch`)

- `export/weights.py` — `ExportedModel`, `extract_weights`
- `export/fortran/emitter.py` — `FortranEmitter` (standalone NN → Fortran 90)
- `export/fortran/hybrid.py` — `HybridUMATEmitter` (NN SEF + analytical mechanics → UMAT)
- `export/fortran/analytical.py` — `UMATHandler` (symbolic → Fortran UMAT via SymPy CSE)

## Conventions

- **Torch optional:** ML imports wrapped in `try/except ImportError` in `__init__.py`
- **Model interface:** All models extend `SurrogateModel` ABC (`forward`, `layer_sequence`, `export_weights`)
- **Export interface:** `ExportedModel` flat dataclass with layers, weights, normalizers, metadata
- **Fortran layout:** Column-major arrays, 15-digit double precision (`:.15e`), `MATMUL`-based forward pass

## Branch Prefixes

- `feat/` — new features
- `fix/` — bug fixes
- `chore/` — maintenance and cleanup
