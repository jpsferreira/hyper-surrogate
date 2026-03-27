# Paper reproduction scripts

Scripts to generate all quantitative results and figures for the hyper-surrogate paper.

## Quick start

```bash
# 1. Run benchmarks (quick mode for testing, ~5 min)
uv run python paper/run_benchmarks.py --quick

# 2. Generate figures from benchmark results
uv run python paper/generate_figures.py

# 3. Generate FE validation files (UMATs + solver inputs)
uv run python paper/fe_validation.py

# 4. After running FE simulations, plot results
uv run python paper/plot_fe_results.py
```

## Full benchmark suite

```bash
# Full run (~30-60 min depending on hardware)
uv run python paper/run_benchmarks.py
```

## Scripts

| Script                | Purpose                                                                        | Outputs                                                         |
| --------------------- | ------------------------------------------------------------------------------ | --------------------------------------------------------------- |
| `run_benchmarks.py`   | Accuracy benchmarks (6 materials × 4+ architectures), scaling study, timing    | `results/benchmarks.json`, `.tex`, `.md`, convergence histories |
| `generate_figures.py` | Publication-quality figures from benchmark data                                | `figures/*.pdf` (or `.svg`/`.png`)                              |
| `fe_validation.py`    | Train surrogate, export analytical + hybrid UMATs, generate Abaqus/FEAP inputs | `results/fe_validation/`                                        |
| `plot_fe_results.py`  | Compare FE simulation results against analytical reference                     | `figures/fe_validation*.pdf`                                    |

## FE validation workflow

1. Run `fe_validation.py` to generate:

   - Analytical UMAT (`.f`) — symbolic, reference solution
   - Hybrid NN UMAT (`.f90`) — neural network surrogate
   - Abaqus `.inp` files (uniaxial, biaxial, shear)
   - FEAP `.inp` files (uniaxial, biaxial, shear)
   - Analytical reference data (`.json`)

2. Run simulations in Abaqus:

   ```bash
   cd paper/results/fe_validation
   abaqus job=uniaxial_analytical user=neohooke_analytical.f
   abaqus job=uniaxial_hybrid    user=neohooke_hybrid_mlp.f90
   ```

3. Run simulations in FEAP:

   ```bash
   feap -i feap_uniaxial.inp
   ```

4. Extract results to CSV (columns: `stretch,sigma11,sigma22,sigma33`) and place in `results/fe_validation/`:

   - `abaqus_uniaxial_analytical.csv`
   - `abaqus_uniaxial_hybrid.csv`
   - `feap_uniaxial_analytical.csv`
   - `feap_uniaxial_hybrid.csv`
   - (same for biaxial and shear)

5. Plot: `uv run python paper/plot_fe_results.py`

## Output structure

```
paper/
├── results/
│   ├── benchmarks.json              # Raw accuracy metrics
│   ├── benchmarks.tex               # LaTeX table
│   ├── benchmarks.md                # Markdown table
│   ├── convergence_*.json           # Per-run training curves
│   ├── scaling_study.json           # Sample size / width study
│   ├── timing.json                  # NN vs analytical speed
│   └── fe_validation/
│       ├── *_analytical.f           # Symbolic UMAT
│       ├── *_hybrid_mlp.f90         # NN-based UMAT
│       ├── abaqus_*.inp             # Abaqus input decks
│       ├── feap_*.inp               # FEAP input files
│       └── validation_reference.json
└── figures/
    ├── accuracy_comparison_*.pdf
    ├── convergence_curves.pdf
    ├── stress_scatter.pdf
    ├── error_distribution.pdf
    ├── scaling_samples.pdf
    ├── scaling_width.pdf
    ├── timing_bar.pdf
    ├── fe_validation.pdf
    ├── fe_validation_pk2.pdf
    └── fe_validation_energy.pdf
```
