"""FE validation: train surrogates, export UMATs, generate solver input files.

Generates:
    results/fe_validation/
        neohooke_analytical.f90         — analytical UMAT (reference)
        neohooke_hybrid_mlp.f90         — hybrid NN UMAT (MLP)
        neohooke_hybrid_mlp.npz         — exported model weights
        abaqus_uniaxial.inp             — Abaqus single-element uniaxial test
        abaqus_biaxial.inp              — Abaqus single-element biaxial test
        abaqus_shear.inp                — Abaqus single-element shear test
        feap_uniaxial.inp               — FEAP single-element uniaxial test
        feap_biaxial.inp                — FEAP single-element biaxial test
        feap_shear.inp                  — FEAP single-element shear test
        validation_reference.json       — analytical stress/energy at test points

After running FE simulations, use:
    uv run python paper/plot_fe_results.py

Usage:
    uv run python paper/fe_validation.py
    uv run python paper/fe_validation.py --material MooneyRivlin
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

# ── paths ──────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
FE_DIR = ROOT / "results" / "fe_validation"
FE_DIR.mkdir(parents=True, exist_ok=True)


# ── hyper-surrogate imports ────────────────────────────────────────
from hyper_surrogate.data.dataset import MaterialDataset, Normalizer, create_datasets
from hyper_surrogate.export.fortran.analytical import UMATHandler
from hyper_surrogate.export.weights import extract_weights
from hyper_surrogate.mechanics.kinematics import Kinematics
from hyper_surrogate.mechanics.materials import MooneyRivlin, NeoHooke, Yeoh
from hyper_surrogate.models.mlp import MLP
from hyper_surrogate.training.losses import EnergyStressLoss
from hyper_surrogate.training.trainer import Trainer

MATERIALS: dict[str, Any] = {
    "NeoHooke": NeoHooke,
    "MooneyRivlin": MooneyRivlin,
    "Yeoh": Yeoh,
}


# ── 1. Train and export UMATs ─────────────────────────────────────
def train_and_export(mat_name: str) -> dict[str, Any]:
    """Train MLP surrogate and export both analytical and hybrid UMATs."""
    MaterialClass = MATERIALS[mat_name]
    material = MaterialClass()
    params = material._params

    print(f"\n══ FE Validation: {mat_name} ══")
    print(f"  Parameters: {params}")

    # --- Analytical UMAT ---
    print("  Generating analytical UMAT ...")
    handler = UMATHandler(material)
    analytical_path = FE_DIR / f"{mat_name.lower()}_analytical.f"
    handler.generate(analytical_path)
    print(f"    -> {analytical_path}")

    # --- Train surrogate ---
    print("  Training hybrid MLP surrogate ...")
    train_ds, val_ds, in_norm, energy_norm = create_datasets(
        material, 10000, input_type="invariants", target_type="energy", seed=42
    )

    model = MLP(input_dim=3, output_dim=1, hidden_dims=[64, 64, 64], activation="softplus")
    tr_result = Trainer(
        model, train_ds, val_ds,
        loss_fn=EnergyStressLoss(alpha=1.0, beta=1.0),
        max_epochs=2000, lr=1e-3, patience=200, batch_size=512,
    ).fit()
    best_val = tr_result.history["val_loss"][tr_result.best_epoch]
    print(f"    Best val loss: {best_val:.6f} (epoch {tr_result.best_epoch})")

    # --- Export hybrid UMAT ---
    from hyper_surrogate.export.fortran.hybrid import HybridUMATEmitter

    exported = extract_weights(tr_result.model, in_norm, energy_norm)
    npz_path = FE_DIR / f"{mat_name.lower()}_hybrid_mlp.npz"
    exported.save(str(npz_path))

    emitter = HybridUMATEmitter(exported)
    hybrid_path = FE_DIR / f"{mat_name.lower()}_hybrid_mlp.f90"
    emitter.write(str(hybrid_path))
    print(f"    -> {hybrid_path}")

    return params


# ── 2. Generate analytical reference data ─────────────────────────
def generate_reference_data(mat_name: str) -> None:
    """Compute stress/energy at discrete stretch levels for post-processing."""
    MaterialClass = MATERIALS[mat_name]
    material = MaterialClass()

    stretches = np.linspace(0.8, 1.5, 50)
    ref: dict[str, Any] = {"material": mat_name, "params": material._params, "tests": {}}

    for test_name, build_F in [
        ("uniaxial", lambda s: _uniaxial_F(s)),
        ("biaxial", lambda s: _biaxial_F(s)),
        ("shear", lambda s: _shear_F(s)),
    ]:
        energies = []
        pk2_11 = []
        pk2_22 = []
        cauchy_11 = []
        stretch_list = stretches.tolist()

        for s in stretches:
            F = build_F(s).reshape(1, 3, 3)
            C = Kinematics.right_cauchy_green(F)
            W = material.evaluate_energy(C)
            S = material.evaluate_pk2(C)
            J = np.linalg.det(F[0])
            sigma = (1.0 / J) * F[0] @ S[0] @ F[0].T

            energies.append(float(W[0]))
            pk2_11.append(float(S[0, 0, 0]))
            pk2_22.append(float(S[0, 1, 1]))
            cauchy_11.append(float(sigma[0, 0]))

        ref["tests"][test_name] = {
            "stretch": stretch_list,
            "energy": energies,
            "pk2_11": pk2_11,
            "pk2_22": pk2_22,
            "cauchy_11": cauchy_11,
        }

    ref_path = FE_DIR / "validation_reference.json"
    with open(ref_path, "w") as f:
        json.dump(ref, f, indent=2)
    print(f"  Reference data -> {ref_path}")


def _uniaxial_F(stretch: float) -> np.ndarray:
    s_t = stretch**-0.5
    return np.diag([stretch, s_t, s_t])


def _biaxial_F(stretch: float) -> np.ndarray:
    s3 = stretch**-2.0
    return np.diag([stretch, stretch, s3])


def _shear_F(gamma: float) -> np.ndarray:
    F = np.eye(3)
    F[0, 1] = gamma - 1.0  # gamma acts as 'stretch' param; shear = gamma - 1
    return F


# ── 3. Abaqus input files ─────────────────────────────────────────
def _abaqus_header(mat_name: str, params: dict[str, float]) -> str:
    """Common Abaqus header: single C3D8 element, node definitions."""
    props_line = ", ".join(f"{v}" for v in params.values())
    n_props = len(params)
    return f"""\
*HEADING
Single-element validation for {mat_name} UMAT
**
*NODE
1,  0.0,  0.0,  0.0
2,  1.0,  0.0,  0.0
3,  1.0,  1.0,  0.0
4,  0.0,  1.0,  0.0
5,  0.0,  0.0,  1.0
6,  1.0,  0.0,  1.0
7,  1.0,  1.0,  1.0
8,  0.0,  1.0,  1.0
**
*ELEMENT, TYPE=C3D8, ELSET=ALL
1, 1, 2, 3, 4, 5, 6, 7, 8
**
*SOLID SECTION, ELSET=ALL, MATERIAL={mat_name.upper()}
1.0,
**
*MATERIAL, NAME={mat_name.upper()}
*USER MATERIAL, CONSTANTS={n_props}
{props_line}
*DEPVAR
1
"""


def _abaqus_uniaxial(mat_name: str, params: dict[str, float], max_stretch: float = 1.3) -> str:
    disp = max_stretch - 1.0
    return _abaqus_header(mat_name, params) + f"""\
**
** ── Uniaxial tension in X ──
**
*BOUNDARY
1, 1, 1, 0.0
1, 2, 2, 0.0
1, 3, 3, 0.0
4, 1, 1, 0.0
5, 1, 1, 0.0
5, 3, 3, 0.0
8, 1, 1, 0.0
4, 3, 3, 0.0
1, 2, 2, 0.0
2, 2, 2, 0.0
5, 2, 2, 0.0
6, 2, 2, 0.0
**
*STEP, NLGEOM=YES
*STATIC
0.05, 1.0, 1e-6, 0.1
**
*BOUNDARY
2, 1, 1, {disp:.6f}
3, 1, 1, {disp:.6f}
6, 1, 1, {disp:.6f}
7, 1, 1, {disp:.6f}
**
*OUTPUT, FIELD
*NODE OUTPUT
U, RF
*ELEMENT OUTPUT
S, E, SDV
**
*OUTPUT, HISTORY
*NODE OUTPUT, NSET=ALL
U, RF
*ELEMENT OUTPUT, ELSET=ALL
S, E
**
*END STEP
"""


def _abaqus_biaxial(mat_name: str, params: dict[str, float], max_stretch: float = 1.2) -> str:
    disp = max_stretch - 1.0
    return _abaqus_header(mat_name, params) + f"""\
**
** ── Equibiaxial tension in X and Y ──
**
*BOUNDARY
1, 1, 1, 0.0
1, 2, 2, 0.0
1, 3, 3, 0.0
4, 1, 1, 0.0
5, 1, 1, 0.0
5, 3, 3, 0.0
8, 1, 1, 0.0
1, 3, 3, 0.0
4, 3, 3, 0.0
5, 2, 2, 0.0
6, 2, 2, 0.0
**
*STEP, NLGEOM=YES
*STATIC
0.05, 1.0, 1e-6, 0.1
**
*BOUNDARY
2, 1, 1, {disp:.6f}
3, 1, 1, {disp:.6f}
6, 1, 1, {disp:.6f}
7, 1, 1, {disp:.6f}
3, 2, 2, {disp:.6f}
4, 2, 2, {disp:.6f}
7, 2, 2, {disp:.6f}
8, 2, 2, {disp:.6f}
**
*OUTPUT, FIELD
*NODE OUTPUT
U, RF
*ELEMENT OUTPUT
S, E, SDV
**
*END STEP
"""


def _abaqus_shear(mat_name: str, params: dict[str, float], max_shear: float = 0.3) -> str:
    return _abaqus_header(mat_name, params) + f"""\
**
** ── Simple shear in XY ──
**
*BOUNDARY
1, 1, 3, 0.0
2, 2, 2, 0.0
2, 3, 3, 0.0
5, 1, 1, 0.0
5, 3, 3, 0.0
6, 2, 2, 0.0
6, 3, 3, 0.0
**
*STEP, NLGEOM=YES
*STATIC
0.05, 1.0, 1e-6, 0.1
**
*BOUNDARY
4, 1, 1, {max_shear:.6f}
3, 1, 1, {max_shear:.6f}
8, 1, 1, {max_shear:.6f}
7, 1, 1, {max_shear:.6f}
**
*OUTPUT, FIELD
*NODE OUTPUT
U, RF
*ELEMENT OUTPUT
S, E, SDV
**
*END STEP
"""


def generate_abaqus_inputs(mat_name: str, params: dict[str, float]) -> None:
    """Write Abaqus .inp files for uniaxial, biaxial, shear single-element tests."""
    tests = {
        "uniaxial": _abaqus_uniaxial,
        "biaxial": _abaqus_biaxial,
        "shear": _abaqus_shear,
    }
    for test_name, gen_fn in tests.items():
        path = FE_DIR / f"abaqus_{test_name}.inp"
        path.write_text(gen_fn(mat_name, params))
        print(f"    Abaqus {test_name} -> {path}")


# ── 4. FEAP input files ───────────────────────────────────────────
def _feap_header(mat_name: str, params: dict[str, float]) -> str:
    """FEAP header for single hex8 element."""
    return f"""\
FEAP * * Single-element {mat_name} validation
  0  0  0  3  8  8

! Node coordinates
COORdinates
  1  0  0.0  0.0  0.0
  2  0  1.0  0.0  0.0
  3  0  1.0  1.0  0.0
  4  0  0.0  1.0  0.0
  5  0  0.0  0.0  1.0
  6  0  1.0  0.0  1.0
  7  0  1.0  1.0  1.0
  8  0  0.0  1.0  1.0

! Element connectivity (8-node hex)
ELEMents
  1  0  0  1  2  3  4  5  6  7  8

! Material: user-defined UMAT
MATErial 1
  USER {len(params)}
"""


def _feap_uniaxial(mat_name: str, params: dict[str, float], max_stretch: float = 1.3) -> str:
    disp = max_stretch - 1.0
    nsteps = 20
    dt = disp / nsteps
    return _feap_header(mat_name, params) + f"""\
! Boundary conditions: fix base, pull top in X
BOUNdary conditions
  1  0  1  1  1    ! node 1: fix x, y, z
  4  0  1  0  1    ! node 4: fix x, z
  5  0  1  0  1    ! node 5: fix x, z
  8  0  1  0  0    ! node 8: fix x
  1  0  0  1  0    ! bottom y-fixed
  2  0  0  1  0
  5  0  0  1  0
  6  0  0  1  0

! Displacement loading in X
FORCe and DISPlacement
  2  0  {disp:.6f}  0.0  0.0
  3  0  {disp:.6f}  0.0  0.0
  6  0  {disp:.6f}  0.0  0.0
  7  0  {disp:.6f}  0.0  0.0

END mesh

BATCh
  LOOP,, {nsteps}
    TIME
    LOOP,, 10
      TANG,,1
    NEXT
    DISP,,ALL
    STRE,,ALL
  NEXT
END batch

STOP
"""


def _feap_biaxial(mat_name: str, params: dict[str, float], max_stretch: float = 1.2) -> str:
    disp = max_stretch - 1.0
    nsteps = 20
    return _feap_header(mat_name, params) + f"""\
! Boundary conditions: equibiaxial
BOUNdary conditions
  1  0  1  1  1
  5  0  1  0  1
  1  0  0  1  0
  2  0  0  1  0
  5  0  0  1  0
  6  0  0  1  0

! Displacement loading in X and Y
FORCe and DISPlacement
  2  0  {disp:.6f}  0.0  0.0
  3  0  {disp:.6f}  {disp:.6f}  0.0
  6  0  {disp:.6f}  0.0  0.0
  7  0  {disp:.6f}  {disp:.6f}  0.0
  4  0  0.0  {disp:.6f}  0.0
  8  0  0.0  {disp:.6f}  0.0

END mesh

BATCh
  LOOP,, {nsteps}
    TIME
    LOOP,, 10
      TANG,,1
    NEXT
    DISP,,ALL
    STRE,,ALL
  NEXT
END batch

STOP
"""


def _feap_shear(mat_name: str, params: dict[str, float], max_shear: float = 0.3) -> str:
    nsteps = 20
    return _feap_header(mat_name, params) + f"""\
! Boundary conditions: simple shear
BOUNdary conditions
  1  0  1  1  1
  2  0  0  1  1
  5  0  1  0  1
  6  0  0  1  1

! Shear displacement on top face (X direction)
FORCe and DISPlacement
  4  0  {max_shear:.6f}  0.0  0.0
  3  0  {max_shear:.6f}  0.0  0.0
  8  0  {max_shear:.6f}  0.0  0.0
  7  0  {max_shear:.6f}  0.0  0.0

END mesh

BATCh
  LOOP,, {nsteps}
    TIME
    LOOP,, 10
      TANG,,1
    NEXT
    DISP,,ALL
    STRE,,ALL
  NEXT
END batch

STOP
"""


def generate_feap_inputs(mat_name: str, params: dict[str, float]) -> None:
    """Write FEAP input files for single-element validation."""
    tests = {
        "uniaxial": _feap_uniaxial,
        "biaxial": _feap_biaxial,
        "shear": _feap_shear,
    }
    for test_name, gen_fn in tests.items():
        path = FE_DIR / f"feap_{test_name}.inp"
        path.write_text(gen_fn(mat_name, params))
        print(f"    FEAP {test_name} -> {path}")


# ── main ───────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Generate FE validation files")
    parser.add_argument("--material", default="NeoHooke", choices=list(MATERIALS.keys()))
    args = parser.parse_args()

    params = train_and_export(args.material)
    generate_reference_data(args.material)

    print("\n  Generating solver input files ...")
    generate_abaqus_inputs(args.material, params)
    generate_feap_inputs(args.material, params)

    print(f"""
══ Next steps ══
1. Run Abaqus simulations:
     abaqus job=uniaxial_analytical user={args.material.lower()}_analytical.f
     abaqus job=uniaxial_hybrid    user={args.material.lower()}_hybrid_mlp.f90

2. Run FEAP simulations:
     feap -i feap_uniaxial.inp

3. Extract stress-displacement results and compare with:
     uv run python paper/plot_fe_results.py
""")


if __name__ == "__main__":
    main()
