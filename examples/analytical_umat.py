"""Generate an analytical Fortran UMAT from a symbolic material model.

Uses SymPy common subexpression elimination (CSE) to produce an optimized
Fortran 90 subroutine with Cauchy stress and spatial tangent stiffness.

No neural network involved -- this is a purely symbolic approach.

Usage:
    uv run python examples/analytical_umat.py
"""

from hyper_surrogate.export.fortran.analytical import UMATHandler
from hyper_surrogate.mechanics.materials import NeoHooke

# ── 1. Define material ────────────────────────────────────────────
print("── 1. Defining NeoHooke material ──")
material = NeoHooke({"C10": 0.5, "KBULK": 1000.0})
print(f"  Parameters: {material._params}")

# ── 2. Generate UMAT ─────────────────────────────────────────────
print("\n── 2. Generating analytical UMAT ──")
handler = UMATHandler(material)
handler.generate("neohooke_analytical_umat.f")

print("  Output: neohooke_analytical_umat.f")
print("\nThe generated UMAT contains:")
print("  - Cauchy stress (Voigt notation) from symbolic PK2")
print("  - Spatial tangent stiffness with Jaumann rate correction")
print("  - Common subexpression elimination for performance")
print("  - Ready for Abaqus or LS-DYNA")
