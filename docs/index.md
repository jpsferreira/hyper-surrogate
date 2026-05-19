# hyper-surrogate

[![Release](https://img.shields.io/github/v/release/jpsferreira/hyper-surrogate)](https://img.shields.io/github/v/release/jpsferreira/hyper-surrogate)
[![Build status](https://img.shields.io/github/actions/workflow/status/jpsferreira/hyper-surrogate/main.yml?branch=main)](https://github.com/jpsferreira/hyper-surrogate/actions/workflows/main.yml?query=branch%3Amain)
[![Commit activity](https://img.shields.io/github/commit-activity/m/jpsferreira/hyper-surrogate)](https://img.shields.io/github/commit-activity/m/jpsferreira/hyper-surrogate)
[![License](https://img.shields.io/github/license/jpsferreira/hyper-surrogate)](https://img.shields.io/github/license/jpsferreira/hyper-surrogate)

**Define hyperelastic materials in Python, deploy them in your finite element solver.**

hyper-surrogate turns a hyperelastic material defined in Python into a
self-contained Fortran 90 user subroutine (UMAT) that links directly to
commercial and research finite element solvers. A material can be
defined in three ways:

- **Path A — built-in symbolic SEF.** Pick from ten classical
  isotropic and anisotropic models and emit a UMAT in one line.
- **Path B — custom symbolic SEF.** Subclass `Material` with a `sef`
  property in `SymPy` and the framework derives stress and consistent
  tangent for you. See the [custom materials tutorial](tutorials/custom_materials.md).
- **Path C — data-driven surrogate.** Train an MLP, ICNN, or polyconvex
  ICNN on energy and stress samples from any ground-truth source and
  emit a hybrid UMAT whose energy is the trained network and whose
  mechanics is computed analytically in Fortran.

All three paths produce the same artefact — a `.f90` file with Cauchy
stress, the analytical consistent tangent, and stress-freedom at the
reference enforced exactly at emission time.

## Where to start

- [Installation](installation.md) — `pip install hyper-surrogate` or `uv sync --extra ml`.
- [Getting started](tutorials/getting_started.md) — your first material, in five minutes.
- [Custom materials](tutorials/custom_materials.md) — write your own SEF and get a UMAT.
- [Fortran export](tutorials/export_fortran.md) — when to pick each of the three emitters.
- [Examples](examples.md) — runnable scripts for every path.

## Runnable examples for each path

| Path | Example                                                                                                                     | What it does                                                           |
| ---- | --------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| A    | [`examples/analytical_umat.py`](https://github.com/jpsferreira/hyper-surrogate/blob/main/examples/analytical_umat.py)       | Built-in NeoHooke → analytical Fortran UMAT in one line                |
| B    | [`examples/custom_sef.py`](https://github.com/jpsferreira/hyper-surrogate/blob/main/examples/custom_sef.py)                 | Custom Ogden-like SEF → analytical Fortran UMAT                        |
| C    | [`examples/export_hybrid_umat.py`](https://github.com/jpsferreira/hyper-surrogate/blob/main/examples/export_hybrid_umat.py) | Train an MLP and emit a hybrid UMAT (NN energy + analytical mechanics) |
