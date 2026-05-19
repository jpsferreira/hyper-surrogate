# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `UMATHandler` exported from the package root for symmetric access to all
  three Fortran emitters (`UMATHandler`, `HybridUMATEmitter`, `FortranEmitter`).
- Custom-SEF tutorial (`docs/tutorials/custom_materials.md`) and runnable
  example (`examples/custom_sef.py`) demonstrating how to subclass
  `Material` with a SymPy `sef` property.
- Regression tests for the polyconvex Fortran emitter:
  - Python emulator of the emitted backward pass vs `torch.autograd.functional.grad`.
  - Emit-string check for the ICNN skip-connection accumulation lines.
- `CHANGELOG.md` for release-history tracking.

### Fixed

- Polyconvex Fortran emitter no longer drops the ICNN per-layer
  skip-connection gradient contributions (`wx_l^T · δ_l` for `l ≥ 1`).
  The omission previously surfaced in Abaqus as a several-MPa spurious
  hydrostatic Cauchy stress on the volumetric branch.

### Changed

- README and JOSS paper rewritten around the "three paths to a Fortran
  UMAT" framing — built-in symbolic SEF, custom symbolic SEF, trained
  surrogate.
- `mkdocs` navigation updated so the Custom Materials tutorial sits
  between Getting Started and Data Generation.

## [0.3.0] — 2026

### Added

- Anisotropic material benchmarks: `HolzapfelOgden` and
  `GasserOgdenHolzapfel` (GOH).
- LaTeX manuscript scaffold under `paper/manuscript/` with one TeX file
  per section.
- Compressible deformation mode (`combined_compressible`) with
  configurable `j_range` and stress-free-reference correction in the
  hybrid UMAT emitter.
- `dW_dI_REF` PARAMETER emitted in the polyconvex Fortran module so
  the analytical stress is zero at the reference configuration.
- Test coverage for `PeirlinckArtery` and the `combined_compressible`
  dataset path.
- Paper reproduction scripts (`paper/run_benchmarks.py`,
  `paper/fe_validation.py`, `paper/generate_figures.py`).

### Fixed

- Hybrid emitter no longer triggers `mypy` `no-any-return` errors on
  the `np.einsum` and `np.log1p` chains.

## [0.2.0]

### Added

- Multi-fiber-family support across the full pipeline (data generator,
  symbolic mechanics, NN architectures, hybrid emitter).
- Comprehensive tutorial suite under `docs/tutorials/`.
- Constitutive theory reference (`docs/constitutive_theory.md`).
- Benchmarking metrics and an `UMATHandler` test suite.

[Unreleased]: https://github.com/jpsferreira/hyper-surrogate/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/jpsferreira/hyper-surrogate/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/jpsferreira/hyper-surrogate/releases/tag/v0.2.0
