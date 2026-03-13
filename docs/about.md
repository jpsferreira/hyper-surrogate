# About

**hyper-surrogate** is a Python library for building data-driven surrogates of hyperelastic constitutive models used in finite element analysis.

## Motivation

Hyperelastic material models (Neo-Hookean, Mooney-Rivlin, etc.) define stress-strain relationships through strain energy functions. Evaluating these symbolically at every integration point is expensive. This library provides:

1. **Symbolic mechanics** -- SymPy-based strain energy differentiation to derive PK2 stress and material stiffness tensors automatically.
2. **ML surrogates** -- Train neural networks (MLP, Input-Convex Neural Networks) to approximate the constitutive response, orders of magnitude faster than symbolic evaluation.
3. **Fortran export** -- Transpile trained models into standalone Fortran 90 subroutines (UMAT-compatible) with baked-in weights, ready for commercial FE solvers (Abaqus, LS-DYNA, etc.).

## Architecture

The library is organized into layered subpackages:

- `mechanics` -- Symbolic tensor algebra, kinematics, and material model definitions
- `data` -- Deformation gradient generation and dataset creation for training
- `models` -- Neural network architectures (MLP, ICNN) with export support
- `training` -- Loss functions (stress, energy-stress) and training loop
- `export` -- Weight extraction, Fortran code generation, and analytical UMAT generation
- `reporting` -- Visualization and statistics for deformation data

## License

See the [LICENSE](https://github.com/jpsferreira/hyper-surrogate/blob/main/LICENSE) file.
