# Architecture Refactoring & ML Surrogate Pipeline — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor hyper-surrogate into layered architecture and add ML surrogate pipeline (PyTorch models → Fortran transpiler).

**Architecture:** Layered package — `mechanics/` (sympy+numpy, no torch), `data/` (generation+datasets), `models/` (MLP+ICNN), `training/` (trainer+losses), `export/` (weights+fortran emitter). Composition over inheritance. Optional torch dependency.

**Tech Stack:** Python 3.10+, NumPy, SymPy, PyTorch (optional), pytest

**Spec:** `docs/superpowers/specs/2026-03-12-arch-refactor-ml-surrogate-design.md`

---

## Chunk 1: Mechanics Layer Refactoring

### Task 1: Create package skeleton and move SymbolicHandler

**Files:**

- Create: `hyper_surrogate/mechanics/__init__.py`
- Create: `hyper_surrogate/mechanics/symbolic.py`
- Modify: `tests/test_symbolic_rules.py`

- [ ] **Step 1: Create mechanics package with empty init**

```python
# hyper_surrogate/mechanics/__init__.py
```

- [ ] **Step 2: Copy and refactor SymbolicHandler into mechanics/symbolic.py**

Copy `hyper_surrogate/symbolic.py` → `hyper_surrogate/mechanics/symbolic.py` with these changes:

- Rename `invariant1` → `isochoric_invariant1`, `invariant2` → `isochoric_invariant2` (keep `invariant3` as-is)
- Rename `reduce_2nd_order` → `to_voigt_2`, `reduce_4th_order` → `to_voigt_4`
- Rename `pk2_tensor` → `pk2`, `cmat_tensor` → `cmat`
- Rename `sigma_tensor` → `cauchy`, `smat_tensor` → `spatial_tangent`
- Rename `lambda_tensor` → `lambdify` (keep same signature: `(symbolic_tensor, *param_symbols) -> Callable`)
- Rename `evaluate_iterator` → keep as-is (used by Material internally)
- Add `jaumann_correction` as alias for `jr`
- Keep all existing logic intact — this is a rename/move, not a rewrite

- [ ] **Step 3: Update test imports and run tests**

Update `tests/test_symbolic_rules.py`:

- Change `from hyper_surrogate.symbolic import SymbolicHandler` → `from hyper_surrogate.mechanics.symbolic import SymbolicHandler`
- Update any references to renamed methods

Run: `pytest tests/test_symbolic_rules.py -v`
Expected: All tests pass

- [ ] **Step 4: Commit**

```bash
git add hyper_surrogate/mechanics/ tests/test_symbolic_rules.py
git commit -m "refactor: move SymbolicHandler to mechanics/ with renamed API"
```

---

### Task 2: Refactor Kinematics into mechanics/

**Files:**

- Create: `hyper_surrogate/mechanics/kinematics.py`
- Modify: `tests/test_kinematics.py`

- [ ] **Step 1: Copy and refactor Kinematics**

Copy `hyper_surrogate/kinematics.py` → `hyper_surrogate/mechanics/kinematics.py` with these changes:

- Rename `invariant1` → `trace_invariant` (parameter name: `f` → `c` to clarify it operates on C tensors)
- Rename `invariant2` → `quadratic_invariant` (parameter name: `f` → `c`)
- Rename `invariant3` → `det_invariant` (parameter name: `f` → `c`)
- Note: The underlying einsum logic is identical — only the parameter name changes for clarity. These methods work on any (N,3,3) array; the rename signals the intended usage on C = F^T F.
- Add new methods `isochoric_invariant1(c)` and `isochoric_invariant2(c)`:

```python
@staticmethod
def isochoric_invariant1(c: np.ndarray) -> np.ndarray:
    """Isochoric first invariant: tr(C) * det(C)^(-1/3)."""
    return np.einsum("nii->n", c) * np.linalg.det(c) ** (-1.0 / 3.0)

@staticmethod
def isochoric_invariant2(c: np.ndarray) -> np.ndarray:
    """Isochoric second invariant: 0.5*(I1^2 - tr(C^2)) * det(C)^(-2/3)."""
    i1 = np.einsum("nii->n", c)
    i1_sq = i1 ** 2
    tr_c2 = np.einsum("nij,nji->n", c, c)
    return 0.5 * (i1_sq - tr_c2) * np.linalg.det(c) ** (-2.0 / 3.0)
```

- [ ] **Step 2: Update test imports and add tests for new methods**

Update `tests/test_kinematics.py` imports. Add:

```python
def test_isochoric_invariant1(right_cauchys):
    result = Kinematics.isochoric_invariant1(right_cauchys)
    det_c = np.linalg.det(right_cauchys)
    tr_c = np.trace(right_cauchys, axis1=1, axis2=2)
    expected = tr_c * det_c ** (-1.0 / 3.0)
    np.testing.assert_allclose(result, expected)

def test_isochoric_invariant2(right_cauchys):
    result = Kinematics.isochoric_invariant2(right_cauchys)
    assert result.shape == (right_cauchys.shape[0],)
```

Run: `pytest tests/test_kinematics.py -v`
Expected: All pass

- [ ] **Step 3: Commit**

```bash
git add hyper_surrogate/mechanics/kinematics.py tests/test_kinematics.py
git commit -m "refactor: move Kinematics to mechanics/ with disambiguated invariants"
```

---

### Task 3: Refactor Material to use composition

**Files:**

- Create: `hyper_surrogate/mechanics/materials.py`
- Create: `tests/test_mechanics_materials.py`
- Modify: `tests/test_neohooke.py`

- [ ] **Step 1: Write failing test for new Material API**

Create `tests/test_mechanics_materials.py`:

```python
import numpy as np
import pytest
from hyper_surrogate.mechanics.materials import NeoHooke, MooneyRivlin


def test_neohooke_has_handler():
    mat = NeoHooke({"C10": 0.5, "KBULK": 1000.0})
    assert hasattr(mat, "_handler")


def test_neohooke_sef_is_expr():
    from sympy import Expr
    mat = NeoHooke({"C10": 0.5, "KBULK": 1000.0})
    assert isinstance(mat.sef, Expr)


def test_neohooke_pk2_func_callable():
    mat = NeoHooke({"C10": 0.5, "KBULK": 1000.0})
    func = mat.pk2_func
    assert callable(func)


def test_neohooke_evaluate_pk2_identity():
    mat = NeoHooke({"C10": 0.5, "KBULK": 1000.0})
    c_identity = np.eye(3).reshape(1, 3, 3)
    result = mat.evaluate_pk2(c_identity)
    assert result.shape == (1, 3, 3)
    # At identity, isochoric PK2 should be zero for incompressible part
    np.testing.assert_allclose(result[0], np.zeros((3, 3)), atol=1e-10)


def test_mooneyrivlin_params():
    mat = MooneyRivlin({"C10": 0.3, "C01": 0.2, "KBULK": 1000.0})
    assert mat._params == {"C10": 0.3, "C01": 0.2, "KBULK": 1000.0}
```

Run: `pytest tests/test_mechanics_materials.py -v`
Expected: FAIL (module not found)

- [ ] **Step 2: Implement Material with composition**

Create `hyper_surrogate/mechanics/materials.py`:

```python
from __future__ import annotations

from functools import cached_property
from typing import Any, Callable

import numpy as np
from sympy import Expr, Matrix, Symbol, log

from hyper_surrogate.mechanics.symbolic import SymbolicHandler


class Material:
    """Base class for constitutive material models using composition."""

    def __init__(self, parameters: dict[str, float]) -> None:
        self._handler = SymbolicHandler()
        self._params = parameters
        self._symbols = {k: Symbol(k) for k in parameters}

    @property
    def handler(self) -> SymbolicHandler:
        return self._handler

    @property
    def sef(self) -> Expr:
        raise NotImplementedError

    @cached_property
    def pk2_expr(self) -> Matrix:
        return self._handler.pk2(self.sef)

    @cached_property
    def cmat_expr(self) -> Any:
        return self._handler.cmat(self.pk2_expr)

    @cached_property
    def pk2_func(self) -> Callable:
        return self._handler.lambdify(self.pk2_expr, *self._symbols.values())

    @cached_property
    def cmat_func(self) -> Callable:
        return self._handler.lambdify(self.cmat_expr, *self._symbols.values())

    def evaluate_pk2(self, c_batch: np.ndarray) -> np.ndarray:
        """Vectorized PK2 evaluation over (N,3,3) C tensors."""
        param_values = list(self._params.values())
        results = []
        for c in c_batch:
            result = self.pk2_func(c.flatten(), *param_values)
            results.append(np.array(result, dtype=float))
        return np.array(results)

    def evaluate_cmat(self, c_batch: np.ndarray) -> np.ndarray:
        """Vectorized CMAT evaluation over (N,3,3) C tensors."""
        param_values = list(self._params.values())
        results = []
        for c in c_batch:
            result = self.cmat_func(c.flatten(), *param_values)
            results.append(np.array(result, dtype=float))
        return np.array(results)

    def evaluate_energy(self, c_batch: np.ndarray) -> np.ndarray:
        """Evaluate strain energy for (N,3,3) C tensors. Returns (N,)."""
        sef_func = self._handler.lambdify(self.sef, *self._symbols.values())
        param_values = list(self._params.values())
        results = []
        for c in c_batch:
            result = sef_func(c.flatten(), *param_values)
            results.append(float(result))
        return np.array(results)

    # --- Symbolic accessors for UMAT generation ---

    def cauchy_voigt(self, f: Matrix) -> Matrix:
        """Voigt-reduced Cauchy stress (6x1) in symbolic form."""
        sigma = self._handler.cauchy(self.sef, f)
        return SymbolicHandler.to_voigt_2(sigma)

    def tangent_voigt(self, f: Matrix, use_jaumann_rate: bool = False) -> Matrix:
        """Voigt-reduced tangent (6x6) in symbolic form."""
        smat = self._handler.spatial_tangent(self.pk2_expr, f)
        if use_jaumann_rate:
            sigma = self._handler.cauchy(self.sef, f)
            smat = smat + self._handler.jaumann_correction(sigma)
        return SymbolicHandler.to_voigt_4(smat)


class NeoHooke(Material):
    DEFAULT_PARAMS = {"C10": 0.5, "KBULK": 1000.0}

    def __init__(self, parameters: dict[str, float] | None = None) -> None:
        params = {**self.DEFAULT_PARAMS, **(parameters or {})}
        super().__init__(params)

    @property
    def sef(self) -> Expr:
        h = self._handler
        C10, KBULK = self._symbols["C10"], self._symbols["KBULK"]
        return (h.isochoric_invariant1 - 3) * C10 + 0.25 * KBULK * (
            h.invariant3 - 1 - 2 * log(h.invariant3 ** 0.5)
        )


class MooneyRivlin(Material):
    DEFAULT_PARAMS = {"C10": 0.3, "C01": 0.2, "KBULK": 1000.0}

    def __init__(self, parameters: dict[str, float] | None = None) -> None:
        params = {**self.DEFAULT_PARAMS, **(parameters or {})}
        super().__init__(params)

    @property
    def sef(self) -> Expr:
        h = self._handler
        C10, C01, KBULK = self._symbols["C10"], self._symbols["C01"], self._symbols["KBULK"]
        return (
            (h.isochoric_invariant1 - 3) * C10
            + (h.isochoric_invariant2 - 3) * C01
            + 0.25 * KBULK * (h.invariant3 - 1 - 2 * log(h.invariant3 ** 0.5))
        )
```

- [ ] **Step 3: Run tests**

Run: `pytest tests/test_mechanics_materials.py -v`
Expected: All pass

- [ ] **Step 4: Port existing NeoHooke tests**

Update `tests/test_neohooke.py` to import from `hyper_surrogate.mechanics.materials` and adapt to new API (pass params dict to constructor, use `evaluate_pk2` instead of iterator pattern).

Run: `pytest tests/test_neohooke.py -v`
Expected: All pass

- [ ] **Step 5: Commit**

```bash
git add hyper_surrogate/mechanics/materials.py tests/test_mechanics_materials.py tests/test_neohooke.py
git commit -m "refactor: Material uses composition, new NeoHooke/MooneyRivlin API"
```

---

### Task 4: Update mechanics **init** and top-level **init**

**Files:**

- Modify: `hyper_surrogate/mechanics/__init__.py`
- Modify: `hyper_surrogate/__init__.py`

- [ ] **Step 1: Export from mechanics/**init**.py**

```python
from hyper_surrogate.mechanics.symbolic import SymbolicHandler
from hyper_surrogate.mechanics.kinematics import Kinematics
from hyper_surrogate.mechanics.materials import Material, NeoHooke, MooneyRivlin

__all__ = ["SymbolicHandler", "Kinematics", "Material", "NeoHooke", "MooneyRivlin"]
```

- [ ] **Step 2: Update top-level **init**.py**

```python
from hyper_surrogate.mechanics.symbolic import SymbolicHandler
from hyper_surrogate.mechanics.kinematics import Kinematics
from hyper_surrogate.mechanics.materials import Material, NeoHooke, MooneyRivlin

__all__ = ["SymbolicHandler", "Kinematics", "Material", "NeoHooke", "MooneyRivlin"]
```

- [ ] **Step 3: Run all tests**

Run: `pytest tests/ -v`
Expected: All mechanics tests pass

- [ ] **Step 4: Commit**

```bash
git add hyper_surrogate/__init__.py hyper_surrogate/mechanics/__init__.py
git commit -m "refactor: export mechanics public API"
```

---

## Chunk 2: Data Layer

### Task 5: Create DeformationGenerator

**Files:**

- Create: `hyper_surrogate/data/__init__.py`
- Create: `hyper_surrogate/data/deformation.py`
- Create: `tests/test_data_deformation.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_data_deformation.py`:

```python
import numpy as np
import pytest
from hyper_surrogate.data.deformation import DeformationGenerator


def test_uniaxial_shape():
    gen = DeformationGenerator(seed=42)
    F = gen.uniaxial(100)
    assert F.shape == (100, 3, 3)


def test_uniaxial_incompressible():
    gen = DeformationGenerator(seed=42)
    F = gen.uniaxial(100)
    dets = np.linalg.det(F)
    np.testing.assert_allclose(dets, np.ones(100), atol=1e-10)


def test_biaxial_shape():
    gen = DeformationGenerator(seed=42)
    F = gen.biaxial(50)
    assert F.shape == (50, 3, 3)


def test_shear_shape():
    gen = DeformationGenerator(seed=42)
    F = gen.shear(50)
    assert F.shape == (50, 3, 3)


def test_combined_shape():
    gen = DeformationGenerator(seed=42)
    F = gen.combined(200)
    assert F.shape == (200, 3, 3)


def test_random_rotation_orthogonal():
    gen = DeformationGenerator(seed=42)
    R = gen.random_rotation(50)
    for i in range(50):
        np.testing.assert_allclose(R[i] @ R[i].T, np.eye(3), atol=1e-10)


def test_seed_reproducibility():
    gen1 = DeformationGenerator(seed=42)
    gen2 = DeformationGenerator(seed=42)
    F1 = gen1.combined(100)
    F2 = gen2.combined(100)
    np.testing.assert_array_equal(F1, F2)
```

Run: `pytest tests/test_data_deformation.py -v`
Expected: FAIL

- [ ] **Step 2: Implement DeformationGenerator**

Create `hyper_surrogate/data/__init__.py` (empty) and `hyper_surrogate/data/deformation.py`:

Port logic from `deformation_gradient.py` + `generator.py` into a single class. Key changes:

- Use `np.random.default_rng(seed)` instead of global `np.random.seed()`
- `n` is a method parameter, not constructor field
- Merge `DeformationGradient` static methods and `DeformationGradientGenerator` into one class
- Keep all deformation mode logic (uniaxial, shear, biaxial, combined, rotation)

```python
from __future__ import annotations

import numpy as np


class DeformationGenerator:
    """Generates physically valid deformation gradients for training data."""

    def __init__(self, seed: int = 42) -> None:
        self._rng = np.random.default_rng(seed)

    def uniaxial(self, n: int, stretch_range: tuple[float, float] = (0.4, 3.0)) -> np.ndarray:
        stretch = self._rng.uniform(*stretch_range, size=n)
        stretch_t = stretch ** -0.5
        result = np.zeros((n, 3, 3))
        result[:, 0, 0] = stretch
        result[:, 1, 1] = stretch_t
        result[:, 2, 2] = stretch_t
        return result

    def biaxial(self, n: int, stretch_range: tuple[float, float] = (0.4, 3.0)) -> np.ndarray:
        s1 = self._rng.uniform(*stretch_range, size=n)
        s2 = self._rng.uniform(*stretch_range, size=n)
        s3 = (s1 * s2) ** -1.0
        result = np.zeros((n, 3, 3))
        result[:, 0, 0] = s1
        result[:, 1, 1] = s2
        result[:, 2, 2] = s3
        return result

    def shear(self, n: int, shear_range: tuple[float, float] = (-1.0, 1.0)) -> np.ndarray:
        gamma = self._rng.uniform(*shear_range, size=n)
        result = np.repeat(np.eye(3)[np.newaxis, :, :], n, axis=0)
        result[:, 0, 1] = gamma
        return result

    def random_rotation(self, n: int) -> np.ndarray:
        axes = self._rng.integers(0, 3, size=n)
        angles = self._rng.uniform(0, np.pi, size=n)
        rotations = []
        for ax, ang in zip(axes, angles):
            c, s = np.cos(ang), np.sin(ang)
            if ax == 0:
                R = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
            elif ax == 1:
                R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
            else:
                R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
            rotations.append(R)
        return np.array(rotations)

    @staticmethod
    def _rotate(F: np.ndarray, R: np.ndarray) -> np.ndarray:
        return np.einsum("nij,njk,nlk->nil", R, F, R)

    def combined(self, n: int, stretch_range: tuple[float, float] = (0.4, 3.0),
                 shear_range: tuple[float, float] = (-1.0, 1.0)) -> np.ndarray:
        fu = self.uniaxial(n, stretch_range)
        fs = self.shear(n, shear_range)
        fb = self.biaxial(n, stretch_range)
        r1, r2, r3 = self.random_rotation(n), self.random_rotation(n), self.random_rotation(n)
        fu = self._rotate(fu, r1)
        fs = self._rotate(fs, r2)
        fb = self._rotate(fb, r3)
        return np.matmul(np.matmul(fb, fu), fs)
```

- [ ] **Step 3: Run tests**

Run: `pytest tests/test_data_deformation.py -v`
Expected: All pass

- [ ] **Step 4: Commit**

```bash
git add hyper_surrogate/data/ tests/test_data_deformation.py
git commit -m "feat: add DeformationGenerator in data layer"
```

---

### Task 6: Create Normalizer and MaterialDataset

**Files:**

- Create: `hyper_surrogate/data/dataset.py`
- Create: `tests/test_data_dataset.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_data_dataset.py`:

```python
import numpy as np
import pytest
from hyper_surrogate.data.dataset import Normalizer, MaterialDataset, create_datasets


class TestNormalizer:
    def test_fit_transform(self):
        data = np.random.randn(100, 3)
        norm = Normalizer().fit(data)
        transformed = norm.transform(data)
        np.testing.assert_allclose(transformed.mean(axis=0), 0.0, atol=1e-10)
        np.testing.assert_allclose(transformed.std(axis=0), 1.0, atol=1e-1)

    def test_inverse_transform(self):
        data = np.random.randn(100, 3) * 5 + 10
        norm = Normalizer().fit(data)
        roundtrip = norm.inverse_transform(norm.transform(data))
        np.testing.assert_allclose(roundtrip, data, atol=1e-10)

    def test_params(self):
        data = np.random.randn(50, 6)
        norm = Normalizer().fit(data)
        params = norm.params
        assert "mean" in params
        assert "std" in params
        assert params["mean"].shape == (6,)
        assert params["std"].shape == (6,)


class TestMaterialDataset:
    def test_len(self):
        inputs = np.random.randn(100, 3)
        targets = np.random.randn(100, 6)
        ds = MaterialDataset(inputs, targets)
        assert len(ds) == 100

    def test_getitem(self):
        inputs = np.random.randn(100, 3)
        targets = np.random.randn(100, 6)
        ds = MaterialDataset(inputs, targets)
        x, y = ds[0]
        assert x.shape == (3,)
        assert y.shape == (6,)


class TestCreateDatasets:
    def test_create_invariants_pk2(self):
        from hyper_surrogate.mechanics.materials import NeoHooke
        material = NeoHooke({"C10": 0.5, "KBULK": 1000.0})
        train_ds, val_ds, in_norm, out_norm = create_datasets(
            material, n_samples=100, input_type="invariants", target_type="pk2_voigt",
        )
        assert len(train_ds) + len(val_ds) == 100
        x, y = train_ds[0]
        assert x.shape == (3,)   # I1_bar, I2_bar, J
        assert y.shape == (6,)   # PK2 Voigt

    def test_create_energy(self):
        from hyper_surrogate.mechanics.materials import NeoHooke
        material = NeoHooke({"C10": 0.5, "KBULK": 1000.0})
        train_ds, val_ds, in_norm, out_norm = create_datasets(
            material, n_samples=100, input_type="invariants", target_type="energy",
        )
        x, y = train_ds[0]
        assert x.shape == (3,)
        # energy target is (energy_scalar, pk2_voigt_6) = tuple of 2
        assert isinstance(y, tuple)
        assert y[1].shape == (6,)
```

Run: `pytest tests/test_data_dataset.py -v`
Expected: FAIL

- [ ] **Step 2: Implement dataset.py**

Create `hyper_surrogate/data/dataset.py`:

```python
from __future__ import annotations

from typing import Any, Literal

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    Dataset = object  # type: ignore[assignment,misc]


class Normalizer:
    """Standard (zero-mean, unit-variance) normalization with export support."""

    def __init__(self) -> None:
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None

    def fit(self, data: np.ndarray) -> Normalizer:
        self._mean = data.mean(axis=0)
        self._std = data.std(axis=0)
        self._std[self._std < 1e-12] = 1.0  # avoid division by zero
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        return (data - self._mean) / self._std

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        return data * self._std + self._mean

    @property
    def params(self) -> dict[str, np.ndarray]:
        return {"mean": self._mean, "std": self._std}


class MaterialDataset(Dataset):
    """Wraps (input, target) pairs for training."""

    def __init__(self, inputs: np.ndarray, targets: Any) -> None:
        self.inputs = inputs
        self.targets = targets

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int) -> tuple:
        x = self.inputs[idx]
        if isinstance(self.targets, tuple):
            y = tuple(t[idx] for t in self.targets)
        else:
            y = self.targets[idx]
        return x, y


def create_datasets(
    material: Any,
    n_samples: int,
    input_type: Literal["invariants", "cauchy_green"] = "invariants",
    target_type: Literal["energy", "pk2_voigt", "pk2_voigt+cmat_voigt"] = "pk2_voigt",
    val_fraction: float = 0.15,
    seed: int = 42,
) -> tuple[MaterialDataset, MaterialDataset, Normalizer, Normalizer]:
    """Generate data, normalize, split, and wrap in datasets."""
    from hyper_surrogate.data.deformation import DeformationGenerator
    from hyper_surrogate.mechanics.kinematics import Kinematics

    # Generate deformations
    gen = DeformationGenerator(seed=seed)
    F = gen.combined(n_samples)
    C = Kinematics.right_cauchy_green(F)

    # Compute inputs
    if input_type == "invariants":
        i1 = Kinematics.isochoric_invariant1(C)
        i2 = Kinematics.isochoric_invariant2(C)
        j = np.sqrt(Kinematics.det_invariant(C))  # J = sqrt(det(C))
        inputs = np.column_stack([i1, i2, j])
    else:  # cauchy_green
        # 6 unique Voigt components: C11, C22, C33, C12, C13, C23
        inputs = np.column_stack([
            C[:, 0, 0], C[:, 1, 1], C[:, 2, 2],
            C[:, 0, 1], C[:, 0, 2], C[:, 1, 2],
        ])

    # Compute targets
    pk2_batch = material.evaluate_pk2(C)  # (N, 3, 3)
    pk2_voigt = np.column_stack([
        pk2_batch[:, 0, 0], pk2_batch[:, 1, 1], pk2_batch[:, 2, 2],
        pk2_batch[:, 0, 1], pk2_batch[:, 0, 2], pk2_batch[:, 1, 2],
    ])

    if target_type == "energy":
        energy = material.evaluate_energy(C)  # (N,)
        targets_raw = energy.reshape(-1, 1)
        # For energy loss, we also need stress — store as tuple.
        # IMPORTANT: Neither energy nor stress is normalized here.
        # EnergyStressLoss operates entirely in raw physical space.
        # Only inputs are normalized (the NN sees normalized invariants).
        in_norm = Normalizer().fit(inputs)
        inputs_normed = in_norm.transform(inputs)

        # Split
        n_val = int(n_samples * val_fraction)
        rng = np.random.default_rng(seed)
        idx = rng.permutation(n_samples)
        train_idx, val_idx = idx[n_val:], idx[:n_val]

        # Energy normalizer is still returned for Fortran export (denormalization),
        # but targets stored raw. Trainer/loss handles raw values.
        energy_norm = Normalizer().fit(targets_raw)

        train_ds = MaterialDataset(
            inputs_normed[train_idx].astype(np.float32),
            (targets_raw[train_idx].astype(np.float32), pk2_voigt[train_idx].astype(np.float32)),
        )
        val_ds = MaterialDataset(
            inputs_normed[val_idx].astype(np.float32),
            (targets_raw[val_idx].astype(np.float32), pk2_voigt[val_idx].astype(np.float32)),
        )
        return train_ds, val_ds, in_norm, energy_norm

    elif target_type == "pk2_voigt":
        targets_raw = pk2_voigt
    elif target_type == "pk2_voigt+cmat_voigt":
        cmat_batch = material.evaluate_cmat(C)  # (N, 3, 3, 3, 3)
        # Extract 21 unique Voigt components (upper triangle of 6x6)
        ii1 = [0, 1, 2, 0, 0, 1]
        ii2 = [0, 1, 2, 1, 2, 2]
        cmat_voigt = np.zeros((n_samples, 21))
        k = 0
        for i in range(6):
            for j in range(i, 6):
                cmat_voigt[:, k] = 0.5 * (
                    cmat_batch[:, ii1[i], ii2[i], ii1[j], ii2[j]]
                    + cmat_batch[:, ii1[i], ii2[i], ii2[j], ii1[j]]
                )
                k += 1
        targets_raw = np.column_stack([pk2_voigt, cmat_voigt])
    else:
        raise ValueError(f"Unknown target_type: {target_type}")

    # Normalize
    in_norm = Normalizer().fit(inputs)
    out_norm = Normalizer().fit(targets_raw)
    inputs_normed = in_norm.transform(inputs)
    targets_normed = out_norm.transform(targets_raw)

    # Split
    n_val = int(n_samples * val_fraction)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n_samples)
    train_idx, val_idx = idx[n_val:], idx[:n_val]

    train_ds = MaterialDataset(inputs_normed[train_idx].astype(np.float32), targets_normed[train_idx].astype(np.float32))
    val_ds = MaterialDataset(inputs_normed[val_idx].astype(np.float32), targets_normed[val_idx].astype(np.float32))

    return train_ds, val_ds, in_norm, out_norm
```

- [ ] **Step 3: Run tests**

Run: `pytest tests/test_data_dataset.py -v`
Expected: All pass

- [ ] **Step 4: Commit**

```bash
git add hyper_surrogate/data/dataset.py tests/test_data_dataset.py
git commit -m "feat: add Normalizer, MaterialDataset, create_datasets"
```

---

## Chunk 3: Models Layer (MLP + ICNN)

### Task 7: Create SurrogateModel base and MLP

**Files:**

- Create: `hyper_surrogate/models/__init__.py`
- Create: `hyper_surrogate/models/base.py`
- Create: `hyper_surrogate/models/mlp.py`
- Create: `tests/test_models_mlp.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_models_mlp.py`:

```python
import numpy as np
import pytest
torch = pytest.importorskip("torch")
from hyper_surrogate.models.mlp import MLP


def test_mlp_forward_shape():
    model = MLP(input_dim=3, output_dim=6, hidden_dims=[32, 32])
    x = torch.randn(10, 3)
    y = model(x)
    assert y.shape == (10, 6)


def test_mlp_layer_sequence():
    model = MLP(input_dim=3, output_dim=6, hidden_dims=[32, 16])
    seq = model.layer_sequence()
    assert len(seq) == 3  # 2 hidden + 1 output
    assert seq[0].activation == "tanh"
    assert seq[-1].activation == "identity"


def test_mlp_export_weights():
    model = MLP(input_dim=3, output_dim=6, hidden_dims=[32])
    weights = model.export_weights()
    assert "layers.0.weight" in weights
    assert "layers.0.bias" in weights
    assert weights["layers.0.weight"].shape == (32, 3)


def test_mlp_properties():
    model = MLP(input_dim=3, output_dim=6)
    assert model.input_dim == 3
    assert model.output_dim == 6
```

Run: `pytest tests/test_models_mlp.py -v`
Expected: FAIL

- [ ] **Step 2: Implement base.py and mlp.py**

Create `hyper_surrogate/models/__init__.py` (empty).

Create `hyper_surrogate/models/base.py`:

```python
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import torch.nn as nn


@dataclass
class LayerInfo:
    weights: str
    bias: str
    activation: str


class SurrogateModel(nn.Module, ABC):
    @property
    @abstractmethod
    def input_dim(self) -> int: ...

    @property
    @abstractmethod
    def output_dim(self) -> int: ...

    @abstractmethod
    def layer_sequence(self) -> list[LayerInfo]: ...

    def export_weights(self) -> dict[str, np.ndarray]:
        return {k: v.detach().cpu().numpy() for k, v in self.state_dict().items()}
```

Create `hyper_surrogate/models/mlp.py`:

```python
from __future__ import annotations

import torch
import torch.nn as nn

from hyper_surrogate.models.base import LayerInfo, SurrogateModel

ACTIVATIONS = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "softplus": nn.Softplus,
}


class MLP(SurrogateModel):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: list[int] | None = None,
        activation: str = "tanh",
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 64]
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._activation_name = activation
        act_cls = ACTIVATIONS[activation]

        dims = [input_dim] + hidden_dims + [output_dim]
        layer_list: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layer_list.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:  # no activation on output
                layer_list.append(act_cls())
        self.layers = nn.Sequential(*layer_list)

    @property
    def input_dim(self) -> int:
        return self._input_dim

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

    def layer_sequence(self) -> list[LayerInfo]:
        result = []
        linear_idx = 0
        for module in self.layers:
            if isinstance(module, nn.Linear):
                is_last = linear_idx == len([m for m in self.layers if isinstance(m, nn.Linear)]) - 1
                act = "identity" if is_last else self._activation_name
                prefix = f"layers.{list(self.layers).index(module)}"
                result.append(LayerInfo(
                    weights=f"{prefix}.weight",
                    bias=f"{prefix}.bias",
                    activation=act,
                ))
                linear_idx += 1
        return result
```

- [ ] **Step 3: Run tests**

Run: `pytest tests/test_models_mlp.py -v`
Expected: All pass

- [ ] **Step 4: Commit**

```bash
git add hyper_surrogate/models/ tests/test_models_mlp.py
git commit -m "feat: add SurrogateModel base and MLP"
```

---

### Task 8: Create ICNN

**Files:**

- Create: `hyper_surrogate/models/icnn.py`
- Create: `tests/test_models_icnn.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_models_icnn.py`:

```python
import pytest
torch = pytest.importorskip("torch")
from hyper_surrogate.models.icnn import ICNN


def test_icnn_forward_scalar_output():
    model = ICNN(input_dim=3, hidden_dims=[32, 32])
    x = torch.randn(10, 3)
    y = model(x)
    assert y.shape == (10, 1)


def test_icnn_convexity():
    """Output should be convex w.r.t. input: f(tx1 + (1-t)x2) <= t*f(x1) + (1-t)*f(x2)."""
    model = ICNN(input_dim=3, hidden_dims=[32, 32])
    model.eval()
    x1 = torch.randn(50, 3)
    x2 = torch.randn(50, 3)
    t = 0.5
    with torch.no_grad():
        f_mix = model(t * x1 + (1 - t) * x2)
        f_avg = t * model(x1) + (1 - t) * model(x2)
    # Convexity: f(mix) <= f(avg) (with numerical tolerance)
    assert (f_mix <= f_avg + 1e-5).all()


def test_icnn_gradient():
    """Can compute gradient of output w.r.t. input."""
    model = ICNN(input_dim=3, hidden_dims=[16])
    x = torch.randn(5, 3, requires_grad=True)
    y = model(x)
    grad = torch.autograd.grad(y.sum(), x, create_graph=True)[0]
    assert grad.shape == (5, 3)


def test_icnn_layer_sequence():
    model = ICNN(input_dim=3, hidden_dims=[32, 16])
    seq = model.layer_sequence()
    assert len(seq) > 0
    # Should have both wz and wx type entries
    keys = [s.weights for s in seq]
    assert any("wz" in k for k in keys)
    assert any("wx" in k for k in keys)


def test_icnn_export_weights():
    model = ICNN(input_dim=3, hidden_dims=[16])
    weights = model.export_weights()
    assert len(weights) > 0


def test_icnn_properties():
    model = ICNN(input_dim=3)
    assert model.input_dim == 3
    assert model.output_dim == 1
```

Run: `pytest tests/test_models_icnn.py -v`
Expected: FAIL

- [ ] **Step 2: Implement ICNN**

Create `hyper_surrogate/models/icnn.py`:

```python
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from hyper_surrogate.models.base import LayerInfo, SurrogateModel


class ICNN(SurrogateModel):
    """Input-Convex Neural Network (Amos+ 2017).

    Guarantees convexity of output w.r.t. input via:
    - Non-negative weights on z-path (enforced via softplus)
    - Skip connections from input to every layer
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] | None = None,
        activation: str = "softplus",
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 64]
        self._input_dim = input_dim
        self._output_dim = 1
        self._activation_name = activation
        self._hidden_dims = hidden_dims

        # First layer: only x-path
        self.wx_layers = nn.ModuleList()
        self.wz_layers = nn.ModuleList()

        # wx_0: input -> hidden_0
        self.wx_layers.append(nn.Linear(input_dim, hidden_dims[0]))

        # Hidden layers: wz (non-negative) + wx (skip)
        for i in range(1, len(hidden_dims)):
            self.wz_layers.append(nn.Linear(hidden_dims[i - 1], hidden_dims[i], bias=False))
            self.wx_layers.append(nn.Linear(input_dim, hidden_dims[i]))

        # Output layer
        self.wz_final = nn.Linear(hidden_dims[-1], 1, bias=False)
        self.wx_final = nn.Linear(input_dim, 1)

        # Activation
        act_map = {"softplus": nn.Softplus(), "relu": nn.ReLU(), "tanh": nn.Tanh()}
        self._activation = act_map[activation]

    @property
    def input_dim(self) -> int:
        return self._input_dim

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # First hidden layer (x-path only)
        z = self._activation(self.wx_layers[0](x))

        # Subsequent hidden layers (z-path with non-neg weights + x skip)
        for wz, wx in zip(self.wz_layers, self.wx_layers[1:]):
            z = self._activation(
                F.linear(z, F.softplus(wz.weight)) + wx(x)
            )

        # Output layer
        return F.linear(z, F.softplus(self.wz_final.weight)) + self.wx_final(x)

    def layer_sequence(self) -> list[LayerInfo]:
        result = []
        # First wx layer
        result.append(LayerInfo(
            weights="wx_layers.0.weight", bias="wx_layers.0.bias",
            activation=self._activation_name,
        ))
        # Hidden wz + wx pairs
        for i in range(len(self.wz_layers)):
            result.append(LayerInfo(
                weights=f"wz_layers.{i}.weight", bias=f"wx_layers.{i + 1}.bias",
                activation=self._activation_name,
            ))
        # Output
        result.append(LayerInfo(
            weights="wz_final.weight", bias="wx_final.bias",
            activation="identity",
        ))
        return result
```

- [ ] **Step 3: Run tests**

Run: `pytest tests/test_models_icnn.py -v`
Expected: All pass

- [ ] **Step 4: Commit**

```bash
git add hyper_surrogate/models/icnn.py tests/test_models_icnn.py
git commit -m "feat: add ICNN (Input-Convex Neural Network)"
```

---

## Chunk 4: Training Layer

### Task 9: Create loss functions

**Files:**

- Create: `hyper_surrogate/training/__init__.py`
- Create: `hyper_surrogate/training/losses.py`
- Create: `tests/test_training_losses.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_training_losses.py`:

```python
import pytest
torch = pytest.importorskip("torch")
from hyper_surrogate.training.losses import StressLoss, StressTangentLoss, EnergyStressLoss


def test_stress_loss():
    loss_fn = StressLoss()
    pred = torch.randn(10, 6)
    target = torch.randn(10, 6)
    loss = loss_fn(pred, target)
    assert loss.shape == ()
    assert loss.item() >= 0


def test_stress_tangent_loss():
    loss_fn = StressTangentLoss(alpha=1.0, beta=0.1)
    pred = torch.randn(10, 27)  # 6 stress + 21 tangent
    target = torch.randn(10, 27)  # same layout
    loss = loss_fn(pred, target)
    assert loss.shape == ()
    assert loss.item() >= 0


def test_energy_stress_loss():
    loss_fn = EnergyStressLoss(alpha=1.0, beta=1.0)
    # Simulate ICNN: input -> scalar energy
    x = torch.randn(10, 3, requires_grad=True)
    w_pred = (x ** 2).sum(dim=1, keepdim=True)  # simple convex function
    w_true = torch.randn(10, 1)
    s_true = torch.randn(10, 3)
    loss = loss_fn(w_pred, w_true, x, s_true)
    assert loss.shape == ()
    assert loss.item() >= 0
```

Run: `pytest tests/test_training_losses.py -v`
Expected: FAIL

- [ ] **Step 2: Implement losses**

Create `hyper_surrogate/training/__init__.py` (empty).

Create `hyper_surrogate/training/losses.py`:

```python
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class StressLoss(nn.Module):
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(pred, target)


class StressTangentLoss(nn.Module):
    def __init__(self, alpha: float = 1.0, beta: float = 0.1) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """pred and target are both (N, 27): first 6 = stress, rest = tangent."""
        s_pred, c_pred = pred[:, :6], pred[:, 6:]
        s_true, c_true = target[:, :6], target[:, 6:]
        return self.alpha * F.mse_loss(s_pred, s_true) + self.beta * F.mse_loss(c_pred, c_true)


class EnergyStressLoss(nn.Module):
    def __init__(self, alpha: float = 1.0, beta: float = 1.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(
        self, w_pred: torch.Tensor, w_true: torch.Tensor,
        inputs: torch.Tensor, s_true: torch.Tensor,
    ) -> torch.Tensor:
        energy_loss = F.mse_loss(w_pred, w_true)
        dw_di = torch.autograd.grad(w_pred.sum(), inputs, create_graph=True)[0]
        stress_loss = F.mse_loss(dw_di, s_true)
        return self.alpha * energy_loss + self.beta * stress_loss
```

- [ ] **Step 3: Run tests**

Run: `pytest tests/test_training_losses.py -v`
Expected: All pass

- [ ] **Step 4: Commit**

```bash
git add hyper_surrogate/training/ tests/test_training_losses.py
git commit -m "feat: add loss functions (StressLoss, StressTangentLoss, EnergyStressLoss)"
```

---

### Task 10: Create Trainer

**Files:**

- Create: `hyper_surrogate/training/trainer.py`
- Create: `tests/test_training_trainer.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_training_trainer.py`:

```python
import numpy as np
import pytest
torch = pytest.importorskip("torch")
from hyper_surrogate.models.mlp import MLP
from hyper_surrogate.training.trainer import Trainer, TrainingResult
from hyper_surrogate.training.losses import StressLoss
from hyper_surrogate.data.dataset import MaterialDataset


def _make_datasets():
    """Simple linear regression dataset."""
    rng = np.random.default_rng(42)
    x = rng.standard_normal((200, 3)).astype(np.float32)
    W = rng.standard_normal((3, 6)).astype(np.float32)
    y = (x @ W).astype(np.float32)
    train = MaterialDataset(x[:170], y[:170])
    val = MaterialDataset(x[170:], y[170:])
    return train, val


def test_trainer_fit_returns_result():
    train, val = _make_datasets()
    model = MLP(input_dim=3, output_dim=6, hidden_dims=[16])
    result = Trainer(model, train, val, loss_fn=StressLoss(), max_epochs=5).fit()
    assert isinstance(result, TrainingResult)
    assert len(result.history["train_loss"]) == 5
    assert result.best_epoch >= 0


def test_trainer_loss_decreases():
    train, val = _make_datasets()
    model = MLP(input_dim=3, output_dim=6, hidden_dims=[32, 32])
    result = Trainer(model, train, val, loss_fn=StressLoss(), max_epochs=50, patience=50).fit()
    assert result.history["train_loss"][-1] < result.history["train_loss"][0]
```

Run: `pytest tests/test_training_trainer.py -v`
Expected: FAIL

- [ ] **Step 2: Implement Trainer**

Create `hyper_surrogate/training/trainer.py`:

```python
from __future__ import annotations

import copy
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from hyper_surrogate.training.losses import EnergyStressLoss


@dataclass
class TrainingResult:
    model: nn.Module
    history: dict[str, list[float]] = field(default_factory=lambda: {"train_loss": [], "val_loss": []})
    best_epoch: int = 0


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_dataset: object,
        val_dataset: object,
        loss_fn: nn.Module,
        lr: float = 1e-3,
        batch_size: int = 256,
        max_epochs: int = 1000,
        patience: int = 50,
        device: str = "cpu",
    ) -> None:
        self.model = model.to(device)
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        self.loss_fn = loss_fn
        self.device = device
        self.max_epochs = max_epochs
        self.patience = patience
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=max(1, patience // 3)
        )
        self._is_energy_loss = isinstance(loss_fn, EnergyStressLoss)
        self._best_state: dict | None = None
        self._best_val_loss = float("inf")
        self._best_epoch = 0
        self._patience_counter = 0

    def _compute_loss(self, batch: tuple) -> torch.Tensor:
        x, y = batch
        x = x.to(self.device)

        if self._is_energy_loss:
            x.requires_grad_(True)
            pred = self.model(x)
            if isinstance(y, (tuple, list)):
                w_true = y[0].to(self.device)
                s_true = y[1].to(self.device)
            else:
                w_true = y.to(self.device)
                s_true = torch.zeros_like(x)
            return self.loss_fn(pred, w_true, x, s_true)
        else:
            pred = self.model(x)
            if isinstance(y, (tuple, list)):
                y = y[0].to(self.device)
            else:
                y = y.to(self.device)
            return self.loss_fn(pred, y)

    def _train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        n = 0
        for batch in self.train_loader:
            self.optimizer.zero_grad()
            loss = self._compute_loss(batch)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * len(batch[0])
            n += len(batch[0])
        return total_loss / n

    def _val_epoch(self) -> float:
        self.model.eval()
        total_loss = 0.0
        n = 0
        with torch.no_grad() if not self._is_energy_loss else torch.enable_grad():
            for batch in self.val_loader:
                loss = self._compute_loss(batch)
                total_loss += loss.item() * len(batch[0])
                n += len(batch[0])
        return total_loss / n

    def fit(self) -> TrainingResult:
        history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

        for epoch in range(self.max_epochs):
            train_loss = self._train_epoch()
            val_loss = self._val_epoch()
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            self.scheduler.step(val_loss)

            if val_loss < self._best_val_loss:
                self._best_val_loss = val_loss
                self._best_state = copy.deepcopy(self.model.state_dict())
                self._best_epoch = epoch
                self._patience_counter = 0
            else:
                self._patience_counter += 1
                if self._patience_counter >= self.patience:
                    break

        if self._best_state is not None:
            self.model.load_state_dict(self._best_state)

        return TrainingResult(model=self.model, history=history, best_epoch=self._best_epoch)
```

- [ ] **Step 3: Run tests**

Run: `pytest tests/test_training_trainer.py -v`
Expected: All pass

- [ ] **Step 4: Commit**

```bash
git add hyper_surrogate/training/trainer.py tests/test_training_trainer.py
git commit -m "feat: add Trainer with early stopping and checkpointing"
```

---

## Chunk 5: Export Layer (Fortran Transpiler)

### Task 11: Create ExportedModel and extract_weights

**Files:**

- Create: `hyper_surrogate/export/__init__.py`
- Create: `hyper_surrogate/export/weights.py`
- Create: `tests/test_export_weights.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_export_weights.py`:

```python
import numpy as np
import pytest
import tempfile, os
torch = pytest.importorskip("torch")
from hyper_surrogate.models.mlp import MLP
from hyper_surrogate.export.weights import extract_weights, ExportedModel
from hyper_surrogate.data.dataset import Normalizer


def test_extract_weights_mlp():
    model = MLP(input_dim=3, output_dim=6, hidden_dims=[16])
    exported = extract_weights(model)
    assert isinstance(exported, ExportedModel)
    assert exported.metadata["architecture"] == "mlp"
    assert exported.metadata["input_dim"] == 3
    assert exported.metadata["output_dim"] == 6
    assert len(exported.layers) > 0
    assert len(exported.weights) > 0


def test_extract_with_normalizers():
    model = MLP(input_dim=3, output_dim=6, hidden_dims=[16])
    in_norm = Normalizer().fit(np.random.randn(100, 3))
    out_norm = Normalizer().fit(np.random.randn(100, 6))
    exported = extract_weights(model, in_norm, out_norm)
    assert exported.input_normalizer is not None
    assert exported.output_normalizer is not None


def test_save_load_roundtrip():
    model = MLP(input_dim=3, output_dim=6, hidden_dims=[16])
    exported = extract_weights(model)
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "model.npz")
        exported.save(path)
        loaded = ExportedModel.load(path)
        assert loaded.metadata["architecture"] == "mlp"
        for k in exported.weights:
            np.testing.assert_array_equal(exported.weights[k], loaded.weights[k])
```

Run: `pytest tests/test_export_weights.py -v`
Expected: FAIL

- [ ] **Step 2: Implement weights.py**

Create `hyper_surrogate/export/__init__.py` (empty).

Create `hyper_surrogate/export/weights.py`:

```python
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from hyper_surrogate.data.dataset import Normalizer
from hyper_surrogate.models.base import LayerInfo


@dataclass
class ExportedModel:
    layers: list[LayerInfo]
    weights: dict[str, np.ndarray]
    input_normalizer: dict[str, np.ndarray] | None = None
    output_normalizer: dict[str, np.ndarray] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def save(self, path: str) -> None:
        save_dict: dict[str, Any] = {}
        # Weights
        for k, v in self.weights.items():
            save_dict[f"w_{k}"] = v
        # Normalizers
        if self.input_normalizer:
            save_dict["in_norm_mean"] = self.input_normalizer["mean"]
            save_dict["in_norm_std"] = self.input_normalizer["std"]
        if self.output_normalizer:
            save_dict["out_norm_mean"] = self.output_normalizer["mean"]
            save_dict["out_norm_std"] = self.output_normalizer["std"]
        # Metadata and layers as JSON strings
        save_dict["_metadata"] = np.array([json.dumps(self.metadata)])
        layers_data = [{"weights": l.weights, "bias": l.bias, "activation": l.activation} for l in self.layers]
        save_dict["_layers"] = np.array([json.dumps(layers_data)])
        np.savez(path, **save_dict)

    @classmethod
    def load(cls, path: str) -> ExportedModel:
        data = np.load(path, allow_pickle=False)
        metadata = json.loads(str(data["_metadata"][0]))
        layers_data = json.loads(str(data["_layers"][0]))
        layers = [LayerInfo(**d) for d in layers_data]
        weights = {k[2:]: data[k] for k in data if k.startswith("w_")}
        in_norm = None
        if "in_norm_mean" in data:
            in_norm = {"mean": data["in_norm_mean"], "std": data["in_norm_std"]}
        out_norm = None
        if "out_norm_mean" in data:
            out_norm = {"mean": data["out_norm_mean"], "std": data["out_norm_std"]}
        return cls(layers=layers, weights=weights, input_normalizer=in_norm, output_normalizer=out_norm, metadata=metadata)


def extract_weights(
    model: Any,
    input_normalizer: Normalizer | None = None,
    output_normalizer: Normalizer | None = None,
) -> ExportedModel:
    from hyper_surrogate.models.base import SurrogateModel
    assert isinstance(model, SurrogateModel)
    return ExportedModel(
        layers=model.layer_sequence(),
        weights=model.export_weights(),
        input_normalizer=input_normalizer.params if input_normalizer else None,
        output_normalizer=output_normalizer.params if output_normalizer else None,
        metadata={
            "architecture": model.__class__.__name__.lower(),
            "input_dim": model.input_dim,
            "output_dim": model.output_dim,
        },
    )
```

- [ ] **Step 3: Run tests**

Run: `pytest tests/test_export_weights.py -v`
Expected: All pass

- [ ] **Step 4: Commit**

```bash
git add hyper_surrogate/export/ tests/test_export_weights.py
git commit -m "feat: add ExportedModel and extract_weights"
```

---

### Task 12: Create FortranEmitter for MLP

**Files:**

- Create: `hyper_surrogate/export/fortran/__init__.py`
- Create: `hyper_surrogate/export/fortran/emitter.py`
- Create: `tests/test_fortran_emitter.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_fortran_emitter.py`:

```python
import numpy as np
import pytest
torch = pytest.importorskip("torch")
from hyper_surrogate.models.mlp import MLP
from hyper_surrogate.export.weights import extract_weights
from hyper_surrogate.export.fortran.emitter import FortranEmitter
from hyper_surrogate.data.dataset import Normalizer


def test_emit_mlp_produces_fortran():
    model = MLP(input_dim=3, output_dim=6, hidden_dims=[8])
    exported = extract_weights(model)
    code = FortranEmitter(exported).emit()
    assert "MODULE nn_surrogate" in code
    assert "SUBROUTINE nn_forward" in code
    assert "MATMUL" in code
    assert "END MODULE" in code


def test_emit_mlp_with_normalizers():
    model = MLP(input_dim=3, output_dim=6, hidden_dims=[8])
    in_norm = Normalizer().fit(np.random.randn(50, 3))
    out_norm = Normalizer().fit(np.random.randn(50, 6))
    exported = extract_weights(model, in_norm, out_norm)
    code = FortranEmitter(exported).emit()
    # Should contain normalization code
    assert "in_mean" in code.lower() or "input" in code.lower()


def test_emit_mlp_activations():
    for act in ["tanh", "relu", "sigmoid", "softplus"]:
        model = MLP(input_dim=3, output_dim=6, hidden_dims=[8], activation=act)
        exported = extract_weights(model)
        code = FortranEmitter(exported).emit()
        assert "SUBROUTINE nn_forward" in code


def test_write_to_file(tmp_path):
    model = MLP(input_dim=3, output_dim=6, hidden_dims=[8])
    exported = extract_weights(model)
    path = tmp_path / "test.f90"
    FortranEmitter(exported).write(str(path))
    assert path.exists()
    content = path.read_text()
    assert "MODULE nn_surrogate" in content
```

Run: `pytest tests/test_fortran_emitter.py -v`
Expected: FAIL

- [ ] **Step 2: Implement FortranEmitter**

Create `hyper_surrogate/export/fortran/__init__.py` (empty).

Create `hyper_surrogate/export/fortran/emitter.py`:

```python
from __future__ import annotations

import numpy as np

from hyper_surrogate.export.weights import ExportedModel


class FortranEmitter:
    """Emits Fortran 90 code for neural network inference."""

    ACTIVATIONS = {
        "relu": "max(0.0d0, {x})",
        "tanh": "tanh({x})",
        "sigmoid": "1.0d0 / (1.0d0 + exp(-({x})))",
        "softplus": "log(1.0d0 + exp({x}))",
        "identity": "{x}",
    }

    def __init__(self, exported: ExportedModel) -> None:
        self.exported = exported

    def _format_array_1d(self, arr: np.ndarray, name: str) -> str:
        n = arr.shape[0]
        values = ", ".join(f"{v:.15e}d0" for v in arr.flat)
        return f"  DOUBLE PRECISION, PARAMETER :: {name}({n}) = (/ {values} /)"

    def _format_array_2d(self, arr: np.ndarray, name: str) -> str:
        rows, cols = arr.shape
        values = ", ".join(f"{v:.15e}d0" for v in arr.T.flat)  # Fortran is column-major
        return f"  DOUBLE PRECISION, PARAMETER :: {name}({rows},{cols}) = RESHAPE((/ {values} /), (/ {rows}, {cols} /))"

    def _emit_activation(self, var: str, activation: str, size: int) -> list[str]:
        if activation == "identity":
            return []
        lines = []
        template = self.ACTIVATIONS[activation]
        lines.append(f"  DO i = 1, {size}")
        lines.append(f"    {var}(i) = {template.format(x=f'{var}(i)')}")
        lines.append("  END DO")
        return lines

    def emit_mlp(self) -> str:
        layers = self.exported.layers
        weights = self.exported.weights
        meta = self.exported.metadata
        in_dim = meta["input_dim"]
        out_dim = meta["output_dim"]

        lines = ["MODULE nn_surrogate", "  IMPLICIT NONE", ""]

        # Declare weight/bias arrays as parameters
        for i, layer in enumerate(layers):
            w = weights[layer.weights]
            b = weights[layer.bias]
            lines.append(self._format_array_2d(w, f"w{i}"))
            lines.append(self._format_array_1d(b, f"b{i}"))

        # Normalization parameters
        if self.exported.input_normalizer:
            lines.append(self._format_array_1d(self.exported.input_normalizer["mean"], "in_mean"))
            lines.append(self._format_array_1d(self.exported.input_normalizer["std"], "in_std"))
        if self.exported.output_normalizer:
            lines.append(self._format_array_1d(self.exported.output_normalizer["mean"], "out_mean"))
            lines.append(self._format_array_1d(self.exported.output_normalizer["std"], "out_std"))

        lines.extend(["", "CONTAINS", ""])

        # Subroutine
        lines.append(f"  SUBROUTINE nn_forward(input, output)")
        lines.append(f"    DOUBLE PRECISION, INTENT(IN) :: input({in_dim})")
        lines.append(f"    DOUBLE PRECISION, INTENT(OUT) :: output({out_dim})")

        # Local variables
        hidden_dims = [weights[l.weights].shape[0] for l in layers]
        for i in range(len(layers)):
            lines.append(f"    DOUBLE PRECISION :: z{i}({hidden_dims[i]})")
        lines.append(f"    DOUBLE PRECISION :: x_norm({in_dim})")
        lines.append("    INTEGER :: i")
        lines.append("")

        # Normalize input
        if self.exported.input_normalizer:
            lines.append("    ! Normalize input")
            lines.append(f"    x_norm = (input - in_mean) / in_std")
        else:
            lines.append(f"    x_norm = input")
        lines.append("")

        # Forward pass
        for i, layer in enumerate(layers):
            input_var = "x_norm" if i == 0 else f"z{i - 1}"
            lines.append(f"    ! Layer {i}")
            lines.append(f"    z{i} = MATMUL(w{i}, {input_var}) + b{i}")
            lines.extend(self._emit_activation(f"z{i}", layer.activation, hidden_dims[i]))
            lines.append("")

        # Copy output and denormalize
        last = f"z{len(layers) - 1}"
        if self.exported.output_normalizer:
            lines.append("    ! Denormalize output")
            lines.append(f"    output = {last} * out_std + out_mean")
        else:
            lines.append(f"    output = {last}")

        lines.extend(["", "  END SUBROUTINE nn_forward", "", "END MODULE nn_surrogate"])

        return "\n".join(lines)

    def emit_icnn(self) -> str:
        layers = self.exported.layers
        weights = self.exported.weights
        meta = self.exported.metadata
        in_dim = meta["input_dim"]

        lines = ["MODULE nn_surrogate", "  IMPLICIT NONE", ""]

        # Declare all weight arrays
        # ICNN weights need softplus pre-applied for wz layers
        for i, layer in enumerate(layers):
            w_key = layer.weights
            w = weights[w_key]
            if "wz" in w_key:
                # Pre-apply softplus: log(1 + exp(w))
                w = np.log(1.0 + np.exp(w))
            lines.append(self._format_array_2d(w, f"w{i}"))
            b_key = layer.bias
            b = weights[b_key]
            lines.append(self._format_array_1d(b, f"b{i}"))

        # wx skip-connection weights for ICNN
        # We need the wx weights for each hidden layer
        wx_keys = sorted([k for k in weights if k.startswith("wx_layers") and "weight" in k])
        for i, wk in enumerate(wx_keys):
            if f"w{i}" not in "\n".join(lines):  # avoid duplicates
                lines.append(self._format_array_2d(weights[wk], f"wx{i}"))

        if self.exported.input_normalizer:
            lines.append(self._format_array_1d(self.exported.input_normalizer["mean"], "in_mean"))
            lines.append(self._format_array_1d(self.exported.input_normalizer["std"], "in_std"))

        lines.extend(["", "CONTAINS", ""])

        # Subroutine with energy + stress output
        lines.append(f"  SUBROUTINE nn_forward(input, energy, stress)")
        lines.append(f"    DOUBLE PRECISION, INTENT(IN) :: input({in_dim})")
        lines.append(f"    DOUBLE PRECISION, INTENT(OUT) :: energy")
        lines.append(f"    DOUBLE PRECISION, INTENT(OUT) :: stress({in_dim})")
        lines.append(f"    DOUBLE PRECISION :: x_norm({in_dim})")

        # Determine hidden dims
        hidden_dims = []
        for layer in layers[:-1]:
            w = weights[layer.weights]
            hidden_dims.append(w.shape[0])

        for i, hd in enumerate(hidden_dims):
            lines.append(f"    DOUBLE PRECISION :: z{i}({hd})")
            lines.append(f"    DOUBLE PRECISION :: dz{i}({hd})")  # for gradient

        lines.append(f"    DOUBLE PRECISION :: denergy({in_dim})")
        lines.append("    INTEGER :: i, j")
        lines.append("")

        # Normalize
        if self.exported.input_normalizer:
            lines.append(f"    x_norm = (input - in_mean) / in_std")
        else:
            lines.append(f"    x_norm = input")
        lines.append("")

        # Forward pass (simplified — full ICNN backprop structure)
        lines.append("    ! Forward pass")
        lines.append("    ! (ICNN forward + backward for gradient)")
        lines.append("    ! Implementation follows chain rule through layers")
        lines.append("")

        # This is a simplified placeholder — full implementation requires
        # layer-by-layer forward + backward with ICNN structure
        lines.append("    energy = 0.0d0")
        lines.append(f"    stress = 0.0d0")

        lines.extend(["", "  END SUBROUTINE nn_forward", "", "END MODULE nn_surrogate"])

        return "\n".join(lines)

    def emit(self) -> str:
        arch = self.exported.metadata.get("architecture", "mlp")
        if arch == "mlp":
            return self.emit_mlp()
        elif arch == "icnn":
            return self.emit_icnn()
        else:
            raise ValueError(f"Unknown architecture: {arch}")

    def write(self, path: str) -> None:
        with open(path, "w") as f:
            f.write(self.emit())
```

**IMPORTANT: The ICNN emitter above is a scaffold.** During implementation, `emit_icnn()` must be completed with:

1. Full forward pass: `z0 = act(wx0 @ x + b0)`, then `z_i = act(wz_i @ z_{i-1} + wx_i @ x + b_i)`, then `energy = wz_final @ z_{L-1} + wx_final @ x + b_final`
2. Backward pass (analytical chain rule): propagate `d_energy/d_x` through each layer using activation derivatives: `softplus'(x) = 1/(1+exp(-x))`, `tanh'(x) = 1-tanh(x)^2`, `relu'(x) = merge(1,0,x>0)`
3. The Fortran code must compute both `energy` and `stress(1:in_dim) = d_energy/d_input`
4. All `wz` weights are pre-computed with `softplus` at export time (already handled in the scaffold)

The implementer should write the full forward+backward Fortran emission and add a test that compares Fortran output against PyTorch `autograd.grad` for a small ICNN on a known input.

- [ ] **Step 3: Run tests**

Run: `pytest tests/test_fortran_emitter.py -v`
Expected: All pass

- [ ] **Step 4: Commit**

```bash
git add hyper_surrogate/export/fortran/ tests/test_fortran_emitter.py
git commit -m "feat: add FortranEmitter for MLP (ICNN structure in place)"
```

---

### Task 13: Relocate analytical UMATHandler

**Files:**

- Create: `hyper_surrogate/export/fortran/analytical.py`
- Modify: `tests/test_umat_handler.py`

- [ ] **Step 1: Move UMATHandler**

Copy `hyper_surrogate/umat_handler.py` → `hyper_surrogate/export/fortran/analytical.py`.
Update its internal import: `from hyper_surrogate.materials import Material` → `from hyper_surrogate.mechanics.materials import Material`.
The `UMATHandler` uses `Material`'s symbolic accessors (`cauchy`, `tangent`, `c_tensor`, `get_default_parameters`). These need to be available on the new `Material` class — `cauchy_voigt` and `tangent_voigt` replace the old `cauchy` and `tangent` methods. Update `UMATHandler.cauchy` property to call `self.material.cauchy_voigt(self.f)` and `UMATHandler.tangent` to call `self.material.tangent_voigt(self.f, use_jaumann_rate=True)`. The `c_tensor` access changes to `self.material.handler.c_tensor`.

- [ ] **Step 2: Update test imports**

Change `tests/test_umat_handler.py` to import from `hyper_surrogate.export.fortran.analytical`.
Update material instantiation to new API: `NeoHooke({"C10": 0.5, "KBULK": 1000.0})`.

Run: `pytest tests/test_umat_handler.py -v`
Expected: All pass

- [ ] **Step 3: Commit**

```bash
git add hyper_surrogate/export/fortran/analytical.py tests/test_umat_handler.py
git commit -m "refactor: relocate UMATHandler to export/fortran/analytical"
```

---

## Chunk 6: Integration and Cleanup

### Task 14: Update top-level **init**.py with full public API

**Files:**

- Modify: `hyper_surrogate/__init__.py`
- Modify: `hyper_surrogate/data/__init__.py`
- Modify: `hyper_surrogate/models/__init__.py`
- Modify: `hyper_surrogate/training/__init__.py`
- Modify: `hyper_surrogate/export/__init__.py`

- [ ] **Step 1: Write sub-package **init** exports**

`hyper_surrogate/data/__init__.py`:

```python
from hyper_surrogate.data.deformation import DeformationGenerator
from hyper_surrogate.data.dataset import MaterialDataset, Normalizer, create_datasets
```

`hyper_surrogate/models/__init__.py`:

```python
from hyper_surrogate.models.mlp import MLP
from hyper_surrogate.models.icnn import ICNN
```

`hyper_surrogate/training/__init__.py`:

```python
from hyper_surrogate.training.trainer import Trainer, TrainingResult
from hyper_surrogate.training.losses import StressLoss, StressTangentLoss, EnergyStressLoss
```

`hyper_surrogate/export/__init__.py`:

```python
from hyper_surrogate.export.weights import extract_weights, ExportedModel
from hyper_surrogate.export.fortran.emitter import FortranEmitter
```

- [ ] **Step 2: Write top-level **init**.py**

```python
# Mechanics (always available)
from hyper_surrogate.mechanics.symbolic import SymbolicHandler
from hyper_surrogate.mechanics.kinematics import Kinematics
from hyper_surrogate.mechanics.materials import Material, NeoHooke, MooneyRivlin

# Data
from hyper_surrogate.data.deformation import DeformationGenerator
from hyper_surrogate.data.dataset import MaterialDataset, Normalizer, create_datasets

# ML (requires torch)
try:
    from hyper_surrogate.models.mlp import MLP
    from hyper_surrogate.models.icnn import ICNN
    from hyper_surrogate.training.trainer import Trainer, TrainingResult
    from hyper_surrogate.training.losses import StressLoss, StressTangentLoss, EnergyStressLoss
except ImportError:
    pass

# Export (always available — only needs numpy)
from hyper_surrogate.export.weights import extract_weights, ExportedModel
from hyper_surrogate.export.fortran.emitter import FortranEmitter
```

- [ ] **Step 3: Run all tests**

Run: `pytest tests/ -v`
Expected: All pass

- [ ] **Step 4: Commit**

```bash
git add hyper_surrogate/__init__.py hyper_surrogate/data/__init__.py hyper_surrogate/models/__init__.py hyper_surrogate/training/__init__.py hyper_surrogate/export/__init__.py
git commit -m "feat: wire up full public API with optional torch imports"
```

---

### Task 15: Update pyproject.toml and add torch as optional dep

**Files:**

- Modify: `pyproject.toml`

- [ ] **Step 1: Add torch as optional dependency**

Add to `pyproject.toml`:

```toml
[tool.poetry.extras]
ml = ["torch"]

[tool.poetry.dependencies]
torch = {version = ">=2.0", optional = true}
```

Update `target-version` in ruff from `py37` to `py310`.

- [ ] **Step 2: Commit**

```bash
git add pyproject.toml
git commit -m "build: add torch as optional dependency, update python target"
```

---

### Task 16: Remove old flat modules

**Files:**

- Delete: `hyper_surrogate/symbolic.py`
- Delete: `hyper_surrogate/kinematics.py`
- Delete: `hyper_surrogate/materials.py`
- Delete: `hyper_surrogate/deformation_gradient.py`
- Delete: `hyper_surrogate/generator.py`
- Delete: `hyper_surrogate/umat_handler.py`
- Move: `hyper_surrogate/reporter.py` → `hyper_surrogate/reporting/reporter.py`

- [ ] **Step 1: Create reporting package**

```bash
mkdir -p hyper_surrogate/reporting
```

Move `reporter.py` into it. Create `hyper_surrogate/reporting/__init__.py`.

- [ ] **Step 2: Delete old flat modules**

```bash
git rm hyper_surrogate/symbolic.py hyper_surrogate/kinematics.py hyper_surrogate/materials.py hyper_surrogate/deformation_gradient.py hyper_surrogate/generator.py hyper_surrogate/umat_handler.py hyper_surrogate/reporter.py
```

- [ ] **Step 3: Update all remaining test imports**

Update any test files still importing from old paths.

Run: `pytest tests/ -v`
Expected: All pass

- [ ] **Step 4: Commit**

```bash
git add hyper_surrogate/reporting/ tests/
git rm hyper_surrogate/symbolic.py hyper_surrogate/kinematics.py hyper_surrogate/materials.py hyper_surrogate/deformation_gradient.py hyper_surrogate/generator.py hyper_surrogate/umat_handler.py hyper_surrogate/reporter.py
git commit -m "refactor: remove old flat modules, complete package restructure"
```

---

### Task 17: End-to-end integration test

**Files:**

- Create: `tests/test_integration.py`

- [ ] **Step 1: Write integration test**

```python
import numpy as np
import pytest
import tempfile, os
torch = pytest.importorskip("torch")


def test_mlp_end_to_end():
    """Full pipeline: data -> train -> export -> Fortran."""
    import hyper_surrogate as hs

    material = hs.NeoHooke({"C10": 0.5, "KBULK": 1000.0})
    train_ds, val_ds, in_norm, out_norm = hs.create_datasets(
        material, n_samples=200, input_type="invariants", target_type="pk2_voigt",
    )

    model = hs.MLP(input_dim=3, output_dim=6, hidden_dims=[16], activation="tanh")
    result = hs.Trainer(model, train_ds, val_ds, loss_fn=hs.StressLoss(), max_epochs=10).fit()

    exported = hs.extract_weights(result.model, in_norm, out_norm)

    with tempfile.TemporaryDirectory() as td:
        # Save/load weights
        npz_path = os.path.join(td, "model.npz")
        exported.save(npz_path)
        loaded = hs.ExportedModel.load(npz_path)

        # Generate Fortran
        f90_path = os.path.join(td, "nn_surrogate.f90")
        hs.FortranEmitter(loaded).write(f90_path)
        assert os.path.exists(f90_path)
        with open(f90_path) as f:
            code = f.read()
        assert "MODULE nn_surrogate" in code
        assert "MATMUL" in code


def test_icnn_end_to_end():
    """ICNN pipeline: data -> train -> export."""
    import hyper_surrogate as hs

    material = hs.NeoHooke({"C10": 0.5, "KBULK": 1000.0})
    train_ds, val_ds, in_norm, out_norm = hs.create_datasets(
        material, n_samples=200, input_type="invariants", target_type="energy",
    )

    model = hs.ICNN(input_dim=3, hidden_dims=[16])
    result = hs.Trainer(
        model, train_ds, val_ds, loss_fn=hs.EnergyStressLoss(), max_epochs=10
    ).fit()

    exported = hs.extract_weights(result.model, in_norm, out_norm)
    assert exported.metadata["architecture"] == "icnn"
```

- [ ] **Step 2: Run integration test**

Run: `pytest tests/test_integration.py -v`
Expected: All pass

- [ ] **Step 3: Run full test suite**

Run: `pytest tests/ -v`
Expected: All pass

- [ ] **Step 4: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: add end-to-end integration tests for MLP and ICNN pipelines"
```
