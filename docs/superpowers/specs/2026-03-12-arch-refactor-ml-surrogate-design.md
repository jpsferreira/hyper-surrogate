# Architecture Refactoring & ML Surrogate Pipeline — Design Spec

**Date:** 2026-03-12
**Scope:** v1 — arch refactoring, ML models, training, Fortran export
**Deferred:** C++ emitter, solver adapter templates, automated validation framework

---

## Goals

1. Refactor the existing codebase from flat inheritance-based structure to layered composition-based architecture
2. Add a PyTorch-based ML surrogate pipeline: model definitions (MLP + ICNN), training loop, and Fortran transpiler
3. The generated Fortran code is a standalone module (solver-agnostic), not tied to any specific FEM solver
4. Support multiple prediction strategies: strain energy (scalar), stress (6 Voigt), stress + tangent (6 + 21)
5. No backward compatibility required — clean slate

---

## Package Structure

```
hyper_surrogate/
├── __init__.py              # Public API exports
├── mechanics/
│   ├── __init__.py
│   ├── symbolic.py          # SymbolicHandler (standalone, no inheritance)
│   ├── kinematics.py        # Batched NumPy kinematic quantities
│   └── materials.py         # Material base + NeoHooke, MooneyRivlin
├── data/
│   ├── __init__.py
│   ├── deformation.py       # DeformationGenerator (merged from two classes)
│   └── dataset.py           # PyTorch Dataset, Normalizer, create_datasets()
├── models/
│   ├── __init__.py
│   ├── base.py              # SurrogateModel ABC + LayerInfo dataclass
│   ├── mlp.py               # Standard feedforward MLP
│   └── icnn.py              # Input-Convex Neural Network
├── training/
│   ├── __init__.py
│   ├── trainer.py           # Training loop, early stopping, checkpointing
│   └── losses.py            # StressLoss, StressTangentLoss, EnergyStressLoss
├── export/
│   ├── __init__.py
│   ├── weights.py           # extract_weights() -> ExportedModel (.npz)
│   └── fortran/
│       ├── __init__.py
│       └── emitter.py       # FortranEmitter: MLP + ICNN -> .f90
└── reporting/
    ├── __init__.py
    └── reporter.py          # PDF reports (existing, cleaned up)
```

### Dependency Rules

| Layer        | Depends on                                | Never imports                       |
| ------------ | ----------------------------------------- | ----------------------------------- |
| `mechanics/` | `numpy`, `sympy`                          | `torch`                             |
| `data/`      | `mechanics/`, `numpy`, optionally `torch` | `sympy`                             |
| `models/`    | `torch`                                   | `sympy`, `numpy` (except via torch) |
| `training/`  | `models/`, `data/`, `torch`               | `sympy`                             |
| `export/`    | `numpy` only                              | `torch`, `sympy`                    |
| `reporting/` | `numpy`, `matplotlib`                     | `torch`, `sympy`                    |

### Optional Dependencies (pyproject.toml)

```toml
[tool.poetry.extras]
ml = ["torch"]
```

Users who only need mechanics + analytical UMAT generation don't need PyTorch installed.

---

## Mechanics Layer

### SymbolicHandler (composition, not inheritance)

Standalone symbolic tensor calculus engine. Material classes hold a reference to it instead of inheriting from it.

```python
class SymbolicHandler:
    def __init__(self):
        self._c = ImmutableMatrix(3, 3, ...)  # C_11..C_33
        self._f = ImmutableMatrix(3, 3, ...)  # F_11..F_33

    # Invariants — explicitly named
    @property
    def isochoric_invariant1(self) -> Expr: ...  # tr(C) * det(C)^(-1/3)
    @property
    def isochoric_invariant2(self) -> Expr: ...  # 0.5*(I1^2-tr(C^2)) * det(C)^(-2/3)
    @property
    def invariant3(self) -> Expr: ...             # det(C)

    # Derivation
    def pk2(self, sef: Expr) -> Matrix: ...
    def cmat(self, pk2: Matrix) -> NDArray: ...
    def cauchy(self, sef: Expr, f: Matrix) -> Matrix: ...
    def spatial_tangent(self, pk2: Matrix, f: Matrix) -> NDArray: ...
    def jaumann_correction(self, sigma: Matrix) -> NDArray: ...

    # Voigt reduction
    @staticmethod
    def to_voigt_2(tensor: Matrix) -> Matrix: ...
    @staticmethod
    def to_voigt_4(tensor: NDArray) -> Matrix: ...

    # Numerical bridge
    def lambdify(self, expr: Expr, *params) -> Callable: ...
```

### Kinematics (disambiguated invariants)

All methods are static, batched over `(N, 3, 3)` arrays via einsum.

```python
class Kinematics:
    @staticmethod
    def right_cauchy_green(f: ndarray) -> ndarray: ...

    # Standard invariants (not isochoric)
    @staticmethod
    def trace_invariant(c: ndarray) -> ndarray: ...
    @staticmethod
    def quadratic_invariant(c: ndarray) -> ndarray: ...
    @staticmethod
    def det_invariant(c: ndarray) -> ndarray: ...

    # Isochoric invariants (matching symbolic layer)
    @staticmethod
    def isochoric_invariant1(c: ndarray) -> ndarray: ...
    @staticmethod
    def isochoric_invariant2(c: ndarray) -> ndarray: ...

    @staticmethod
    def jacobian(f: ndarray) -> ndarray: ...
```

### Material (composition over inheritance)

```python
class Material:
    def __init__(self, parameters: dict[str, float]):
        self._handler = SymbolicHandler()
        self._params = parameters
        self._symbols = {k: Symbol(k) for k in parameters}

    @property
    def sef(self) -> Expr:
        raise NotImplementedError

    @cached_property
    def pk2_expr(self) -> Matrix: ...

    @cached_property
    def pk2_func(self) -> Callable: ...

    def evaluate_pk2(self, c_batch: ndarray) -> ndarray:
        """Vectorized evaluation over (N,3,3) C tensors."""
        ...
```

Subclasses (`NeoHooke`, `MooneyRivlin`) override only `sef`. Default parameters are class-level constants.

---

## Data Layer

### DeformationGenerator

Merges `DeformationGradient` + `DeformationGradientGenerator` + drops the `Generator` wrapper class.

```python
class DeformationGenerator:
    def __init__(self, seed: int = 42):
        self._rng = np.random.default_rng(seed)

    def uniaxial(self, n: int, stretch_range=(0.4, 3.0)) -> ndarray: ...
    def biaxial(self, n: int, stretch_range=(0.4, 3.0)) -> ndarray: ...
    def shear(self, n: int, shear_range=(-1.0, 1.0)) -> ndarray: ...
    def combined(self, n: int, **kwargs) -> ndarray: ...
    def random_rotation(self, n: int) -> ndarray: ...
```

`n` (batch size) is a method parameter, not a constructor field.

### Dataset & Normalization

```python
class MaterialDataset(torch.utils.data.Dataset):
    def __init__(self, inputs, targets, input_transform=None, target_transform=None): ...

class Normalizer:
    def fit(self, data: ndarray) -> "Normalizer": ...
    def transform(self, data: ndarray) -> ndarray: ...
    def inverse_transform(self, data: ndarray) -> ndarray: ...

    @property
    def params(self) -> dict:
        """{"mean": ndarray, "std": ndarray} — exported to Fortran."""
        ...

def create_datasets(
    material: Material,
    n_samples: int,
    input_type: Literal["invariants", "cauchy_green"],
    target_type: Literal["energy", "pk2_voigt", "pk2_voigt+cmat_voigt"],
    val_fraction: float = 0.15,
    seed: int = 42,
) -> tuple[MaterialDataset, MaterialDataset, Normalizer, Normalizer]:
    """End-to-end: generate deformations -> evaluate material -> normalize -> split -> wrap."""
    ...
```

`input_type="invariants"` produces `[I1_bar, I2_bar, J]` (3 values, frame-indifferent).
`input_type="cauchy_green"` produces 6 unique Voigt components of C.

---

## Models Layer

### SurrogateModel (base interface)

```python
class SurrogateModel(nn.Module, ABC):
    @property
    @abstractmethod
    def input_dim(self) -> int: ...

    @property
    @abstractmethod
    def output_dim(self) -> int: ...

    @abstractmethod
    def layer_sequence(self) -> list[LayerInfo]:
        """Ordered (weights_key, bias_key, activation) per layer.
        This is the contract the Fortran exporter relies on."""
        ...

    def export_weights(self) -> dict[str, ndarray]: ...

@dataclass
class LayerInfo:
    weights: str    # key, e.g. "layer_0.weight"
    bias: str       # key, e.g. "layer_0.bias"
    activation: str # "relu", "tanh", "sigmoid", "softplus", "identity"
```

### MLP

```python
class MLP(SurrogateModel):
    def __init__(self, input_dim: int, output_dim: int,
                 hidden_dims: list[int] = [64, 64],
                 activation: str = "tanh"): ...
```

Default `tanh` — smooth, differentiable, suitable for constitutive models.

### ICNN (Input-Convex Neural Network)

```python
class ICNN(SurrogateModel):
    """Amos+ 2017. Guarantees convexity of output w.r.t. input.

    z_0 = sigma(W_0^x @ x + b_0)
    z_i = sigma(softplus(W_i^z) @ z_{i-1} + W_i^x @ x + b_i)
    y   = softplus(W_L^z) @ z_{L-1} + W_L^x @ x + b_L

    Non-negative weights on wz layers enforced via softplus at forward time.
    """

    def __init__(self, input_dim: int, hidden_dims: list[int] = [64, 64],
                 activation: str = "softplus"): ...
```

Output is always scalar (strain energy W). Stress obtained via `torch.autograd.grad` during training.

---

## Training Layer

### Losses

| Loss                | Use case                        | Inputs                                                        |
| ------------------- | ------------------------------- | ------------------------------------------------------------- |
| `StressLoss`        | MLP predicting stress           | `(s_pred, s_true)`                                            |
| `StressTangentLoss` | MLP predicting stress + tangent | `(pred, s_true, c_true)` with `alpha=1.0, beta=0.1`           |
| `EnergyStressLoss`  | ICNN predicting energy          | `(w_pred, w_true, inputs, s_true)` with `alpha=1.0, beta=1.0` |

`EnergyStressLoss` computes `dW/dI` via `torch.autograd.grad(create_graph=True)` to enforce stress-energy consistency.
`StressTangentLoss` weights tangent lower (`beta=0.1`) because tangent values span a larger range.

### Trainer

```python
class Trainer:
    def __init__(self, model, train_dataset, val_dataset, loss_fn,
                 lr=1e-3, batch_size=256, max_epochs=1000, patience=50, device="cpu"): ...

    def fit(self) -> TrainingResult: ...

@dataclass
class TrainingResult:
    model: SurrogateModel
    history: dict[str, list[float]]
    best_epoch: int
```

- Adam optimizer
- ReduceLROnPlateau scheduler (patience // 3)
- Early stopping with patience
- Best-model checkpointing

Deliberately simple — small models, in-memory datasets.

---

## Export Layer (Fortran Transpiler)

### ExportedModel

```python
@dataclass
class ExportedModel:
    layers: list[LayerInfo]
    weights: dict[str, ndarray]
    input_normalizer: dict | None   # {"mean": ndarray, "std": ndarray}
    output_normalizer: dict | None
    metadata: dict                   # architecture, input_dim, output_dim

    def save(self, path: str): ...   # .npz
    @classmethod
    def load(cls, path: str) -> "ExportedModel": ...
```

The `.npz` file is the single artifact that crosses the Python-Fortran boundary. No PyTorch needed to load it.

### FortranEmitter

```python
class FortranEmitter:
    ACTIVATIONS = {
        "relu":     "max(0.0d0, {x})",
        "tanh":     "tanh({x})",
        "sigmoid":  "1.0d0 / (1.0d0 + exp(-{x}))",
        "softplus": "log(1.0d0 + exp({x}))",
        "identity": "{x}",
    }

    def __init__(self, exported: ExportedModel): ...

    def emit_mlp(self) -> str: ...
    def emit_icnn(self) -> str: ...
    def emit(self) -> str: ...        # dispatch by metadata["architecture"]
    def write(self, path: str): ...
```

Generated code structure:

```fortran
MODULE nn_surrogate
  IMPLICIT NONE
  DOUBLE PRECISION, PARAMETER :: w0(64,3) = RESHAPE([...], [64,3])
  DOUBLE PRECISION, PARAMETER :: b0(64) = [...]
  ...
CONTAINS
  SUBROUTINE nn_forward(input, output)
    DOUBLE PRECISION, INTENT(IN)  :: input(3)
    DOUBLE PRECISION, INTENT(OUT) :: output(6)
    DOUBLE PRECISION :: z0(64), z1(64)
    ! normalize input
    ! layer-by-layer: z = activation(matmul(w, z_prev) + b)
    ! denormalize output
  END SUBROUTINE
END MODULE
```

Design decisions:

1. **Weights baked as `PARAMETER` arrays** — compiled into the binary. No file I/O at runtime. Necessary because UMAT environments typically cannot do I/O.
2. **ICNN: `softplus(wz)` pre-computed at export time** — stored as plain arrays. Generated Fortran uses plain `MATMUL`, no softplus-on-weights at inference.
3. **Normalization baked in** — subroutine normalizes inputs and denormalizes outputs internally. Caller passes raw physical quantities.
4. **Standalone Fortran module** — not a UMAT. Users write a thin solver-specific wrapper that calls `nn_forward`. Solver adapters are deferred to a later version.
5. **`MATMUL` for matrix-vector products** — Fortran intrinsic, compiler-optimized. Appropriate for the small matrices involved.

---

## Existing Analytical UMAT Path

The current `UMATHandler` (symbolic math -> Fortran UMAT via CSE + `sympy.fcode`) is preserved and relocated to `export/fortran/analytical.py`. It remains a separate code path for users who want exact symbolic UMATs without ML.

---

## Public API

```python
# hyper_surrogate/__init__.py

# Mechanics
from .mechanics.symbolic import SymbolicHandler
from .mechanics.kinematics import Kinematics
from .mechanics.materials import Material, NeoHooke, MooneyRivlin

# Data
from .data.deformation import DeformationGenerator
from .data.dataset import MaterialDataset, Normalizer, create_datasets

# Models
from .models.mlp import MLP
from .models.icnn import ICNN

# Training
from .training.trainer import Trainer, TrainingResult
from .training.losses import StressLoss, StressTangentLoss, EnergyStressLoss

# Export
from .export.weights import extract_weights, ExportedModel
from .export.fortran.emitter import FortranEmitter
```

---

## End-to-End Usage Examples

### MLP predicting stress

```python
import hyper_surrogate as hs

material = hs.NeoHooke({"C10": 0.5, "KBULK": 1000.0})
train_ds, val_ds, in_norm, out_norm = hs.create_datasets(
    material, n_samples=50_000,
    input_type="invariants", target_type="pk2_voigt",
)

model = hs.MLP(input_dim=3, output_dim=6, hidden_dims=[64, 64], activation="tanh")
result = hs.Trainer(model, train_ds, val_ds, loss_fn=hs.StressLoss()).fit()

exported = hs.extract_weights(result.model)
exported.input_normalizer = in_norm.params
exported.output_normalizer = out_norm.params
hs.FortranEmitter(exported).write("nn_neohooke.f90")
```

### ICNN predicting energy (physics-informed)

```python
train_ds, val_ds, in_norm, out_norm = hs.create_datasets(
    material, n_samples=50_000,
    input_type="invariants", target_type="energy",
)

model = hs.ICNN(input_dim=3, hidden_dims=[64, 64], activation="softplus")
loss = hs.EnergyStressLoss(alpha=1.0, beta=1.0)
result = hs.Trainer(model, train_ds, val_ds, loss_fn=loss).fit()

exported = hs.extract_weights(result.model)
exported.input_normalizer = in_norm.params
exported.output_normalizer = out_norm.params
hs.FortranEmitter(exported).write("nn_neohooke_icnn.f90")
```

---

## Deferred (Post-v1)

- **C++ emitter** — same `ExportedModel` input, different backend in `export/cpp/`
- **Solver adapter templates** — `export/fortran/adapters/abaqus.py`, `ansys.py`, etc. Generate thin UMAT/USERMAT wrappers calling `nn_forward`
- **Validation framework** — automated comparison of NN UMAT vs analytical UMAT on canonical deformation paths, convergence tests
