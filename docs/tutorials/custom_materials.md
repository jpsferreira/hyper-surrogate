# Defining your own SEF

When the hyperelastic model you need is not in the built-in catalogue
but you can write its strain-energy function (SEF) in closed form,
**hyper-surrogate** lets you plug it into the framework by subclassing
the `Material` base class. From there, every downstream capability —
the symbolic Fortran UMAT, the data generator, the surrogate trainer —
works on your model exactly as it does on `NeoHooke` or
`HolzapfelOgden`.

This page walks through that workflow end to end.

---

## What `Material` expects from a subclass

A custom material needs three things:

1. A `DEFAULT_PARAMS` class attribute listing every parameter the SEF
   depends on, with default values.
2. An `__init__` that merges the user's overrides onto the defaults
   and calls `super().__init__(params)`.
3. A `sef` property that returns a single SymPy expression in the
   standard invariants. The expression is built from the symbolic
   invariants exposed by `self._handler` and the parameter symbols
   exposed by `self._symbols`.

That is it. The base class handles symbolic differentiation,
batch-tensor evaluation, code generation, and integration with the
data pipeline.

### Invariants exposed by `self._handler`

| Attribute                       | Symbol      | Description                                |
| ------------------------------- | ----------- | ------------------------------------------ |
| `_handler.isochoric_invariant1` | $\bar{I}_1$ | First isochoric invariant of $\mathbf{C}$  |
| `_handler.isochoric_invariant2` | $\bar{I}_2$ | Second isochoric invariant of $\mathbf{C}$ |
| `_handler.invariant3`           | $I_3 = J^2$ | Determinant of $\mathbf{C}$                |

For anisotropic models with fiber families you also have access to
$I_4, I_5$ — see [Anisotropic materials](anisotropic_materials.md).

### Parameter symbols via `self._symbols`

Every key you list in `DEFAULT_PARAMS` becomes a SymPy symbol with the
same name, accessible as `self._symbols["..."]`. Use these inside the
`sef` expression so that the generated Fortran knows the parameters
are model inputs (eventually surfaced as the `PROPS` array in the
emitted UMAT).

---

## Example: a one-term Ogden-like SEF

The SEF

$$
W(\bar{I}_1, J) = \frac{\mu}{\alpha} \left( \bar{I}_1^{\alpha/2} - 3 \right)
                + \frac{K}{2} (J - 1)^2
$$

is not in the built-in catalogue. Here is the full subclass — fewer
than fifteen lines:

```python
import sympy as sp
from hyper_surrogate import Material

class MyOgdenLike(Material):
    DEFAULT_PARAMS = {"mu": 1.0, "alpha": 2.0, "KBULK": 1000.0}

    def __init__(self, parameters=None):
        super().__init__({**self.DEFAULT_PARAMS, **(parameters or {})})

    @property
    def sef(self):
        h = self._handler
        mu, alpha, K = (self._symbols[k] for k in ("mu", "alpha", "KBULK"))
        return (
            (mu / alpha) * (h.isochoric_invariant1 ** (alpha / 2) - 3)
            + 0.5 * K * (sp.sqrt(h.invariant3) - 1) ** 2
        )
```

Construct it like any built-in material:

```python
material = MyOgdenLike({"mu": 0.8, "alpha": 3.0, "KBULK": 2000.0})
```

---

## Going to Fortran in one line

Hand the instance to `UMATHandler` and you get a complete analytical
Abaqus-compatible UMAT — Cauchy stress, consistent tangent, Jaumann
correction, all derived symbolically and emitted with common-subexpression
elimination:

```python
from hyper_surrogate import UMATHandler

UMATHandler(material).generate("mysef.f90")
```

No neural network is involved. No Python runtime is needed at solve
time. The `PARAMETER` arrays inside the emitted `.f90` are determined
entirely by the symbolic expression in your `sef` property.

A runnable end-to-end version of this example lives at
[`examples/custom_sef.py`](https://github.com/jpsferreira/hyper-surrogate/blob/main/examples/custom_sef.py).

---

## Driving a surrogate from your custom material

If the SEF is expensive to evaluate (multi-scale, homogenisation, or
just a heavy SymPy expression), you may prefer to train a neural-network
surrogate and emit a hybrid UMAT instead. The pipeline is exactly the
same as for the built-in materials:

```python
from hyper_surrogate import (
    MLP, Trainer, EnergyStressLoss,
    create_datasets, extract_weights, HybridUMATEmitter,
)

train_ds, val_ds, in_norm, energy_norm = create_datasets(
    MyOgdenLike(), n_samples=4000,
    input_type="invariants", target_type="energy",
    deformation_mode="combined_compressible",
)

model = MLP(input_dim=3, output_dim=1, hidden_dims=[64, 64, 64], activation="softplus")
result = Trainer(
    model, train_ds, val_ds,
    loss_fn=EnergyStressLoss(alpha=1.0, beta=1.0),
    max_epochs=2000, patience=200,
).fit()

exported = extract_weights(result.model, in_norm, energy_norm)
HybridUMATEmitter(exported).write("mysef_hybrid.f90")
```

See [Training surrogates](training_surrogates.md) for loss-function
choices and architecture options, and [Exporting to Fortran](export_fortran.md)
for the trade-offs between the analytical and hybrid emitters.

---

## Anisotropic and multi-fiber materials

A `Material` subclass with one or more fiber families needs to declare
the fiber directions and override `sef_from_all_invariants` instead of
`sef`. See [Anisotropic materials](anisotropic_materials.md) for the
fiber-aware pattern and the `PeirlinckArtery` source code for a
worked two-fiber example.
