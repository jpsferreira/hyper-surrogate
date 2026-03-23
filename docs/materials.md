# Material Model Catalog

## Overview

hyper-surrogate provides a comprehensive library of hyperelastic constitutive models for both isotropic and anisotropic materials.

| Model | Type | Invariants | Parameters | Use Case |
|-------|------|-----------|------------|----------|
| `NeoHooke` | Isotropic | $\bar{I}_1$ | 1 | Simple rubber-like materials |
| `MooneyRivlin` | Isotropic | $\bar{I}_1, \bar{I}_2$ | 2 | Rubber with moderate strains |
| `Yeoh` | Isotropic | $\bar{I}_1$ | 3 | Rubber at large strains |
| `Demiray` | Isotropic | $\bar{I}_1$ | 2 | Soft biological tissues |
| `Ogden` | Isotropic | $\bar{\lambda}_i$ | $2N$ | General rubber (any strain range) |
| `Fung` | Isotropic | $\mathbf{E}$ | 3 | Soft tissues (exponential stiffening) |
| `HolzapfelOgden` | Anisotropic | $\bar{I}_1, I_4$ | 4 | Arterial wall (single fiber) |
| `GasserOgdenHolzapfel` | Anisotropic | $\bar{I}_1, I_4$ | 5 | Arterial wall (dispersed fibers) |
| `Guccione` | Anisotropic | $\mathbf{E}$ (fiber frame) | 4 | Cardiac tissue |

## Isotropic Models

### Neo-Hooke

```python
from hyper_surrogate import NeoHooke

mat = NeoHooke(parameters={"C10": 0.5, "KBULK": 1000.0})
```

The simplest hyperelastic model, linear in the first invariant.

### Mooney-Rivlin

```python
from hyper_surrogate import MooneyRivlin

mat = MooneyRivlin(parameters={"C10": 0.3, "C01": 0.2, "KBULK": 1000.0})
```

Extends Neo-Hooke with second invariant dependence.

### Yeoh

```python
from hyper_surrogate.mechanics.materials import Yeoh

mat = Yeoh(parameters={"C10": 0.5, "C20": -0.01, "C30": 0.001, "KBULK": 1000.0})
```

Third-order polynomial in $\bar{I}_1 - 3$. Captures large-strain stiffening.

### Demiray

```python
from hyper_surrogate.mechanics.materials import Demiray

mat = Demiray(parameters={"C1": 0.05, "C2": 8.0, "KBULK": 1000.0})
```

Single-parameter exponential model commonly used for soft tissues.

### Ogden

```python
from hyper_surrogate.mechanics.materials import Ogden

# 1-term Ogden (equivalent to Neo-Hooke when alpha=2)
mat = Ogden(parameters={"mu1": 1.0, "alpha1": 2.0, "KBULK": 1000.0})

# 3-term Ogden
mat = Ogden(parameters={
    "mu1": 1.491, "alpha1": 1.3,
    "mu2": 0.003, "alpha2": 5.0,
    "mu3": -0.024, "alpha3": -2.0,
    "KBULK": 1000.0,
})
```

Principal-stretch-based model. Note: uses numerical evaluation (no symbolic SEF).

### Fung

```python
from hyper_surrogate.mechanics.materials import Fung

mat = Fung(parameters={"c": 1.0, "b1": 10.0, "b2": 5.0, "KBULK": 1000.0})
```

Exponential model expressed in Green strain components.

## Anisotropic Models

### Holzapfel-Ogden

```python
from hyper_surrogate import HolzapfelOgden
import numpy as np

mat = HolzapfelOgden(
    parameters={"a": 0.059, "b": 8.023, "af": 18.472, "bf": 16.026, "KBULK": 1000.0},
    fiber_direction=np.array([1.0, 0.0, 0.0]),
)
```

Single fiber family model for arterial walls.

### Gasser-Ogden-Holzapfel (GOH)

```python
from hyper_surrogate.mechanics.materials import GasserOgdenHolzapfel
import numpy as np

mat = GasserOgdenHolzapfel(
    parameters={
        "a": 0.059, "b": 8.023,
        "af": 18.472, "bf": 16.026,
        "kappa": 0.226,  # fiber dispersion
        "KBULK": 1000.0,
    },
    fiber_direction=np.array([1.0, 0.0, 0.0]),
)
```

Extends HolzapfelOgden with fiber dispersion parameter $\kappa$. When $\kappa = 0$, reduces to HolzapfelOgden. When $\kappa = 1/3$, fibers are isotropically distributed.

### Guccione

```python
from hyper_surrogate.mechanics.materials import Guccione
import numpy as np

mat = Guccione(
    parameters={"C": 0.876, "bf": 18.48, "bt": 3.58, "bfs": 1.627, "KBULK": 1000.0},
    fiber_direction=np.array([1.0, 0.0, 0.0]),
    sheet_direction=np.array([0.0, 1.0, 0.0]),
)
```

Transversely isotropic model for cardiac tissue, expressed in the fiber-sheet-normal frame.
