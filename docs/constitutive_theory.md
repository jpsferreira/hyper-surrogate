# Constitutive Theory of Hyperelasticity

This document provides a self-contained reference on the continuum mechanics theory underpinning **hyper-surrogate**. It covers kinematics, strain energy functions, stress measures, material tangents, and the specific constitutive models implemented in the library.

---

## 1. Kinematics of Finite Deformation

### 1.1 Deformation Gradient

The deformation gradient $\mathbf{F}$ maps an infinitesimal material line element $d\mathbf{X}$ in the reference configuration to its image $d\mathbf{x}$ in the current configuration:

$$d\mathbf{x} = \mathbf{F}\, d\mathbf{X}, \qquad F_{iJ} = \frac{\partial x_i}{\partial X_J}$$

The Jacobian determinant $J = \det(\mathbf{F}) > 0$ measures volume change. For incompressible materials, $J = 1$.

### 1.2 Strain Tensors

| Tensor | Symbol | Definition | Configuration |
|--------|--------|------------|---------------|
| Right Cauchy-Green | $\mathbf{C}$ | $\mathbf{F}^T \mathbf{F}$ | Reference (Lagrangian) |
| Left Cauchy-Green | $\mathbf{b}$ | $\mathbf{F}\, \mathbf{F}^T$ | Current (Eulerian) |
| Green-Lagrange strain | $\mathbf{E}$ | $\frac{1}{2}(\mathbf{C} - \mathbf{I})$ | Reference |

In `hyper-surrogate`, all constitutive evaluations are based on $\mathbf{C}$:

```python
import hyper_surrogate as hs

C = hs.Kinematics.right_cauchy_green(F)  # (N, 3, 3)
B = hs.Kinematics.left_cauchy_green(F)   # (N, 3, 3)
```

### 1.3 Principal Invariants of $\mathbf{C}$

The three principal invariants encode the deformation state in a rotation-invariant manner:

$$I_1 = \text{tr}(\mathbf{C}) = \lambda_1^2 + \lambda_2^2 + \lambda_3^2$$

$$I_2 = \frac{1}{2}\left[(\text{tr}\,\mathbf{C})^2 - \text{tr}(\mathbf{C}^2)\right] = \lambda_1^2\lambda_2^2 + \lambda_2^2\lambda_3^2 + \lambda_3^2\lambda_1^2$$

$$I_3 = \det(\mathbf{C}) = J^2 = \lambda_1^2 \lambda_2^2 \lambda_3^2$$

where $\lambda_1, \lambda_2, \lambda_3$ are the principal stretches.

### 1.4 Isochoric-Volumetric Split

To decouple shape change from volume change, the deformation gradient is multiplicatively decomposed:

$$\mathbf{F} = J^{1/3}\, \bar{\mathbf{F}}, \qquad \bar{\mathbf{C}} = J^{-2/3}\, \mathbf{C}$$

The **isochoric (modified) invariants** are:

| Invariant | Expression | Physical Meaning |
|-----------|------------|-----------------|
| $\bar{I}_1$ | $J^{-2/3}\, I_1$ | Isochoric shape change (trace) |
| $\bar{I}_2$ | $J^{-4/3}\, I_2$ | Isochoric shape change (cofactor) |
| $J$ | $\sqrt{I_3}$ | Volume ratio |

```python
I1_bar = hs.Kinematics.isochoric_invariant1(C)  # (N,)
I2_bar = hs.Kinematics.isochoric_invariant2(C)  # (N,)
J      = hs.Kinematics.jacobian(F)              # (N,)
```

### 1.5 Fiber (Pseudo-)Invariants

For anisotropic materials with a preferred fiber direction $\mathbf{a}_0$ (unit vector in the reference configuration):

$$I_4 = \mathbf{a}_0 \cdot \mathbf{C}\, \mathbf{a}_0 = \lambda_f^2$$

$$I_5 = \mathbf{a}_0 \cdot \mathbf{C}^2\, \mathbf{a}_0$$

where $\lambda_f$ is the fiber stretch. $I_4$ measures stretch along the fiber; $I_5$ captures coupling between fiber stretch and shear.

```python
fiber_dir = np.array([1.0, 0.0, 0.0])
I4 = hs.Kinematics.fiber_invariant4(C, fiber_dir)  # (N,)
I5 = hs.Kinematics.fiber_invariant5(C, fiber_dir)  # (N,)
```

### 1.6 Principal Stretches

The principal stretches $\lambda_i$ are the square roots of the eigenvalues of $\mathbf{C}$:

$$\mathbf{C}\, \mathbf{N}_i = \lambda_i^2\, \mathbf{N}_i, \qquad i = 1, 2, 3$$

```python
stretches = hs.Kinematics.principal_stretches(C)  # (N, 3), sorted descending
```

---

## 2. Stress Measures

### 2.1 Second Piola-Kirchhoff Stress

For a hyperelastic material defined by a strain energy function $W(\mathbf{C})$, the second Piola-Kirchhoff (PK2) stress is:

$$\mathbf{S} = 2\frac{\partial W}{\partial \mathbf{C}}$$

This is a **symmetric** tensor in the reference configuration.

### 2.2 Cauchy Stress (Push-Forward)

The Cauchy (true) stress in the current configuration is obtained by the push-forward operation:

$$\boldsymbol{\sigma} = \frac{1}{J}\, \mathbf{F}\, \mathbf{S}\, \mathbf{F}^T$$

### 2.3 Material and Spatial Tangents

The **material tangent** (elasticity tensor in the reference configuration):

$$\mathbb{C} = 2\frac{\partial \mathbf{S}}{\partial \mathbf{C}} = 4\frac{\partial^2 W}{\partial \mathbf{C}\, \partial \mathbf{C}}$$

The **spatial tangent** is obtained via push-forward to the current configuration and includes the **Jaumann rate correction** required by most FE solvers (e.g. Abaqus):

$$c_{ijkl} = \frac{1}{J} F_{iI} F_{jJ} F_{kK} F_{lL}\, \mathbb{C}_{IJKL} + \text{Jaumann correction}$$

### 2.4 Voigt Notation

For FE implementation, symmetric tensors and tangent matrices are stored in Voigt notation:

| Voigt Index | Tensor Components |
|:-----------:|:-----------------:|
| 1 | 11 (xx) |
| 2 | 22 (yy) |
| 3 | 33 (zz) |
| 4 | 12 (xy) |
| 5 | 13 (xz) |
| 6 | 23 (yz) |

- **Stress**: $\mathbf{S} \to [S_{11}, S_{22}, S_{33}, S_{12}, S_{13}, S_{23}]$ (6 components)
- **Tangent**: $\mathbb{C} \to 6 \times 6$ matrix

---

## 3. Strain Energy Functions

### 3.1 General Structure

All isotropic models in `hyper-surrogate` follow the decoupled form:

$$W(\mathbf{C}) = W_{\text{iso}}(\bar{I}_1, \bar{I}_2) + U(J)$$

where $W_{\text{iso}}$ is the isochoric (shape-changing) part and $U(J)$ is the volumetric penalty. Anisotropic models add fiber contributions:

$$W(\mathbf{C}) = W_{\text{iso}}(\bar{I}_1, \bar{I}_2) + W_{\text{fiber}}(I_4, I_5) + U(J)$$

### 3.2 Volumetric Penalty

All models share the same volumetric term:

$$U(J) = \frac{K}{4}\left(J^2 - 1 - 2\ln J\right)$$

| Property | Value |
|----------|-------|
| $U(1) = 0$ | Zero energy at $J=1$ |
| $U'(1) = 0$ | Zero pressure at $J=1$ |
| $U''(1) = K$ | Bulk modulus at $J=1$ |
| $U(J) \to \infty$ as $J \to 0^+$ or $J \to \infty$ | Penalty barrier |

The parameter $K$ (bulk modulus, `KBULK` in code) controls the degree of near-incompressibility. Typical values: $K = 100$--$10000 \times$ shear modulus.

---

## 4. Isotropic Material Models

### 4.1 Neo-Hooke

The simplest invariant-based hyperelastic model:

$$W = C_{10}(\bar{I}_1 - 3) + U(J)$$

| Parameter | Symbol | Meaning | Typical Range |
|-----------|--------|---------|---------------|
| `C10` | $C_{10}$ | Half the initial shear modulus ($\mu/2$) | 0.1 -- 10 MPa |
| `KBULK` | $K$ | Bulk modulus | 100 -- 10000 MPa |

**Stress derivatives**:

$$\frac{\partial W}{\partial \bar{I}_1} = C_{10}, \qquad \frac{\partial W}{\partial \bar{I}_2} = 0$$

**Use case**: Simple rubber-like materials at moderate strains ($< 100\%$).

```python
from hyper_surrogate import NeoHooke
mat = NeoHooke(parameters={"C10": 0.5, "KBULK": 1000.0})
```

### 4.2 Mooney-Rivlin

Extends Neo-Hooke with dependence on the second invariant:

$$W = C_{10}(\bar{I}_1 - 3) + C_{01}(\bar{I}_2 - 3) + U(J)$$

| Parameter | Symbol | Meaning |
|-----------|--------|---------|
| `C10` | $C_{10}$ | First invariant coefficient |
| `C01` | $C_{01}$ | Second invariant coefficient |
| `KBULK` | $K$ | Bulk modulus |

The initial shear modulus is $\mu = 2(C_{10} + C_{01})$.

**Use case**: Rubber with improved accuracy at moderate-to-large strains.

```python
from hyper_surrogate import MooneyRivlin
mat = MooneyRivlin(parameters={"C10": 0.3, "C01": 0.2, "KBULK": 1000.0})
```

### 4.3 Yeoh (Reduced Polynomial)

Third-order polynomial in $(\bar{I}_1 - 3)$:

$$W = C_{10}(\bar{I}_1 - 3) + C_{20}(\bar{I}_1 - 3)^2 + C_{30}(\bar{I}_1 - 3)^3 + U(J)$$

| Parameter | Symbol | Role |
|-----------|--------|------|
| `C10` | $C_{10}$ | Linear (small-strain) response |
| `C20` | $C_{20}$ | Softening at moderate strains (often $< 0$) |
| `C30` | $C_{30}$ | Stiffening at large strains (often $> 0$) |

**Use case**: Rubber undergoing large deformations with strain-stiffening. The S-shaped stress-strain curve of filled rubbers is well captured.

```python
from hyper_surrogate.mechanics.materials import Yeoh
mat = Yeoh(parameters={"C10": 0.5, "C20": -0.01, "C30": 0.001, "KBULK": 1000.0})
```

### 4.4 Demiray (Exponential)

Exponential model for soft biological tissues:

$$W = \frac{C_1}{C_2}\left[\exp\!\left(C_2(\bar{I}_1 - 3)\right) - 1\right] + U(J)$$

| Parameter | Symbol | Role |
|-----------|--------|------|
| `C1` | $C_1$ | Ground-state stiffness |
| `C2` | $C_2$ | Exponential stiffening rate |

At small strains ($\bar{I}_1 \approx 3$), the Taylor expansion gives $W \approx C_1(\bar{I}_1 - 3)$, recovering a Neo-Hooke response.

**Use case**: Isotropic soft tissues (skin, blood vessel ground substance).

```python
from hyper_surrogate.mechanics.materials import Demiray
mat = Demiray(parameters={"C1": 0.05, "C2": 8.0, "KBULK": 1000.0})
```

### 4.5 Ogden

Principal-stretch-based formulation:

$$W = \sum_{p=1}^{N} \frac{\mu_p}{\alpha_p}\left(\bar{\lambda}_1^{\alpha_p} + \bar{\lambda}_2^{\alpha_p} + \bar{\lambda}_3^{\alpha_p} - 3\right) + U(J)$$

where $\bar{\lambda}_i = J^{-1/3} \lambda_i$ are the isochoric principal stretches.

| Parameters | Constraint |
|-----------|-----------|
| $\mu_p, \alpha_p$ pairs | $\sum_p \mu_p \alpha_p = 2\mu$ (consistency with shear modulus) |

Special cases:

- $N=1,\ \alpha_1=2$: Neo-Hooke ($C_{10} = \mu_1/2$)
- $N=2,\ \alpha_1=2,\ \alpha_2=-2$: Mooney-Rivlin

**Note**: The Ogden model uses numerical eigenvalue decomposition (no closed-form symbolic SEF in terms of $I_1, I_2$).

```python
from hyper_surrogate.mechanics.materials import Ogden
mat = Ogden(parameters={
    "mu1": 1.491, "alpha1": 1.3,
    "mu2": 0.003, "alpha2": 5.0,
    "mu3": -0.024, "alpha3": -2.0,
    "KBULK": 1000.0,
})
```

### 4.6 Fung (Exponential, Green Strain)

Exponential model formulated in terms of Green-Lagrange strain components:

$$W = \frac{c}{2}\left[\exp(Q) - 1\right] + U(J)$$

$$Q = b_1 E_{11}^2 + b_2(E_{22}^2 + E_{33}^2 + 2E_{12}^2)$$

| Parameter | Symbol | Role |
|-----------|--------|------|
| `c` | $c$ | Overall stiffness scaling |
| `b1` | $b_1$ | Axial (fiber direction) stiffening |
| `b2` | $b_2$ | Transverse stiffening |

**Use case**: Arterial tissue, especially when the exponential stiffening is important in multiple directions.

```python
from hyper_surrogate.mechanics.materials import Fung
mat = Fung(parameters={"c": 1.0, "b1": 10.0, "b2": 5.0, "KBULK": 1000.0})
```

---

## 5. Anisotropic Material Models

### 5.1 Holzapfel-Ogden (Single Fiber Family)

Models arterial tissue with one family of collagen fibers embedded in a soft ground matrix:

$$W = \underbrace{\frac{a}{2b}\left[\exp\!\left(b(\bar{I}_1 - 3)\right) - 1\right]}_{\text{ground substance}} + \underbrace{\frac{a_f}{2b_f}\left[\exp\!\left(b_f\langle I_4 - 1\rangle^2\right) - 1\right]}_{\text{fiber contribution}} + U(J)$$

The **Macaulay bracket** $\langle \cdot \rangle = \max(0, \cdot)$ ensures fibers only contribute under **tension** ($I_4 > 1$, i.e. the fiber is stretched).

| Parameter | Symbol | Meaning |
|-----------|--------|---------|
| `a` | $a$ | Ground substance stiffness (kPa) |
| `b` | $b$ | Ground substance exponential coefficient |
| `af` | $a_f$ | Fiber stiffness (kPa) |
| `bf` | $b_f$ | Fiber exponential coefficient |
| `KBULK` | $K$ | Bulk modulus |

```python
from hyper_surrogate import HolzapfelOgden
mat = HolzapfelOgden(
    parameters={"a": 0.059, "b": 8.023, "af": 18.472, "bf": 16.026, "KBULK": 1000.0},
    fiber_direction=np.array([1.0, 0.0, 0.0]),
)
```

### 5.2 Gasser-Ogden-Holzapfel (GOH) -- Dispersed Fibers

Extends Holzapfel-Ogden with a **fiber dispersion** parameter $\kappa$:

$$W = \frac{a}{2b}\left[\exp\!\left(b(\bar{I}_1 - 3)\right) - 1\right] + \frac{a_f}{2b_f}\left[\exp\!\left(b_f \bar{E}^2\right) - 1\right] + U(J)$$

where the **generalized strain invariant** accounts for dispersion:

$$\bar{E} = \kappa(\bar{I}_1 - 3) + (1 - 3\kappa)(I_4 - 1)$$

| $\kappa$ Value | Physical Meaning |
|:-------------:|------------------|
| $0$ | Perfectly aligned fibers (reduces to Holzapfel-Ogden) |
| $1/3$ | Isotropically dispersed (no preferred direction) |
| $0 < \kappa < 1/3$ | Partially dispersed (realistic arterial tissue) |

| Parameter | Symbol | Meaning |
|-----------|--------|---------|
| `kappa` | $\kappa$ | Fiber dispersion ($0 \le \kappa \le 1/3$) |

Typical values for human arteries: $\kappa \approx 0.1$--$0.3$.

```python
from hyper_surrogate.mechanics.materials import GasserOgdenHolzapfel
mat = GasserOgdenHolzapfel(
    parameters={"a": 0.059, "b": 8.023, "af": 18.472, "bf": 16.026,
                "kappa": 0.226, "KBULK": 1000.0},
    fiber_direction=np.array([1.0, 0.0, 0.0]),
)
```

### 5.3 Guccione (Cardiac Tissue)

Transversely isotropic model for myocardial tissue, expressed in a **fiber-sheet-normal** local frame:

$$W = \frac{C}{2}\left[\exp(Q) - 1\right] + U(J)$$

$$Q = b_f E_{ff}^2 + b_t(E_{ss}^2 + E_{nn}^2 + 2E_{sn}^2) + b_{fs}(2E_{fs}^2 + 2E_{fn}^2)$$

where $E_{ff}, E_{ss}, E_{nn}, E_{fs}, E_{fn}, E_{sn}$ are the Green-Lagrange strain components in the local fiber ($f$), sheet ($s$), and normal ($n$) frame.

| Parameter | Symbol | Role |
|-----------|--------|------|
| `C` | $C$ | Overall stiffness |
| `bf` | $b_f$ | Fiber direction stiffening |
| `bt` | $b_t$ | Transverse (sheet/normal) stiffening |
| `bfs` | $b_{fs}$ | Fiber-sheet shear stiffening |

This model requires **two** direction vectors: `fiber_direction` and `sheet_direction`. The normal direction is computed as $\mathbf{n}_0 = \mathbf{f}_0 \times \mathbf{s}_0$.

```python
from hyper_surrogate.mechanics.materials import Guccione
mat = Guccione(
    parameters={"C": 0.876, "bf": 18.48, "bt": 3.58, "bfs": 1.627, "KBULK": 1000.0},
    fiber_direction=np.array([1.0, 0.0, 0.0]),
    sheet_direction=np.array([0.0, 1.0, 0.0]),
)
```

---

## 6. Summary: Model Selection Guide

| Tissue / Application | Recommended Model | Why |
|----------------------|-------------------|-----|
| Simple rubber (small strain) | Neo-Hooke | 1 parameter, sufficient for $< 50\%$ strain |
| Rubber (moderate strain) | Mooney-Rivlin | 2 parameters, captures $I_2$-dependence |
| Rubber (large strain, stiffening) | Yeoh or Ogden | Captures S-shaped stress-strain |
| Isotropic soft tissue | Demiray | Exponential stiffening, 2 parameters |
| Arterial wall (single fiber family) | Holzapfel-Ogden | Fiber tension-only via Macaulay bracket |
| Arterial wall (dispersed fibers) | GOH | Adds realistic fiber dispersion |
| Cardiac tissue | Guccione | Fiber-sheet-normal anisotropy |
| General isotropic (data-driven fit) | Ogden ($N$-term) | Very flexible, stretch-based |

---

## 7. Chain Rule for Invariant-Based SEFs

When the SEF is written in terms of invariants $W(\bar{I}_1, \bar{I}_2, J, I_4, I_5)$, the PK2 stress is computed via the chain rule:

$$\mathbf{S} = 2\frac{\partial W}{\partial \mathbf{C}} = 2\sum_{\alpha} \frac{\partial W}{\partial I_\alpha} \frac{\partial I_\alpha}{\partial \mathbf{C}}$$

The invariant derivatives with respect to $\mathbf{C}$ are:

| $I_\alpha$ | $\displaystyle\frac{\partial I_\alpha}{\partial \mathbf{C}}$ |
|:----------:|:-----------------------------------------------------------:|
| $I_1$ | $\mathbf{I}$ |
| $I_2$ | $I_1\,\mathbf{I} - \mathbf{C}$ |
| $I_3$ | $I_3\,\mathbf{C}^{-1}$ |
| $I_4$ | $\mathbf{a}_0 \otimes \mathbf{a}_0$ |
| $I_5$ | $\mathbf{a}_0 \otimes (\mathbf{C}\,\mathbf{a}_0) + (\mathbf{C}\,\mathbf{a}_0) \otimes \mathbf{a}_0$ |

This is the fundamental relationship exploited in the **hybrid UMAT**: the neural network learns $W(I_\alpha)$ and its gradient $\partial W / \partial I_\alpha$, while the invariant-to-stress mapping uses the analytical expressions above.

---

## 8. Thermodynamic Consistency

### 8.1 Requirements

A hyperelastic strain energy function must satisfy:

1. **Objectivity**: $W(\mathbf{Q}\mathbf{F}) = W(\mathbf{F})$ for all rotations $\mathbf{Q}$ -- guaranteed by expressing $W$ in terms of $\mathbf{C}$ (or its invariants).
2. **Normalization**: $W(\mathbf{I}) = 0$ -- zero energy in the reference state.
3. **Growth condition**: $W \to \infty$ as $J \to 0^+$ or $J \to \infty$ -- prevents material collapse.
4. **Polyconvexity** (stronger than quasiconvexity): ensures existence of minimizers in boundary value problems.

### 8.2 Convexity vs. Polyconvexity

$W(\mathbf{F})$ is **polyconvex** if there exists a convex function $P$ such that:

$$W(\mathbf{F}) = P(\mathbf{F}, \text{cof}\,\mathbf{F}, \det\,\mathbf{F})$$

In terms of invariants of $\mathbf{C}$, a sufficient condition for polyconvexity is that $W$ is **separately convex** in $I_1$, $I_2$, and $J$.

This is the motivation for the `PolyconvexICNN` architecture: each branch handles one invariant (or group), and the sum of convex functions is convex.

---

## 9. Symbolic Computation in hyper-surrogate

The `SymbolicHandler` class uses **SymPy** to compute exact symbolic expressions for stress and tangent:

```
SymPy symbolic C_ij  -->  W(C)  -->  S = dW/dC  -->  C_mat = dS/dC
         |                                                    |
    lambdify()                                          lambdify()
         |                                                    |
    NumPy function  <-----  batch evaluate over N samples ---'
```

This approach:

- Eliminates numerical differentiation errors
- Enables **Common Subexpression Elimination (CSE)** for optimized Fortran code
- Provides exact reference solutions for validating neural network surrogates

---

## 10. References

1. **Holzapfel, G.A.** (2000). *Nonlinear Solid Mechanics: A Continuum Approach for Engineering*. Wiley.
2. **Holzapfel, G.A., Gasser, T.C., Ogden, R.W.** (2000). A new constitutive framework for arterial wall mechanics and a comparative study of material models. *J. Elasticity*, 61, 1--48.
3. **Gasser, T.C., Ogden, R.W., Holzapfel, G.A.** (2006). Hyperelastic modelling of arterial layers with distributed collagen fibre orientations. *J. R. Soc. Interface*, 3, 15--35.
4. **Guccione, J.M., McCulloch, A.D., Waldman, L.K.** (1991). Passive material properties of intact ventricular myocardium determined from a cylindrical model. *J. Biomech. Eng.*, 113, 42--55.
5. **Treloar, L.R.G.** (1944). Stress-strain data for vulcanised rubber under various types of deformation. *Trans. Faraday Soc.*, 40, 59--70.
6. **Amos, B., Xu, L., Kolter, J.Z.** (2017). Input convex neural networks. *ICML*.
7. **Linka, K., et al.** (2023). A new family of Constitutive Artificial Neural Networks towards automated model discovery. *CMAME*, 403, 115731.
