# Theory

## Continuum Mechanics Background

### Deformation

The deformation gradient $\mathbf{F}$ maps material points from the reference to the current configuration. The right Cauchy-Green tensor $\mathbf{C} = \mathbf{F}^T \mathbf{F}$ characterizes the deformation state.

### Invariants

The principal invariants of $\mathbf{C}$ are:

- $I_1 = \text{tr}(\mathbf{C})$
- $I_2 = \frac{1}{2}[(\text{tr}\,\mathbf{C})^2 - \text{tr}(\mathbf{C}^2)]$
- $I_3 = \det(\mathbf{C}) = J^2$

Their isochoric (volume-preserving) counterparts:

- $\bar{I}_1 = J^{-2/3} I_1$
- $\bar{I}_2 = J^{-4/3} I_2$

For anisotropic materials with fiber direction $\mathbf{a}_0$:

- $I_4 = \mathbf{a}_0 \cdot \mathbf{C} \mathbf{a}_0$ (fiber stretch squared)
- $I_5 = \mathbf{a}_0 \cdot \mathbf{C}^2 \mathbf{a}_0$

### Strain Energy Function (SEF)

Hyperelastic materials are defined by a strain energy function $W(\mathbf{C})$ from which all stress and tangent quantities derive:

- **Second Piola-Kirchhoff stress**: $\mathbf{S} = 2 \frac{\partial W}{\partial \mathbf{C}}$
- **Cauchy stress**: $\boldsymbol{\sigma} = \frac{1}{J} \mathbf{F} \mathbf{S} \mathbf{F}^T$
- **Material tangent**: $\mathbb{C} = 4 \frac{\partial^2 W}{\partial \mathbf{C} \partial \mathbf{C}}$

## Supported Material Models

### Isotropic Models

| Model             | SEF                                                                                                                                  | Parameters               |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------ | ------------------------ |
| **Neo-Hooke**     | $W = C_{10}(\bar{I}_1 - 3) + U(J)$                                                                                                   | $C_{10}$                 |
| **Mooney-Rivlin** | $W = C_{10}(\bar{I}_1 - 3) + C_{01}(\bar{I}_2 - 3) + U(J)$                                                                           | $C_{10}, C_{01}$         |
| **Yeoh**          | $W = \sum_{i=1}^{3} C_{i0}(\bar{I}_1 - 3)^i + U(J)$                                                                                  | $C_{10}, C_{20}, C_{30}$ |
| **Demiray**       | $W = \frac{C_1}{C_2}[\exp(C_2(\bar{I}_1-3)) - 1] + U(J)$                                                                             | $C_1, C_2$               |
| **Ogden**         | $W = \sum_p \frac{\mu_p}{\alpha_p}(\bar{\lambda}_1^{\alpha_p} + \bar{\lambda}_2^{\alpha_p} + \bar{\lambda}_3^{\alpha_p} - 3) + U(J)$ | $\mu_p, \alpha_p$        |
| **Fung**          | $W = \frac{c}{2}[\exp(Q) - 1] + U(J)$                                                                                                | $c, b_1, b_2$            |

### Anisotropic Models

| Model               | SEF                                                                                         | Parameters               |
| ------------------- | ------------------------------------------------------------------------------------------- | ------------------------ |
| **Holzapfel-Ogden** | $W = \frac{a}{2b}[\exp(b(\bar{I}_1-3))-1] + \frac{a_f}{2b_f}[\exp(b_f(I_4-1)^2)-1] + U(J)$  | $a, b, a_f, b_f$         |
| **GOH**             | $W = \frac{a}{2b}[\exp(b(\bar{I}_1-3))-1] + \frac{a_f}{2b_f}[\exp(b_f \bar{E}^2)-1] + U(J)$ | $a, b, a_f, b_f, \kappa$ |
| **Guccione**        | $W = \frac{C}{2}[\exp(Q)-1] + U(J)$ with fiber-frame Q                                      | $C, b_f, b_t, b_{fs}$    |

where $\bar{E} = \kappa(\bar{I}_1 - 3) + (1-3\kappa)(I_4 - 1)$ is the GOH generalized strain invariant with dispersion $\kappa \in [0, 1/3]$.

### Volumetric Contribution

All models use the standard volumetric penalty:

$$U(J) = \frac{K}{4}(J^2 - 1 - 2\ln J)$$

where $K$ is the bulk modulus.

## Surrogate Pipeline

### Architecture

```
Material -> DeformationGenerator -> Dataset -> Model -> Trainer -> FortranEmitter -> .f90
```

1. **Define** a `Material` with its SEF and parameters
2. **Generate** synthetic deformation gradients (uniaxial, biaxial, shear, combined)
3. **Compute** invariants and stress/energy targets
4. **Train** a neural network (MLP, ICNN, PolyconvexICNN, or CANN)
5. **Export** to Fortran 90 UMAT subroutine

### Neural Network Architectures

- **MLP**: Standard feedforward network. Flexible but no physics guarantees.
- **ICNN**: Input-Convex Neural Network. Guarantees convexity of energy w.r.t. inputs via non-negative weights on the z-path.
- **PolyconvexICNN**: Polyconvex ICNN with separate branches for invariant groups.
- **CANN**: Constitutive ANN with interpretable basis functions and non-negative weights. Enables model discovery through sparsification.

### Thermodynamically Consistent Training

The `EnergyStressLoss` enforces thermodynamic consistency by jointly minimizing:

$$\mathcal{L} = \alpha \|W_{pred} - W_{true}\|^2 + \beta \left\|\frac{\partial W_{pred}}{\partial \mathbf{I}} - \frac{\partial W_{true}}{\partial \mathbf{I}}\right\|^2$$

where the stress (gradient) term is computed via automatic differentiation through the network.

## Experimental Data Integration

The `ExperimentalData` class loads biomechanical test data (uniaxial, biaxial) and converts to deformation gradients for integration with the surrogate pipeline.

The `fit_material` function uses `scipy.optimize.minimize` to fit material parameters to experimental data by minimizing the stress residual:

$$\min_{\theta} \sum_i \|\boldsymbol{\sigma}_{model}(\lambda_i; \theta) - \boldsymbol{\sigma}_{exp,i}\|^2$$

## References

1. Holzapfel, G.A., Gasser, T.C., Ogden, R.W. (2000). A new constitutive framework for arterial wall mechanics. _J. Elasticity_, 61, 1-48.
2. Gasser, T.C., Ogden, R.W., Holzapfel, G.A. (2006). Hyperelastic modelling of arterial layers with distributed collagen fibre orientations. _J. R. Soc. Interface_, 3, 15-35.
3. Linka, K., et al. (2023). A new family of Constitutive Artificial Neural Networks towards automated model discovery. _Comput. Methods Appl. Mech. Eng._, 403, 115731.
4. Amos, B., Xu, L., Kolter, J.Z. (2017). Input convex neural networks. _ICML_.
5. Guccione, J.M., McCulloch, A.D., Waldman, L.K. (1991). Passive material properties of intact ventricular myocardium determined from a cylindrical model. _J. Biomech. Eng._, 113, 42-55.
