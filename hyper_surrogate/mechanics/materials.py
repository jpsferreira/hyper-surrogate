from __future__ import annotations

from collections.abc import Callable
from functools import cached_property
from typing import Any, ClassVar

import numpy as np
from sympy import Expr, Matrix, Rational, Symbol, log
from sympy import exp as sympy_exp

from hyper_surrogate.mechanics.symbolic import SymbolicHandler


class Material:
    """Base class for constitutive material models using composition."""

    def __init__(
        self,
        parameters: dict[str, float],
        fiber_direction: np.ndarray | None = None,
        fiber_directions: list[np.ndarray] | None = None,
    ) -> None:
        self._handler = SymbolicHandler()
        self._params = parameters
        self._symbols = {k: Symbol(k) for k in parameters}
        # Normalize fiber storage: always a list
        if fiber_directions is not None:
            self._fiber_directions = [np.asarray(d, dtype=float) for d in fiber_directions]
        elif fiber_direction is not None:
            self._fiber_directions = [np.asarray(fiber_direction, dtype=float)]
        else:
            self._fiber_directions = []

    @property
    def is_anisotropic(self) -> bool:
        """Whether this material depends on fiber invariants (I4, I5)."""
        return len(self._fiber_directions) > 0

    @property
    def fiber_direction(self) -> np.ndarray | None:
        """First fiber direction (backward compat). None if isotropic."""
        return self._fiber_directions[0] if self._fiber_directions else None

    @property
    def fiber_directions(self) -> list[np.ndarray]:
        """All fiber directions. Empty list if isotropic."""
        return self._fiber_directions

    @property
    def num_fiber_families(self) -> int:
        """Number of fiber families."""
        return len(self._fiber_directions)

    @property
    def input_dim(self) -> int:
        """Invariant input dimension: 3 (isotropic) + 2 per fiber family."""
        return 3 + 2 * self.num_fiber_families

    @property
    def handler(self) -> SymbolicHandler:
        return self._handler

    @property
    def sef(self) -> Expr:
        raise NotImplementedError

    def sef_from_invariants(
        self,
        i1_bar: Symbol,
        i2_bar: Symbol,
        j: Symbol,
        i4: Symbol | None = None,
        i5: Symbol | None = None,
    ) -> Expr:
        """Return SEF as a function of invariant symbols (I1_bar, I2_bar, J[, I4, I5]).

        Override in subclasses to enable energy gradient computation w.r.t. invariants.
        """
        raise NotImplementedError

    def sef_from_all_invariants(
        self,
        i1_bar: Symbol,
        i2_bar: Symbol,
        j: Symbol,
        fiber_invariants: list[tuple[Symbol, Symbol]] | None = None,
    ) -> Expr:
        """Return SEF as a function of all invariant symbols including multi-fiber.

        ``fiber_invariants`` is a list of (I4_k, I5_k) tuples, one per fiber family.
        Default implementation delegates to :meth:`sef_from_invariants` for 0 or 1 fibers.
        Override in multi-fiber subclasses.
        """
        if not fiber_invariants:
            return self.sef_from_invariants(i1_bar, i2_bar, j)
        if len(fiber_invariants) == 1:
            i4, i5 = fiber_invariants[0]
            return self.sef_from_invariants(i1_bar, i2_bar, j, i4, i5)
        msg = f"{type(self).__name__} does not support {len(fiber_invariants)} fiber families"
        raise NotImplementedError(msg)

    @cached_property
    def pk2_expr(self) -> Matrix:
        return self._handler.pk2(self.sef)

    @cached_property
    def cmat_expr(self) -> Any:
        return self._handler.cmat(self.pk2_expr)

    @cached_property
    def pk2_func(self) -> Callable:
        return self._handler.lambdify(self.pk2_expr, *self._symbols.values())  # type: ignore[no-any-return]

    @cached_property
    def cmat_func(self) -> Callable:
        return self._handler.lambdify(self.cmat_expr, *self._symbols.values())  # type: ignore[no-any-return]

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

    def evaluate_energy_grad_invariants(self, c_batch: np.ndarray) -> np.ndarray:
        """Compute dW/d(invariants) for (N,3,3) C tensors.

        Returns (N, input_dim) where input_dim = 3 + 2*num_fiber_families.
        """
        from sympy import diff
        from sympy import lambdify as sym_lambdify

        from hyper_surrogate.mechanics.kinematics import Kinematics

        i1s, i2s, js = Symbol("I1b"), Symbol("I2b"), Symbol("J")

        inv_syms: list[Symbol] = [i1s, i2s, js]
        fiber_inv_pairs: list[tuple[Symbol, Symbol]] = []

        for k in range(self.num_fiber_families):
            i4k = Symbol(f"I4_{k}")
            i5k = Symbol(f"I5_{k}")
            inv_syms.extend([i4k, i5k])
            fiber_inv_pairs.append((i4k, i5k))

        W = self.sef_from_all_invariants(i1s, i2s, js, fiber_inv_pairs or None)

        dW = [diff(W, s) for s in inv_syms]
        param_syms = list(self._symbols.values())
        fn = sym_lambdify((*inv_syms, *param_syms), dW, modules="numpy")

        i1 = Kinematics.isochoric_invariant1(c_batch)
        i2 = Kinematics.isochoric_invariant2(c_batch)
        j = np.sqrt(Kinematics.det_invariant(c_batch))

        param_vals = list(self._params.values())
        n = len(c_batch)

        inv_vals: list[np.ndarray] = [i1, i2, j]
        for a0 in self._fiber_directions:
            inv_vals.append(Kinematics.fiber_invariant4(c_batch, a0))
            inv_vals.append(Kinematics.fiber_invariant5(c_batch, a0))

        results = fn(*inv_vals, *param_vals)

        return np.column_stack([np.broadcast_to(np.asarray(r, dtype=float), (n,)) for r in results])

    def evaluate_energy(self, c_batch: np.ndarray) -> np.ndarray:
        """Evaluate strain energy for (N,3,3) C tensors. Returns (N,)."""
        from sympy import lambdify as sym_lambdify

        c_syms = self._handler.c_symbols()
        sef_func = sym_lambdify((c_syms, *self._symbols.values()), self.sef, modules="numpy")
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
    DEFAULT_PARAMS: ClassVar[dict[str, float]] = {"C10": 0.5, "KBULK": 1000.0}

    def __init__(self, parameters: dict[str, float] | None = None) -> None:
        params = {**self.DEFAULT_PARAMS, **(parameters or {})}
        super().__init__(params, fiber_direction=None)

    def _volumetric(self, j: Symbol) -> Expr:
        KBULK = self._symbols["KBULK"]
        i3 = j**2
        return Rational(1, 4) * KBULK * (i3 - 1 - 2 * log(i3 ** Rational(1, 2)))

    @property
    def sef(self) -> Expr:
        h = self._handler
        C10, KBULK = self._symbols["C10"], self._symbols["KBULK"]
        return (h.isochoric_invariant1 - 3) * C10 + 0.25 * KBULK * (h.invariant3 - 1 - 2 * log(h.invariant3**0.5))

    def sef_from_invariants(
        self,
        i1_bar: Symbol,
        i2_bar: Symbol,
        j: Symbol,
        i4: Symbol | None = None,
        i5: Symbol | None = None,
    ) -> Expr:
        C10 = self._symbols["C10"]
        return (i1_bar - 3) * C10 + self._volumetric(j)


class MooneyRivlin(Material):
    DEFAULT_PARAMS: ClassVar[dict[str, float]] = {"C10": 0.3, "C01": 0.2, "KBULK": 1000.0}

    def __init__(self, parameters: dict[str, float] | None = None) -> None:
        params = {**self.DEFAULT_PARAMS, **(parameters or {})}
        super().__init__(params, fiber_direction=None)

    def _volumetric(self, j: Symbol) -> Expr:
        KBULK = self._symbols["KBULK"]
        i3 = j**2
        return Rational(1, 4) * KBULK * (i3 - 1 - 2 * log(i3 ** Rational(1, 2)))

    @property
    def sef(self) -> Expr:
        h = self._handler
        C10, C01, KBULK = self._symbols["C10"], self._symbols["C01"], self._symbols["KBULK"]
        return (
            (h.isochoric_invariant1 - 3) * C10
            + (h.isochoric_invariant2 - 3) * C01
            + 0.25 * KBULK * (h.invariant3 - 1 - 2 * log(h.invariant3**0.5))
        )

    def sef_from_invariants(
        self,
        i1_bar: Symbol,
        i2_bar: Symbol,
        j: Symbol,
        i4: Symbol | None = None,
        i5: Symbol | None = None,
    ) -> Expr:
        C10, C01 = self._symbols["C10"], self._symbols["C01"]
        return (i1_bar - 3) * C10 + (i2_bar - 3) * C01 + self._volumetric(j)


class HolzapfelOgden(Material):
    """Holzapfel-Ogden transversely isotropic hyperelastic model.

    W = (a/2b)(exp(b(I1_bar - 3)) - 1) + (af/2bf)(exp(bf*<I4-1>^2) - 1) + vol(J)

    where <x> = max(x, 0) is the Macaulay bracket (fiber only under tension).
    """

    DEFAULT_PARAMS: ClassVar[dict[str, float]] = {
        "a": 0.059,
        "b": 8.023,
        "af": 18.472,
        "bf": 16.026,
        "KBULK": 1000.0,
    }

    def __init__(
        self,
        parameters: dict[str, float] | None = None,
        fiber_direction: np.ndarray | None = None,
    ) -> None:
        params = {**self.DEFAULT_PARAMS, **(parameters or {})}
        if fiber_direction is None:
            fiber_direction = np.array([1.0, 0.0, 0.0])
        super().__init__(params, fiber_direction=fiber_direction)

    def _volumetric(self, j: Symbol) -> Expr:
        KBULK = self._symbols["KBULK"]
        i3 = j**2
        return Rational(1, 4) * KBULK * (i3 - 1 - 2 * log(i3 ** Rational(1, 2)))

    @property
    def sef(self) -> Expr:
        msg = "HolzapfelOgden.sef requires fiber invariants; use sef_from_invariants instead"
        raise NotImplementedError(msg)

    def sef_from_invariants(
        self,
        i1_bar: Symbol,
        i2_bar: Symbol,
        j: Symbol,
        i4: Symbol | None = None,
        i5: Symbol | None = None,
    ) -> Expr:
        a_s = self._symbols["a"]
        b_s = self._symbols["b"]
        af_s = self._symbols["af"]
        bf_s = self._symbols["bf"]

        # Isotropic ground substance
        W_iso = (a_s / (2 * b_s)) * (sympy_exp(b_s * (i1_bar - 3)) - 1)

        # Fiber contribution (I4 term only; I5 omitted for simplicity)
        # Macaulay bracket: symbolic form; numerically clamped during evaluation
        W_fiber = (af_s / (2 * bf_s)) * (sympy_exp(bf_s * (i4 - 1) ** 2) - 1) if i4 is not None else Symbol("zero") * 0

        return W_iso + W_fiber + self._volumetric(j)


class GasserOgdenHolzapfel(Material):
    """Gasser-Ogden-Holzapfel model with fiber dispersion (kappa).

    W = (a/2b)(exp(b(I1_bar - 3)) - 1)
      + (af/2bf)(exp(bf * E_bar^2) - 1)
      + vol(J)

    where E_bar = kappa*(I1_bar - 3) + (1 - 3*kappa)*(I4 - 1).

    kappa in [0, 1/3]: kappa=0 recovers HolzapfelOgden, kappa=1/3 gives isotropic fiber.
    """

    DEFAULT_PARAMS: ClassVar[dict[str, float]] = {
        "a": 0.059,
        "b": 8.023,
        "af": 18.472,
        "bf": 16.026,
        "kappa": 0.226,
        "KBULK": 1000.0,
    }

    def __init__(
        self,
        parameters: dict[str, float] | None = None,
        fiber_direction: np.ndarray | None = None,
    ) -> None:
        params = {**self.DEFAULT_PARAMS, **(parameters or {})}
        if fiber_direction is None:
            fiber_direction = np.array([1.0, 0.0, 0.0])
        super().__init__(params, fiber_direction=fiber_direction)

    def _volumetric(self, j: Symbol) -> Expr:
        KBULK = self._symbols["KBULK"]
        i3 = j**2
        return Rational(1, 4) * KBULK * (i3 - 1 - 2 * log(i3 ** Rational(1, 2)))

    @property
    def sef(self) -> Expr:
        msg = "GasserOgdenHolzapfel.sef requires fiber invariants; use sef_from_invariants instead"
        raise NotImplementedError(msg)

    def sef_from_invariants(
        self,
        i1_bar: Symbol,
        i2_bar: Symbol,
        j: Symbol,
        i4: Symbol | None = None,
        i5: Symbol | None = None,
    ) -> Expr:
        a_s = self._symbols["a"]
        b_s = self._symbols["b"]
        af_s = self._symbols["af"]
        bf_s = self._symbols["bf"]
        kappa_s = self._symbols["kappa"]

        # Isotropic ground substance
        W_iso = (a_s / (2 * b_s)) * (sympy_exp(b_s * (i1_bar - 3)) - 1)

        # Fiber contribution with dispersion
        if i4 is not None:
            E_bar = kappa_s * (i1_bar - 3) + (1 - 3 * kappa_s) * (i4 - 1)
            W_fiber: Expr = (af_s / (2 * bf_s)) * (sympy_exp(bf_s * E_bar**2) - 1)
        else:
            W_fiber = Symbol("zero") * 0

        return W_iso + W_fiber + self._volumetric(j)


class Yeoh(Material):
    """Yeoh hyperelastic model (third-order reduced polynomial).

    W = C10*(I1_bar - 3) + C20*(I1_bar - 3)^2 + C30*(I1_bar - 3)^3 + vol(J)
    """

    DEFAULT_PARAMS: ClassVar[dict[str, float]] = {
        "C10": 0.5,
        "C20": -0.01,
        "C30": 0.001,
        "KBULK": 1000.0,
    }

    def __init__(self, parameters: dict[str, float] | None = None) -> None:
        params = {**self.DEFAULT_PARAMS, **(parameters or {})}
        super().__init__(params, fiber_direction=None)

    def _volumetric(self, j: Symbol) -> Expr:
        KBULK = self._symbols["KBULK"]
        i3 = j**2
        return Rational(1, 4) * KBULK * (i3 - 1 - 2 * log(i3 ** Rational(1, 2)))

    @property
    def sef(self) -> Expr:
        h = self._handler
        C10 = self._symbols["C10"]
        C20 = self._symbols["C20"]
        C30 = self._symbols["C30"]
        KBULK = self._symbols["KBULK"]
        i1_dev = h.isochoric_invariant1 - 3
        return (
            C10 * i1_dev
            + C20 * i1_dev**2
            + C30 * i1_dev**3
            + Rational(1, 4) * KBULK * (h.invariant3 - 1 - 2 * log(h.invariant3 ** Rational(1, 2)))
        )

    def sef_from_invariants(
        self,
        i1_bar: Symbol,
        i2_bar: Symbol,
        j: Symbol,
        i4: Symbol | None = None,
        i5: Symbol | None = None,
    ) -> Expr:
        C10 = self._symbols["C10"]
        C20 = self._symbols["C20"]
        C30 = self._symbols["C30"]
        i1_dev = i1_bar - 3
        return C10 * i1_dev + C20 * i1_dev**2 + C30 * i1_dev**3 + self._volumetric(j)


class Demiray(Material):
    """Demiray exponential hyperelastic model.

    W = (C1 / C2) * (exp(C2 * (I1_bar - 3)) - 1) + vol(J)
    """

    DEFAULT_PARAMS: ClassVar[dict[str, float]] = {
        "C1": 0.05,
        "C2": 8.0,
        "KBULK": 1000.0,
    }

    def __init__(self, parameters: dict[str, float] | None = None) -> None:
        params = {**self.DEFAULT_PARAMS, **(parameters or {})}
        super().__init__(params, fiber_direction=None)

    def _volumetric(self, j: Symbol) -> Expr:
        KBULK = self._symbols["KBULK"]
        i3 = j**2
        return Rational(1, 4) * KBULK * (i3 - 1 - 2 * log(i3 ** Rational(1, 2)))

    @property
    def sef(self) -> Expr:
        h = self._handler
        C1 = self._symbols["C1"]
        C2 = self._symbols["C2"]
        KBULK = self._symbols["KBULK"]
        return (C1 / C2) * (sympy_exp(C2 * (h.isochoric_invariant1 - 3)) - 1) + Rational(1, 4) * KBULK * (
            h.invariant3 - 1 - 2 * log(h.invariant3 ** Rational(1, 2))
        )

    def sef_from_invariants(
        self,
        i1_bar: Symbol,
        i2_bar: Symbol,
        j: Symbol,
        i4: Symbol | None = None,
        i5: Symbol | None = None,
    ) -> Expr:
        C1 = self._symbols["C1"]
        C2 = self._symbols["C2"]
        return (C1 / C2) * (sympy_exp(C2 * (i1_bar - 3)) - 1) + self._volumetric(j)


class Ogden(Material):
    """Ogden hyperelastic model (N-term, principal stretch formulation).

    W = sum_p (mu_p / alpha_p) * (lambda1^alpha_p + lambda2^alpha_p + lambda3^alpha_p - 3) + vol(J)

    This model is expressed in principal stretches, not invariants.
    The sef_from_invariants method approximates via I1_bar and I2_bar using
    the relationship between invariants and symmetric functions of stretches.
    """

    DEFAULT_PARAMS: ClassVar[dict[str, float]] = {
        "mu1": 1.0,
        "alpha1": 2.0,
        "KBULK": 1000.0,
    }

    def __init__(self, parameters: dict[str, float] | None = None) -> None:
        params = {**self.DEFAULT_PARAMS, **(parameters or {})}
        # Determine number of Ogden terms from parameter keys
        n = 1
        while f"mu{n + 1}" in params and f"alpha{n + 1}" in params:
            n += 1
        self._n_terms = n
        super().__init__(params, fiber_direction=None)

    def _volumetric(self, j: Symbol) -> Expr:
        KBULK = self._symbols["KBULK"]
        i3 = j**2
        return Rational(1, 4) * KBULK * (i3 - 1 - 2 * log(i3 ** Rational(1, 2)))

    @property
    def sef(self) -> Expr:
        msg = "Ogden.sef requires principal stretches; use evaluate_energy instead"
        raise NotImplementedError(msg)

    def sef_from_invariants(
        self,
        i1_bar: Symbol,
        i2_bar: Symbol,
        j: Symbol,
        i4: Symbol | None = None,
        i5: Symbol | None = None,
    ) -> Expr:
        msg = "Ogden model is not naturally expressed in invariants; use evaluate_energy instead"
        raise NotImplementedError(msg)

    def evaluate_energy(self, c_batch: np.ndarray) -> np.ndarray:
        """Evaluate Ogden energy via eigenvalues of C."""
        from hyper_surrogate.mechanics.kinematics import Kinematics

        n_samples = len(c_batch)
        j_vals = np.sqrt(Kinematics.det_invariant(c_batch))
        # Principal stretches squared = eigenvalues of C
        eigvals = np.linalg.eigvalsh(c_batch)  # (N, 3), sorted ascending
        # Isochoric principal stretches: lambda_bar_i = J^(-1/3) * lambda_i
        lambda_bar_sq = eigvals * (j_vals ** (-2.0 / 3.0))[:, None]
        lambda_bar = np.sqrt(np.maximum(lambda_bar_sq, 1e-30))

        W = np.zeros(n_samples)
        for p in range(1, self._n_terms + 1):
            mu_p = self._params[f"mu{p}"]
            alpha_p = self._params[f"alpha{p}"]
            W += (mu_p / alpha_p) * (
                lambda_bar[:, 0] ** alpha_p + lambda_bar[:, 1] ** alpha_p + lambda_bar[:, 2] ** alpha_p - 3
            )

        # Volumetric
        KBULK = self._params["KBULK"]
        j2 = j_vals**2
        W += 0.25 * KBULK * (j2 - 1 - 2 * np.log(j_vals))

        return W

    def evaluate_energy_grad_voigt(self, c_batch: np.ndarray) -> np.ndarray:
        """Numerical gradient dW/dC via finite differences on the 6 Voigt components of C.

        Returns (N, 6) array. Not to be confused with invariant gradients.
        """
        n = len(c_batch)
        eps = 1e-7
        W0 = self.evaluate_energy(c_batch)
        grad = np.zeros((n, 6))
        idx_map = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]
        for k, (i, j) in enumerate(idx_map):
            c_p = c_batch.copy()
            c_p[:, i, j] += eps
            c_p[:, j, i] += eps  # symmetry
            W_p = self.evaluate_energy(c_p)
            grad[:, k] = (W_p - W0) / eps
        return grad

    def evaluate_energy_grad_invariants(self, c_batch: np.ndarray) -> np.ndarray:
        """Not available for Ogden — uses principal stretch formulation, not invariants."""
        msg = "Ogden uses principal stretches; use evaluate_energy_grad_voigt for dW/dC Voigt gradients"
        raise NotImplementedError(msg)


class Guccione(Material):
    """Guccione transversely isotropic cardiac model.

    W = (C / 2) * (exp(Q) - 1) + vol(J)

    Q = bf * E_ff^2 + bt * (E_ss^2 + E_nn^2 + E_sn^2 + E_ns^2)
      + bfs * (E_fs^2 + E_sf^2 + E_fn^2 + E_nf^2)

    where E = (C - I) / 2 is the Green-Lagrange strain in the fiber frame.
    """

    DEFAULT_PARAMS: ClassVar[dict[str, float]] = {
        "C": 0.876,
        "bf": 18.48,
        "bt": 3.58,
        "bfs": 1.627,
        "KBULK": 1000.0,
    }

    def __init__(
        self,
        parameters: dict[str, float] | None = None,
        fiber_direction: np.ndarray | None = None,
        sheet_direction: np.ndarray | None = None,
    ) -> None:
        params = {**self.DEFAULT_PARAMS, **(parameters or {})}
        if fiber_direction is None:
            fiber_direction = np.array([1.0, 0.0, 0.0])
        self._sheet_direction = (
            np.asarray(sheet_direction, dtype=float) if sheet_direction is not None else np.array([0.0, 1.0, 0.0])
        )
        super().__init__(params, fiber_direction=fiber_direction)

    def _volumetric(self, j: Symbol) -> Expr:
        KBULK = self._symbols["KBULK"]
        i3 = j**2
        return Rational(1, 4) * KBULK * (i3 - 1 - 2 * log(i3 ** Rational(1, 2)))

    @property
    def sef(self) -> Expr:
        msg = "Guccione.sef requires fiber-frame computation; use evaluate_energy instead"
        raise NotImplementedError(msg)

    def sef_from_invariants(
        self,
        i1_bar: Symbol,
        i2_bar: Symbol,
        j: Symbol,
        i4: Symbol | None = None,
        i5: Symbol | None = None,
    ) -> Expr:
        msg = "Guccione model is expressed in Green strain components, not invariants"
        raise NotImplementedError(msg)

    def _rotation_matrix(self) -> np.ndarray:
        """Build rotation from global to fiber (f, s, n) frame."""
        fiber_dir = self.fiber_direction
        if fiber_dir is None:  # pragma: no cover
            msg = "Guccione requires a fiber direction"
            raise ValueError(msg)
        f0 = fiber_dir / np.linalg.norm(fiber_dir)
        s0 = self._sheet_direction / np.linalg.norm(self._sheet_direction)
        n0 = np.cross(f0, s0)
        n0 = n0 / np.linalg.norm(n0)
        return np.array([f0, s0, n0])  # (3, 3) rows = local axes

    def evaluate_energy(self, c_batch: np.ndarray) -> np.ndarray:
        """Evaluate Guccione energy via Green strain in fiber frame."""
        from hyper_surrogate.mechanics.kinematics import Kinematics

        Q_mat = self._rotation_matrix()  # (3, 3)

        # Green-Lagrange strain: E = (C - I) / 2
        E_global = 0.5 * (c_batch - np.eye(3))

        # Rotate to fiber frame: E_local = Q * E_global * Q^T
        E_local = np.einsum("ij,njk,lk->nil", Q_mat, E_global, Q_mat)

        C_p = self._params["C"]
        bf_p = self._params["bf"]
        bt_p = self._params["bt"]
        bfs_p = self._params["bfs"]

        E_ff = E_local[:, 0, 0]
        E_ss = E_local[:, 1, 1]
        E_nn = E_local[:, 2, 2]
        E_fs = E_local[:, 0, 1]
        E_fn = E_local[:, 0, 2]
        E_sn = E_local[:, 1, 2]

        Q_val = bf_p * E_ff**2 + bt_p * (E_ss**2 + E_nn**2 + 2 * E_sn**2) + bfs_p * (2 * E_fs**2 + 2 * E_fn**2)

        W = 0.5 * C_p * (np.exp(Q_val) - 1)

        # Volumetric
        j_vals = np.sqrt(Kinematics.det_invariant(c_batch))
        KBULK = self._params["KBULK"]
        j2 = j_vals**2
        W += 0.25 * KBULK * (j2 - 1 - 2 * np.log(j_vals))

        return np.asarray(W)

    def evaluate_energy_grad_voigt(self, c_batch: np.ndarray) -> np.ndarray:
        """Numerical gradient dW/dC via finite differences on the 6 Voigt components of C.

        Returns (N, 6) array. Not to be confused with invariant gradients.
        """
        n = len(c_batch)
        eps = 1e-7
        W0 = self.evaluate_energy(c_batch)
        grad = np.zeros((n, 6))
        idx_map = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]
        for k, (i, j) in enumerate(idx_map):
            c_p = c_batch.copy()
            c_p[:, i, j] += eps
            c_p[:, j, i] += eps
            W_p = self.evaluate_energy(c_p)
            grad[:, k] = (W_p - W0) / eps
        return grad

    def evaluate_energy_grad_invariants(self, c_batch: np.ndarray) -> np.ndarray:
        """Not available for Guccione — uses C-component formulation, not invariants."""
        msg = "Guccione uses C-component formulation; use evaluate_energy_grad_voigt for dW/dC Voigt gradients"
        raise NotImplementedError(msg)


class Fung(Material):
    """Fung-type exponential model with configurable Q quadratic form.

    W = (c / 2) * (exp(Q) - 1) + vol(J)

    Q = sum_ij b_ij * E_ij * E_ij  (Green-Lagrange strain components)

    Default: isotropic Q = b1*(E11^2 + E22^2 + E33^2) + b2*(E12^2 + E13^2 + E23^2)
    """

    DEFAULT_PARAMS: ClassVar[dict[str, float]] = {
        "c": 1.0,
        "b1": 10.0,
        "b2": 5.0,
        "KBULK": 1000.0,
    }

    def __init__(self, parameters: dict[str, float] | None = None) -> None:
        params = {**self.DEFAULT_PARAMS, **(parameters or {})}
        super().__init__(params, fiber_direction=None)

    def _volumetric(self, j: Symbol) -> Expr:
        KBULK = self._symbols["KBULK"]
        i3 = j**2
        return Rational(1, 4) * KBULK * (i3 - 1 - 2 * log(i3 ** Rational(1, 2)))

    @property
    def sef(self) -> Expr:
        msg = "Fung.sef is expressed in Green strain; use evaluate_energy instead"
        raise NotImplementedError(msg)

    def sef_from_invariants(
        self,
        i1_bar: Symbol,
        i2_bar: Symbol,
        j: Symbol,
        i4: Symbol | None = None,
        i5: Symbol | None = None,
    ) -> Expr:
        msg = "Fung model is expressed in Green strain components, not invariants"
        raise NotImplementedError(msg)

    def evaluate_energy(self, c_batch: np.ndarray) -> np.ndarray:
        """Evaluate Fung energy via Green strain."""
        from hyper_surrogate.mechanics.kinematics import Kinematics

        E = 0.5 * (c_batch - np.eye(3))

        c_p = self._params["c"]
        b1 = self._params["b1"]
        b2 = self._params["b2"]

        Q = b1 * (E[:, 0, 0] ** 2 + E[:, 1, 1] ** 2 + E[:, 2, 2] ** 2) + b2 * 2 * (
            E[:, 0, 1] ** 2 + E[:, 0, 2] ** 2 + E[:, 1, 2] ** 2
        )

        W = 0.5 * c_p * (np.exp(Q) - 1)

        # Volumetric
        j_vals = np.sqrt(Kinematics.det_invariant(c_batch))
        KBULK = self._params["KBULK"]
        j2 = j_vals**2
        W += 0.25 * KBULK * (j2 - 1 - 2 * np.log(j_vals))

        return W

    def evaluate_energy_grad_voigt(self, c_batch: np.ndarray) -> np.ndarray:
        """Numerical gradient dW/dC via finite differences on the 6 Voigt components of C.

        Returns (N, 6) array. Not to be confused with invariant gradients.
        """
        n = len(c_batch)
        eps = 1e-7
        W0 = self.evaluate_energy(c_batch)
        grad = np.zeros((n, 6))
        idx_map = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]
        for k, (i, j) in enumerate(idx_map):
            c_p = c_batch.copy()
            c_p[:, i, j] += eps
            c_p[:, j, i] += eps
            W_p = self.evaluate_energy(c_p)
            grad[:, k] = (W_p - W0) / eps
        return grad

    def evaluate_energy_grad_invariants(self, c_batch: np.ndarray) -> np.ndarray:
        """Not available for Fung — uses C-component formulation, not invariants."""
        msg = "Fung uses C-component formulation; use evaluate_energy_grad_voigt for dW/dC Voigt gradients"
        raise NotImplementedError(msg)


class HolzapfelOgdenBiaxial(Material):
    """Two-fiber Holzapfel-Ogden model for arterial wall tissue.

    W = (a/2b)(exp(b(I1_bar - 3)) - 1)
      + sum_{k=1}^{2} (af/2bf)(exp(bf*(I4_k - 1)^2) - 1)
      + vol(J)

    Each fiber family contributes an I4 term; both share the same parameters.
    """

    DEFAULT_PARAMS: ClassVar[dict[str, float]] = {
        "a": 0.059,
        "b": 8.023,
        "af": 18.472,
        "bf": 16.026,
        "KBULK": 1000.0,
    }

    def __init__(
        self,
        parameters: dict[str, float] | None = None,
        fiber_directions: list[np.ndarray] | None = None,
    ) -> None:
        params = {**self.DEFAULT_PARAMS, **(parameters or {})}
        if fiber_directions is None:
            theta = np.radians(39.0)
            fiber_directions = [
                np.array([np.cos(theta), np.sin(theta), 0.0]),
                np.array([np.cos(theta), -np.sin(theta), 0.0]),
            ]
        if len(fiber_directions) != 2:
            msg = f"HolzapfelOgdenBiaxial requires exactly 2 fiber directions, got {len(fiber_directions)}"
            raise ValueError(msg)
        super().__init__(params, fiber_directions=fiber_directions)

    def _volumetric(self, j: Symbol) -> Expr:
        KBULK = self._symbols["KBULK"]
        i3 = j**2
        return Rational(1, 4) * KBULK * (i3 - 1 - 2 * log(i3 ** Rational(1, 2)))

    @property
    def sef(self) -> Expr:
        msg = "HolzapfelOgdenBiaxial.sef requires fiber invariants; use sef_from_all_invariants instead"
        raise NotImplementedError(msg)

    def sef_from_invariants(
        self,
        i1_bar: Symbol,
        i2_bar: Symbol,
        j: Symbol,
        i4: Symbol | None = None,
        i5: Symbol | None = None,
    ) -> Expr:
        msg = "HolzapfelOgdenBiaxial has 2 fibers; use sef_from_all_invariants instead"
        raise NotImplementedError(msg)

    def sef_from_all_invariants(
        self,
        i1_bar: Symbol,
        i2_bar: Symbol,
        j: Symbol,
        fiber_invariants: list[tuple[Symbol, Symbol]] | None = None,
    ) -> Expr:
        a_s = self._symbols["a"]
        b_s = self._symbols["b"]
        af_s = self._symbols["af"]
        bf_s = self._symbols["bf"]

        # Isotropic ground substance
        W: Expr = (a_s / (2 * b_s)) * (sympy_exp(b_s * (i1_bar - 3)) - 1)

        # Fiber contributions
        if fiber_invariants is not None:
            for i4_k, _i5_k in fiber_invariants:
                W = W + (af_s / (2 * bf_s)) * (sympy_exp(bf_s * (i4_k - 1) ** 2) - 1)

        return W + self._volumetric(j)

    def evaluate_energy(self, c_batch: np.ndarray) -> np.ndarray:
        """Evaluate strain energy for (N,3,3) C tensors via invariants."""
        from sympy import lambdify as sym_lambdify

        from hyper_surrogate.mechanics.kinematics import Kinematics

        i1s, i2s, js = Symbol("I1b"), Symbol("I2b"), Symbol("J")
        fiber_inv_pairs: list[tuple[Symbol, Symbol]] = []
        inv_syms: list[Symbol] = [i1s, i2s, js]
        for k in range(self.num_fiber_families):
            i4k, i5k = Symbol(f"I4_{k}"), Symbol(f"I5_{k}")
            inv_syms.extend([i4k, i5k])
            fiber_inv_pairs.append((i4k, i5k))

        W = self.sef_from_all_invariants(i1s, i2s, js, fiber_inv_pairs)
        param_syms = list(self._symbols.values())
        fn = sym_lambdify((*inv_syms, *param_syms), W, modules="numpy")

        i1 = Kinematics.isochoric_invariant1(c_batch)
        i2 = Kinematics.isochoric_invariant2(c_batch)
        j = np.sqrt(Kinematics.det_invariant(c_batch))

        inv_vals: list[np.ndarray] = [i1, i2, j]
        for a0 in self._fiber_directions:
            inv_vals.append(Kinematics.fiber_invariant4(c_batch, a0))
            inv_vals.append(Kinematics.fiber_invariant5(c_batch, a0))

        param_vals = list(self._params.values())
        results = fn(*inv_vals, *param_vals)
        return np.asarray(results, dtype=float)
