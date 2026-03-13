from __future__ import annotations

from collections.abc import Callable
from functools import cached_property
from typing import Any, ClassVar

import numpy as np
from sympy import Expr, Matrix, Rational, Symbol, log

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

    def sef_from_invariants(self, i1_bar: Symbol, i2_bar: Symbol, j: Symbol) -> Expr:
        """Return SEF as a function of invariant symbols (I1_bar, I2_bar, J).

        Override in subclasses to enable energy gradient computation w.r.t. invariants.
        """
        raise NotImplementedError

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
        """Compute dW/d(I1_bar, I2_bar, J) for (N,3,3) C tensors. Returns (N,3)."""
        from sympy import diff
        from sympy import lambdify as sym_lambdify

        from hyper_surrogate.mechanics.kinematics import Kinematics

        i1s, i2s, js = Symbol("I1b"), Symbol("I2b"), Symbol("J")
        W = self.sef_from_invariants(i1s, i2s, js)
        dW = [diff(W, s) for s in (i1s, i2s, js)]

        param_syms = list(self._symbols.values())
        fn = sym_lambdify((i1s, i2s, js, *param_syms), dW, modules="numpy")

        i1 = Kinematics.isochoric_invariant1(c_batch)
        i2 = Kinematics.isochoric_invariant2(c_batch)
        j = np.sqrt(Kinematics.det_invariant(c_batch))

        param_vals = list(self._params.values())
        results = fn(i1, i2, j, *param_vals)
        n = len(c_batch)
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
        super().__init__(params)

    def _volumetric(self, j: Symbol) -> Expr:
        KBULK = self._symbols["KBULK"]
        i3 = j**2
        return Rational(1, 4) * KBULK * (i3 - 1 - 2 * log(i3 ** Rational(1, 2)))

    @property
    def sef(self) -> Expr:
        h = self._handler
        C10, KBULK = self._symbols["C10"], self._symbols["KBULK"]
        return (h.isochoric_invariant1 - 3) * C10 + 0.25 * KBULK * (h.invariant3 - 1 - 2 * log(h.invariant3**0.5))

    def sef_from_invariants(self, i1_bar: Symbol, i2_bar: Symbol, j: Symbol) -> Expr:
        C10 = self._symbols["C10"]
        return (i1_bar - 3) * C10 + self._volumetric(j)


class MooneyRivlin(Material):
    DEFAULT_PARAMS: ClassVar[dict[str, float]] = {"C10": 0.3, "C01": 0.2, "KBULK": 1000.0}

    def __init__(self, parameters: dict[str, float] | None = None) -> None:
        params = {**self.DEFAULT_PARAMS, **(parameters or {})}
        super().__init__(params)

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

    def sef_from_invariants(self, i1_bar: Symbol, i2_bar: Symbol, j: Symbol) -> Expr:
        C10, C01 = self._symbols["C10"], self._symbols["C01"]
        return (i1_bar - 3) * C10 + (i2_bar - 3) * C01 + self._volumetric(j)
