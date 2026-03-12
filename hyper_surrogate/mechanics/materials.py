from __future__ import annotations

from functools import cached_property
from typing import Any, Callable, ClassVar

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

    @property
    def sef(self) -> Expr:
        h = self._handler
        C10, KBULK = self._symbols["C10"], self._symbols["KBULK"]
        return (h.isochoric_invariant1 - 3) * C10 + 0.25 * KBULK * (h.invariant3 - 1 - 2 * log(h.invariant3**0.5))


class MooneyRivlin(Material):
    DEFAULT_PARAMS: ClassVar[dict[str, float]] = {"C10": 0.3, "C01": 0.2, "KBULK": 1000.0}

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
            + 0.25 * KBULK * (h.invariant3 - 1 - 2 * log(h.invariant3**0.5))
        )
