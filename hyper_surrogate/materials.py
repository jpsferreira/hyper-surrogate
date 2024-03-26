from typing import Any

import sympy as sym

from hyper_surrogate.symbolic import SymbolicHandler


class NeoHooke(SymbolicHandler):
    def __init__(self) -> None:
        super().__init__()
        self.parameters = ["C10", "C01"]

    @property
    def sef(self) -> Any:
        return (self.invariant1 - 3) * sym.Symbol("C10") + (self.invariant2 - 3) * sym.Symbol("C01")

    @property
    def pk2_symb(self) -> Any:
        return self.pk2_tensor(self.sef)

    @property
    def cmat_symb(self) -> Any:
        return self.cmat_tensor(self.pk2_symb)

    def pk2(self) -> Any:
        return self.lambdify(self.pk2_symb, *self.parameters)

    def cmat(self) -> Any:
        return self.lambdify(self.cmat_symb, *self.parameters)
