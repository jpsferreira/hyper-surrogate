from typing import Any

import numpy as np


class Generator:
    def __init__(self, seed: int | None = None, size: int | None = None) -> None:
        self.seed = seed
        self.size = size
        np.random.seed(self.seed)

    def uniform(self, low: float, high: float) -> np.ndarray:
        return np.random.uniform(low, high, size=self.size)

    def integer_in_interval(self, low: int = 0, high: int = 3) -> np.ndarray[Any, Any]:
        """Returns a random integer in the interval [low, high]."""
        return np.random.randint(low, high, size=self.size)

    def float_in_interval(self, a: float = 0, b: float = 180, interval: float = 5) -> np.ndarray[Any, Any]:
        """Returns a random number in the interval [a, b] with a given interval."""
        if interval <= 0 or interval >= 180 or interval == 0:
            return np.array([0])
        return np.random.choice(np.arange(a, b + interval, interval), size=self.size)

    def normal(self, loc: float, scale: float) -> np.ndarray:
        return np.random.normal(loc, scale, size=self.size)

    def lognormal(self, mean: float, sigma: float) -> np.ndarray:
        return np.random.lognormal(mean, sigma, size=self.size)

    def beta(self, a: float, b: float) -> np.ndarray:
        return np.random.beta(a, b, size=self.size)

    def gamma(self, shape: float, scale: float) -> np.ndarray:
        return np.random.gamma(shape, scale, size=self.size)

    def weibull(self, a: float) -> np.ndarray:
        return np.random.weibull(a, size=self.size)
