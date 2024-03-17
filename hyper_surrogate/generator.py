from typing import Any

import numpy as np


class Generator:
    """A class that provides various random number generation methods."""

    def __init__(self, seed: int | None = None, size: int | None = None) -> None:
        """
        Initialize the Generator object.

        Args:
            seed (int | None): The seed value for random number generation. If None, a random seed will be used.
            size (int | None): The size of the generated random numbers. If None, a single random number will be generated.

        Returns:
            None
        """
        self.seed = seed
        self.size = size
        np.random.seed(self.seed)

    def uniform(self, low: float, high: float) -> np.ndarray:
        """
        Generate random numbers from a uniform distribution.

        Args:
            low (float): The lower bound of the distribution.
            high (float): The upper bound of the distribution.

        Returns:
            np.ndarray: An array of random numbers from the uniform distribution.
        """
        return np.random.uniform(low, high, size=self.size)

    def integer_in_interval(self, low: int = 0, high: int = 3) -> np.ndarray[Any, Any]:
        """
        Generate random integers in the specified interval.

        Args:
            low (int): The lower bound of the interval (inclusive).
            high (int): The upper bound of the interval (exclusive).

        Returns:
            np.ndarray: An array of random integers in the specified interval.
        """
        return np.random.randint(low, high, size=self.size)

    def float_in_interval(self, a: float = 0, b: float = 180, interval: float = 5) -> np.ndarray[Any, Any]:
        """
        Generate random numbers in the specified interval with a given interval.

        Args:
            a (float): The lower bound of the interval.
            b (float): The upper bound of the interval.
            interval (float): The interval between the generated numbers.

        Returns:
            np.ndarray: An array of random numbers in the specified interval.
        """
        if interval <= 0 or interval >= 180 or interval == 0:
            return np.array([0])
        return np.random.choice(np.arange(a, b + interval, interval), size=self.size)

    def normal(self, loc: float, scale: float) -> np.ndarray:
        """
        Generate random numbers from a normal distribution.

        Args:
            loc (float): The mean of the distribution.
            scale (float): The standard deviation of the distribution.

        Returns:
            np.ndarray: An array of random numbers from the normal distribution.
        """
        return np.random.normal(loc, scale, size=self.size)

    def lognormal(self, mean: float, sigma: float) -> np.ndarray:
        """
        Generate random numbers from a log-normal distribution.

        Args:
            mean (float): The mean of the underlying normal distribution.
            sigma (float): The standard deviation of the underlying normal distribution.

        Returns:
            np.ndarray: An array of random numbers from the log-normal distribution.
        """
        return np.random.lognormal(mean, sigma, size=self.size)

    def beta(self, a: float, b: float) -> np.ndarray:
        """
        Generate random numbers from a beta distribution.

        Args:
            a (float): The shape parameter (alpha) of the distribution.
            b (float): The shape parameter (beta) of the distribution.

        Returns:
            np.ndarray: An array of random numbers from the beta distribution.
        """
        return np.random.beta(a, b, size=self.size)

    def gamma(self, shape: float, scale: float) -> np.ndarray:
        """
        Generate random numbers from a gamma distribution.

        Args:
            shape (float): The shape parameter (k) of the distribution.
            scale (float): The scale parameter (theta) of the distribution.

        Returns:
            np.ndarray: An array of random numbers from the gamma distribution.
        """
        return np.random.gamma(shape, scale, size=self.size)

    def weibull(self, a: float) -> np.ndarray:
        """
        Generate random numbers from a Weibull distribution.

        Args:
            a (float): The shape parameter (k) of the distribution.

        Returns:
            np.ndarray: An array of random numbers from the Weibull distribution.
        """
        return np.random.weibull(a, size=self.size)
