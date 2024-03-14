import numpy as np


class Generator:
    def __init__(self, seed=None, size=None):
        self.seed = seed
        self.size = size

    # init np whenever a method is called
    @property
    def np(self):
        return np.random.seed(self.seed)

    def uniform(self, low, high):
        return np.random.uniform(low, high, size=self.size)

    def int(self, low: int = 0, high: int = 3):
        """Returns a random integer in the interval [low, high]."""
        return np.random.randint(low, high, size=self.size)

    def in_interval(self, a: float = 0, b: float = 180, interval: float = 5):
        """Returns a random number in the interval [a, b] with a given interval."""
        if interval <= 0 or interval >= 180 or interval == 0:
            return 0
        return np.random.choice(np.arange(a, b+interval, interval), size=self.size)

    def normal(self, loc, scale):
        return np.random.normal(loc, scale, size=self.size)

    def lognormal(self, mean, sigma):
        return np.random.lognormal(mean, sigma, size=self.size)

    def beta(self, a, b):
        return np.random.beta(a, b, size=self.size)

    def gamma(self, shape, scale):
        return np.random.gamma(shape, scale, size=self.size)

    def weibull(self, a):
        return np.random.weibull(a, size=self.size)
