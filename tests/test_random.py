import numpy as np

from hyper_surrogate.generator import Generator


class TestGenerator:
    # The class can be instantiated without any arguments.
    def test_instantiation_without_arguments(self):
        random = Generator()
        assert isinstance(random, Generator)

    # The class can be instantiated with a seed of 0.
    def test_instantiation_with_seed_zero(self):
        random = Generator(seed=0)
        assert isinstance(random, Generator)

    # The class can be instantiated with a seed argument.
    def test_instantiation_with_seed_argument(self):
        random = Generator(seed=123)
        assert isinstance(random, Generator)

    # The class can be instantiated with a size argument.
    def test_instantiation_with_size_argument(self):
        random = Generator(size=10)
        assert isinstance(random, Generator)

    # The uniform method returns an array of random numbers between low and high.
    def test_uniform_method(self):
        random = Generator(seed=42, size=10)
        result = random.uniform(0, 1)
        assert isinstance(result, np.ndarray)
        assert len(result) == 10
        assert np.all(result >= 0) and np.all(result <= 1)

    # The normal method returns an array of normally distributed random numbers.
    def test_normal_method_returns_array_of_normally_distributed_random_numbers(self):
        random = Generator(seed=42, size=10)
        result = random.normal(0, 1)
        assert isinstance(result, np.ndarray)
        assert len(result) == 10
        assert all(isinstance(x, float) for x in result)
        assert all(-3 <= x <= 3 for x in result)

    # The lognormal method returns an array of log-normally distributed random numbers.
    def test_lognormal_method(self):
        random = Generator(seed=42, size=10)
        result = random.lognormal(0, 1)
        assert isinstance(result, np.ndarray)
        assert len(result) == 10
        assert all(result > 0)

    # The gamma method returns an array of gamma distributed random numbers.
    def test_gamma_method_returns_array(self):
        random = Generator(seed=42, size=10)
        result = random.gamma(shape=2, scale=1)
        assert isinstance(result, np.ndarray)
        assert len(result) == 10
        assert all(isinstance(x, float) for x in result)

    # The beta method returns an array of beta distributed random numbers.
    def test_beta_method_returns_array(self):
        random = Generator(seed=42, size=10)
        result = random.beta(2, 3)
        assert isinstance(result, np.ndarray)
        assert len(result) == 10
        assert all(0 <= x <= 1 for x in result)

    # The class can be instantiated with a size of 0.
    def test_instantiation_with_size_zero(self):
        random = Generator(size=0)
        assert isinstance(random, Generator)

    # The weibull method returns an array of Weibull distributed random numbers.
    def test_weibull_method_returns_array(self):
        random = Generator(seed=42, size=10)
        result = random.weibull(2)
        assert isinstance(result, np.ndarray)
        assert len(result) == 10
        assert all(isinstance(x, float) for x in result)

    # The uniform method returns an array of random numbers when low and high are equal.
    def test_uniform_method_returns_array_when_low_and_high_are_equal(self):
        random = Generator(seed=42, size=10)
        result = random.uniform(5, 5)
        assert isinstance(result, np.ndarray)
        assert len(result) == 10
        assert np.all(result == 5)

    # The normal method returns an array of normally distributed random numbers when scale is 0.
    def test_normal_method_with_scale_zero(self):
        random = Generator(seed=42, size=10)
        result = random.normal(0, 0)
        assert isinstance(result, np.ndarray)
        assert len(result) == 10
        assert np.all(result == 0)

    # The lognormal method returns an array of log-normally distributed random numbers when sigma is 0.
    def test_lognormal_method_with_sigma_zero(self):
        random = Generator(seed=42, size=10)
        result = random.lognormal(0, 0)
        assert isinstance(result, np.ndarray)
        assert len(result) == 10
        assert np.all(result == 1.0)

    # The gamma method returns an array of gamma distributed random numbers when shape or scale is 0.
    def test_gamma_method_with_zero_shape_or_scale(self):
        random = Generator(seed=42)
        random.size = 10
        result = random.gamma(0, 2)
        assert isinstance(result, np.ndarray)
        assert len(result) == 10
        assert all(x >= 0 for x in result)

    # The class can be instantiated with both seed and size arguments.
    def test_instantiation_with_seed_and_size_arguments(self):
        random = Generator(seed=123, size=10)
        assert isinstance(random, Generator)
        assert random.seed == 123
        assert random.size == 10

    # The beta method returns an array of beta distributed random numbers when a or b is 1.
    def test_beta_method_with_zero_b(self):
        random = Generator(seed=42, size=10)
        result = random.beta(2, 1)
        assert isinstance(result, np.ndarray)
        assert len(result) == 10
        assert all(0 <= x <= 1 for x in result)

    # The weibull method returns an array of Weibull distributed random numbers when a is 0.
    def test_weibull_method_with_a_equal_to_zero(self):
        random = Generator(seed=42)
        random.size = 10
        result = random.weibull(0)
        assert isinstance(result, np.ndarray)
        assert len(result) == 10
        assert all(x >= 0 for x in result)
