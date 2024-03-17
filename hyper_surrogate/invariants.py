from typing import Any

import numpy as np


class TensorInvariants:
    """
    A class to compute invariants of a tensor.

    Attributes:
        tensor (np.ndarray): The input tensor.

    Methods:
        invariant1: Compute the first invariant of the tensor.
        invariant2: Compute the second invariant of the tensor.
    """

    def __init__(self, tensor: np.ndarray) -> None:
        """
        Initialize the TensorInvariants object.

        Args:
            tensor (np.ndarray): The input tensor.
        """
        self.tensor = tensor

    @property
    def invariant1(self) -> Any:
        """
        Compute the first invariant of the tensor.

        Returns:
            Any: The computed first invariant.
        """
        return self.tensor.trace()

    @property
    def invariant2(self) -> Any:
        """
        Compute the second invariant of the tensor.

        Returns:
            Any: The computed second invariant.
        """
        return (self.tensor.trace() ** 2 - (self.tensor**2).trace()) / 2

    # @property
    # def invariant3(self) -> Any:
    #     return self.tensor.det()
