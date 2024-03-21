from typing import Any

import numpy as np

from hyper_surrogate.generator import Generator


class DeformationGradient:
    """A class representing deformation gradient operations."""

    def __init__(self) -> None:
        pass

    @staticmethod
    def uniaxial(stretch: np.ndarray) -> np.ndarray:
        """
        Calculate the deformation gradient tensor for uniaxial deformation.

        Args:
            stretch: A 1D array representing the stretch factor.
        Returns:
            The deformation gradient tensor as a 3D array.
        """
        stretch = np.atleast_1d(stretch)
        # Calculate the transverse stretch factor for the entire array
        stretch_t = stretch**-0.5
        # Initialize the resulting 3D array with zeros
        result = np.zeros((stretch.size, 3, 3))
        # Fill in the diagonal values for each 2D sub-array
        result[:, 0, 0] = stretch  # Set the first diagonal elements to stretch
        result[:, 1, 1] = stretch_t  # Set the second diagonal elements to stretch_t
        result[:, 2, 2] = stretch_t  # Set the third diagonal elements to stretch_t

        return result

    @staticmethod
    def shear(shear: np.ndarray) -> np.ndarray:
        """
        Calculate the deformation gradient tensor for shear deformation.

        Args:
            shear: A 1D array representing the shear factor.

        Returns:
            The deformation gradient tensor as a 3D array.
        """
        shear = np.atleast_1d(shear)
        # Initialize the resulting 3D array with the identity matrix replicated for each shear value
        result = np.repeat(np.eye(3)[np.newaxis, :, :], shear.size, axis=0)

        # Set the shear values in the appropriate position for each 2D sub-array
        result[:, 0, 1] = shear

        return result

    @staticmethod
    def biaxial(stretch1: np.ndarray, stretch2: np.ndarray) -> np.ndarray:
        """
        Calculate the deformation gradient tensor for biaxial deformation.
        latex equation:

        Args:
            stretch1: A 1D array representing the first stretch factor.
            stretch2: A 1D array representing the second stretch factor.

        Returns:
            The deformation gradient tensor as a 3D array.
        """
        # Calculate the third stretch factor for the entire arrays
        stretch1 = np.atleast_1d(stretch1)
        stretch2 = np.atleast_1d(stretch2)
        stretch3 = (stretch1 * stretch2) ** -1.0

        # Initialize the resulting 3D array with zeros
        result = np.zeros((stretch1.size, 3, 3))

        # Fill in the diagonal values for each 2D sub-array
        result[:, 0, 0] = stretch1  # Set the first diagonal elements to stretch1
        result[:, 1, 1] = stretch2  # Set the second diagonal elements to stretch2
        result[:, 2, 2] = stretch3  # Set the third diagonal elements to stretch3

        return result

    @staticmethod
    def _axis_rotation(axis: int, angle: float) -> np.ndarray:
        """
        Calculate the rotation matrix for a given axis and angle.

        Args:
            axis: An integer representing the axis of rotation (0 for x-axis, 1 for y-axis, 2 for z-axis).
            angle: A float representing the angle of rotation in radians.

        Returns:
            The rotation matrix as a 2D array.
        """
        c, s = np.cos(angle), np.sin(angle)
        dict_axis = {
            0: np.array(
                [
                    [1, 0, 0],
                    [0, c, -s],
                    [0, s, c],
                ]
            ),
            1: np.array(
                [
                    [c, 0, s],
                    [0, 1, 0],
                    [-s, 0, c],
                ]
            ),
            2: np.array(
                [
                    [c, -s, 0],
                    [s, c, 0],
                    [0, 0, 1],
                ]
            ),
        }
        return dict_axis[axis] if axis in dict_axis else np.eye(3)

    def rotation(self, axis: np.ndarray, angle: np.ndarray) -> np.ndarray:
        """
        Calculate the rotation matrix for multiple axes and angles.

        Args:
            axis: A 1D array representing the axes of rotation (0 for x-axis, 1 for y-axis, 2 for z-axis).
            angle: A 1D array representing the angles of rotation in radians.

        Returns:
            The rotation matrix as a 3D array.
        """
        axis, angle = np.atleast_1d(axis), np.atleast_1d(angle)
        rotation = []
        for ax, ang in zip(axis, angle):
            rotation.append(self._axis_rotation(ax, ang))
        return np.array(rotation)

    def rescale(self, F: np.ndarray) -> Any:
        """
        Rescale the deformation gradient tensor.

        Args:
            F: The deformation gradient tensor as a 3D array.

        Returns:
            The rescaled deformation gradient tensor.
        """
        return F / self.invariant3(F) ** (1.0 / 3.0)

    @staticmethod
    def invariant1(F: np.ndarray) -> Any:
        """
        Calculate the first invariant of the deformation gradient tensor.

        Args:
            F: The deformation gradient tensor as a 3D array.

        Returns:
            The first invariant.
        """
        return np.trace(F)

    @staticmethod
    def invariant2(F: np.ndarray) -> Any:
        """
        Calculate the second invariant of the deformation gradient tensor.

        Args:
            F: The deformation gradient tensor as a 3D array.

        Returns:
            The second invariant.
        """
        return 0.5 * (np.trace(F) ** 2 - np.trace(np.matmul(F, F)))

    @staticmethod
    def invariant3(F: np.ndarray) -> Any:
        """
        Calculate the third invariant of the deformation gradient tensor.

        Args:
            F: The deformation gradient tensor as a 3D array.

        Returns:
            The third invariant.
        """
        return np.linalg.det(F)

    @staticmethod
    def to_radians(degree: float) -> float:
        """
        Convert degrees to radians.

        Args:
            degree: The angle in degrees.

        Returns:
            The angle in radians.
        """
        return degree * np.pi / 180

    @staticmethod
    def rotate(F: np.ndarray, R: np.ndarray) -> Any:
        """
        Rotate the deformation gradient tensor.

        Args:
            F: The deformation gradient tensor as a 3D array.
            R: The rotation matrix as a 3D array.

        Returns:
            The rotated deformation gradient tensor.
        """
        F = np.atleast_3d(F)
        R = np.atleast_3d(R)
        return np.einsum("nij,njk,nlk->nil", R, F, R)


class DeformationGradientGenerator(DeformationGradient):
    """
    Generates deformation gradients for hyper-surrogate modeling.

    Args:
        seed (int | None): Seed value for the random number generator. Default is None.
        size (int | None): Size of the generator. Default is None.
        generator (Generator | None): Random number generator. Default is None.

    Attributes:
        seed (int | None): Seed value for the random number generator.
        size (int | None): Size of the generator.
        generator (Generator): Random number generator.

    Methods:
        axis(n_axis: int = 3) -> Any:
            Generates a random axis.

        angle(min_interval: float = 5) -> Any:
            Generates a random angle.

        generate_rotation(n_axis: int = 3, min_interval: float = 5) -> np.ndarray:
            Generates a random rotation matrix.

        generate(stretch_min: float = 0.4, stretch_max: float = 3.0, shear_min: float = -1, shear_max: float = 1) -> Any:
            Generates a deformation gradient.

    """

    def __init__(
        self,
        seed: int | None = None,
        size: int | None = None,
        generator: Generator | None = None,
    ) -> None:
        self.seed = seed
        self.size = size
        self.generator = generator if generator else Generator(seed=seed, size=size)

    def axis(self, n_axis: int = 3) -> Any:
        """
        Generates a random axis.

        Args:
            n_axis (int): Number of axes to choose from. Default is 3.

        Returns:
            Any: Randomly generated axis.

        """
        return self.generator.integer_in_interval(low=0, high=n_axis)

    def angle(self, min_interval: float = 5) -> Any:
        """
        Generates a random angle.

        Args:
            min_interval (float): Minimum interval for the angle. Default is 5.

        Returns:
            Any: Randomly generated angle.

        """
        min_interval = self.to_radians(min_interval)
        return self.generator.float_in_interval(a=0, b=np.pi, interval=min_interval)

    def generate_rotation(self, n_axis: int = 3, min_interval: float = 5) -> np.ndarray:
        """
        Generates a random rotation matrix.

        Args:
            n_axis (int): Number of axes to choose from. Default is 3.
            min_interval (float): Minimum interval for the angle. Default is 5.

        Returns:
            np.ndarray: Randomly generated rotation matrix.

        """
        axis = self.axis(n_axis=n_axis)
        angle = self.angle(min_interval=min_interval)
        return self.rotation(axis, angle)

    def generate(
        self,
        stretch_min: float = 0.4,
        stretch_max: float = 3.0,
        shear_min: float = -1,
        shear_max: float = 1,
    ) -> Any:
        """
        Generates a random deformation gradient.

        Args:
            stretch_min (float): Minimum value for stretch. Default is 0.4.
            stretch_max (float): Maximum value for stretch. Default is 3.0.
            shear_min (float): Minimum value for shear. Default is -1.
            shear_max (float): Maximum value for shear. Default is 1.

        Returns:
            Any: Generated random deformation gradient.

        """
        u, s, b1, b2 = (
            self.generator.uniform(stretch_min, stretch_max),
            self.generator.uniform(shear_min, shear_max),
            self.generator.uniform(stretch_min, stretch_max),
            self.generator.uniform(stretch_min, stretch_max),
        )
        fu, fs, fb = (
            self.uniaxial(u),
            self.shear(s),
            self.biaxial(b1, b2),
        )
        r1, r2, r3 = (
            self.generate_rotation(),
            self.generate_rotation(),
            self.generate_rotation(),
        )

        # rotate deformation gradients
        fu = self.rotate(fu, r1)
        fs = self.rotate(fs, r2)
        fb = self.rotate(fb, r3)

        # Compute deformation gradient
        return np.matmul(np.matmul(fb, fu), fs)
