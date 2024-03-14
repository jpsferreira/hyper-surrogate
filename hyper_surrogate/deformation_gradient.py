import numpy as np

from hyper_surrogate.generator import Generator


class DeformationGradient:
    def __init__(self):
        pass

    @staticmethod
    def uniaxial(stretch: np.ndarray) -> np.ndarray:
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
        shear = np.atleast_1d(shear)
        # Initialize the resulting 3D array with the identity matrix replicated for each shear value
        result = np.repeat(np.eye(3)[np.newaxis, :, :], shear.size, axis=0)

        # Set the shear values in the appropriate position for each 2D sub-array
        result[:, 0, 1] = shear

        return result

    @staticmethod
    def biaxial(stretch1: float, stretch2: float) -> np.ndarray:
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

    def rotation(self, axis: int, angle: np.ndarray) -> np.ndarray:
        axis, angle = np.atleast_1d(axis), np.atleast_1d(angle)
        rotations = []
        for ax, ang in zip(axis, angle):
            rotations.append(self._axis_rotation(ax, ang))
            return np.array(rotations)

    def rescale(self, F: np.ndarray) -> np.ndarray:
        return F / self.invariant3(F) ** (1.0 / 3.0)

    @staticmethod
    def invariant1(F: np.ndarray) -> float:
        return np.trace(F)

    @staticmethod
    def invariant2(F: np.ndarray) -> float:
        return 0.5 * (np.trace(F) ** 2 - np.trace(np.matmul(F, F)))

    @staticmethod
    def invariant3(F: np.ndarray) -> float:
        return np.linalg.det(F)

    @staticmethod
    def to_radians(degree: float) -> float:
        return degree * np.pi / 180


class DeformationGradientGenerator(DeformationGradient):
    def __init__(self, seed=None, size=None, generator=Generator):
        self.seed = seed
        self.size = size
        self.generator = generator(seed=seed, size=size)

    def axis(self, n_axis: int = 3) -> int:
        return self.generator.int(low=0, high=n_axis)

    def angle(self, min_interval: float = 5) -> float:
        min_interval = self.to_radians(min_interval)
        return self.generator.in_interval(a=0, b=np.pi, interval=min_interval)

    def rotate(self, n_axis: int = 3, min_interval: float = 5) -> np.ndarray:
        axis = self.axis(n_axis=n_axis)
        angle = self.angle(min_interval=min_interval)
        return self.rotation(axis, angle)
