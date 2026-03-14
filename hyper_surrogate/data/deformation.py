from __future__ import annotations

import numpy as np


class DeformationGenerator:
    """Generates physically valid deformation gradients for training data."""

    def __init__(self, seed: int = 42) -> None:
        self._rng = np.random.default_rng(seed)

    def uniaxial(self, n: int, stretch_range: tuple[float, float] = (0.4, 3.0)) -> np.ndarray:
        stretch = self._rng.uniform(*stretch_range, size=n)
        stretch_t = stretch**-0.5
        result = np.zeros((n, 3, 3))
        result[:, 0, 0] = stretch
        result[:, 1, 1] = stretch_t
        result[:, 2, 2] = stretch_t
        return result

    def biaxial(self, n: int, stretch_range: tuple[float, float] = (0.4, 3.0)) -> np.ndarray:
        s1 = self._rng.uniform(*stretch_range, size=n)
        s2 = self._rng.uniform(*stretch_range, size=n)
        s3 = (s1 * s2) ** -1.0
        result = np.zeros((n, 3, 3))
        result[:, 0, 0] = s1
        result[:, 1, 1] = s2
        result[:, 2, 2] = s3
        return result

    def shear(self, n: int, shear_range: tuple[float, float] = (-1.0, 1.0)) -> np.ndarray:
        gamma = self._rng.uniform(*shear_range, size=n)
        result = np.repeat(np.eye(3)[np.newaxis, :, :], n, axis=0)
        result[:, 0, 1] = gamma
        return result

    def random_rotation(self, n: int) -> np.ndarray:
        axes = self._rng.integers(0, 3, size=n)
        angles = self._rng.uniform(0, np.pi, size=n)
        rotations = []
        for ax, ang in zip(axes, angles, strict=False):
            c, s = np.cos(ang), np.sin(ang)
            if ax == 0:
                R = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
            elif ax == 1:
                R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
            else:
                R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
            rotations.append(R)
        return np.array(rotations)

    @staticmethod
    def _rotate(F: np.ndarray, R: np.ndarray) -> np.ndarray:
        return np.einsum("nij,njk,nlk->nil", R, F, R)  # type: ignore[no-any-return]

    def fiber_directions(
        self,
        n: int,
        preferred: np.ndarray | None = None,
        dispersion: float = 0.0,
    ) -> np.ndarray:
        """Generate fiber direction vectors.

        Args:
            n: Number of samples.
            preferred: Preferred fiber direction (3,). Default: [1, 0, 0].
            dispersion: Cone half-angle in radians. 0 = all aligned.

        Returns:
            Fiber directions (N, 3), unit vectors.
        """
        if preferred is None:
            preferred = np.array([1.0, 0.0, 0.0])
        preferred = preferred / np.linalg.norm(preferred)

        if dispersion <= 0.0:
            return np.tile(preferred, (n, 1))

        # Sample directions in a cone around preferred
        # Build local frame: preferred = e3, find orthogonal e1, e2
        if abs(preferred[2]) < 0.9:
            t = np.cross(preferred, np.array([0.0, 0.0, 1.0]))
        else:
            t = np.cross(preferred, np.array([1.0, 0.0, 0.0]))
        e1 = t / np.linalg.norm(t)
        e2 = np.cross(preferred, e1)

        phi = self._rng.uniform(0, 2 * np.pi, size=n)
        theta = self._rng.uniform(0, dispersion, size=n)

        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)

        dirs = x[:, None] * e1 + y[:, None] * e2 + z[:, None] * preferred
        norms = np.linalg.norm(dirs, axis=1, keepdims=True)
        result: np.ndarray = dirs / norms
        return result

    def combined(
        self,
        n: int,
        stretch_range: tuple[float, float] = (0.4, 3.0),
        shear_range: tuple[float, float] = (-1.0, 1.0),
    ) -> np.ndarray:
        fu = self.uniaxial(n, stretch_range)
        fs = self.shear(n, shear_range)
        fb = self.biaxial(n, stretch_range)
        r1, r2, r3 = self.random_rotation(n), self.random_rotation(n), self.random_rotation(n)
        fu = self._rotate(fu, r1)
        fs = self._rotate(fs, r2)
        fb = self._rotate(fb, r3)
        return np.matmul(np.matmul(fb, fu), fs)  # type: ignore[no-any-return]
