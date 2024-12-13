from typing import Any

import numpy as np


class Kinematics:
    """
    A class that provides various kinematic methods.

    Attributes:
        None: This class does not have any attributes.

    Methods:
        jacobian: Compute the Jacobian of the deformation gradient.
        invariant1: Calculate the first invariant of each tensor in the batch.
        invariant2: Calculate the second invariant of the deformation gradient tensor.
        invariant3: Calculate the third invariant of the deformation gradient tensor.
        right_cauchy_green: Compute the right Cauchy-Green deformation tensor for a batch of deformation gradients.
        left_cauchy_green: Compute the left Cauchy-Green deformation tensor for a batch of deformation gradients.
        rotation_tensor: Compute the rotation tensors.
        pushforward: Forward tensor configuration.
        principal_stretches: Compute the principal stretches.
        principal_directions: Compute the principal directions.

    """

    @staticmethod
    def jacobian(f: np.ndarray) -> Any:
        """
        Compute the Jacobian of the deformation gradient.

        Args:
            f: 4D tensor of shape (N, 3, 3, 3).

        Returns:
            np.ndarray: The Jacobian of the deformation gradient.
        """
        return np.linalg.det(f)

    @staticmethod
    def invariant1(f: np.ndarray) -> Any:
        """
        Calculate the first invariant of each tensor in the batch.

        Args:
            f: 4D tensor of shape (N, 3, 3, 3).

        Returns:
            The first invariant of each tensor in the batch.
        """
        # einsum
        return np.einsum("nii->n", f)

    @staticmethod
    def invariant2(f: np.ndarray) -> Any:
        """
        Calculate the second invariant of the deformation gradient tensor.

        Args:
            f: 4D tensor of shape (N, 3, 3, 3).

        Returns:
            The second invariant.
        """
        # use einsum to calculate the second invariant: 0.5 * (np.trace(F) ** 2 - np.trace(np.matmul(F, F)))
        return 0.5 * (np.einsum("nii->n", f) ** 2 - np.einsum("nij,nji->n", f, f))

    @staticmethod
    def invariant3(f: np.ndarray) -> Any:
        """
        Calculate the third invariant of the deformation gradient tensor.

        Args:
            f: The deformation gradient tensor as a 3D array.

        Returns:
            The third invariant.
        """
        return np.linalg.det(f)

    @staticmethod
    def right_cauchy_green(f: np.ndarray) -> Any:
        """
        Compute the right Cauchy-Green deformation tensor for a batch of deformation gradients
        using a more efficient vectorized approach.
        $$C = F^T F$$

        Args:
            f (np.ndarray): The deformation gradient tensor with shape (N, 3, 3),
                            where N is the number of deformation gradients.

        Returns:
            np.ndarray: The batch of right Cauchy-Green deformation tensors, shape (N, 3, 3).
        """
        # Use np.einsum to perform batch matrix multiplication: f's transpose @ f
        # The einsum subscript 'nij,nkj->nik' denotes batched matrix multiplication
        # where 'n' iterates over each matrix in the batch,
        # 'ji' are the indices of the transposed matrix,
        # and 'jk' are the indices for the second matrix.
        # Note: The difference from the left Cauchy-Green tensor is in the order of multiplication.
        return np.einsum("nji,njk->nik", f, f)

    @staticmethod
    def left_cauchy_green(f: np.ndarray) -> Any:
        """
        Compute the left Cauchy-Green deformation tensor for a batch of deformation gradients
        using a more efficient vectorized approach.

        Args:
            f (np.ndarray): The deformation gradient tensor with shape (N, 3, 3),
                            where N is the number of deformation gradients.

        Returns:
            np.ndarray: The batch of left Cauchy-Green deformation tensors, shape (N, 3, 3).
        """
        # Use np.einsum to perform batch matrix multiplication: f @ f's transpose
        # The einsum subscript 'nij,njk->nik' denotes batched matrix multiplication
        # where 'n' iterates over each matrix in the batch,
        # 'ij' are the indices of the first matrix,
        # and 'jk' are the indices for the second matrix (transposed to 'kj' for multiplication).
        return np.einsum("nij,nkj->nik", f, f)

    @staticmethod
    def rotation_tensor(f: np.ndarray) -> Any:
        """
        Compute the rotation tensors.

        Args:
            f (np.ndarray): The deformation gradients. batched with shape (N, 3, 3).

        Returns:
            np.ndarray: The rotation tensors. batched with shape (N, 3, 3).
        """
        return np.einsum("nij,njk->nik", f, np.linalg.inv(f))

    @staticmethod
    def pushforward(f: np.ndarray, tensor2D: np.ndarray) -> Any:
        """
        Forward tensor configuration.
        F*tensor2D*F^T. This is the forward transformation of a 2D tensor.

        Args:
            f (np.ndarray): deformation gradient # (N, 3, 3)
            tensor2D (np.ndarray): The 2D tensor to be mapped # (N, 3, 3)

        Returns:
            np.ndarray: The transformed tensor.
        """
        return np.einsum("nik,njl,nkl->nij", f, f, tensor2D)

    def principal_stretches(self, f: np.ndarray) -> np.ndarray:
        """
        Compute the principal stretches.

        Args:
            f (np.ndarray): The deformation gradient.

        Returns:
            np.ndarray: The principal stretches.
        """
        return np.sqrt(np.linalg.eigvals(self.right_cauchy_green(f)))

    def principal_directions(self, f: np.ndarray) -> np.ndarray:
        """
        Compute the principal directions.

        Args:
            f (np.ndarray): The deformation gradient.

        Returns:
            np.ndarray: The principal directions.
        """
        return np.linalg.eig(self.right_cauchy_green(f))[1]
