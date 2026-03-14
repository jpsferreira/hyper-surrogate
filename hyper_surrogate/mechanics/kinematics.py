from typing import Any

import numpy as np


class Kinematics:
    """
    A class that provides various kinematic methods.

    Attributes:
        None: This class does not have any attributes.

    Methods:
        jacobian: Compute the Jacobian of the deformation gradient.
        trace_invariant: Calculate the first invariant (trace) of each tensor in the batch.
        quadratic_invariant: Calculate the second invariant of the tensor.
        det_invariant: Calculate the third invariant (determinant) of the tensor.
        isochoric_invariant1: Calculate the isochoric first invariant.
        isochoric_invariant2: Calculate the isochoric second invariant.
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
            f: 4D tensor of shape (N, 3, 3).

        Returns:
            np.ndarray: The Jacobian of the deformation gradient.
        """
        return np.linalg.det(f)

    @staticmethod
    def trace_invariant(c: np.ndarray) -> Any:
        """
        Calculate the first invariant (trace) of each tensor in the batch.

        Args:
            c: Tensor of shape (N, 3, 3).

        Returns:
            The first invariant of each tensor in the batch.
        """
        return np.einsum("nii->n", c)

    @staticmethod
    def quadratic_invariant(c: np.ndarray) -> Any:
        """
        Calculate the second invariant of the tensor.

        Args:
            c: Tensor of shape (N, 3, 3).

        Returns:
            The second invariant.
        """
        return 0.5 * (np.einsum("nii->n", c) ** 2 - np.einsum("nij,nji->n", c, c))

    @staticmethod
    def det_invariant(c: np.ndarray) -> Any:
        """
        Calculate the third invariant (determinant) of the tensor.

        Args:
            c: Tensor of shape (N, 3, 3).

        Returns:
            The third invariant.
        """
        return np.linalg.det(c)

    @staticmethod
    def isochoric_invariant1(c: np.ndarray) -> np.ndarray:
        """Isochoric first invariant: tr(C) * det(C)^(-1/3)."""
        return np.einsum("nii->n", c) * np.linalg.det(c) ** (-1.0 / 3.0)  # type: ignore[no-any-return]

    @staticmethod
    def isochoric_invariant2(c: np.ndarray) -> np.ndarray:
        """Isochoric second invariant: 0.5*(I1^2 - tr(C^2)) * det(C)^(-2/3)."""
        i1 = np.einsum("nii->n", c)
        i1_sq = i1**2
        tr_c2 = np.einsum("nij,nji->n", c, c)
        return 0.5 * (i1_sq - tr_c2) * np.linalg.det(c) ** (-2.0 / 3.0)  # type: ignore[no-any-return]

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

    def right_stretch_tensor(self, f: np.ndarray) -> Any:
        """
        Compute the right stretch tensor.

        Args:
            f (np.ndarray): The deformation gradient.

        Returns:
            np.ndarray: The right stretch tensor.
        """
        v, vv = np.linalg.eig(self.right_cauchy_green(f))
        return np.einsum("...ij,...j->...ij", vv, v)

    def left_stretch_tensor(self, f: np.ndarray) -> Any:
        """
        Compute the left stretch tensor.

        Args:
            f (np.ndarray): The deformation gradient.

        Returns:
            np.ndarray: The left stretch tensor.
        """
        v, vv = np.linalg.eig(self.left_cauchy_green(f))
        return np.einsum("...ij,...j->...ij", vv, v)

    @staticmethod
    def fiber_invariant4(c: np.ndarray, a0: np.ndarray) -> np.ndarray:
        """Fiber invariant I4 = a0 . C . a0.

        Args:
            c: Right Cauchy-Green tensor (N, 3, 3).
            a0: Fiber direction (3,) or (N, 3).

        Returns:
            I4 values (N,).
        """
        a0 = np.asarray(a0)
        if a0.ndim == 1:
            return np.einsum("i,nij,j->n", a0, c, a0)  # type: ignore[no-any-return]
        return np.einsum("ni,nij,nj->n", a0, c, a0)  # type: ignore[no-any-return]

    @staticmethod
    def fiber_invariant5(c: np.ndarray, a0: np.ndarray) -> np.ndarray:
        """Fiber invariant I5 = a0 . C^2 . a0.

        Args:
            c: Right Cauchy-Green tensor (N, 3, 3).
            a0: Fiber direction (3,) or (N, 3).

        Returns:
            I5 values (N,).
        """
        a0 = np.asarray(a0)
        c2 = np.einsum("nij,njk->nik", c, c)
        if a0.ndim == 1:
            return np.einsum("i,nij,j->n", a0, c2, a0)  # type: ignore[no-any-return]
        return np.einsum("ni,nij,nj->n", a0, c2, a0)  # type: ignore[no-any-return]

    def rotation_tensor(self, f: np.ndarray) -> Any:
        """
        Compute the rotation tensors.

        Args:
            f (np.ndarray): The deformation gradients. batched with shape (N, 3, 3).

        Returns:
            np.ndarray: The rotation tensors. batched with shape (N, 3, 3).
        """
        return np.einsum("nij,njk->nik", f, np.linalg.inv(self.right_stretch_tensor(f)))
