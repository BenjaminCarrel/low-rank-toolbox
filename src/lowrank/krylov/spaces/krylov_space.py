"""Standard Krylov subspace construction and operations.

Author: Benjamin Carrel, University of Geneva, 2022-2023
"""

# %% Imports
from __future__ import annotations

import warnings

import numpy as np
import numpy.linalg as la
import scipy.linalg as sla
from numpy import ndarray
from scipy.sparse import spmatrix

from .space_structure import SpaceStructure


# %% Class definition
class KrylovSpace(SpaceStructure):
    """
    Class for (block) Krylov spaces.
    The definition of a Krylov space of size $m$ is the following:
        K_m(A,x) = span{x, A x, A^2 x, ..., A^(m-1) x}
    where $A$ is a sparse matrix, and $x$ is a vector or a matrix.

    **Algorithm Selection:**
    - If A is symmetric (is_symmetric=True): Uses Lanczos algorithm with 3-term recurrence
      - Stores tridiagonal coefficients (alpha, beta) in O(m) space
      - Orthogonalization cost: O(n*r) per iteration
    - If A is non-symmetric (is_symmetric=False): Uses Arnoldi algorithm with full orthogonalization
      - Stores upper Hessenberg matrix H in O(m²) space
      - Orthogonalization cost: O(n*m*r) per iteration

    **Performance:** For symmetric problems, Lanczos is faster and uses significantly
    less memory than Arnoldi.

    How to use
    ----------
    1. Initialize the Krylov space with the matrix $A$ and the vector $x$.
    2. Augment the basis of the Krylov space as needed with the method `augment_basis`.
    3. The basis is stored in the attribute 'basis', or 'Q' for short.

    Attributes
    ----------
    A : spmatrix
        Matrix of shape (n,n)
    X : ndarray
        Vector of shape (n,1) or (n,r)
    m : int
        Size of the Krylov space
    is_symmetric : bool
        True if A is symmetric (uses Lanczos), False otherwise (uses Arnoldi)
    Q : ndarray
        Matrix of shape (n,m) or (n,m*r) containing the basis of the Krylov space
    basis : ndarray
        Pointer to Q
    _alpha : ndarray (symmetric only)
        Lanczos diagonal coefficients
    _beta : ndarray (symmetric only)
        Lanczos off-diagonal coefficients
    H : ndarray (non-symmetric only)
        Upper Hessenberg matrix from Arnoldi

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.sparse import csr_matrix
    >>> from lowrank.krylov import KrylovSpace
    >>> # Non-symmetric case (uses Arnoldi)
    >>> A = csr_matrix([[1, 2, 0], [0, 3, 1], [1, 0, 2]])
    >>> x = np.array([1.0, 0.0, 0.0])
    >>> K = KrylovSpace(A, x, is_symmetric=False)
    >>> K.augment_basis()  # Add next basis vector
    >>> K.Q.shape
    (3, 2)
    >>> # Symmetric case (uses Lanczos - more efficient)
    >>> A_sym = csr_matrix([[4, 1, 0], [1, 3, 1], [0, 1, 2]])
    >>> K_sym = KrylovSpace(A_sym, x, is_symmetric=True)
    >>> K_sym.augment_basis()
    >>> K_sym.Q.shape
    (3, 2)
    """

    # %% INITIALIZATION
    def __init__(self, A: spmatrix, X: ndarray, **extra_args) -> None:
        """
        Initialize a Krylov Space where X is a vector or a matrix

        Parameters
        ----------
        A : spmatrix
            Sparse matrix of shape (n,n)
        X : ndarray
            Vector or matrix of shape (n,) or (n,r)
        extra_args : dict
            Extra arguments

        Extra arguments
        ---------------
        is_symmetric : bool
            True if A is symmetric (uses Lanczos), False otherwise (uses Arnoldi).
            Lanczos is much cheaper for symmetric problems.
        matvec: callable
            Function for the matrix-vector product
        """
        # Call parent class
        super().__init__(A, X, **extra_args)

        # Check for a function to compute the matrix-vector product
        if "matvec" in extra_args:
            self.matvec = extra_args["matvec"]
        else:
            self.matvec = lambda x: A.dot(x)

        # Symmetric case -> Lanczos algorithm
        if self.is_symmetric:
            self._alpha = np.empty(self.n, dtype=object)
            self._beta = np.empty(self.n, dtype=object)
            self.Q, self._beta[0] = la.qr(X, mode="reduced")
        # Non symmetric case -> Arnoldi algorithm
        else:
            self.Q, self.H = la.qr(X, mode="reduced")
        self.Q = np.array(self.Q, dtype=self.dtype)

    # %% PROPERTIES
    @property
    def basis(self):
        """The orthonormal basis of the Krylov space.

        Returns
        -------
        ndarray
            Matrix of shape (n, m*r) containing the basis vectors.
        """
        return self.Q

    @property
    def size(self):
        """The size of the Krylov space.

        Returns
        -------
        int
            The number of basis vectors (m * r).
        """
        return self.m * self.r

    # %% AUGMENT BASIS
    def augment_basis(self) -> None:
        """Augment the basis of the Krylov space.

        Adds the next block of r basis vectors to the Krylov space using:
        - Lanczos algorithm if A is symmetric
        - Arnoldi algorithm if A is non-symmetric

        The new basis vectors are computed as A * Q[:, (m-1)*r:m*r] and then
        orthogonalized against the existing basis.

        Notes
        -----
        If the next basis would exceed the dimension of the matrix, a warning
        is issued and the method returns without modifying the basis.
        """
        # Check the next size does not exceed the dimension of the matrix
        if self.n < self.r * (self.m + 1):
            # warn user and do nothing
            warnings.warn("The next basis would exceed the dimension of the matrix.")
            return

        # Initialize
        r = self.r
        self.m += 1
        m = self.m
        AQ = self.matvec(self.Q[:, (m - 2) * r : (m - 1) * r])

        # Symmetric case (Lanczos)
        if self.is_symmetric:
            Q = np.zeros((self.n, m * r), dtype=self.Q.dtype)
            Q[:, : (m - 1) * r] = self.Q
            self._alpha[m - 1] = Q[:, (m - 2) * r : (m - 1) * r].T.dot(AQ)
            AQ -= Q[:, (m - 2) * r : (m - 1) * r].dot(self._alpha[m - 1])
            if m > 2:
                AQ -= Q[:, (m - 3) * r : (m - 2) * r].dot(self._beta[m - 2].T)
            Q[:, (m - 1) * r : m * r], self._beta[m - 1] = la.qr(AQ, mode="reduced")
            self.Q = Q

        # Non-symmetric case (scipy's qr_insert)
        else:
            self.Q, self.H = sla.qr_insert(self.Q, self.H, AQ, (m - 1) * r, which="col")
