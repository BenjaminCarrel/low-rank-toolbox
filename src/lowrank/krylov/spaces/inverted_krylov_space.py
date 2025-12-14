"""Inverted Krylov subspace construction.

Author: Benjamin Carrel, University of Geneva, 2022-2023
"""

# %% Imports
import numpy as np
from numpy import ndarray
import scipy.sparse.linalg as spsla
from scipy.sparse import spmatrix
from .krylov_space import KrylovSpace

# %% Class definition
class InvertedKrylovSpace(KrylovSpace):
    """Inverted Krylov space.
    
    Constructs the inverted Krylov space:
        IK_m(A, X) = span{A^(-1)X, A^(-2)X, ..., A^(-m)X}
    
    This class wraps the KrylovSpace class by providing a custom matvec function
    that computes A^(-1) * v instead of A * v.
    
    How to use
    ----------
    1. Initialize the inverted Krylov space with matrix A and vector/matrix X.
    2. Augment the basis as needed with the `augment_basis` method.
    3. Access the basis via the `basis` or `Q` attribute.
    
    Attributes
    ----------
    A : spmatrix
        Sparse matrix of shape (n, n)
    X : ndarray
        Initial vector or matrix of shape (n, r)
    invA : callable
        Function that computes A^(-1) * v for a given vector/matrix v
    Q : ndarray
        Orthonormal basis of the inverted Krylov space
    basis : ndarray
        Pointer to Q
    
    Examples
    --------
    >>> import numpy as np
    >>> from scipy.sparse import csr_matrix
    >>> from lowrank.krylov import InvertedKrylovSpace
    >>> # Create a sparse matrix
    >>> A = csr_matrix([[4, 1, 0], [1, 3, 1], [0, 1, 2]])
    >>> X = np.array([[1.0], [0.0], [0.0]])
    >>> # Build inverted Krylov space
    >>> IK = InvertedKrylovSpace(A, X)
    >>> # Start with A^(-1)*X
    >>> IK.Q.shape
    (3, 1)
    >>> # Augment with A^(-2)*X
    >>> IK.augment_basis()
    >>> IK.Q.shape
    (3, 2)
    >>> # Basis is orthonormal
    >>> np.allclose(IK.Q.T @ IK.Q, np.eye(2))
    True
    """

    def __init__(self, A: spmatrix, X: ndarray, invA: callable = None, **extra_args) -> None:
        """
        Parameters
        ----------
        A : spmatrix
            The matrix A of the linear system.
        X : ndarray
            The basis of the Krylov space.
        invA: callable
            The function that computes the action of A^(-1) on a vector, or a matrix.
        """
        # Validate inputs first (before using them)
        if not isinstance(A, spmatrix):
            raise TypeError("A must be a sparse matrix")
        if not isinstance(X, ndarray):
            raise TypeError("X must be a numpy array")
        if A.shape[0] != A.shape[1]:
            raise ValueError("A must be a square matrix")
        if A.shape[0] != X.shape[0]:
            raise ValueError("A and X must have the same number of rows")
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            raise ValueError("X contains NaN or Inf values")
        if np.any(np.isnan(A.data)) or np.any(np.isinf(A.data)):
            raise ValueError("A contains NaN or Inf values")
        
        # Check if invA is provided
        if invA is None:
            spluA = spsla.splu(A)
            invA = lambda x: spluA.solve(x).reshape(x.shape) # the reshape is needed for the case where x is a vector (because of the QR)

        # Define the matvec function that ensures output is 2D
        def matvec(v):
            result = invA(v)
            if result.ndim == 1:
                result = result.reshape(-1, 1)
            return result

        # Call the KrylovSpace class
        X = invA(X) # the inverted Krylov space starts from A^(-1)X
        # Ensure X is 2D
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        super().__init__(A, X, matvec=matvec, **extra_args)