"""
Author: Benjamin Carrel, University of Geneva, 2022

This module contains the RationalKrylovSpace class and methods
"""

# %% Imports
import numpy as np
from numpy import ndarray
import scipy.linalg as la
import scipy.sparse as sps
import scipy.sparse.linalg as spsla
from scipy.sparse import spmatrix
from .space_structure import SpaceStructure


class RationalKrylovSpace(SpaceStructure):
    """Rational Krylov space.
    
    Constructs the rational Krylov space:
        RK_m(A, X) = span{X, (A - p_1*I)^{-1}X, ..., prod_{i=1}^{m-1}(A - p_i*I)^{-1}X}
    
    where p_1, ..., p_{m-1} are specified poles (shifts). This space generalizes the
    standard Krylov space by using rational functions of A instead of polynomials.
    
    How to use
    ----------
    1. Initialize with matrix A, vector/matrix X, and list of poles.
    2. Augment the basis as needed with `augment_basis`.
    3. Access the basis via `basis` or `Q`.
    
    Attributes
    ----------
    A : spmatrix
        Sparse matrix of shape (n, n)
    X : ndarray
        Initial vector or matrix of shape (n, r)
    poles : ndarray
        Array of poles (shifts) for the rational Krylov space
    max_iter : int
        Maximum number of iterations (length of poles list)
    m : int
        Current size of the rational Krylov space
    Q : ndarray
        Orthonormal basis of shape (n, m*r)
    H : ndarray
        Upper triangular matrix from QR factorization
    basis : ndarray
        Pointer to Q
    
    Examples
    --------
    >>> import numpy as np
    >>> from scipy.sparse import csr_matrix
    >>> A = csr_matrix([[4, 1], [1, 3]])
    >>> X = np.array([[1.0], [0.0]])
    >>> poles = [1.0, 2.0, 3.0]
    >>> RK = RationalKrylovSpace(A, X, poles)
    >>> RK.augment_basis()  # Adds next basis vector using first pole
    """

    #%% INITIALIZATION
    def __init__(self, A: spmatrix, X: ndarray, poles: list, inverse_only: bool = False, **extra_args) -> None:
        """
        Initialize a rational Krylov space

        Parameters
        ----------
        A : spmatrix
            Sparse matrix of shape (n, n)
        X : ndarray
            Vector or matrix of shape (n, r)
        poles : list
            Poles (shifts) for the rational Krylov space
        inverse_only : bool, optional
            If True, solves (A - p*I) * v_new = v_old.
            If False, solves (A - p*I) * v_new = A * v_old.
            Default is False. The False case is useful for approximating matrix
            functions, while True is for solving linear systems.
        extra_args : dict
            Extra arguments

        Extra arguments
        ---------------
        symmetric : bool
            True if A is symmetric, False otherwise (not used for rational Krylov)
        """
        # Call parent class
        super().__init__(A, X, **extra_args)
        # Check and store specific parameters
        if not poles:
            raise ValueError("poles list cannot be empty")
        self.poles = np.array(poles)
        # Validate poles
        if np.any(np.isnan(self.poles)) or np.any(np.isinf(self.poles)):
            raise ValueError("poles contain NaN or Inf values")
        # Override max_iter with the number of poles
        self.max_iter = len(poles)
        # dtype depends on the dtype of the matrix A, X and poles
        for pole in poles:
            if np.iscomplex(pole):
                self.dtype = np.promote_types(self.dtype, np.complex128)
        # For rational Krylov, the symmetric flag is not used
        Q, H = la.qr(X, mode="economic")
        self.Q, self.H = np.array(Q, dtype=self.dtype), np.array(H, dtype=self.dtype)
        if inverse_only:
            self.small_matvec = lambda x: x
        else:
            self.small_matvec = lambda x: A.dot(x)
       

    #%% PROPERTIES
    @property
    def basis(self) -> ndarray:
        """The orthonormal basis of the rational Krylov space.
        
        Returns
        -------
        ndarray
            Matrix of shape (n, m*r) containing the basis vectors.
        """
        return self.Q

    @property
    def size(self) -> int:
        """The size of the rational Krylov space.
        
        Returns
        -------
        int
            The number of basis vectors (m * r).
        """
        return self.m * self.r

    #%% BASIS AUGMENTATION
    def augment_basis(self):
        """Augment the basis of the rational Krylov space.
        
        Adds the next block of r basis vectors by solving:
            (A - p_{m-1}*I) * v_new = v_old
        
        where p_{m-1} is the next pole in the sequence. The new vectors are then
        orthogonalized against the existing basis using QR factorization.
        
        Raises
        ------
        ValueError
            If the space would exceed the problem dimension or if there are not
            enough poles specified.
        """
        # Check if the basis is already full
        if self.m * self.r >= self.n:
            raise ValueError("The space is exceeding the dimension of the problem")
        # Check the poles
        if self.m-1 >= len(self.poles):
            raise ValueError("Not enough poles specified for the requested space size")
        
         # Initialize
        A = self.A
        r = self.r
        self.m += 1
        m = self.m
        Q = np.zeros((self.n, m * r), dtype=self.dtype)
        Q[:, : (m - 1) * r] = self.Q

        # Solve the next linear system
        matvec = lambda x: spsla.spsolve((self.A - self.poles[self.m-2] * sps.eye(self.n, format='csc')), self.small_matvec(x)).reshape(x.shape)
        Wm = matvec(Q[:, (m - 2) * r : (m - 1) * r])

        # Update-orthogonalization
        self.Q, self.H = la.qr_insert(self.Q, self.H, Wm, (m-1)*r, 'col')
        
