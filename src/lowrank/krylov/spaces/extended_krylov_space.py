"""
Author: Benjamin Carrel, University of Geneva, 2022

This module contains the definition of the ExtendedKrylovSpace class.
"""

# %% Imports
from __future__ import annotations
import numpy as np
from numpy import ndarray
import scipy.linalg as la
import scipy.sparse.linalg as spsla
from scipy.sparse import spmatrix
from .space_structure import SpaceStructure
from .krylov_space import KrylovSpace
from .inverted_krylov_space import InvertedKrylovSpace



# %% Class definition
class ExtendedKrylovSpace(SpaceStructure):
    """Extended Krylov space.
    
    Constructs the extended Krylov space by combining a standard Krylov space
    and an inverted Krylov space:
        EK_m(A, X) = span{X, AX, A^2X, ..., A^(m-1)X, A^(-1)X, A^(-2)X, ..., A^(-m)X}
    
    This space is particularly useful for problems requiring information from both
    the range of A and its inverse.
    
    **Algorithm Selection:** When is_symmetric=True, both component Krylov spaces 
    (standard and inverted) use the Lanczos algorithm, providing approximately 2x 
    speedup over the non-symmetric case. The combined basis maintains orthogonality 
    across both components.
    
    How to use
    ----------
    1. Create an instance: EK = ExtendedKrylovSpace(A, X, invA=None)
    2. Augment the basis as needed: EK.augment_basis()
    3. Access the basis via EK.basis or EK.Q
    
    Attributes
    ----------
    A : spmatrix
        Sparse matrix of shape (n, n)
    X : ndarray
        Initial vector or matrix of shape (n, r)
    invA : callable
        Function that computes A^(-1) * v for a given vector/matrix v
    krylov_space : KrylovSpace
        The standard Krylov space component
    inverted_krylov_space : InvertedKrylovSpace
        The inverted Krylov space component
    Q : ndarray
        Combined orthonormal basis of the extended Krylov space
    Q1 : ndarray
        Basis of the standard Krylov space
    Q2 : ndarray
        Basis of the inverted Krylov space
    basis : ndarray
        Pointer to Q
    
    Examples
    --------
    >>> import numpy as np
    >>> from scipy.sparse import csr_matrix
    >>> A = csr_matrix([[4, 1], [1, 3]])
    >>> X = np.array([[1.0], [0.0]])
    >>> EK = ExtendedKrylovSpace(A, X)
    >>> EK.augment_basis()  # Adds both A^k*X and A^(-k-1)*X
    """

    def __init__(self, A: spmatrix, X: ndarray, invA: callable = None, **extra_args):
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
        # Validate inputs first (before accessing any attributes)
        if not isinstance(X, ndarray):
            raise TypeError("X must be a numpy array")
        
        # Ensure X is 2D before calling parent constructor
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        # Call parent constructor
        super().__init__(A, X, **extra_args)

        # Check specific inputs
        assert invA is None or callable(invA), "invA must be a function."
        if invA is None:
            spluA = spsla.splu(A)
            invA = spluA.solve

        # Store specific inputs
        self.invA = invA

        # Krylov space
        self.krylov_space = KrylovSpace(A, X, is_symmetric=self.is_symmetric)

        # Inverted Krylov space
        self.inverted_krylov_space = InvertedKrylovSpace(A, X, invA=invA, is_symmetric=self.is_symmetric)
        
        # Cache for the combined basis
        self._Q_cache = None
        self._cache_size = 0

    # %% Properties
    @property
    def Q1(self) -> ndarray:
        """Basis of the standard Krylov space component.
        
        Returns
        -------
        ndarray
            Matrix of shape (n, m*r) containing the Krylov basis vectors.
        """
        return self.krylov_space.Q
    
    @property
    def H1(self) -> ndarray:
        """Upper Hessenberg matrix from Arnoldi for the Krylov space.
        
        Returns
        -------
        ndarray
            The Hessenberg matrix (non-symmetric case only).
        """
        return self.krylov_space.H

    @property
    def Q2(self) -> ndarray:
        """Basis of the inverted Krylov space component.
        
        Returns
        -------
        ndarray
            Matrix of shape (n, m*r) containing the inverted Krylov basis vectors.
        """
        return self.inverted_krylov_space.Q
    
    @property
    def H2(self) -> ndarray:
        """Upper Hessenberg matrix from Arnoldi for the inverted Krylov space.
        
        Returns
        -------
        ndarray
            The Hessenberg matrix (non-symmetric case only).
        """
        return self.inverted_krylov_space.H
    
    @property
    def Q(self) -> ndarray:
        """Combined orthonormal basis of the extended Krylov space.
        
        Concatenates Q1 and Q2, then orthonormalizes the result. The result is
        cached and only recomputed when the space size changes.
        
        Returns
        -------
        ndarray
            Matrix of shape (n, size) containing the orthonormal basis.
        """
        # Check if cache is valid
        current_size = self.size
        if self._Q_cache is None or self._cache_size != current_size:
            # Recompute and cache
            self._Q_cache = la.qr(np.hstack((self.Q1, self.Q2)), mode="economic")[0]
            self._cache_size = current_size
        return self._Q_cache

    @property
    def basis(self) -> ndarray:
        """The orthonormal basis of the extended Krylov space.
        
        Returns
        -------
        ndarray
            Matrix of shape (n, size) (same as Q).
        """
        return self.Q

    @property
    def size(self) -> int:
        """Total size of the extended Krylov space.
        
        Returns
        -------
        int
            Sum of the sizes of the Krylov and inverted Krylov components.
        """
        return self.krylov_space.size + self.inverted_krylov_space.size

    # %% Methods
    def augment_basis(self):
        """Augment the basis of the extended Krylov space.
        
        Augments both the standard Krylov space and the inverted Krylov space
        by adding the next block of basis vectors to each component.
        """
        self.krylov_space.augment_basis()
        self.inverted_krylov_space.augment_basis()
        

    


        




