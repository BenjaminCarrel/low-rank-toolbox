"""Krylov-based solvers for the Lyapunov equation.

Author: Benjamin Carrel, University of Geneva, 2022-2023
"""

#%% Imports
import numpy as np
from numpy import ndarray
import scipy.linalg as la
from scipy.sparse import spmatrix, issparse
import scipy.sparse as sps
import scipy.sparse.linalg as spsla
from lowrank import LowRankMatrix, QuasiSVD
from lowrank.matrices.low_rank_matrix import LowRankEfficiencyWarning
from lowrank.krylov import KrylovSpace, ExtendedKrylovSpace, RationalKrylovSpace

# Warnings
from warnings import warn


machine_precision = np.finfo(float).eps
Matrix = ndarray | spmatrix | LowRankMatrix

#%% FUNCTIONS

def solve_small_lyapunov(A: ndarray | spmatrix, C: ndarray) -> ndarray:
    "Solve the Lyapunov equation AX + XA = C for small matrices."
    # Convert sparse A to dense for scipy.linalg.solve_lyapunov
    A_dense = A.toarray() if issparse(A) else A
    return la.solve_lyapunov(A_dense, C)

def solve_sparse_low_rank_symmetric_lyapunov(A: spmatrix,
                                             C: LowRankMatrix,
                                             tol: float = 1e-12,
                                             max_iter: int = None,
                                             krylov_kwargs: dict = None,
                                             ) -> QuasiSVD:
    """
    Low-rank solver for the symmetric Lyapunov equation:
    
    Find X such that A X + X A = C.
    
    NOTE: Matrices A and C must be symmetric, which is exploited to halve the computational cost.
    Uses a single Krylov space since the solution X is also symmetric.
    
    **Performance:** Uses the Lanczos algorithm (3-term recurrence) instead of Arnoldi 
    (full orthogonalization), reducing computational cost by approximately 2x and storage 
    by O(m²) → O(m) for the recurrence coefficients.
    
    Parameters
    ----------
    A : spmatrix
        Symmetric sparse matrix of shape (n, n)
    C : LowRankMatrix
        Symmetric low-rank matrix of shape (n, n)
    tol : float, optional
        Convergence tolerance, by default 1e-12
    max_iter : int, optional
        Maximum iterations, by default n / rank(C)
    krylov_kwargs : dict, optional
        Krylov space configuration with keys:
        - 'extended' (bool): Use extended Krylov space (default True)
          Extended Krylov uses both A and A^(-1), providing faster convergence.
          Uses Lanczos algorithm for symmetric A (efficient 3-term recurrence).
        - 'invA' (callable): Custom inverse function for A
        - 'poles' (array_like): Poles for rational Krylov space
          WARNING: Rational Krylov currently uses Arnoldi even for symmetric A
          (Lanczos not yet implemented for rational case).
        
    Returns
    -------
    QuasiSVD
        Symmetric low-rank solution X
    
    Examples
    --------
    >>> import numpy as np
    >>> from scipy.sparse import csr_matrix
    >>> from lowrank import LowRankMatrix
    >>> from lowrank.krylov import solve_sparse_low_rank_symmetric_lyapunov
    >>> # Create a symmetric sparse matrix
    >>> A = csr_matrix([[4, 1, 0], [1, 3, 1], [0, 1, 2]])
    >>> # Create a symmetric low-rank right-hand side
    >>> U = np.array([[1.0], [0.0], [0.0]])
    >>> C = LowRankMatrix(U, U.T)  # Rank-1 symmetric matrix
    >>> # Solve AX + XA = C
    >>> X = solve_sparse_low_rank_symmetric_lyapunov(A, C, tol=1e-10)
    >>> # Verify the solution
    >>> residual = A @ X + X @ A - C
    >>> residual.norm() < 1e-9
    True
    >>> # X is also symmetric and low-rank
    >>> X.is_symmetric()
    True
    """
    # Check inputs
    assert isinstance(A, spmatrix), "A must be a sparse matrix"
    assert isinstance(C, LowRankMatrix), "C must be a low-rank matrix"
    assert tol > machine_precision, "tol must be larger than machine precision"
    if max_iter is None:
        max_iter = max(2, int(A.shape[0] / C.rank))
    assert max_iter >= 2, "max_iter must be at least 2"

    # Parse Krylov kwargs
    if krylov_kwargs is None:
        krylov_kwargs = {}
    extended = krylov_kwargs.get("extended", True)
    invA = krylov_kwargs.get("invA", None)
    poles = krylov_kwargs.get("poles", None)

    if extended and poles is not None:
        raise ValueError("Cannot use rational Krylov space with extended Krylov space")

    # Precompute some quantities
    normA = spsla.norm(A)
    normC = C.norm()

    # Define the Krylov space
    # Ensure initial vector is 2D
    X0 = C._matrices[0]
    if X0.ndim == 1:
        X0 = X0.reshape(-1, 1)
    
    if extended:
        if invA is None:
            invA = lambda x: spsla.spsolve(A, x)
        krylov_space = ExtendedKrylovSpace(A, X0, invA, is_symmetric=True)
    elif poles is not None:
        krylov_space = RationalKrylovSpace(A, X0, poles, is_symmetric=True)
    else:
        krylov_space = KrylovSpace(A, X0, is_symmetric=True)
        warn('Warning: standard Krylov space may not converge. Consider using extended or rational Krylov space.')
        
    # Current basis
    Uk = krylov_space.Q

    # SOLVE SMALL PROJECTED LYAPUNOV IN LOOP
    for k in np.arange(1, max_iter):
        # SOLVE PROJECTED LYAPUNOV Ak Y + Y Ak = Ck
        Ak = Uk.T.dot(A.dot(Uk))
        # Ck = U^T @ C @ U (convert to dense)
        CUk = C.dot(Uk)  # C @ U
        if isinstance(CUk, LowRankMatrix):
            CUk = CUk.to_dense()
        Ck = Uk.T @ CUk  # U^T @ (C @ U)
        Yk = la.solve_lyapunov(Ak, Ck)

        # CHECK CONVERGENCE
        Xk = QuasiSVD(Uk, Yk, Uk)
        AXk = Xk.dot_sparse(A, side="opposite")
        XkA = Xk.dot_sparse(A)
        # Compute residual norm
        crit = (C - AXk - XkA).norm() / (2 * normA * la.norm(Yk) + normC)
        
        if crit < tol:
            # Truncate up to machine precision since the criterion overestimates the error
            return Xk.to_svd().truncate()
        else:
            krylov_space.augment_basis()
            Uk = krylov_space.Q

    warn('No convergence before max_iter')
    # Need to solve with final basis
    Ak = Uk.T.dot(A.dot(Uk))
    CUk = C.dot(Uk)
    if isinstance(CUk, LowRankMatrix):
        CUk = CUk.to_dense()
    Ck = Uk.T @ CUk
    Yk = la.solve_lyapunov(Ak, Ck)
    X = QuasiSVD(Uk, Yk, Uk)
    return X




def solve_sparse_low_rank_non_symmetric_lyapunov(A: spmatrix,
                                   C: LowRankMatrix,
                                   tol: float = 1e-12,
                                   max_iter: int = None,
                                   krylov_kwargs: dict = None
                                   ) -> QuasiSVD:
    """
    Low-rank solver for the general Lyapunov equation:
    
    Find X such that A X + X A^H = C.
    
    Uses two Krylov spaces (left for A, right for A^H). The projected problem is solved
    as a Sylvester equation: Ak Y + Y Bk^H = Ck.
    
    Parameters
    ----------
    A : spmatrix
        Sparse matrix of shape (n, n)
    C : LowRankMatrix
        Low-rank right-hand side of shape (n, n)
    tol : float, optional
        Convergence tolerance, by default 1e-12
    max_iter : int, optional
        Maximum iterations, by default n / rank(C)
    krylov_kwargs : dict, optional
        Krylov space configuration with keys:
        - 'extended' (bool): Use extended Krylov space (default True)
        - 'invA' (callable): Custom inverse function for A
        - 'invAH' (callable): Custom inverse function for A^H
        - 'poles_A' (array_like): Poles for rational Krylov space for A
        - 'poles_AH' (array_like): Poles for rational Krylov space for A^H
        
    Returns
    -------
    QuasiSVD
        Low-rank solution X (not necessarily symmetric)
    """
    # Check inputs
    assert isinstance(A, spmatrix), "A must be a sparse matrix"
    assert isinstance(C, LowRankMatrix), "C must be a low-rank matrix"
    assert tol > machine_precision, "tol must be larger than machine precision"
    if max_iter is None:
        max_iter = max(2, int(A.shape[0] / C.rank))
    assert max_iter >= 2, "max_iter must be at least 2"

    # Parse Krylov kwargs
    if krylov_kwargs is None:
        krylov_kwargs = {}
    extended = krylov_kwargs.get("extended", True)
    invA = krylov_kwargs.get("invA", None)
    invAH = krylov_kwargs.get("invAH", None)
    poles_A = krylov_kwargs.get("poles_A", None)
    poles_AH = krylov_kwargs.get("poles_AH", None)

    if extended and (poles_A is not None or poles_AH is not None):
        raise ValueError("Cannot use rational Krylov space with extended Krylov space")

    # Precompute some quantities
    normA = spsla.norm(A)
    normC = C.norm()
    AH = A.T.conj()  # Hermitian transpose (conjugate transpose)
    U, V = C._matrices[0], C._matrices[-1].T.conj()
    
    # Ensure U and V are 2D
    if U.ndim == 1:
        U = U.reshape(-1, 1)
    if V.ndim == 1:
        V = V.reshape(-1, 1)

    # Define the left Krylov space (for A)
    if extended:
        if invA is None:
            invA = lambda x: spsla.spsolve(A, x)
        left_space = ExtendedKrylovSpace(A, U, invA)
    elif poles_A is not None:
        left_space = RationalKrylovSpace(A, U, poles_A)
    else:
        warn('Warning: standard Krylov space may not converge. Consider using extended or rational Krylov space.')
        left_space = KrylovSpace(A, U)

    # Define the right Krylov space (for A^H)
    if extended:
        if invAH is None:
            invAH = lambda x: spsla.spsolve(AH, x)
        right_space = ExtendedKrylovSpace(AH, V, invAH)
    elif poles_AH is not None:
        right_space = RationalKrylovSpace(AH, V, poles_AH)
    else:
        right_space = KrylovSpace(AH, V)
    
    # Current basis
    Uk = left_space.Q
    Vk = right_space.Q

    # SOLVE SMALL PROJECTED SYLVESTER IN LOOP
    for k in np.arange(1, max_iter):
        # SOLVE PROJECTED SYLVESTER Ak Y + Y Bk^H = Ck
        # where Ak = U^T @ A @ U and Bk = V^T @ A^H @ V
        Ak = Uk.T.dot(A.dot(Uk))
        Bk = Vk.T.dot(AH.dot(Vk))
        # Ck = U^T @ C @ V (convert to dense)
        CVk = C.dot(Vk)  # C @ V
        if isinstance(CVk, LowRankMatrix):
            CVk = CVk.to_dense()
        Ck = Uk.conj().T @ CVk  # U^H @ (C @ V)
        # Solve Ak Y + Y Bk^H = Ck, which is a Sylvester equation
        Yk = la.solve_sylvester(Ak, Bk.conj().T, Ck)

        # CHECK CONVERGENCE
        Xk = QuasiSVD(Uk, Yk, Vk)
        AXk = Xk.dot_sparse(A, side="opposite")  # A @ Xk
        XkAH = Xk.dot_sparse(AH)  # Xk @ A^H
        # Compute residual norm
        crit = (C - AXk - XkAH).norm() / (2 * normA * la.norm(Yk) + normC)
        
        if crit < tol or k == max_iter - 1:
            # Truncate to machine precision since the criterion overestimates the error
            return Xk.to_svd().truncate()
        else:
            left_space.augment_basis()
            Uk = left_space.Q
            right_space.augment_basis()
            Vk = right_space.Q

    warn('No convergence before max_iter')
    # Need to solve with final basis
    Ak = Uk.T.dot(A.dot(Uk))
    Bk = Vk.T.dot(AH.dot(Vk))
    CVk = C.dot(Vk)
    if isinstance(CVk, LowRankMatrix):
        CVk = CVk.to_dense()
    Ck = Uk.conj().T @ CVk
    Yk = la.solve_sylvester(Ak, Bk.conj().T, Ck)
    X = QuasiSVD(Uk, Yk, Vk)
    return X
    

def solve_lyapunov(A: ndarray | spmatrix,
                   C: ndarray | LowRankMatrix,
                   tol: float = 1e-12,
                   max_iter: int = None,
                   is_symmetric: bool = None,
                   krylov_kwargs: dict = None,
                   ) -> ndarray | LowRankMatrix:
    """
    Efficient low-rank compatible solver for the Lyapunov equation.
    
    Find X such that AX + XA^H = C.
    
    This function is a wrapper that selects the appropriate solver based on the types of A and C.
    
    For lower computational cost, if A and C are symmetric, set is_symmetric=True.
    This enables the use of the Lanczos algorithm (3-term recurrence) instead of 
    Arnoldi (full orthogonalization), approximately halving the computational cost 
    and memory usage per iteration.

    Parameters
    ----------
    A : ndarray | spmatrix
        The matrix A of shape (n, n)
    C : ndarray | LowRankMatrix
        The matrix C of shape (n, n)
    tol : float, optional
        Convergence tolerance for Krylov solver, by default 1e-12
    max_iter : int, optional
        Maximum iterations for Krylov solver, by default None
    is_symmetric : bool, optional
        If True, use symmetric solver (only for sparse A and low-rank C).
        Exploits symmetry using Lanczos algorithm for ~2x speedup.
        If None, symmetry is auto-detected (with efficiency warning).
    krylov_kwargs : dict, optional
        Krylov space configuration (see solve_sparse_low_rank_symmetric_lyapunov 
        or solve_sparse_low_rank_non_symmetric_lyapunov for details)
        
    Returns
    -------
    ndarray | LowRankMatrix
        The solution X, either dense or low-rank and symmetric if and only if C is symmetric.
    
    Examples
    --------
    >>> import numpy as np
    >>> from scipy.sparse import csr_matrix
    >>> from lowrank import LowRankMatrix
    >>> from lowrank.krylov import solve_lyapunov
    >>> # Small dense case
    >>> A_small = np.array([[4, 1], [1, 3]])
    >>> C_small = np.array([[1, 0], [0, 1]])
    >>> X_small = solve_lyapunov(A_small, C_small)
    >>> # Verify: AX + XA = C
    >>> np.allclose(A_small @ X_small + X_small @ A_small, C_small)
    True
    >>> # Large sparse case with low-rank RHS
    >>> A = csr_matrix([[4, 1, 0], [1, 3, 1], [0, 1, 2]])
    >>> U = np.array([[1.0], [0.0], [0.0]])
    >>> C = LowRankMatrix(U, U.T)
    >>> X = solve_lyapunov(A, C, is_symmetric=True, tol=1e-10)
    >>> # Solution is low-rank and symmetric
    >>> type(X).__name__
    'SVD'
    >>> X.rank <= 10  # Much lower rank than n
    True
    """
    # Check Krylov kwargs
    if krylov_kwargs is None:
        # Default parameters for Krylov solver
        krylov_kwargs = {'extended': True}
    # Check symmetry input
    if is_symmetric is None:
        # Check symmetry of low-rank matrix C
        if isinstance(C, ndarray):
            is_symmetric = np.allclose(C, C.T.conj())
        else:
            try:
                warn("Checking symmetry of low-rank matrix C. For better performance, please provide is_symmetric bool input.", LowRankEfficiencyWarning)
                is_symmetric = C.is_symmetric()
            except:
                Cd = C.to_dense()
                warn("Checking symmetry of low-rank C via dense conversion. For better performance, please provide is_symmetric bool input.", LowRankEfficiencyWarning)
                is_symmetric = np.allclose(Cd, Cd.T.conj())
        # Check symmetry of A if needed
        if is_symmetric:
            # Sparse matrix case
            if isinstance(A, spmatrix):
                if not (A != A.T).nnz == 0:
                    is_symmetric = False
            # Dense matrix case
            elif isinstance(A, ndarray):
                if not np.allclose(A, A.T.conj()):
                    is_symmetric = False
                    
    # Low rank solver
    if isinstance(C, LowRankMatrix):
        # Convert dense A to sparse if needed
        if isinstance(A, ndarray):
            A = sps.csc_matrix(A)
        if is_symmetric:
            X = solve_sparse_low_rank_symmetric_lyapunov(A, C, tol, max_iter, krylov_kwargs)
        else:
            X = solve_sparse_low_rank_non_symmetric_lyapunov(A, C, tol, max_iter, krylov_kwargs)
            
    # Dense solver
    else:
        X = solve_small_lyapunov(A, C)
    return X




