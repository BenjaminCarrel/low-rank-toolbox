"""
Author: Benjamin Carrel, University of Geneva, 2022

This module contains solvers for the Sylvester equation.
"""

#%% Imports
import numpy as np
from numpy import ndarray
import scipy.linalg as la
from scipy.sparse import spmatrix, issparse
import scipy.sparse as sps
import scipy.sparse.linalg as spsla
from lowrank import LowRankMatrix, QuasiSVD
from lowrank.krylov import KrylovSpace, ExtendedKrylovSpace, RationalKrylovSpace

# Warnings
from warnings import warn

machine_precision = np.finfo(float).eps
Matrix = ndarray | spmatrix | LowRankMatrix

#%% Functions

def solve_small_sylvester(A: ndarray | spmatrix, B: ndarray | spmatrix, C: ndarray) -> ndarray:
    """
    Solve the Sylvester equation for small systems.
    
    Find X such that AX + XB = C. Wrapper to scipy.linalg.solve_sylvester.
    For larger systems, use solve_sparse_low_rank_sylvester or solve_sylvester_large_A_small_B.
    
    Parameters
    ----------
    A : ndarray | spmatrix
        Matrix of shape (m, m)
    B : ndarray | spmatrix
        Matrix of shape (n, n)
    C : ndarray
        Matrix of shape (m, n)
    
    Returns
    -------
    ndarray
        Dense solution of shape (m, n)
    """
    # Convert sparse matrices to dense for scipy.linalg.solve_sylvester
    A_dense = A.toarray() if issparse(A) else A
    B_dense = B.toarray() if issparse(B) else B
    return la.solve_sylvester(A_dense, B_dense, C)

def solve_sylvester_large_A_small_B(A: spmatrix,
                                    B: ndarray,
                                    C: ndarray) -> ndarray:
    """
    Solve the Sylvester equation when A is large and B is small.
    
    Find X such that AX + XB = C using eigenvalue decomposition of B.
    
    Reference:
    Simoncini, V., 2016. Computational Methods for Linear Matrix Equations. 
    SIAM Rev. 58, 377-441. https://doi.org/10.1137/130912839
    
    Parameters
    ----------
    A : spmatrix
        Sparse matrix of shape (m, m)
    B : ndarray
        Dense matrix of shape (n, n)
    C : ndarray
        Dense matrix of shape (m, n)
    
    Returns
    -------
    ndarray
        Dense solution of the Sylvester equation of shape (m, n)
    """
    # Check inputs
    assert isinstance(A, spmatrix), "A must be a sparse matrix"
    assert isinstance(B, ndarray), "B must be a dense matrix"
    assert isinstance(C, ndarray), "C must be a dense matrix"

    S, W = la.eigh(B)
    C_hat = C.dot(W)
    X_hat = np.zeros(C.shape)
    I = sps.eye(*A.shape)
    for i in np.arange(len(S)):
        X_hat[:, i] = spsla.spsolve(A + S[i] * I, C_hat[:, i])
    return X_hat @ W.T

def solve_sparse_low_rank_sylvester(A: spmatrix,
                                    B: spmatrix,
                                    C: LowRankMatrix,
                                    tol: float = 1e-12,
                                    max_iter: int = None,
                                    krylov_kwargs: dict = None,
                                    is_A_symmetric: bool = None,
                                    is_B_symmetric: bool = None,
                                    ) -> QuasiSVD:
    """
    Low-rank solver for the Sylvester equation.
    
    Find X such that AX + XB = C.
    
    Uses two Krylov spaces (left for A, right for B). The projected problem is solved
    as a small Sylvester equation: Ak Y + Y Bk = Ck.
    
    **Algorithm Note:** Each Krylov space independently uses Lanczos if the corresponding
    matrix is symmetric. If A is symmetric, the left Krylov space uses Lanczos.
    If B is symmetric, the right Krylov space uses Lanczos. Both can use Lanczos independently.
    
    **Special Case:** When A = B and both are symmetric, the problem becomes a Lyapunov
    equation (AX + XA = C). Use solve_lyapunov() instead to exploit the symmetry of X
    and use only one Krylov space instead of two.
    
    Reference:
    Simoncini, V., 2016. Computational Methods for Linear Matrix Equations. 
    SIAM Rev. 58, 377-441. https://doi.org/10.1137/130912839

    Parameters
    ----------
    A : spmatrix
        Sparse matrix of shape (m, m)
    B : spmatrix
        Sparse matrix of shape (n, n)
    C : LowRankMatrix
        Low-rank right-hand side of shape (m, n)
    tol : float, optional
        Convergence tolerance, by default 1e-12
    max_iter : int, optional
        Maximum iterations, by default m / rank(C)
    krylov_kwargs : dict, optional
        Krylov space configuration with keys:
        - 'extended' (bool): Use extended Krylov space (default True)
        - 'invA' (callable): Custom inverse function for A
        - 'invB' (callable): Custom inverse function for B
        - 'poles_A' (array_like): Poles for rational Krylov space for A
        - 'poles_B' (array_like): Poles for rational Krylov space for B
    is_A_symmetric : bool, optional
        If True, A is symmetric and Lanczos is used for left Krylov space.
        If None (default), symmetry is auto-detected.
    is_B_symmetric : bool, optional
        If True, B is symmetric and Lanczos is used for right Krylov space.
        If None (default), symmetry is auto-detected.

    Returns
    -------
    QuasiSVD
        Low-rank solution X of shape (m, n)
    """
    # Check inputs
    assert isinstance(A, spmatrix), "A must be a sparse matrix"
    assert isinstance(B, spmatrix), "B must be a sparse matrix"
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
    invB = krylov_kwargs.get("invB", None)
    poles_A = krylov_kwargs.get("poles_A", None)
    poles_B = krylov_kwargs.get("poles_B", None)

    if extended and (poles_A is not None or poles_B is not None):
        raise ValueError("Cannot use rational Krylov space with extended Krylov space")

    # Check symmetry of A and B
    if is_A_symmetric is None:
        # Auto-detect symmetry of A
        is_A_symmetric = (A != A.T).nnz == 0
    
    if is_B_symmetric is None:
        # Auto-detect symmetry of B
        is_B_symmetric = (B != B.T).nnz == 0

    # Precompute some quantities
    normA = spsla.norm(A)
    normB = spsla.norm(B)
    normC = C.norm()
    U, V = C._matrices[0], C._matrices[-1].T
    
    # Ensure U and V are 2D
    if U.ndim == 1:
        U = U.reshape(-1, 1)
    if V.ndim == 1:
        V = V.reshape(-1, 1)

    # Define the left Krylov space (for A)
    if extended:
        if invA is None:
            invA = lambda x: spsla.spsolve(A, x)
        left_space = ExtendedKrylovSpace(A, U, invA, is_symmetric=is_A_symmetric)
    elif poles_A is not None:
        left_space = RationalKrylovSpace(A, U, poles_A, is_symmetric=is_A_symmetric)
    else:
        warn('Warning: standard Krylov space may not converge. Consider using extended or rational Krylov space.')
        left_space = KrylovSpace(A, U, is_symmetric=is_A_symmetric)

    # Define the right Krylov space (for B)
    if extended:
        if invB is None:
            invB = lambda x: spsla.spsolve(B, x)
        right_space = ExtendedKrylovSpace(B, V, invB, is_symmetric=is_B_symmetric)
    elif poles_B is not None:
        right_space = RationalKrylovSpace(B, V, poles_B, is_symmetric=is_B_symmetric)
    else:
        right_space = KrylovSpace(B, V, is_symmetric=is_B_symmetric)
    
    # Current basis
    Uk = left_space.Q
    Vk = right_space.Q

    # SOLVE SMALL PROJECTED SYLVESTER IN LOOP
    for k in np.arange(1, max_iter):
        # SOLVE PROJECTED SYLVESTER Ak Y + Y Bk = Ck
        Ak = Uk.T.dot(A.dot(Uk))
        Bk = Vk.T.dot(B.dot(Vk))
        # Ck = U^T @ C @ V (convert to dense)
        CVk = C.dot(Vk)  # C @ V
        if isinstance(CVk, LowRankMatrix):
            CVk = CVk.to_dense()
        Ck = Uk.T @ CVk  # U^T @ (C @ V)
        Yk = la.solve_sylvester(Ak, Bk, Ck)

        # CHECK CONVERGENCE
        Xk = QuasiSVD(Uk, Yk, Vk)
        AXk = Xk.dot_sparse(A, side="opposite")  # A @ Xk
        XkB = Xk.dot_sparse(B)  # Xk @ B
        # Compute residual norm
        crit = (AXk + XkB - C).norm() / \
            ((normA + normB) * la.norm(Yk) + normC)
        
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
    Bk = Vk.T.dot(B.dot(Vk))
    CVk = C.dot(Vk)
    if isinstance(CVk, LowRankMatrix):
        CVk = CVk.to_dense()
    Ck = Uk.T @ CVk
    Yk = la.solve_sylvester(Ak, Bk, Ck)
    X = QuasiSVD(Uk, Yk, Vk)
    return X

def solve_sylvester(A: ndarray | spmatrix,
                    B: ndarray | spmatrix,
                    C: ndarray | LowRankMatrix,
                    tol: float = 1e-12,
                    max_iter: int = None,
                    krylov_kwargs: dict = None,
                    is_A_symmetric: bool = None,
                    is_B_symmetric: bool = None,
                    ) -> ndarray | LowRankMatrix:
    """
    Efficient low-rank compatible solver for the Sylvester equation.
    
    Find X such that AX + XB = C.
    
    This function is a wrapper that selects the appropriate solver based on the types of A, B, and C.

    Parameters
    ----------
    A : ndarray | spmatrix
        Matrix of shape (m, m)
    B : ndarray | spmatrix
        Matrix of shape (n, n)
    C : ndarray | LowRankMatrix
        Right-hand side of shape (m, n)
    tol : float, optional
        Convergence tolerance for Krylov solver, by default 1e-12
    max_iter : int, optional
        Maximum iterations for Krylov solver, by default None
    krylov_kwargs : dict, optional
        Krylov space configuration for sparse low-rank solver
    is_A_symmetric : bool, optional
        If True, A is symmetric. Auto-detected if None.
    is_B_symmetric : bool, optional
        If True, B is symmetric. Auto-detected if None.

    Returns
    -------
    ndarray | LowRankMatrix
        Solution X of shape (m, n), either dense or low-rank depending on inputs
    """
    # Check Krylov kwargs
    if krylov_kwargs is None:
        # Default parameters for Krylov solver
        krylov_kwargs = {'extended': True}
    
    # Low rank solver
    if isinstance(C, LowRankMatrix):
        # Convert dense A and B to sparse if needed
        if isinstance(A, ndarray):
            A = sps.csc_matrix(A)
        if isinstance(B, ndarray):
            B = sps.csc_matrix(B)
        # Both A and B must be sparse for low-rank solver
        assert isinstance(A, spmatrix) and isinstance(B, spmatrix), \
            "For low-rank C, both A and B should be sparse (or will be converted)"
        X = solve_sparse_low_rank_sylvester(A, B, C, tol, max_iter, krylov_kwargs, 
                                           is_A_symmetric, is_B_symmetric)
    
    # Dense solver
    else:
        # C is dense
        if isinstance(A, spmatrix) and isinstance(B, ndarray):
            # A large/sparse, B small/dense
            X = solve_sylvester_large_A_small_B(A, B, C)
        elif isinstance(A, ndarray) and isinstance(B, spmatrix):
            # A small/dense, B large/sparse - transpose the problem
            # AX + XB = C  =>  B^T X^T + X^T A^T = C^T
            X_T = solve_sylvester_large_A_small_B(B.T, A, C.T)
            X = X_T.T
        else:
            # Both A and B are small (dense or small sparse)
            X = solve_small_sylvester(A, B, C)
    return X
