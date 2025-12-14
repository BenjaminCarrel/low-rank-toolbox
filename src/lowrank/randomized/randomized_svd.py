"""Randomized SVD algorithms for low-rank matrix approximation.

Author: Benjamin Carrel, University of Geneva, 2024
"""

from scipy.sparse.linalg import LinearOperator
import numpy as np
from ..matrices.svd import SVD
from ..matrices.low_rank_matrix import LowRankMatrix
from .rangefinder import adaptive_rangefinder, rangefinder


def randomized_svd(X: LinearOperator, r: int, p: int = 10, q: int = 0, truncate: bool = True, seed: int = 1234, **extra_data) -> SVD:
    """
    Randomized SVD algorithm.
    
    Reference:
        "Finding structure with randomness: Probabilistic algorithms for constructing approximate matrix decompositions",
        Halko, Martinsson and Tropp 2010.
        
    The randomized SVD computes an approximate SVD of a matrix X with the following steps:
    1. Estimate the range with Q = rangefinder(X, r, p, q, seed)
    2. Form the smaller matrix C = Q^H X
    3. Compute the SVD of C: C = U_C S V^H
    4. Form the approximate SVD of X: X_approx = Q U_C S V^H
    

    Parameters
    ----------
    X : SVD | LowRankMatrix | ndarray | LinearOperator
        Matrix or operator to be decomposed
    r : int
        Rank of the decomposition, must be positive
    p : int, optional
        Oversampling parameter (default: 10)
    q : int, optional
        Number of power iterations (default: 0)
    truncate : bool, optional
        If True, truncate the SVD to rank r (default: True)
    seed : int, optional
        Random seed for reproducibility (default: 1234)
    **extra_data : dict
        Additional arguments passed to SVD constructor. If 'Omega' is provided, use it as the sketching matrix.
    
    Returns
    -------
    SVD
        Near-optimal best rank r approximation of X
    """
    # Input validation
    if r < 1:
        raise ValueError(f'Rank must be at least 1, got r={r}.')
    if p < 0:
        raise ValueError(f'Oversampling parameter must be non-negative, got p={p}.')
    if q < 0:
        raise ValueError(f'Number of power iterations must be non-negative, got q={q}.')
    if r + p > min(X.shape):
        raise ValueError(f'Rank + oversampling ({r + p}) exceeds minimum matrix dimension ({min(X.shape)}).')
    
    _, n = X.shape
    # Draw or extract the random matrix Omega
    if 'Omega' in extra_data:
        Omega = extra_data['Omega']
    else:
        np.random.seed(seed)
        Omega = np.random.randn(n, r + p)

    # Step 1: find the range of X
    Q = rangefinder(X, r, p, q, seed, Omega=Omega)
    
    # Step 2: Form smaller matrix B = Q^H @ X
    if isinstance(X, LowRankMatrix): # support for low-rank matrices
        C = X.dot(Q.T.conj(), side='left')
    else:
        C = Q.T.conj().dot(X)

    # Step 3: Compute SVD of C
    if truncate:
        Xr = SVD.truncated_svd(C, r, **extra_data)
    else:
        Xr = SVD.from_dense(C, **extra_data)
    
    # Step 4: Form U = Q @ U_C (V and S are unchanged)
    Xr.U = Q.dot(Xr.U)
    return Xr



def adaptive_randomized_svd(X: LinearOperator, tol: float = 1e-6, failure_prob: float = 1e-6, max_rank: int = None, seed: int = 1234, **extra_data) -> SVD:
    """
    Adaptive randomized SVD algorithm.
    
    Reference: 
        "Finding structure with randomness: Probabilistic algorithms for constructing approximate matrix decompositions",
        Halko, Martinsson and Tropp 2010.
        
    The adaptive randomized SVD computes an approximate SVD of a matrix X with the following steps:
    1. Estimate the range with Q = adaptive_rangefinder(X, tol, failure_prob, seed)
    2. Form the smaller matrix C = Q^H X
    3. Compute the SVD of C: C = U_C S V^H
    4. Form the approximate SVD of X: X_approx = Q U_C S V^H
    

    Parameters
    ----------
    X : SVD | LowRankMatrix | ndarray | LinearOperator
        Matrix or operator to be decomposed
    tol : float, optional
        Target approximation error (default: 1e-6)
    failure_prob : float, optional
        Failure probability for the approximation error, must be in (0, 1) (default: 1e-6)
    max_rank : int, optional
        Maximum rank for the approximation (default: None, no limit)
    seed : int, optional
        Random seed for reproducibility (default: 1234)
    **extra_data : dict
        Additional arguments passed to SVD constructor
    
    Returns
    -------
    SVD
        Near-optimal approximation of X with ||X - X_approx||_F <= tol with probability >= 1 - failure_prob
    """
    # Input validation
    if tol <= 0:
        raise ValueError(f'Tolerance must be positive, got tol={tol}.')
    if not (0 < failure_prob < 1):
        raise ValueError(f'Failure probability must be in (0, 1), got failure_prob={failure_prob}.')
    if max_rank is not None and max_rank < 1:
        raise ValueError(f'Maximum rank must be at least 1, got max_rank={max_rank}.')
    
    # Step 1: find the range of X adaptively
    Q = adaptive_rangefinder(X, tol, failure_prob, seed)
    if max_rank is not None:
        Q = Q[:, :max_rank]
    
    # Step 2: Form smaller matrix C = Q^H @ X
    if isinstance(X, LowRankMatrix): # support for low-rank matrices
        C = X.dot(Q.T.conj(), side='left')
    else:
        C = Q.T.conj().dot(X)

    # Step 3: Compute SVD of C
    Xr = SVD.from_dense(C, **extra_data)
    
    # Step 4: Form U = Q @ U_C (V and S are unchanged)
    Xr.U = Q.dot(Xr.U)
    return Xr