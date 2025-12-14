"""Generalized Nyström method for low-rank matrix approximation.

Author: Benjamin Carrel, University of Geneva, 2024
"""

import numpy as np
from scipy.sparse.linalg import LinearOperator
from ..matrices.quasi_svd import QuasiSVD
from ..matrices.svd import SVD
from ..matrices.low_rank_matrix import LowRankMatrix


def generalized_nystrom(X: LinearOperator, 
                        r: int, 
                        oversampling_params: tuple = (10, 15), 
                        epsilon: float = None,
                        seed: int = 1234,
                        **extra_data) -> QuasiSVD:
    """
    Generalized Nyström method

    Reference:
        "Fast and stable randomized low-rank matrix approximation"
        Nakatsukasa, 2019

    Approximation with formula
        X ~= X J (K^T X J)^{dagger} K^T X

    Parameters
    ----------
    X : LinearOperator
        Matrix to approximate (real-valued only, complex matrices not supported)
    r : int
        Rank of approximation, must be positive
    oversampling_params : tuple, optional
        Oversampling parameters (p1, p2) for the two sketch matrices (default: (10, 15))
    epsilon : float, optional
        When given, perform stable GN with epsilon-truncation for SVD (default: None)
    seed : int, optional
        Random seed for reproducibility (default: 1234)
    **extra_data : dict
        Additional arguments passed to QuasiSVD constructor

    Returns
    -------
    QuasiSVD
        Near optimal best rank r approximation of X in QuasiSVD format
        
    Notes
    -----
    This method returns a QuasiSVD (not SVD) because the middle matrix S
    is typically inverted, making it non-diagonal. Convert to SVD if needed:
        result = generalized_nystrom(X, r).to_svd()
    """
    # Input validation
    if r < 1:
        raise ValueError(f'Rank must be at least 1, got r={r}.')
    if epsilon is not None and epsilon <= 0:
        raise ValueError(f'Epsilon must be positive when provided, got epsilon={epsilon}.')
    
    m, n = X.shape
    p1, p2 = oversampling_params
    
    if r + p1 > n:
        raise ValueError(f'Rank + p1 ({r + p1}) exceeds number of columns ({n}).')
    if r + p2 > m:
        raise ValueError(f'Rank + p2 ({r + p2}) exceeds number of rows ({m}).')
    # Draw the two random matrices
    np.random.seed(seed)
    J= np.random.randn(n, r + p1)
    K = np.random.randn(m, r + p2)

    # Compute the factors
    if isinstance(X, LowRankMatrix):
        XJ = X.dot(J, dense_output=True)
        KtX = X.dot(K.T, side='left', dense_output=True)
    else:
        XJ = X.dot(J)
        KtX = K.T.dot(X)
    KtXJ = KtX.dot(J)
    
    # Compute SVD of middle term and truncate for stable version
    if epsilon is None:
        C = SVD.truncated_svd(KtXJ, r=r)
    else:
        C = SVD.truncated_svd(KtXJ, rtol=epsilon)

    # Return in QuasiSVD format
    U = XJ.dot(C.V)
    S_inv = np.diag(1.0 / C.s)  # Inverse of diagonal matrix
    V = (C.U.T.dot(KtX)).T

    return QuasiSVD(U, S_inv, V, **extra_data)
