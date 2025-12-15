"""Generalized Nyström method for low-rank matrix approximation.

Author: Benjamin Carrel, University of Geneva, 2024
"""

from typing import Optional

import numpy as np
from scipy.sparse.linalg import LinearOperator

from ..matrices.low_rank_matrix import LowRankMatrix
from ..matrices.quasi_svd import QuasiSVD
from ..matrices.svd import SVD


def generalized_nystrom(
    X: LinearOperator,
    r: int,
    oversampling_params: tuple = (10, 15),
    epsilon: Optional[float] = None,
    seed: int = 1234,
    **extra_data,
) -> QuasiSVD:
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
        raise ValueError(f"Rank must be at least 1, got r={r}.")
    if epsilon is not None and epsilon <= 0:
        raise ValueError(
            f"Epsilon must be positive when provided, got epsilon={epsilon}."
        )

    m, n = X.shape
    p1, p2 = oversampling_params

    if r + p1 > n:
        raise ValueError(f"Rank + p1 ({r + p1}) exceeds number of columns ({n}).")
    if r + p2 > m:
        raise ValueError(f"Rank + p2 ({r + p2}) exceeds number of rows ({m}).")
    # Draw the two random matrices
    np.random.seed(seed)
    J = np.random.randn(n, r + p1)
    K = np.random.randn(m, r + p2)

    # Compute the factors
    if isinstance(X, LowRankMatrix):
        XJ = X.dot(J, dense_output=True)
        KtX = X.dot(K.T, side="left", dense_output=True)
    else:
        XJ = X.dot(J)
        KtX = K.T.dot(X)
    KtXJ = KtX.dot(J)

    # Compute SVD of middle term using numpy
    U_C, s_C, Vh_C = np.linalg.svd(KtXJ, full_matrices=False)

    # Truncate based on rank or tolerance
    if epsilon is None:
        # Truncate to rank r
        U_C = U_C[:, :r]
        s_C = s_C[:r]
        Vh_C = Vh_C[:r, :]
    else:
        # Truncate based on relative tolerance
        threshold = epsilon * s_C[0]
        r_effective = int(np.sum(s_C > threshold))
        U_C = U_C[:, :r_effective]
        s_C = s_C[:r_effective]
        Vh_C = Vh_C[:r_effective, :]

    # Construct final QuasiSVD
    V_C = Vh_C.T.conj()
    U = XJ.dot(V_C)
    S_inv = np.diag(1.0 / s_C)  # Inverse of diagonal matrix
    V = (U_C.T.dot(KtX)).T

    # Skip memory check for final result - QuasiSVD with inverted S matrix
    # can have inflated storage, but this is expected for generalized Nyström
    return QuasiSVD(U, S_inv, V, **extra_data)
