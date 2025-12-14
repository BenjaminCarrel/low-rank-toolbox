"""Strong QDEIM - Strong Rank-Revealing QR based DEIM.

Author: Benjamin Carrel, University of Geneva, 2024
"""

import numpy as np
from numpy import ndarray
from .utils import sRRQR
import scipy.linalg as la


def sQDEIM(U: ndarray, 
           eta: float = 2, 
           return_projector: bool = False,
           return_inverse: bool = False,
           **extra_args) -> list:
    """
    sQDEIM - Strong RRQR based DEIM of U (size n x k)

    Key advantage: the selection of the indexes is guaranteed to satisfy the condition:
    sigma_{min}(U[p, :])^{-1} <= sqrt(1 + eta * r (n-k))
    
    By default, eta = 2

    Parameters
    ----------
    U: ndarray
        Orthonormal matrix of size n x k
    eta: float
        Bounding factor for R_11^{-1} R_12, must be >= 1
    mode: str
        Specifies the truncation criterion. Must be 'rank' or 'tol'.
    param: int or float
        The parameter for the chosen mode.
                           - If mode is 'rank', `param` is the desired rank `k`.
                           - If mode is 'tol', `param` is the error tolerance.
    return_projector: bool
        If True, return also the matrix U @ inv(U[S, :])
    return_inverse: bool
        If True, return also the matrix inv(U[S, :])


    Returns
    -------
    p: list
        Selection of m row indices with guaranteed upper bound: norm(inv(U[S,:])) <= sqrt(n-k+1) * O(2^m).
    P_U: ndarray (n x k) (optional)
        Matrix U @ inv(U[S, :])
    inv_U: ndarray (k x k) (optional)
        Matrix inv(U[S, :])
    """
    # Initialisation
    _, k = U.shape
    Q, R, P = sRRQR(U.T.conj(), eta=eta, mode="rank", param=k)
    p = P[:k]
    if return_projector:
        L = la.solve(R[:, :k], R[:, k:]).T.conj()
        P_U = np.vstack((np.eye(k), L))
        Q = np.argsort(P)
        P_U = P_U[Q, :]
        if return_inverse:
            inv_U = U.T.conj().dot(P_U)
            return p, P_U, inv_U
        else:
            return p, P_U
    else:
        return p
