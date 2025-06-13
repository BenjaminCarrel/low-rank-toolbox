import numpy as np
from numpy import ndarray
from ..utils import sRRQR
import scipy.linalg as la

    
def sQDEIM(U: ndarray, tol: float = None, compute_M: bool = False) -> list:
    """
    sQDEIM - Strong RRQR based DEIM of U (size n x k)

    Key advantage: the selection of the indexes is guaranteed to satisfy the condition:
    sigma_{min}(U[p, :])^{-1} <= sqrt(1 + f * r (n-k))
    
    By default, f = 2

    Parameters
    ----------
    U: ndarray
        Orthonormal matrix of size n x k
    tol: float
        Tolerance for the quantity sigma_{min}(U[p, :])^{-1}
    compute_M: bool
        If True, return also the matrix U @ inv(U[S, :])

    Returns
    -------
    p: list
        Selection of m row indices with guaranteed upper bound: norm(inv(U[S,:])) <= sqrt(n-k+1) * O(2^m).
    M: ndarray (optional)
        Matrix U @ inv(U[S, :])
    """
    # Initialisation
    _, k = U.shape
    if tol is None:
        Q, R, P = sRRQR(U.T.conj(), f=2, mode="rank", param=k)
        p = P[:k]
    else:
        Q, R, P = sRRQR(U.T.conj(), f=2, mode="tol", param=tol)
        k = Q.shape[1]
        p = P[:k]
    if compute_M:
        L = la.solve(R[:, :k], R[:, k:]).T.conj()
        M = np.row_stack((np.eye(k), L))
        Q = np.argsort(P)
        M = M[Q, :]
        return p, M
    else:
        return p
