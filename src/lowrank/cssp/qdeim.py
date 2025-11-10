import numpy as np
from numpy import ndarray
import scipy.linalg as la


def QDEIM(U: ndarray, compute_M: bool = False, **extra_args) -> list:
    """
    QDEIM - QR based DEIM of U (size n x k)

    Reference
        A new selection operator for the discrete empirical interpolation method - improved a priori error bound and extensions.
        Zlatko Drmač and Serkan Gugercin.
        SIAM Journal on Scientific Computing, 38(2), A631-A648.

    Original Matlab code from Zlatko Drmač 

    Parameters
    ----------
    U: ndarray
        Orthonormal matrix of size n x k
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
    (_, R, P) = la.qr(U.T.conj(), pivoting=True)
    p = P[0:k]
    if compute_M:
        L = la.solve(R[:, :k], R[:, k:]).T.conj()
        M = np.vstack((np.eye(k), L))
        Q = np.argsort(P)
        M = M[Q, :]
        return p, M
    else:
        return p
