import numpy as np
from numpy import ndarray
from ..utils import sRRQR
import scipy.linalg as la

    
def oversampling_sQDEIM(U: ndarray, oversampling_size: int, tol: float = None, compute_M: bool = False) -> list:
    """
    Oversampling sQDEIM - Oversampled version of sQDEIM

    Reference:
    ACCURACY AND STABILITY OF CUR DECOMPOSITIONS WITH OVERSAMPLING 
    (Taejun Park and Yuji Nakatsukasa)

    Parameters
    ----------
    U: ndarray
        Orthonormal matrix of size n x k
    oversampling_size: int
        Oversampling size p < k such that m = k + p
    tol: float
        Tolerance for the strong rank-revealing QR factorization
        If None, use the rank-revealing QR factorization with f=2
    compute_M: bool
        If True, return also the matrix U @ pseudoinv(U[S, :])

    Returns
    -------
    p: list
        Selection of m row indices.
    M: ndarray (optional)
        Matrix U @ pseudoinv(U[S, :])
    """
    # Sanity check
    if oversampling_size < 0:
        raise ValueError("Oversampling size must be positive")
    if oversampling_size > U.shape[1]:
        raise ValueError("Oversampling size must be smaller than the number of columns of U")
    # sQDEIM
    _, k = U.shape
    m = k + oversampling_size
    if tol is None:
        Q, R, P = sRRQR(U.T.conj(), f=2, mode="rank", param=k)
        p1 = P[:k]
    else:
        Q, R, P = sRRQR(U.T.conj(), f=2, mode="tol", param=tol)
        k = Q.shape[1]
        p1 = P[:k]

    ## Algorithm 4.2 in reference
    _, _, vt = la.svd(U[p1, :], full_matrices=False)
    Vp = vt.T.conj()[:, k-oversampling_size:]

    # Apply again SQDEIM on the unchosen rows
    M = U[P[k:], :].dot(Vp)
    if tol is None:
        Q, R, P = sRRQR(M.T.conj(), f=2, mode="rank", param=oversampling_size)
        p2 = P[:oversampling_size]
    else:
        Q, R, P = sRRQR(M.T.conj(), f=2, mode="tol", param=tol)
        p2 = P[:oversampling_size]
    # Concatenate the two selections
    p = np.concatenate((p1, p2))

    if compute_M:
        M = la.lstsq(U[p, :].T.conj(), U.T.conj())[0].T.conj()
        return p, M
    else:
        return p
