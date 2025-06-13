import numpy as np
import scipy.linalg as la


def gpode(U, oversampling_size: int, compute_M: bool = False, **extra_args) -> list:
    """
    Gappy POD+E - Greedy algorithm for the selection of m rows of U
    Minimize the norm of the pseudoinverse of U[S, :]
    Total cost is O(k^2 + m^2) + QR of U.T.conj() of cost O(nk^2)

    Reference
        Stability of discrete empirical interpolation and gappy proper orthogonal decomposition with randomized and deterministic sampling points
        Benjamin Peherstorfer, Zlatko Drmač, and Serkan Gugercin
        SIAM Journal on Scientific Computing, 42(5), A2837-A2864.

    Parameters
    ----------
    U: ndarray
        Orthonormal matrix of size n x k
    oversampling_size: int
        Oversampling size p such that m = k + p
    compute_M: bool 
        If True, return also the matrix U @ pinv(U[p, :])

    Returns
    -------
    p: list
        Selection of m row indices
    M: ndarray (optional)
        Matrix U @ pinv(U[p, :])
    """
    # QDEIM
    _, k = U.shape
    m = k + oversampling_size
    _, _, P = la.qr(U.T.conj(), pivoting=True)
    p = P[0:k]
    for _ in np.arange(k, m):
        # Compute SVD
        _, s, Wh = la.svd(U[p, :], full_matrices=False)
        # Compute the last gap
        g = s[-2]**2 - s[-1]**2
        Ub = Wh.dot(U.T.conj())
        su = np.sum(np.abs(Ub)**2, axis=0)
        r = g + su
        r = r - np.sqrt((g + su)**2 - 4 * g * Ub[-1, :]**2)
        # Descending sort indexes
        I = np.argsort(r)[::-1]
        e = 0
        # Update selection operator p
        while np.any(I[e] in p):
            e += 1
        p = np.append(p, I[e])
    if compute_M:
        M = la.lstsq(U[p, :].T.conj(), U.T.conj())[0].T.conj()
        return p, M
    else:
        return p