import numpy as np
import scipy.linalg as la


def gpode(U, 
          oversampling_size: int, 
          return_projector: bool = False, 
          return_inverse: bool = False, 
          **extra_args) -> list:
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
    return_projector: bool
        If True, return also the matrix U @ pinv(U[p, :])
    return_inverse: bool
        If True, return also the inverse of U[p, :]
    extra_args: dict
        Additional arguments: 
            qr_kwargs: dict
                Additional arguments for the QR factorization
            lstsq_kwargs: dict
                Additional arguments for the lstsq function
    Returns
    -------
    p: list
        Selection of m row indices
    P_U: ndarray (n x k) (optional)
        Matrix U @ pinv(U[p, :])
    inv_U: ndarray (k x k) (optional)
        Inverse of U[p, :]
    """
    # QDEIM
    _, k = U.shape
    m = k + oversampling_size
    _, _, P = la.qr(U.T.conj(), pivoting=True, **extra_args.get('qr_kwargs', {}))
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
    if return_projector:
        P_U = la.lstsq(U[p, :].T.conj(), U.T.conj(), **extra_args.get('lstsq_kwargs', {}))[0].T.conj()
        if return_inverse:
            inv_U = U.T.conj().dot(P_U)
            return p, P_U, inv_U
        else:
            return p, P_U
    else:
        return p