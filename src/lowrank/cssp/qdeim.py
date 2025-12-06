import numpy as np
from numpy import ndarray
import scipy.linalg as la


def QDEIM(U: ndarray, 
          return_projector: bool = False, 
          return_inverse: bool = False, 
          **extra_args) -> list:
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
    return_projector: bool
        If True, return also the matrix U @ inv(U[S, :])
    return_inverse: bool
        If True, return also the matrix inv(U[S, :])
    extra_args: dict
        Additional arguments: 
            qr_kwargs: dict
                Additional arguments for the QR factorization
            solve_kwargs: dict
                Additional arguments for the solve function

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
    (_, R, P) = la.qr(U.T.conj(), pivoting=True, **extra_args.get('qr_kwargs', {}))
    p = P[0:k]
    if return_projector:
        L = la.solve(R[:, :k], R[:, k:], **extra_args.get('solve_kwargs', {})).T.conj()
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
