"""Oversampling Strong QDEIM algorithm.

Author: Benjamin Carrel, University of Geneva, 2024
"""

from typing import Optional, Union

import numpy as np
import scipy.linalg as la
from numpy import ndarray

from .utils import sRRQR


def oversampling_sQDEIM(
    U: ndarray,
    oversampling_size: int,
    tol: Optional[float] = None,
    return_projection: bool = False,
    return_inverse: bool = False,
) -> Union[tuple, list]:
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
        If None, use the rank-revealing QR factorization with eta=2
    return_projection: bool
        If True, return also the matrix U @ pseudoinv(U[S, :])
    return_inverse: bool
        If True, return also the inverse of U[S, :]

    Returns
    -------
    p: list
        Selection of m = k + oversampling_size row indices.
    P_U: ndarray (n x m) (optional)
        Matrix U @ pseudoinv(U[p, :]) where U[p, :] is the (m x k) submatrix.
        Only returned if return_projection=True.
    inv_U: ndarray (k x m) (optional)
        Matrix U.T @ P_U, representing the pseudoinverse relationship.
        Only returned if return_inverse=True (requires return_projection=True).
    """
    # Sanity check
    if oversampling_size < 0:
        raise ValueError("Oversampling size must be positive")
    if oversampling_size > U.shape[1]:
        raise ValueError(
            "Oversampling size must be smaller than the number of columns of U"
        )
    # sQDEIM
    _, k = U.shape
    m = k + oversampling_size
    if tol is None:
        Q, R, P = sRRQR(U.T.conj(), eta=2, mode="rank", param=k)
        p1 = P[:k]
    else:
        Q, R, P = sRRQR(U.T.conj(), eta=2, mode="tol", param=tol)
        k = Q.shape[1]
        p1 = P[:k]

    ## Algorithm 4.2 in reference
    _, _, vt = la.svd(U[p1, :], full_matrices=False)
    Vp = vt.T.conj()[:, k - oversampling_size :]

    # Apply again SQDEIM on the unchosen rows
    # Store the first permutation for mapping back indices
    P_first = P
    P_U_temp = U[P_first[k:], :].dot(Vp)
    if tol is None:
        Q, _, P_second = sRRQR(
            P_U_temp.T.conj(), eta=2, mode="rank", param=oversampling_size
        )
        # Map back to original indices
        p2 = P_first[k:][P_second[:oversampling_size]]
    else:
        Q, _, P_second = sRRQR(P_U_temp.T.conj(), eta=2, mode="tol", param=tol)
        # Map back to original indices
        p2 = P_first[k:][P_second[:oversampling_size]]
    # Concatenate the two selections
    p = np.concatenate((p1, p2))

    if return_projection:
        P_U = la.lstsq(U[p, :].T.conj(), U.T.conj())[0].T.conj()
        if return_inverse:
            inv_U = U.T.conj().dot(P_U)
            return p, P_U, inv_U
        else:
            return p, P_U
    else:
        return p
