"""Discrete Empirical Interpolation Method (DEIM).

Author: Benjamin Carrel, University of Geneva, 2024
"""

from typing import Union

import numpy as np
from numpy import ndarray
from scipy import linalg as la


def DEIM(
    U: ndarray,
    return_projector: bool = False,
    return_inverse: bool = False,
    **extra_args,
) -> Union[list, tuple]:
    """
    DEIM - Discrete empirical interpolation method

    Construct a matrix P = [e_p1, e_p2, ..., e_pk]
    where the indexes pi are obtained via the DEIM procedure.

    Reference
        Nonlinear Model Reduction via Discrete Empirical Interpolation
        Saifon Chaturantabut and Danny C. Sorensen
        SIAM Journal on Scientific Computing 2010 32:5, 2737-2764


    Parameters
    ----------
    U: ndarray
        Orthonormal matrix of size n x k
    return_projector: bool
        If True, return also the matrix U @ inv(U[S, :])
    return_inverse: bool
        If True, return also the inverse matrix inv(U[S, :])
    extra_args: dict
        Additional arguments:
            solve_kwargs: dict
                Additional arguments for the solve function

    Returns
    -------
    p: list
        List of indexes selected by DEIM
    P_U: ndarray (n x k) (optional)
        Projector matrix U @ inv(U[S, :])
    inv_U: ndarray (k x k) (optional)
        Inverse matrix inv(U[S, :])
    """

    # Initialisation
    k = U.shape[1]

    p1 = np.argmax(np.abs(U[:, 0]))
    p = [p1]

    # Loop of DEIM
    for i in np.arange(1, k):
        # Solve linear system
        c = np.linalg.solve(U[p, :i], U[p, i], **extra_args.get("solve_kwargs", {}))
        # Compute the residual and extract new max
        r = U[:, i] - U[:, :i].dot(c)
        pi = np.argmax(np.abs(r))
        # Update the indexes
        p += [pi]

    if return_projector:
        P_U = la.solve(U[p, :].T.conj(), U.T.conj()).T.conj()
        if return_inverse:
            inv_U = U.T.conj().dot(P_U)
            return p, P_U, inv_U
        else:
            return p, P_U
    else:
        return p
