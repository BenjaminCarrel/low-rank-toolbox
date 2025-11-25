import numpy as np
from numpy import ndarray
from scipy import linalg as la


def DEIM(U: ndarray, compute_M: bool = False, **extra_args) -> ndarray:
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
    compute_M: bool
        If True, return also the matrix U @ inv(U[S, :])
    
    Returns
    -------
    p: list
        List of indexes selected by DEIM
    M: ndarray (optional)
        Matrix U @ inv(U[S, :])
    """

    # Initialisation
    k = U.shape[1]

    p1 = np.argmax(np.abs(U[:, 0]))
    p = [p1]

    # Loop of DEIM
    for i in np.arange(1, k):
        # Solve linear system
        c =  np.linalg.solve(U[p, :i], U[p, i])
        # Compute the residual and extract new max
        r = U[:, i] - U[:, :i].dot(c)
        pi = np.argmax(np.abs(r))
        # Update the indexes
        p += [pi]

    if compute_M:
        M = la.solve(U[p, :].T.conj(), U.T.conj()).T.conj()
        return p, M
    else:
        return p