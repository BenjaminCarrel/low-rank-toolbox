import scipy.linalg as la


def gpodr(U, oversampling_size:int = None, tol: float = None, max_iter: int = None, compute_M: bool = False, **extra_args):
    """
    Gappy POD+R - QDEIM and randomized oversampling.

    When tol is given, then p is ignored and the algorithm will select the number of rows to satisfy the condition:
    sigma_{min}(U[p, :])^{-1} <= tol
    
    Reference
        Stability of discrete empirical interpolation and gappy proper orthogonal decomposition with randomized and deterministic sampling points
        Benjamin Peherstorfer, Zlatko Drmač, and Serkan Gugercin
        SIAM Journal on Scientific Computing, 42(5), A2837-A2864.
        
    Parameters
    ----------
    U: ndarray
        Orthonormal matrix of size n x k
    oversampling_size: int
        Oversampling size 
    tol: float
        Tolerance for the quantity sigma_{min}(U[p, :])^{-1}
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
    n, k = U.shape
    _, _, P = la.qr(U.T.conj(), pivoting=True)
    p = P[0:k]
    # With tol
    if tol is not None:
        # Compute SVD
        i = 1
        s = la.svdvals(U[p, :])
        # print(1/s[-1], i)
        if max_iter is None:
            max_iter = n - k
        while 1/s[-1] > tol and i <= max_iter:
            # Add next index from P
            p = P[0:k+i]
            i += 1
            s = la.svdvals(U[p, :])
    else:
        # With l
        if oversampling_size is None:
            oversampling_size = k # Default value
        p = P[0:k+oversampling_size]
    if compute_M:
        M = la.lstsq(U[p, :].T.conj(), U.T.conj())[0].T.conj()
        return p, M
    else:
        return p
