import scipy.linalg as la


def gpodr(U, 
          oversampling_size: int = None, 
          tol: float = None, 
          max_iter: int = None, 
          return_projector: bool = False, 
          return_inverse: bool = False,
          **extra_args):
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
    n, k = U.shape
    _, _, P = la.qr(U.T.conj(), pivoting=True, **extra_args.get('qr_kwargs', {}))
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
    if return_projector:
        P_U = la.lstsq(U[p, :].T.conj(), U.T.conj(), **extra_args.get('lstsq_kwargs', {}))[0].T.conj()
        if return_inverse:
            inv_U = U.T.conj().dot(P_U)
            return p, P_U, inv_U
        else:
            return p, P_U
    else:
        return p
