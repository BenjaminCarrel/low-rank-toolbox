

from scipy.sparse.linalg import LinearOperator
from scipy import linalg as la
import numpy as np
from ..matrices.svd import SVD
from ..matrices.low_rank_matrix import LowRankMatrix


def randomized_svd(X: LinearOperator, r: int, p: int = 10, nb_subspace_iter: int = 0, **extra_data) -> SVD:
    """Randomized SVD algorithm.
    
    Reference
    ---------
    "Finding structure with randomness: Probabilistic algorithms for constructing approximate matrix decompositions",
    Halko, Martinsson and Tropp 2010.
    
    Error bounds
    ------------

    Parameters
    ----------
    X : SVD | LowRankMatrix | ndarray
        Matrix or operator to be decomposed
    r : int
        Rank of the decomposition
    p : int
        Oversampling parameter
    
    Returns
    -------
    SVD
        Near-optimal best rank r approximation of X
    """
    _, n = X.shape
    # Draw the random matrix
    np.random.seed(123)
    Omega = np.random.randn(n, r + p)

    # Step 1: find the range of X
    Y = X.dot(Omega)
    if isinstance(Y, LowRankMatrix): # support for low-rank matrices
        Y = Y.todense()
    Q, _ = la.qr(Y, mode='economic')

    # Subspace iteration
    for _ in range(nb_subspace_iter):
        Y = X.T.conj().dot(Q)
        if isinstance(Y, LowRankMatrix):
            Y = Y.todense()
        Q, _ = la.qr(Y, mode='economic')
        Y = X.dot(Q)
        if isinstance(Y, LowRankMatrix):
            Y = Y.todense()
        Q, _ = la.qr(Y, mode='economic')
    
    # Randomized SVD routine
    if isinstance(X, LowRankMatrix): # support for low-rank matrices
        C = X.dot(Q.T.conj(), side='left')
    else:
        C = Q.T.conj().dot(X)

    Xr = cls.truncated_svd(C, r, **extra_data)
    Xr.U = Q.dot(Xr.U)
    return Xr