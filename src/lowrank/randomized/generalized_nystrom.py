
import numpy as np
from scipy.sparse.linalg import LinearOperator
from ..matrices.quasi_svd import QuasiSVD
from ..matrices.svd import SVD
from ..matrices.low_rank_matrix import LowRankMatrix


def generalized_nystrom(X: LinearOperator, 
                        r: int, 
                        oversampling_params: tuple = (10, 15), 
                        epsilon: float = None,
                        seed: int = 1234,
                        **extra_data) -> QuasiSVD:
    """General Nystroem method

    Reference
    "Fast and stable randomized low-rank matrix approximation"
    Nakatsukasa, 2019

    X ~= X J (K^T X J)^{dagger} K^T X

    Parameters
    ----------
    X: LinearOperator
        Matrix to approximate
    r: int
        Rank of approximation
    oversamplings: tuple
        Oversampling parameters (p1, p2) for the two sketch matrices
    epsilon: float (default is None)
        When given, perform stable GN with epsilon-pseudoinverse

    Returns
    -------
    QuasiSVD
        Near optimal best rank r approximation of X in QuasiSVD format
        
    Notes
    -----
    This method returns a QuasiSVD (not SVD) because the middle matrix S
    is typically inverted, making it non-diagonal. Convert to SVD if needed:
        result = QuasiSVD.generalized_nystroem(X, r).to_svd()
    """
    m, n = X.shape
    p1, p2 = oversampling_params
    # Draw the two random matrices
    np.random.seed(seed)
    J= np.random.randn(n, r + p1)
    K = np.random.randn(m, r + p2)

    # Compute the factors
    if isinstance(X, LowRankMatrix):
        XJ = X.dot(J, dense_output=True)
        KtX = X.dot(K.T, side='left', dense_output=True)
    else:
        XJ = X.dot(J)
        KtX = K.T.dot(X)
    KtXJ = KtX.dot(J)
    
    # Compute SVD of middle term and truncate for stable version
    if epsilon is None:
        C = SVD.truncated_svd(KtXJ, r=r)
    else:
        C = SVD.truncated_svd(KtXJ, rtol=epsilon)

    # Return in QuasiSVD format
    U = XJ.dot(C.V)
    S = np.linalg.inv(C.S)
    V = (C.U.T.dot(KtX)).T

    return QuasiSVD(U, S, V, **extra_data)
