"""
Author: Benjamin Carrel, University of Geneva, 2022

Lanczos related functions.
"""

# %% Imports
from __future__ import annotations
import numpy as np
from numpy import ndarray
import scipy.linalg as la
import scipy.sparse as sps
import scipy.sparse.linalg as spsla
from scipy.sparse import spmatrix, diags, csc_matrix

# %% Functions
def Lanczos(A: ndarray | spmatrix, x: ndarray, m: int) -> tuple[ndarray, spmatrix]:
    """Lanczos algorithm. 
    Computes orthogonal basis of a Krylov space:
        K_m(A,x) = span{x, A x, A^2 x, ..., A^(m-1) x}
    where $A$ is a symmetric matrix, and $x$ is a vector.
    If $A$ is non-symmetric, use the Arnoldi algorithm instead.

    Inspired from Martin J. Gander's lecture.

    Parameters
    ----------
    A : ndarray | spmatrix
        Matrix of shape (n,n)
    x : ndarray
        Vector of shape (n,)
    m : int
        Size of the Krylov space

    Returns
    -------
    Q : ndarray
        Matrix of shape (n,m) containing the basis of the Krylov space.
    T : spmatrix
        Tridiagonal matrix of shape (m,m). It is also the projection of A on the Krylov space.
    """
    # Check inputs
    assert isinstance(A, (np.ndarray, spmatrix)), "A must be a numpy array or a scipy sparse matrix"
    assert isinstance(x, np.ndarray), "x must be a numpy array"
    assert A.shape[0] == A.shape[1], "A must be a square matrix"
    # Sanity check
    x = x.reshape(-1)
    assert x.ndim == 1, "x must be a vector"
    assert x.shape[0] == A.shape[0], "x and A must have the same size"
    assert m <= A.shape[0], "The size of the Krylov space is too large"

    # Initialize
    n = A.shape[0]
    Q = np.zeros((n, m), dtype=A.dtype)
    alpha = np.zeros(m, dtype=A.dtype)
    beta = np.zeros(m-1, dtype=A.dtype)
    Q[:, 0] = x / la.norm(x)

    # Lanczos algorithm
    for j in np.arange(m):
        u = A.dot(Q[:, j])
        alpha[j] = Q[:, j].T.dot(u)
        u = u - alpha[j] * Q[:, j]
        if j > 0:
            u = u - beta[j-1] * Q[:, j-1]
        if j < m-1:
            beta[j] = la.norm(u)
            Q[:, j+1] = u / beta[j]
    T = diags([alpha, beta, beta], [0, -1, 1], format='csc')
    return Q, T

# NOTE: rational Lanczos is not implemented since rational Arnoldi is more efficient in general.
# Indeed, rational Lanczos requires two inversion per iteration, while rational Arnoldi requires only one.
# It is, therefore, more expensive when the poles change often.
# Moreover, full orthogonalization is needed when projecting on the Krylov space.
# See "Rational Krylov Methods for Operator Functions", S. Güttel, 2010.


def block_Lanczos(A: ndarray | spmatrix, X: ndarray, m: int) -> tuple[ndarray, spmatrix]:
    """ Block Lanczos algorithm.
    Initialize a Krylov Space where X is a matrix

    Parameters
    ----------
    A : ndarray
        Matrix of shape (n,n)
    X : ndarray
        Matrix of shape (n,r)
    m : int
        Size of the Krylov space

    Returns
    -------
    Q : ndarray
        Matrix of shape (n,m*r) containing the basis of the Krylov space
    T : spmatrix
        Tridiagonal matrix of shape (m*r,m*r). It is also the projection of A on the Krylov space.
    """
    # Check inputs
    assert isinstance(A, (ndarray, spmatrix)), "A must be a numpy array or a scipy sparse matrix"
    assert isinstance(X, ndarray), "X must be a numpy array"
    assert A.shape[0] == A.shape[1], "A must be a square matrix"
    # Sanity check
    if X.ndim != 2:
        raise ValueError("X must be a matrix")
    (n, r) = X.shape
    if m*r > A.shape[0]:
        raise ValueError("The size of the Krylov space is too large")

    # Initialize
    Q = np.zeros((n, m*r), dtype=A.dtype)
    alpha = np.empty(m, dtype=object)
    beta = np.empty(m-1, dtype=object)
    Q[:, :r] = la.orth(X)

    # Block Lanczos algorithm
    for j in np.arange(m):
        u = A.dot(Q[:, j*r:(j+1)*r])
        alpha[j] = Q[:, j*r:(j+1)*r].T.dot(u)
        u = u - Q[:, j*r:(j+1)*r].dot(alpha[j])
        if j > 0:
            u = u - Q[:, (j-1)*r:j*r].dot(beta[j-1].T)
        if j < m-1:
            Q[:, (j+1)*r:(j+2)*r], beta[j] = la.qr(u, mode='economic')

    # Sparse block tridiagonal matrix T
    in_bmat = np.empty((m, m), dtype=object)
    # First row
    in_bmat[0, 0] = alpha[0]
    in_bmat[0, 1] = beta[0].T
    # Middle rows
    for k in np.arange(1, m-1):
        in_bmat[k, k-1] = beta[k-1]
        in_bmat[k, k] = alpha[k]
        in_bmat[k, k+1] = beta[k].T
    # Last row
    in_bmat[m-1, m-2] = beta[m-2]
    in_bmat[m-1, m-1] = alpha[m-1]

    T = sps.bmat(in_bmat, format='csc')

    return Q, T

    
    return Q, T

    # # Reconstruction of the tridiagonal matrix
    # T = np.zeros((m*r, m*r), dtype=A.dtype)
    # for k in np.arange(m):
    #     T[k*r:(k+1)*r, k*r:(k+1)*r] = alpha[k]
    #     if k > 0:
    #         T[k*r:(k+1)*r, (k-1)*r:k*r] = beta[k-1]
    #     if k < m-1:
    #         T[k*r:(k+1)*r, (k+1)*r:(k+2)*r] = beta[k].T
    # T = csc_matrix(T)

# %%
