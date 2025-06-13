"""
File for DEIM related functions

Author: Benjamin Carrel, University of Geneva
"""

#%% Importations
import numpy as np
from numpy import ndarray
from scipy import linalg as la
from .sRRQR import sRRQR, sRRQR_rank, sRRQR_tol

#%% Original DEIM
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
        # # M as a solver
        # class M:
        #     def __init__(self, U, p):
        #         self.U = U
        #         self.p = p

        #     def dot(self, x):
        #         return self.U.dot(la.solve(self.U[p, :], x))
        return p, M
    else:
        return p


def QDEIM(U: ndarray, compute_M: bool = False, **extra_args) -> list:
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
    compute_M: bool
        If True, return also the matrix U @ inv(U[S, :])

    Returns
    -------
    p: list
        Selection of m row indices with guaranteed upper bound: norm(inv(U[S,:])) <= sqrt(n-k+1) * O(2^m).
    M: ndarray (optional)
        Matrix U @ inv(U[S, :])
    """
    # Initialisation
    _, k = U.shape
    (_, R, P) = la.qr(U.T.conj(), pivoting=True)
    p = P[0:k]
    if compute_M:
        L = la.solve(R[:, :k], R[:, k:]).T.conj()
        M = np.row_stack((np.eye(k), L))
        Q = np.argsort(P)
        M = M[Q, :]
        return p, M
    else:
        return p
    
def sQDEIM(U: ndarray, compute_M: bool = False, **extra_args) -> list:
    """
    sQDEIM - Strong RRQR based DEIM of U (size n x k)

    Key advantage: the selection of the indexes is guaranteed to satisfy the condition:
    sigma_{min}(U[p, :])^{-1} <= sqrt(1 + f * r (n-k))
    
    By default, f = 2

    Parameters
    ----------
    U: ndarray
        Orthonormal matrix of size n x k
    tol: float
        Tolerance for the quantity sigma_{min}(U[p, :])^{-1}
    compute_M: bool
        If True, return also the matrix U @ inv(U[S, :])

    Returns
    -------
    p: list
        Selection of m row indices with guaranteed upper bound: norm(inv(U[S,:])) <= sqrt(n-k+1) * O(2^m).
    M: ndarray (optional)
        Matrix U @ inv(U[S, :])
    """
    # Initialisation
    _, k = U.shape
    p = sRRQR_rank(U.T.conj(), 2, k) [2][:k]
    if compute_M:
        M = la.solve(U[p, :].T.conj(), U.T.conj()).T.conj()
        return p, M
    else:
        return p

def gpode(U, oversampling_size: int, compute_M: bool = False, **extra_args) -> list:
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
        Oversampling size l such that m = k + l
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
    m = k + oversampling_size
    _, _, P = la.qr(U.T.conj(), pivoting=True)
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
    if compute_M:
        M = la.lstsq(U[p, :].T.conj(), U.T.conj())[0].T.conj()
        return p, M
    else:
        return p

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
    

def osinsky_arp_cssp(V, compute_M: bool = False, **extra_args):
    """
    Implements Osinsky's Adaptive Randomized Pivoting (ARP) algorithm for Column Subset Selection Problem (CSSP).

    Reference: ADAPTIVE RANDOMIZED PIVOTING FOR COLUMN SUBSET SELECTION, DEIM, AND LOW-RANK APPROXIMATION by Cortinovis and Kressner. 
    
    Parameters:
        V (numpy.ndarray): An (n x r) matrix with orthonormal columns.
        compute_M (bool): If True, return the matrix V @ inv(V[S, :]).
        
    Returns:
        J (list): Indices of the selected columns.
    """
    n = V.shape[0]  # Number of rows
    r = V.shape[1]  # Number of columns
    J = []           # Selected column indices
    Ir = np.eye(r)   # Identity matrix of size r
    Vk = V.copy()    # Copy of V to modify during iterations
    
    for k in range(r):
        # Compute probabilities
        p = np.linalg.norm(Vk[:, k:], axis=1) ** 2 / (r - k)  # Squared norms of remaining columns
        # print(p)
        
        # Sample an index according to probabilities
        jk = np.random.choice(n, p=p)
        # Check if the index is already selected
        while jk in J:
            jk = np.random.choice(n, p=p)
        J.append(jk)
        
        # Construct Householder reflector to annihilate Vk(jk, k+1:r)
        v = Vk[jk, k:].copy()
        v[0] -= np.linalg.norm(v)
        v = v / np.linalg.norm(v) if np.linalg.norm(v) != 0 else v
        Qk = np.eye(r)
        Qk[k:, k:] -= 2 * np.outer(v, v)
        
        # Update V_k
        Vk = Vk.dot(Qk)

    if compute_M:
        M = la.solve(V[J, :].T.conj(), V.T.conj()).T.conj()
        return J, M
    else:
        return J



def osinsky_cssp(A, V, compute_M: bool = False, **extra_args):
    """
    Osinsky's deterministic algorithm for Column Subset Selection Problem (CSSP)
    
    Parameters:
    V : numpy.ndarray
        Orthonormal matrix of shape (n, r) defining a row space approximation
    
    Returns:
    J : list
        Selected column indices
    """
    n, r = V.shape
    J = []
    A_tilde = A - A.dot(V).dot(V.T) # Compute \tilde{A}_0
    Vk = V.copy()
    
    for k in range(r):
        # Compute index that minimizes the given ratio
        norms_A = np.linalg.norm(A_tilde, axis=0) ** 2  # Column-wise squared norm
        norms_V = np.linalg.norm(Vk[:, k:], axis=1) ** 2  # Row-wise squared norm
        # print(norms_A / norms_V)
        # For loop to avoid division by zero
        score = np.ones(n) * np.inf
        for j in range(n):
            if j in J:
                continue
            elif norms_V[j] == 0:
                score[j] = np.inf
            else:
                score[j] = norms_A[j] / norms_V[j]
        jk = np.argmin(score)
        J.append(jk)
        
        # Construct Householder reflector of the j_k-th row to annihilate the j_k-th row of V_k
        v = Vk[jk, k:].copy()
        v[0] -= np.linalg.norm(v)
        v = v / np.linalg.norm(v) if np.linalg.norm(v) != 0 else v
        Qk = np.eye(r)
        Qk[k:, k:] -= 2 * np.outer(v, v)
        
        # Update Vk
        Vk = Vk.dot(Qk)
        # print('V_k=', Vk)
        
        # Update A_tilde
        A_tilde -= np.outer(A_tilde[:, jk], Vk[:, k]) / Vk[jk, k] if Vk[jk, k] != 0 else 1
        # print('A_tilde=', A_tilde)

    if compute_M:
        M = la.solve(V[J, :].T.conj(), V.T.conj()).T.conj()
        return J, M
    
    return J

