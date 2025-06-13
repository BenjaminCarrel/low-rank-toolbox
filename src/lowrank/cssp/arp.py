import numpy as np
import scipy.linalg as la

def _householder_vector(x):
    norm_x = np.linalg.norm(x) # Full norm
    if norm_x < 1e-15:
        return np.zeros_like(x), 0.0 # Zero vector input

    # Standard Householder vector calculation
    # v = x + sign(x[0]) * norm(x) * e1
    alpha = np.copysign(norm_x, x[0])
    v = x.copy()
    v[0] += alpha

    # Normalize v such that v[0] = 1
    v_norm_sq = np.dot(v, v)
    if v_norm_sq < 1e-15:
        return np.zeros_like(x), 0.0 # Should not happen if norm_x > 0

    beta = 2.0 / v_norm_sq
    return v, beta

def _apply_householder_right(A, v, beta):
    v = v.reshape(-1, 1) # Ensure v is n x 1
    # Apply the Householder transformation
    Av = A.dot(v)
    A_updated = A - beta * (Av @ v.T)
    return A_updated


def ARP(U: np.ndarray, compute_M: bool = False, seed: int=None):
    """
    Implements the Adaptive Randomized Pivoting (ARP) algorithm for Column Subset Selection Problem (CSSP).

    Reference: ADAPTIVE RANDOMIZED PIVOTING FOR COLUMN SUBSET SELECTION, DEIM, AND LOW-RANK APPROXIMATION by Cortinovis and Kressner. 

    Note: The algorithm is similar to Osinsky's algorithm for the CSSP. The randomization step allows for better error bounds (in expectation).
    (See Algorithm 2.1)
    
    Parameters
    ----------
    U: ndarray
        An (n x r) matrix with orthonormal columns.
    compute_M: bool
        If True, return also the matrix U @ inv(U[S, :])
    seed: int, optional
        Random seed for reproducibility.

    Returns
    -------
    J: list
        Selection of m column indices.
    M: ndarray (optional)
        Matrix U @ inv(U[S, :])
    """
    # Seed for reproducibility
    if seed is not None:
        np.random.seed(seed)
    
    # Adaptive Randomized Pivoting (ARP) algorithm
    n, r = U.shape
    J = []
    Uk = U.copy()

    for k in range(r):

        # Calculate squared norms of rows for the remaining columns
        row_norms_sq = np.sum(Uk[:, k:r]**2, axis=1)
        total_norm_sq = np.sum(row_norms_sq)
        probs = row_norms_sq / total_norm_sq

        # Sample an index jk according to the probabilities
        jk = np.random.choice(n, p=probs)
        J.append(jk)

        # Householder reflection
        v, beta = _householder_vector(Uk[jk, k:r])
        if beta != 0: # Apply only if reflection is non-trivial
            Uk[:, k:r] = _apply_householder_right(Uk[:, k:r], v, beta)

    if compute_M:
        M = la.lstsq(U[J, :].T.conj(), U.T.conj())[0].T.conj()
        return J, M
    else:
        return J


#%% MY OWN IMPLEMENTATION, WORKS BUT A BIT SLOWER THAN ABOVE FOR LARGE MATRICES


# def ARP(V, compute_M: bool = False, **extra_args):
#     """
#     Implements the Adaptive Randomized Pivoting (ARP) algorithm for Column Subset Selection Problem (CSSP).

#     Reference: ADAPTIVE RANDOMIZED PIVOTING FOR COLUMN SUBSET SELECTION, DEIM, AND LOW-RANK APPROXIMATION by Cortinovis and Kressner. 

#     Note: The algorithm is similar to Osinsky's algorithm for the CSSP. The randomization step allows for better error bounds (in expectation).
#     (See Algorithm 2.1)
    
#     Parameters
#     ----------
#     V: ndarray
#         An (n x r) matrix with orthonormal columns.
#     compute_M: bool
#         If True, return also the matrix V @ inv(V[S, :])

#     Returns
#     -------
#     J: list
#         Selection of m column indices.
#     M: ndarray (optional)
#         Matrix V @ inv(V[S, :])
#     """
#     n = V.shape[0]  # Number of rows
#     r = V.shape[1]  # Number of columns
#     J = []           # Selected column indices
#     Ir = np.eye(r)   # Identity matrix of size r
#     Vk = V.copy()    # Copy of V to modify during iterations
    
#     for k in range(r):
#         # Compute probabilities
#         p = np.linalg.norm(Vk[:, k:], axis=1) ** 2 / (r - k)  # Squared norms of remaining columns
#         # print(p)
        
#         # Sample an index according to probabilities
#         jk = np.random.choice(n, p=p)
#         # Check if the index is already selected
#         while jk in J:
#             jk = np.random.choice(n, p=p)
#         J.append(jk)
        
#         # Construct Householder reflector to annihilate Vk(jk, k+1:r)
#         v = Vk[jk, k:].copy()
#         v[0] -= np.linalg.norm(v)
#         v = v / np.linalg.norm(v) if np.linalg.norm(v) > 1e-12 else v
#         Qk = np.eye(r)
#         Qk[k:, k:] -= 2 * np.outer(v, v)
        
#         # Update V_k
#         Vk = Vk.dot(Qk)

#     if compute_M:
#         M = la.solve(V[J, :].T.conj(), V.T.conj()).T.conj()
#         return J, M
#     else:
#         return J

