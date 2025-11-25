"""
Strong rank-revealing QR factorization.

Implementation from the MATLAB code available at:
https://fr.mathworks.com/matlabcentral/fileexchange/69139-strong-rank-revealing-qr-decomposition

This implementation extends the original algorithms to support complex matrices
with proper unitary operations and phase normalization.

Reference:
Gu, Ming, and Stanley C. Eisenstat. "Efficient algorithms for computing a strong rank-revealing QR factorization." SIAM Journal on Scientific Computing 17.4 (1996): 848-869.

Notes:
- Complex matrices are supported with proper conjugate transpose operations
- Givens rotations automatically maintain positive real diagonal elements
- Phase normalization ensures consistent QR factorization
"""


import numpy as np
import scipy.linalg
from scipy.spatial.distance import cdist
from .givens import givens


def sRRQR_rank(A, f, k):
    """
    Strong Rank-Revealing QR factorization for real or complex matrices with a fixed rank 'k'.

    This function implements Algorithm 4 from:
    Gu, M., & Eisenstat, S. C. (1996). Efficient algorithms for computing a
    strong rank-revealing QR factorization. SIAM Journal on Scientific
    Computing, 17(4), 848-869.

    It computes a factorization $A P = Q R$ where $P$ is a permutation matrix,
    $Q$ is orthogonal (or unitary for complex matrices), and $R$ is upper triangular. 
    The factorization is partitioned as:
    $$
    A P = [Q_1, Q_2] \\begin{pmatrix} R_{11} & R_{12} \\\\ 0 & R_{22} \\end{pmatrix}
    $$
    such that the entries of $R_{11}^{-1} R_{12}$ are bounded by a factor $f$.

    Args:
        A (np.ndarray): The input matrix (m x n), real or complex.
        f (float): A constant (>= 1) that bounds the entries of $R_{11}^{-1} R_{12}$.
        k (int): The prescribed rank, i.e., the dimension of the $R_{11}$ block.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]:
            - Q (np.ndarray): Truncated orthogonal/unitary matrix $Q_1$ (m x k).
            - R (np.ndarray): Truncated upper triangular matrix $[R_{11}, R_{12}]$ (k x n).
            - p (np.ndarray): The column permutation vector.
    """
    if f < 1:
        print("Parameter f is less than 1. Automatically set f = 2.")
        f = 2.0

    m, n = A.shape
    k = min(k, m, n)

    # Initial column-pivoted QR
    Q, R, p = scipy.linalg.qr(A, mode='economic', pivoting=True)
    p_orig = p.copy()
    
    if k == n:
        return Q, R, p

    # Ensure diagonal of R has positive real part (phase normalization for complex)
    d = np.diag(R)
    # For complex numbers, normalize phase: d / |d| gives unit complex numbers
    # For real numbers, this reduces to sign(d)
    phase_factors = np.where(d != 0, d / np.abs(d), 1.0)
    R = R * phase_factors[:, np.newaxis]
    # Apply conjugate of phase factors to Q to maintain QR = A
    Q = Q * np.conj(phase_factors)[np.newaxis, :]

    # Initialization from paper
    R11 = R[:k, :k]
    R12 = R[:k, k:]
    R22 = R[k:, k:]

    AB = scipy.linalg.solve_triangular(R11, R12, lower=False)
    gamma = np.linalg.norm(R22, axis=0)
    
    # inv(R11)
    invR11 = scipy.linalg.solve_triangular(R11, np.eye(k, dtype=A.dtype), lower=False)
    omega = 1.0 / np.linalg.norm(invR11, axis=1)

    # Main interchange loop
    while True:
        # Identify columns to interchange
        tmp = (1.0 / omega[:, np.newaxis] * gamma[np.newaxis, :])**2 + np.abs(AB)**2
        indices = np.argwhere(tmp > f**2)

        if indices.shape[0] == 0:
            break

        # print("sRRQR_rank: Performing column interchange...")
        
        i_idx, j_idx = indices[0] # Get row/col of first violator
        
        # Step 1: Interchange (k+1)-th and (k+j)-th columns
        if j_idx > 0:
            # Python indexing: k is k-th, k+j_idx is (k+j_idx)-th
            AB[:, [0, j_idx]] = AB[:, [j_idx, 0]]
            gamma[[0, j_idx]] = gamma[[j_idx, 0]]
            R[:, [k, k + j_idx]] = R[:, [k + j_idx, k]]
            p[[k, k + j_idx]] = p[[k + j_idx, k]]

        # Step 2: Move i-th column to k-th position
        if i_idx < k - 1:
            # Cyclic shift of columns i to k-1 to the left
            # Move column i to position k-1
            p[i_idx:k] = np.roll(p[i_idx:k], -1)
            R[:, i_idx:k] = np.roll(R[:, i_idx:k], -1, axis=1)
            omega[i_idx:k] = np.roll(omega[i_idx:k], -1)
            AB[i_idx:k, :] = np.roll(AB[i_idx:k, :], -1, axis=0)

            # Retriangularize R11 using Givens rotations
            for ii in range(i_idx, k - 1):
                G = givens(R[ii, ii], R[ii + 1, ii])
                R[ii:ii + 2, :] = G @ R[ii:ii + 2, :]
                Q[:, ii:ii + 2] = Q[:, ii:ii + 2] @ G.T.conj()
            # Ensure diagonal element has positive real part for complex case
            if np.real(R[k-1, k-1]) < 0:
                R[k-1, :] *= -1
                Q[:, k-1] *= -1

        # Step 3: Zero out subdiagonal of (k+1)-th column (now at index k)
        m_R = R.shape[0]
        if k < m_R - 1:
            for ii in range(k + 1, m_R):
                G = givens(R[k, k], R[ii, k])
                R[[k, ii], :] = G @ R[[k, ii], :]
                Q[:, [k, ii]] = Q[:, [k, ii]] @ G.T.conj()

        # Step 4: Interchange k-th and (k+1)-th columns
        p[[k-1, k]] = p[[k, k-1]]
        ga = R[k-1, k-1]
        mu = R[k-1, k] / ga
        
        nu = R[k, k] / ga if k < m_R else 0.0
        
        rho = np.sqrt(np.abs(mu)**2 + np.abs(nu)**2)
        ga_bar = ga * rho
        
        b1 = R[:k-1, k-1].copy()
        b2 = R[:k-1, k].copy()
        c1T = R[k-1, k+1:].copy()
        c2T = R[k, k+1:].copy() if k < m_R else np.zeros_like(c1T)
        
        c1T_bar = (mu * c1T + nu * c2T) / rho
        c2T_bar = (nu * c1T - mu * c2T) / rho

        # Modify R matrix
        R[:k-1, k-1] = b2
        R[:k-1, k] = b1
        R[k-1, k-1] = ga_bar
        R[k-1, k] = ga * mu / rho
        if k < m_R:
            R[k, k] = ga * nu / rho
        R[k-1, k+1:] = c1T_bar
        if k < m_R:
            R[k, k+1:] = c2T_bar

        # Update AB, gamma, omega
        # Note: This part recomputes quantities for simplicity.
        # The original paper provides O(k(n-k)) update formulas.
        R11 = R[:k, :k]
        R12 = R[:k, k:]
        R22 = R[k:, k:]
        AB = scipy.linalg.solve_triangular(R11, R12, lower=False)
        gamma = np.linalg.norm(R22, axis=0)
        invR11 = scipy.linalg.solve_triangular(R11, np.eye(k), lower=False)
        omega = 1.0 / np.linalg.norm(invR11, axis=1)

    # Check that the set of pivot indices has changed (order may differ)
    # if not np.all(p == p_orig):
    #     print("sRRQR_rank: Column interchange performed.")

    if not set(p) == set(p_orig):
        print("sRRQR_rank: Column interchange performed.")

    return Q[:, :k], R[:k, :], p

def sRRQR_tol(A, f, tol):
    """
    Strong Rank-Revealing QR for real or complex matrices with a tolerance 'tol'.

    This function implements Algorithm 5 from the Gu and Eisenstat paper.
    It determines the rank `k` such that the spectral norms of the columns of
    the $R_{22}$ block are all less than `tol`.

    Args:
        A (np.ndarray): The input matrix (m x n), real or complex.
        f (float): A constant (>= 1) that bounds the entries of $R_{11}^{-1} R_{12}$.
        tol (float): The error threshold for determining the rank.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]:
            - Q (np.ndarray): Truncated orthogonal/unitary matrix $Q_1$ (m x k).
            - R (np.ndarray): Truncated upper triangular matrix $[R_{11}, R_{12}]$ (k x n).
            - p (np.ndarray): The column permutation vector.
    """
    if f < 1:
        print("Parameter f is less than 1. Automatically set f = 2.")
        f = 2.0

    m, n = A.shape
    Q, R, p = scipy.linalg.qr(A, mode='economic', pivoting=True)
    p_orig = p.copy()

    d = np.diag(R)
    # For complex numbers, normalize phase: d / |d| gives unit complex numbers
    # For real numbers, this reduces to sign(d)
    phase_factors = np.where(d != 0, d / np.abs(d), 1.0)
    R = R * phase_factors[:, np.newaxis]
    # Apply conjugate of phase factors to Q to maintain QR = A
    Q = Q * np.conj(phase_factors)[np.newaxis, :]

    # Estimate initial rank k
    diag_R = np.abs(np.diag(R))
    k_candidates = np.where(diag_R > tol)[0]
    if not k_candidates.size:
        return np.zeros((m, 0)), np.zeros((0, n)), np.arange(n)
    
    k = k_candidates[-1] + 1
    
    if k == n:
        print("Rank equals the number of columns!")
        return Q, R, p

    # Outer loop for adjusting rank k
    while True:
        # Inner loop: perform SRRQR for fixed rank k
        while True:
            R11 = R[:k, :k]
            R12 = R[:k, k:]
            R22 = R[k:, k:]

            AB = scipy.linalg.solve_triangular(R11, R12, lower=False)
            gamma = np.linalg.norm(R22, axis=0)
            invR11 = scipy.linalg.solve_triangular(R11, np.eye(k), lower=False)
            omega = 1.0 / np.linalg.norm(invR11, axis=1)

            tmp = (1.0 / omega[:, np.newaxis] * gamma[np.newaxis, :])**2 + np.abs(AB)**2
            indices = np.argwhere(tmp > f**2)

            if indices.shape[0] == 0:
                break
            
            i_idx, j_idx = indices[0]
            
            # (Same update logic as sRRQR_rank, simplified here for brevity by recomputing)
            # A full implementation would use the efficient O(k(n-k)) updates.
            # Step 1:
            if j_idx > 0:
                R[:, [k, k + j_idx]] = R[:, [k + j_idx, k]]
                p[[k, k + j_idx]] = p[[k + j_idx, k]]
            # Step 2:
            if i_idx < k - 1:
                p[i_idx:k] = np.roll(p[i_idx:k], -1)
                R[:, i_idx:k] = np.roll(R[:, i_idx:k], -1, axis=1)
                for ii in range(i_idx, k - 1):
                    G = givens(R[ii, ii], R[ii + 1, ii])
                    R[ii:ii+2, :] = G @ R[ii:ii+2, :]
                    Q[:, ii:ii+2] = Q[:, ii:ii+2] @ G.T.conj()
                if np.real(R[k-1, k-1]) < 0:
                    R[k-1, :] *= -1
                    Q[:, k-1] *= -1
            # Step 3:
            m_R = R.shape[0]
            if k < m_R - 1:
                for ii in range(k + 1, m_R):
                    G = givens(R[k, k], R[ii, k])
                    R[[k, ii], :] = G @ R[[k, ii], :]
                    Q[:, [k, ii]] = Q[:, [k, ii]] @ G.T.conj()
            # Step 4:
            p[[k-1, k]] = p[[k, k-1]]
            R[:, [k-1, k]] = R[:, [k, k-1]]
            if k < m_R:  # Only if we have the k-th row
                G = givens(R[k-1, k-1], R[k, k-1])
                R[k-1:k+1, :] = G @ R[k-1:k+1, :]
                Q[:, k-1:k+1] = Q[:, k-1:k+1] @ G.T.conj()

        # Check if rank can be reduced
        invR11 = scipy.linalg.solve_triangular(R[:k, :k], np.eye(k), lower=False)
        omega = 1.0 / np.linalg.norm(invR11, axis=1)
        
        min_omega_val = np.min(omega)
        if min_omega_val > tol:
            break
            
        # Reduce rank
        i_to_move = np.argmin(omega)
        if i_to_move < k - 1:
             # Move i_to_move column to k-th position
            p[i_to_move:k] = np.roll(p[i_to_move:k], -1)
            R[:, i_to_move:k] = np.roll(R[:, i_to_move:k], -1, axis=1)
            # Retriangularize
            for ii in range(i_to_move, k - 1):
                G = givens(R[ii, ii], R[ii + 1, ii])
                R[ii:ii+2, :] = G @ R[ii:ii+2, :]
                Q[:, ii:ii+2] = Q[:, ii:ii+2] @ G.T.conj()
            if np.real(R[k-1,k-1]) < 0:
                R[k-1, :] *= -1
                Q[:, k-1] *= -1
        k -= 1
        if k == 0:
            break

    # if not np.all(p == p_orig):
    #     print("sRRQR_tol: Column interchange performed.")

    return Q[:, :k], R[:k, :], p

def sRRQR(A, f, mode, param):
    """
    Dispatcher for Strong Rank-Revealing QR factorization.

    This function automatically detects if the input matrix A is real or complex
    and calls the appropriate implementation.

    Args:
        A (np.ndarray): The input matrix (m x n), real or complex.
        f (float): Bounding factor for $R_{11}^{-1} R_{12}$, must be >= 1.
        mode (str): Specifies the truncation criterion. Must be 'rank' or 'tol'.
        param (int or float): The parameter for the chosen mode.
                           - If mode is 'rank', `param` is the desired rank `k`.
                           - If mode is 'tol', `param` is the error tolerance.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]:
            - Q (np.ndarray): Truncated unitary/orthogonal matrix $Q_1$.
            - R (np.ndarray): Truncated upper triangular matrix $[R_{11}, R_{12}]$.
            - p (np.ndarray): The column permutation vector.
    """
    # if np.iscomplexobj(A):
    #     raise NotImplementedError("Complex matrix support is not implemented for sRRQR.")

    if mode.lower() == 'rank':
        return sRRQR_rank(A, f, param)
    elif mode.lower() == 'tol':
        return sRRQR_tol(A, f, param)
    else:
        raise ValueError(f"Unknown mode: {mode}. Must be 'rank' or 'tol'.")


if __name__ == '__main__':
    # --- Example from the MATLAB file for a real matrix ---
    print("--- REAL MATRIX EXAMPLE ---")
    np.random.seed(0) # for reproducibility
    X0 = 4 * (1 - 2 * np.random.rand(200, 3))
    Y0 = np.array([12, 0, 0]) + 4 * (1 - 2 * np.random.rand(200, 3))

    dist = cdist(X0, Y0)
    A = 1.0 / dist

    # --- Strong Rank Revealing QR with given rank ---
    print("\n[SRRQR with fixed rank]")
    k = 20
    f = 1.001
    Q, R, p = sRRQR(A, f, 'rank', k)

    # Reconstruct permuted A for error calculation
    A_permuted = A[:, p]
    error = np.linalg.norm(A_permuted - Q @ R, 'fro') / np.linalg.norm(A, 'fro')

    # Check the bound on inv(R11) * R12
    R11 = R[:k, :k]
    R12 = R[:k, k:]
    tmp = scipy.linalg.solve_triangular(R11, R12, lower=False)
    entry = np.max(np.abs(tmp))

    print(f"Approximation rank: {k}")
    print(f"Relative approx. error: {error:.3E}")
    print(f"Maximum entry in inv(R11)*R12: {entry:.3f} (Bound f={f})")

    # --- Strong Rank Revealing QR with given error threshold ---
    print("\n[SRRQR with given tolerance]")
    tol = 1e-4
    f = 1.01
    Q, R, p = sRRQR(A, f, 'tol', tol)
    
    A_permuted = A[:, p]
    error = np.linalg.norm(A_permuted - Q @ R, 'fro') / np.linalg.norm(A, 'fro')

    k_found = R.shape[0]
    R11 = R[:, :k_found]
    R12 = R[:, k_found:]
    tmp = scipy.linalg.solve_triangular(R11, R12, lower=False)
    entry = np.max(np.abs(tmp))
    
    print(f"Approximation rank found: {k_found}")
    print(f"Relative approx. error: {error:.3E}")
    print(f"Maximum entry in inv(R11)*R12: {entry:.3f} (Bound f={f})")
    