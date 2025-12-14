"""Adaptive Randomized Pivoting (ARP) algorithm.

Author: Benjamin Carrel, University of Geneva, 2024
"""

import numpy as np
import scipy.linalg as la

def _householder_vector(x):
    """
    Compute Householder reflection vector.
    
    Parameters
    ----------
    x : ndarray
        Input vector
        
    Returns
    -------
    v : ndarray
        Householder vector
    beta : float
        Scaling factor (2 / ||v||^2)
    """
    norm_x = np.linalg.norm(x) # Full norm
    if norm_x < 1e-15:
        return np.zeros_like(x), 0.0 # Zero vector input

    # Householder vector computation
    if x[0] == 0:
        alpha = norm_x
    else:
        alpha = np.sign(x[0]) * norm_x
    
    v = x.copy()
    v[0] -= alpha

    # Normalize v using complex dot product
    v_norm_sq = np.linalg.norm(v)**2
    if v_norm_sq < 1e-15:
        return np.zeros_like(x), 0.0 # Should not happen if norm_x > 0

    beta = 2.0 / v_norm_sq
    return v, beta

def _apply_householder_right(A, v, beta):
    """
    Apply Householder transformation from the right: A @ (I - beta * v * v^H).
    
    Parameters
    ----------
    A : ndarray
        Matrix to transform
    v : ndarray
        Householder vector
    beta : float
        Scaling factor
        
    Returns
    -------
    A_updated : ndarray
        Transformed matrix
    """
    v = v.reshape(-1, 1) # Ensure v is n x 1
    # Apply the Householder transformation
    Av = A.dot(v)
    A_updated = A - beta * (Av @ v.T.conj())
    return A_updated


def ARP(U: np.ndarray, seed: int = None, return_projector: bool = False, return_inverse: bool = False, **extra_args) -> np.ndarray:
    """
    Implements the Adaptive Randomized Pivoting (ARP) algorithm for Column Subset Selection Problem (CSSP).

    Reference: ADAPTIVE RANDOMIZED PIVOTING FOR COLUMN SUBSET SELECTION, DEIM, AND LOW-RANK APPROXIMATION by Cortinovis and Kressner. 

    Note: The algorithm is similar to Osinsky's algorithm for the CSSP. The randomization step allows for better error bounds (in expectation).
    (See Algorithm 2.1)
    
    Parameters
    ----------
    U: ndarray
        An (n x r) matrix with orthonormal columns.
    seed: int, optional
        Random seed for reproducibility. 
    return_projector: bool, optional
        If True, return also the matrix U @ inv(U[S, :])
    return_inverse: bool, optional
        If True, return also the inverse matrix inv(U[S, :])
        
    Returns
    -------
    J: ndarray
        Selection of r row indices.
    P_U: ndarray (n x r) (optional)
        Matrix U @ inv(U[J, :]) where U[J, :] is the (r x r) submatrix.
        Only returned if return_projector=True.
    inv_U: ndarray (r x r) (optional)
        Matrix inv(U[J, :]).
        Only returned if return_inverse=True (requires return_projector=True).
    """
    # Seed for reproducibility
    if seed is not None:
        np.random.seed(seed)
    
    # Adaptive Randomized Pivoting (ARP) algorithm
    n, r = U.shape
    if r > n:
        raise ValueError("Number of columns r must be less than or equal to number of rows n.")
    J = []
    Uk = U.copy()

    for k in range(r):

        # Calculate squared norms of rows for the remaining columns
        row_norms_sq = np.sum(np.abs(Uk[:, k:r])**2, axis=1)
        
        # Set probabilities to zero for already selected indices
        for idx in J:
            row_norms_sq[idx] = 0.0
            
        total_norm_sq = np.sum(row_norms_sq)
        
        # Avoid division by zero
        if total_norm_sq < 1e-15:
            # If all remaining norms are negligible, pick remaining indices arbitrarily
            remaining_indices = [i for i in range(n) if i not in J]
            jk = remaining_indices[0] if remaining_indices else 0
        else:
            probs = row_norms_sq / total_norm_sq
            # Sample an index jk according to the probabilities
            jk = np.random.choice(n, p=probs)
            
        J.append(jk)

        # Householder reflection to zero out the jk-th row from column k onwards
        x = Uk[jk, k:r].copy()  # Extract the row vector to be zeroed out
        v, beta = _householder_vector(x.conj())
        if beta != 0: # Apply only if reflection is non-trivial
            Uk[:, k:r] = _apply_householder_right(Uk[:, k:r], v, beta)

    J = np.array(J)  # Convert to ndarray
    
    if return_projector:
        M = la.lstsq(U[J, :].T.conj(), U.T.conj())[0].T.conj()
        if return_inverse:
            inv_U = U.T.conj().dot(M)
            return J, M, inv_U
        else:
            return J, M
    return J
    

if __name__ == "__main__":
    # Example usage - real case
    np.random.seed(0)
    ## Tall matrix with orthonormal columns
    print("Real case -- Tall matrix with orthonormal columns")
    n, r = 10, 4
    U = np.random.randn(n, r)
    U, _ = la.qr(U, mode='economic')

    J = ARP(U, return_projector=False, seed=42)
    print("Selected indices J:", J)
    print("U[J, :]:\n", U[J, :])

    # Example usage - complex case
    print("\nComplex case -- Tall matrix with orthonormal columns")
    U_complex = np.random.randn(n, r) + 1j * np.random.randn(n, r)
    U_complex, _ = la.qr(U_complex, mode='economic')
    J_complex = ARP(U_complex, return_projector=False, seed=42)
    print("Selected indices J (complex case):", J_complex)
    print("U[J, :] (complex case):\n", U_complex[J_complex, :])