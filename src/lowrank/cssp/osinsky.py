import numpy as np
from numpy import ndarray
import scipy.linalg as la

def Osinsky(U: ndarray, compute_M: bool = False, return_Uk: bool = False) -> list:
    """
    Osinsky's quasi optimal column subset selection algorithm.

    Reference:
        "Close to optimal column approximations with a single SVD." by A.I. Osinsky, 2023.

    Parameters
    ----------
    U : numpy.ndarray
        Orthonormal real matrix of shape (n, r) defining a row space approximation
    compute_M : bool
        If True, return also the matrix U @ inv(U[S, :])
    return_Uk : bool
        If True, return also the modified matrix Uk after applying Householder reflections.

    Returns
    -------
    J : list
        Selected column indices
    M : numpy.ndarray (optional)
        Matrix U @ inv(U[S, :])
    """
    n, r = U.shape
    if r > n:
        raise ValueError("Number of columns r must be less than or equal to number of rows n.")
    Uk = U.T.copy()
    P = np.arange(n) 
    l_scores = np.zeros(n)
    eps = np.finfo(np.float64).eps # Machine epsilon for float64

    for k in range(r):
        # --- Pivot Selection ---
        current_sub_Uk = Uk[k:r, k:n] 
        norms_sq = np.linalg.norm(current_sub_Uk, axis=0)**2
        current_l = l_scores[k:n]
        
        # Criterion: norms_sq / (1 + l_j), handle small norms/denominators
        denominators = 1.0 + current_l
        pivot_vals = np.zeros_like(denominators)
        valid_mask = norms_sq > eps 
        pivot_vals[valid_mask] = norms_sq[valid_mask] / denominators[valid_mask]
            
        best_idx_in_slice = np.argmax(pivot_vals)
        j_pivot = k + best_idx_in_slice

        # --- Swap ---
        if k != j_pivot:
            Uk[:, [k, j_pivot]] = Uk[:, [j_pivot, k]]
            P[k], P[j_pivot] = P[j_pivot], P[k]
            l_scores[k], l_scores[j_pivot] = l_scores[j_pivot], l_scores[k]
            
        # --- Householder Reflection ---
        x = Uk[k:r, k].copy()
        norm_x = np.linalg.norm(x)
        d_update = Uk[k, k:n].copy() # Initialize update scores from current row

        if norm_x > eps: 
            alpha = x[0]
            v = x
            v[0] = alpha - np.sign(x[0]) * norm_x
            norm_v = np.linalg.norm(v)

            if norm_v > eps:
                v /= norm_v
                # Apply reflection: A_sub = A_sub - 2 * v * (v.H @ A_sub)
                sub_matrix_to_update = Uk[k:r, k:n]
                vT_Asub = v.T.conj().dot(sub_matrix_to_update)
                Uk[k:r, k:n] -= 2 * np.outer(v, vT_Asub)
                d_update = Uk[k, k:n].copy()

        # --- Update Scores ---
        # Use **2 for real scores update
        l_scores[k:n] += np.abs(d_update)**2  # CHANGED to absolute value for complex matrices
        #l_scores[k:n] += d_update**2  # CHANGED to absolute value for complex matrices
    # --- End Loop ---

    if compute_M:
        M = la.solve(U[P[:r], :].T.conj(), U.T.conj()).T.conj()
        if return_Uk:
            return P[:r], M, Uk
        return P[:r], M
    if return_Uk:
        return P[:r], Uk
    return P[:r]

if __name__ == "__main__":
    # Example usage - real case
    np.random.seed(0)
    ## Tall matrix with orthonormal columns
    print("Real case -- Tall matrix with orthonormal columns")
    n, r = 10, 4
    U = np.random.randn(n, r)
    U, _ = la.qr(U, mode='economic')

    J, Uk = Osinsky(U, compute_M=False, return_Uk=True)
    print("Selected indices J:", J)
    print("Matrix U:\n", Uk)

    # Example usage - complex case
    print("\nComplex case -- Tall matrix with orthonormal columns")
    U_complex = np.random.randn(n, r) + 1j * np.random.randn(n, r)
    U_complex, _ = la.qr(U_complex, mode='economic')

    J_complex, Uk_complex = Osinsky(U_complex, compute_M=False, return_Uk=True)
    print("Selected indices J (complex case):", J_complex)
    print("Matrix U (complex case):\n", Uk_complex)


