import numpy as np
from numpy import ndarray
import scipy.linalg as la

def OCSS(U: ndarray, compute_M: bool = False) -> list:
    """
    Osinsky's OCSS algorithm for row selection via QR decomposition.

    Reference:
        "Close to optimal column approximations with a single SVD." by A.I. Osinsky, 2023.

    Parameters
    ----------
    U : numpy.ndarray
        Orthonormal real matrix of shape (n, r) defining a row space approximation
    compute_M : bool
        If True, return also the matrix U @ inv(U[S, :])

    Returns
    -------
    J : list
        Selected column indices
    M : numpy.ndarray (optional)
        Matrix U @ inv(U[S, :])
    """
    n, r = U.shape
    A = U.T.copy()
    P = np.arange(n) 
    l_scores = np.zeros(n)
    eps = np.finfo(np.float64).eps # Machine epsilon for float64

    for k in range(r):
        # --- Pivot Selection ---
        current_sub_A = A[k:r, k:n] 
        # Use **2 for real matrices instead of np.abs()**2
        norms_sq = np.sum(current_sub_A**2, axis=0) 
        current_l = l_scores[k:n]
        
        # Criterion: norms_sq / (1 + l_j), handle small norms/denominators
        denominators = 1.0 + current_l
        pivot_vals = np.zeros_like(denominators)
        # Check against epsilon^2 for squared norms? Or just eps? Use eps for norm check.
        valid_mask = norms_sq > eps 
        pivot_vals[valid_mask] = norms_sq[valid_mask] / denominators[valid_mask]
            
        best_idx_in_slice = np.argmax(pivot_vals)
        j_pivot = k + best_idx_in_slice

        # --- Swap ---
        if k != j_pivot:
            A[:, [k, j_pivot]] = A[:, [j_pivot, k]]
            P[k], P[j_pivot] = P[j_pivot], P[k]
            l_scores[k], l_scores[j_pivot] = l_scores[j_pivot], l_scores[k]
            
        # --- Householder Reflection ---
        x = A[k:r, k].copy()
        norm_x = np.linalg.norm(x)
        d_update = A[k, k:n].copy() # Initialize update scores from current row

        if norm_x > eps: 
            alpha = x[0]
            v = x 
            
            # Real Householder update: v[0] = x[0] + sign(x[0]) * ||x||
            # np.copysign handles sign(0) returning 1 (or -1 depending on impl)
            # If alpha is exactly 0, copysign(norm_x, 0.0) returns norm_x.
            v[0] = alpha + np.copysign(norm_x, alpha if alpha != 0 else 1.0)
            
            norm_v = np.linalg.norm(v)

            if norm_v > eps:
                v /= norm_v
                # Apply reflection: A_sub = A_sub - 2 * v * (v.T @ A_sub)
                sub_matrix_to_update = A[k:r, k:n]
                # Use v.T @ ... for dot product with real vector v
                vT_Asub = v.T.dot(sub_matrix_to_update)
                A[k:r, k:n] -= 2 * np.outer(v, vT_Asub)
                d_update = A[k, k:n].copy() # Get updated row slice

        # --- Update Scores ---
        # Use **2 for real scores update
        l_scores[k:n] += d_update**2 
    # --- End Loop ---

    if compute_M:
        M = la.solve(U[P[:r], :].T.conj(), U.T.conj()).T.conj()
        return P[:r], M
    return P[:r]


# def OCSS(U, compute_M=False):
#     """
#     Osinsky's OCSS algorithm for row selection via QR decomposition.

#     Reference:
#     "Close to optimal column approximations with a single SVD." by A.I. Osinsky, 2023.

                    
#     """
#     n, r = U.shape
#     P = np.arange(n)
#     l_scores = np.zeros(n, dtype=np.float64)

#     for k in range(r):
        
#         # --- Pivot Selection (Step 3) ---
#         # Consider columns from index k to n-1.
        
#         # Calculate squared norms of sub-columns A[k:r, j] for j >= k
#         # Slice A to get the relevant submatrix for norms and Householder
#         current_sub_A_for_norms = A[k:r, k:n] 
        
#         # norms_sq will have length n-k
#         norms_sq = np.sum(np.abs(current_sub_A_for_norms)**2, axis=0)

#         # Get the scores l_j for j >= k
#         current_l = l_scores[k:n]
        
#         # Calculate pivot criterion: norms_sq / (1 + l_j)
#         # Handle division by zero (denominator >= 1) and norm_sq = 0 safely
#         pivot_vals = np.zeros(n - k, dtype=np.float64)
#         # Avoid calculation for columns with effectively zero norm in the submatrix
#         valid_indices = norms_sq > 1e-14 # Indices within the norms_sq array

#         if np.any(valid_indices):
#              denominators = 1.0 + current_l[valid_indices]
#              # Ensure denominator isn't zero (shouldn't happen as l>=0)
#              denominators[denominators < 1e-14] = 1e-14 
#              pivot_vals[valid_indices] = norms_sq[valid_indices] / denominators
        
#         # Find the index within the *slice* k:n that maximizes the criterion
#         # If all pivot_vals are 0 or negative (unlikely), argmax returns 0
#         best_idx_in_slice = np.argmax(pivot_vals)
        
#         # The actual column index in the full matrix A
#         j_pivot = k + best_idx_in_slice

#         # --- Swap Columns (Step 4) ---
#         if k != j_pivot:
#             # Swap columns in A
#             A[:, [k, j_pivot]] = A[:, [j_pivot, k]]
#             # Swap corresponding permutation indices
#             P[k], P[j_pivot] = P[j_pivot], P[k]
#             # Swap corresponding scores
#             l_scores[k], l_scores[j_pivot] = l_scores[j_pivot], l_scores[k]
            
#         # --- Householder Reflection (Steps 9-12) ---
#         # Vector to apply Householder to (k-th column, from row k down)
#         # Use A[:, k] after swap, slice from row k
#         x = A[k:r, k].copy() # Important: use copy()
        
#         norm_x = np.linalg.norm(x)

#         if norm_x < 1e-14: 
#             # If the column segment is zero, skip reflection (already zeroed)
#             # Scores still need update based on A[k, k:n] (which might be non-zero)
#              d_update = A[k, k:n].copy() # Elements for score update
#         else:
#             alpha = x[0]
#             # Compute Householder vector v (normalized)
#             # Follows logic of steps 9-11 for stable calculation
#             v = x # Start with x
            
#             v_0_update = 0.0 # Default value
#             if np.abs(alpha) < 1e-14: 
#                  # alpha is zero, standard choice is ||x|| * e^i*0 = ||x||
#                  v_0_update = norm_x 
#             else:
#                 # alpha is non-zero
#                 if dtype == np.complex128:
#                     # Complex case: alpha + exp(i*arg(alpha)) * ||x||
#                     v_0_update = alpha + np.exp(1j * np.angle(alpha)) * norm_x
#                 else: # Real case: alpha + sign(alpha) * ||x||
#                     v_0_update = alpha + np.copysign(norm_x, alpha) 

#             v[0] = v_0_update
            
#             # Normalize v (Step 11)
#             norm_v = np.linalg.norm(v)
#             if norm_v < 1e-14:
#                  # Should not happen if norm_x > 0, but safety check
#                  print(f"Warning: Householder vector norm is near zero at step k={k}")
#                  d_update = A[k, k:n].copy() # No reflection applied
#             else:
#                 v /= norm_v # Normalize v in place
                
#                 # Apply reflection H = I - 2vv* to A[k:r, k:n] (Step 12)
#                 # Efficient calculation: A_sub = A_sub - 2 * v * (v* @ A_sub)
#                 sub_matrix_to_update = A[k:r, k:n] # View for update
                
#                 # v_star_dot_sub = v.conj().T @ sub_matrix_to_update # Row vector
#                 # np.outer(v, v_star_dot_sub) # Outer product
#                 # A[k:r, k:n] -= 2 * np.outer(v, v_star_dot_sub) # Update A
                
#                 # Alternative way using einsum for potentially better clarity/performance
#                 # A[k:r, k:n] -= 2 * np.einsum('i,j->ij', v, np.einsum('i,ik->k', v.conj(), sub_matrix_to_update))

#                 # Standard way using matmul (@) and outer
#                 vH_Asub = v.conj().T @ sub_matrix_to_update # 1 x (n-k) row vector
#                 A[k:r, k:n] -= 2 * np.outer(v, vH_Asub) # Subtract rank-1 update

#                 # The elements used for score update are from the k-th row AFTER reflection
#                 d_update = A[k, k:n].copy() # Copy the updated row slice

#         # --- Update Scores (Steps 6-8) ---
#         # Update l_j for j >= k based on |A_{k,j}|^2 after reflection (like R_kj)
#         l_scores[k:n] += np.abs(d_update)**2

#     # --- End Loop ---

#     if compute_M:
#         M = la.solve(U[P[:r], :].T.conj(), U.T.conj()).T.conj()
#         return P[:r], M

#     # The first r elements of P contain the indices of the selected columns of A
#     # which correspond to the selected rows of V.
#     return P[:r]

# # --- Example Usage ---
# # Since you have a background in applied math, let's use a slightly more
# # structured example where row selection might be non-trivial.

# n_rows = 10
# r_rank = 4

# # Construct V with some structure: first r rows are dominant, others decay
# # Use SVD for controlled construction
# np.random.seed(42)
# U_full, _ = np.linalg.qr(np.random.randn(n_rows, n_rows))
# Sigma_vals = np.zeros(n_rows)
# Sigma_vals[:r_rank] = 1.0 # Orthonormal columns basis

# # Create a V where the first r rows essentially form an identity block
# # (scaled/rotated), and other rows have smaller contributions.
# # We want the algorithm to ideally pick the first r rows.
# temp_V = U_full[:, :r_rank] * Sigma_vals[:r_rank] # n x r

# # Let's make the first r rows more distinct by scaling them
# scaling_factors = np.linspace(2, 1, r_rank) # e.g., [2, 1.66, 1.33, 1]
# temp_V[:r_rank, :] *= scaling_factors[:, np.newaxis] 

# # Now, make the columns orthonormal using QR decomposition
# V_example, _ = np.linalg.qr(temp_V)

# print("--- Example Usage ---")
# print(f"Constructed V: n={n_rows}, r={r_rank}")
# # print("Input matrix V (n x r, orthonormal columns):")
# # print(V_example)
# # print(f"Shape: {V_example.shape}")
# print("Check V.T @ V (should be close to identity):")
# print(np.round(V_example.T @ V_example, 5)) 

# try:
#     selected_indices = select_rows_via_qr_pivot(V_example)
#     print("\nSelected row indices:", selected_indices) # Expect something like [0, 1, 2, 3]

#     # Verify the selected submatrix 
#     V_selected = V_example[selected_indices, :]
#     print("\nSelected submatrix V_hat (V[selected_indices, :]):")
#     # print(np.round(V_selected, 3))
    
#     # Check condition number if needed (requires scipy)
#     # from scipy.linalg import svd
#     # _, s, _ = svd(V_selected)
#     # print(f"\nCondition number of V_hat: {s[0]/s[-1]:.2f}")

# except ValueError as e:
#     print(f"\nError during execution: {e}")

# # --- Simple Test Case from Thought Process ---
# V_simple = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
# print("\n--- Simple Test Case ---")
# print("V = \n", V_simple)
# selected_indices_simple = select_rows_via_qr_pivot(V_simple)
# print("Selected row indices (simple case):", selected_indices_simple) # Expect [0, 1]

# # --- Complex Test Case ---
# n_rows_c = 5
# r_rank_c = 3
# np.random.seed(123)
# Zc = np.random.rand(n_rows_c, r_rank_c) + 1j * np.random.rand(n_rows_c, r_rank_c)
# Vc, _ = np.linalg.qr(Zc)
# print("\n--- Complex Test Case ---")
# print(f"Constructed Vc: n={n_rows_c}, r={r_rank_c}")
# print("Check Vc.conj().T @ Vc (should be close to identity):")
# print(np.round(Vc.conj().T @ Vc, 5))
# try:
#     selected_indices_c = select_rows_via_qr_pivot(Vc)
#     print("\nSelected row indices (complex case):", selected_indices_c) 
# except ValueError as e:
#      print(f"\nError during execution: {e}")

# def OCSS(V, compute_M: bool = False, **extra_args):
#     """
#     Osinsky's deterministic algorithm for Optimal Column Subset Selection (OCSS)

#     Reference: Close to optimal column approximations with a single SVD, A.I. Osinsky
#     (See Algorithm 2)
    
#     Parameters:
#     V : numpy.ndarray
#         Orthonormal matrix of shape (n, r) defining a row space approximation
    
#     Returns:
#     J : list
#         Selected column indices
#     """
#     Vk = V.copy()
#     n, r = V.shape
#     l = np.zeros(n)
#     J = []
#     for k in np.arange(r):
#         # Norm of each row, length n
#         scores = (1 + l) / la.norm(V[:, k:], axis=1)
#         j = np.argmin(scores[k:])
#         J.append(j)
#         # Swap k-th and j-th rows of V
#         Vk[[k, j], :] = Vk[[j, k], :]
#         d = Vk[k, k:r].dot(Vk[:, k:r].T) / la.norm(Vk[k, k:r])
#         for i in np.arange(n):
#             l[i] += d[i] ** 2
#         v = Vk[k, k:r].copy()
#         v[0] -= np.sign(v[0]) * la.norm(v)
#         v = v / la.norm(v) if la.norm(v) != 0 else v
#         Vk[:, k:r] -= 2 * np.outer(Vk[:, k:r].dot(v.T), v)

#     if compute_M:
#         M = la.solve(V[J, :].T.conj(), V.T.conj()).T.conj()
#         return J, M
#     else:
#         return J

# # def osinsky_cssp(A: ndarray, V: ndarray, compute_M: bool = False, **extra_args):
# #     """
# #     Osinsky's deterministic algorithm for Column Subset Selection Problem (CSSP)

# #     References: 
# #     1. Close to optimal column approximations with a single SVD, A.I. Osinsky
# #     (See Algorithm 1)
# #     2. ADAPTIVE RANDOMIZED PIVOTING FOR COLUMN SUBSET SELECTION, DEIM, AND LOW-RANK APPROXIMATION, Cortinovis and Kressner.
# #     (See Algorithm 2.2)
    
# #     Parameters
# #     ----------
# #     V : numpy.ndarray
# #         Orthonormal matrix of shape (n, r) defining a row space approximation
# #     compute_M : bool
# #         If True, return also the matrix V @ inv(V[S, :])
    
# #     Returns
# #     -------
# #     J : list
# #         Selected column indices of A
# #     M : numpy.ndarray (optional)
# #         Matrix V @ inv(V[S, :])
# #     """
# #     n, r = V.shape
# #     J = []
# #     A_tilde = A - A.dot(V).dot(V.T) # Compute \tilde{A}_0
# #     Vk = V.copy()
    
# #     for k in range(r):
# #         # Compute index that minimizes the given ratio
# #         norms_A = np.linalg.norm(A_tilde, axis=0) ** 2  # Column-wise squared norm
# #         norms_V = np.linalg.norm(Vk[:, k:], axis=1) ** 2  # Row-wise squared norm
# #         # print(norms_A / norms_V)
# #         # For loop to avoid division by zero
# #         score = np.ones(n) * np.inf
# #         for j in range(n):
# #             if j in J:
# #                 continue
# #             elif norms_V[j] == 0:
# #                 score[j] = np.inf
# #             else:
# #                 score[j] = norms_A[j] / norms_V[j]
# #         jk = np.argmin(score)
# #         J.append(jk)
        
# #         # Construct Householder reflector of the j_k-th row to annihilate the j_k-th row of V_k
# #         v = Vk[jk, k:].copy()
# #         v[0] -= np.linalg.norm(v)
# #         v = v / np.linalg.norm(v) if np.linalg.norm(v) != 0 else v
# #         Qk = np.eye(r)
# #         Qk[k:, k:] -= 2 * np.outer(v, v)
        
# #         # Update Vk
# #         Vk = Vk.dot(Qk)
# #         # print('V_k=', Vk)
        
# #         # Update A_tilde
# #         A_tilde -= np.outer(A_tilde[:, jk], Vk[:, k]) / Vk[jk, k] if Vk[jk, k] != 0 else 1
# #         # print('A_tilde=', A_tilde)

# #     if compute_M:
# #         M = la.solve(V[J, :].T.conj(), V.T.conj()).T.conj()
# #         return J, M
    
# #     return J

