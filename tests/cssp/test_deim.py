"""
Test file for DEIM.py

Author: Benjamin Carrel, University of Geneva, 2024
"""

import numpy as np
import pytest
import scipy.linalg as la

from lowrank import DEIM

# ===========================
# FIXTURES
# ===========================


@pytest.fixture
def orthonormal_matrix():
    """Create a random orthonormal matrix."""
    np.random.seed(42)
    A = np.random.randn(20, 5)
    Q, _ = la.qr(A, mode="economic")
    return Q


@pytest.fixture
def complex_orthonormal_matrix():
    """Create a random complex orthonormal matrix."""
    np.random.seed(43)
    A = np.random.randn(30, 7) + 1j * np.random.randn(30, 7)
    Q, _ = la.qr(A, mode="economic")
    return Q


@pytest.fixture
def tall_orthonormal_matrix():
    """Create a tall orthonormal matrix (n >> k)."""
    np.random.seed(44)
    A = np.random.randn(100, 3)
    Q, _ = la.qr(A, mode="economic")
    return Q


@pytest.fixture
def square_orthonormal_matrix():
    """Create a square orthonormal matrix."""
    np.random.seed(45)
    A = np.random.randn(10, 10)
    Q, _ = la.qr(A, mode="economic")
    return Q


# ===========================
# BASIC FUNCTIONALITY TESTS
# ===========================


def test_deim_basic(orthonormal_matrix):
    """Test basic DEIM functionality."""
    Q = orthonormal_matrix
    n, k = Q.shape

    # Test without optional returns
    p = DEIM(Q)

    assert isinstance(p, list), "p should be a list"
    assert len(p) == k, f"Expected {k} indices, got {len(p)}"
    assert len(set(p)) == k, "Indices should be unique"
    assert all(0 <= idx < n for idx in p), "Indices should be in valid range [0, n)"


def test_deim_return_projector(orthonormal_matrix):
    """Test DEIM with return_projector=True."""
    Q = orthonormal_matrix
    n, k = Q.shape

    p, P_U = DEIM(Q, return_projector=True)

    assert isinstance(p, list), "p should be a list"
    assert isinstance(P_U, np.ndarray), "P_U should be a numpy array"
    assert P_U.shape == (n, k), f"P_U shape should be ({n}, {k}), got {P_U.shape}"
    assert len(p) == k, f"Expected {k} indices, got {len(p)}"

    # Verify that P_U = U @ inv(U[p, :])
    U_p = Q[p, :]
    inv_U_p = la.inv(U_p)
    expected_P_U = Q @ inv_U_p

    assert np.allclose(P_U, expected_P_U, rtol=1e-10), "P_U != U @ inv(U[p, :])"


def test_deim_return_both(orthonormal_matrix):
    """Test DEIM with both return_projector=True and return_inverse=True."""
    Q = orthonormal_matrix
    n, k = Q.shape

    p, P_U, inv_U = DEIM(Q, return_projector=True, return_inverse=True)

    assert isinstance(p, list), "p should be a list"
    assert isinstance(P_U, np.ndarray), "P_U should be a numpy array"
    assert isinstance(inv_U, np.ndarray), "inv_U should be a numpy array"
    assert P_U.shape == (n, k), f"P_U shape should be ({n}, {k})"
    assert inv_U.shape == (k, k), f"inv_U shape should be ({k}, {k})"

    # Verify inv_U = inv(U[p, :])
    U_p = Q[p, :]
    expected_inv_U = la.inv(U_p)

    assert np.allclose(inv_U, expected_inv_U, rtol=1e-10), "inv_U != inv(U[p, :])"

    # Verify relationship: P_U = U @ inv_U
    assert np.allclose(P_U, Q @ inv_U, rtol=1e-10), "P_U != U @ inv_U"


def test_deim_interpolation_property(orthonormal_matrix):
    """Test that DEIM satisfies the interpolation property."""
    Q = orthonormal_matrix
    k = Q.shape[1]

    p = DEIM(Q)
    U_p = Q[p, :]

    # Check that U[p, :] is invertible
    det = np.linalg.det(U_p)
    assert abs(det) > 1e-10, f"U[p,:] should be invertible, det={det}"

    # The selected submatrix should have reasonable condition number
    cond = np.linalg.cond(U_p)
    assert cond < 1e12, f"Condition number {cond} is too large"


def test_deim_greedy_selection(orthonormal_matrix):
    """Test that DEIM follows the greedy selection process."""
    Q = orthonormal_matrix
    k = Q.shape[1]

    p = DEIM(Q)

    # First index should be max of first column
    expected_p1 = np.argmax(np.abs(Q[:, 0]))
    assert p[0] == expected_p1, f"First index should be {expected_p1}, got {p[0]}"

    # Indices should be selected greedily
    for i in range(1, k):
        # Verify the greedy choice was made correctly
        c = np.linalg.solve(Q[p[:i], :i], Q[p[:i], i])
        r = Q[:, i] - Q[:, :i] @ c
        expected_pi = np.argmax(np.abs(r))
        assert p[i] == expected_pi, f"Index {i} should be {expected_pi}, got {p[i]}"


# ===========================
# COMPLEX MATRIX TESTS
# ===========================


def test_deim_complex(complex_orthonormal_matrix):
    """Test DEIM with complex matrices."""
    Q = complex_orthonormal_matrix
    n, k = Q.shape

    p = DEIM(Q)

    assert len(p) == k, f"Expected {k} indices for complex matrix"
    assert len(set(p)) == k, "Complex matrix indices should be unique"

    # Test with projector
    p, P_U = DEIM(Q, return_projector=True)

    U_p = Q[p, :]
    inv_U_p = la.inv(U_p)
    expected_P_U = Q @ inv_U_p

    assert np.allclose(P_U, expected_P_U, rtol=1e-10), "Complex P_U != U @ inv(U[p, :])"


def test_deim_complex_return_inverse(complex_orthonormal_matrix):
    """Test DEIM with complex matrices and return_inverse=True."""
    Q = complex_orthonormal_matrix

    p, P_U, inv_U = DEIM(Q, return_projector=True, return_inverse=True)

    U_p = Q[p, :]
    expected_inv_U = la.inv(U_p)

    assert np.allclose(
        inv_U, expected_inv_U, rtol=1e-10
    ), "Complex inv_U != inv(U[p, :])"
    assert np.allclose(P_U, Q @ inv_U, rtol=1e-10), "Complex P_U != U @ inv_U"


# ===========================
# EDGE CASES
# ===========================


def test_deim_rank_one():
    """Test DEIM with rank-1 matrix."""
    np.random.seed(46)
    v = np.random.randn(15, 1)
    Q, _ = la.qr(v, mode="economic")

    p = DEIM(Q)

    assert len(p) == 1, "Rank-1 matrix should return 1 index"
    assert 0 <= p[0] < 15, "Index should be valid"

    # The selected row should be the maximum absolute value
    expected_idx = np.argmax(np.abs(Q[:, 0]))
    assert p[0] == expected_idx, f"Should select max abs value index {expected_idx}"


def test_deim_square_matrix(square_orthonormal_matrix):
    """Test DEIM with square orthonormal matrix (n = k)."""
    Q = square_orthonormal_matrix
    n, k = Q.shape
    assert n == k, "Should be square"

    p = DEIM(Q)

    assert len(p) == k, f"Expected {k} indices"
    assert len(set(p)) == k, "All indices should be unique"

    # For a square orthonormal matrix, U[p, :] should be well-conditioned
    U_p = Q[p, :]
    cond_number = np.linalg.cond(U_p)
    assert cond_number < 1e12, f"Square matrix condition number {cond_number} too large"


def test_deim_tall_matrix(tall_orthonormal_matrix):
    """Test DEIM with very tall matrix (n >> k)."""
    Q = tall_orthonormal_matrix
    n, k = Q.shape
    assert n > 10 * k, "Should be very tall"

    p = DEIM(Q)

    assert len(p) == k, f"Expected {k} indices"
    assert len(set(p)) == k, "All indices should be unique"
    assert all(0 <= idx < n for idx in p), "Indices in valid range"


def test_deim_rank_two():
    """Test DEIM with rank-2 matrix to verify greedy algorithm."""
    np.random.seed(47)
    A = np.random.randn(25, 2)
    Q, _ = la.qr(A, mode="economic")

    p = DEIM(Q)

    assert len(p) == 2, "Should return 2 indices"
    assert p[0] != p[1], "Indices should be different"

    # Verify first index is max of first column
    assert p[0] == np.argmax(np.abs(Q[:, 0]))


# ===========================
# ORTHOGONALITY VERIFICATION
# ===========================


def test_deim_input_orthogonality_check(orthonormal_matrix):
    """Verify input matrix is actually orthonormal."""
    Q = orthonormal_matrix

    # Check orthonormality
    eye_k = np.eye(Q.shape[1])
    assert np.allclose(Q.T @ Q, eye_k, rtol=1e-10), "Input should be orthonormal"


def test_deim_preserves_span(orthonormal_matrix):
    """Test that the selected rows preserve the span via interpolation."""
    Q = orthonormal_matrix
    k = Q.shape[1]

    p, P_U = DEIM(Q, return_projector=True)

    # For any vector in span(Q), interpolation at p should reconstruct it
    test_vec = Q @ np.random.randn(k)
    reconstructed = P_U @ test_vec[p]

    assert np.allclose(
        test_vec, reconstructed, rtol=1e-10
    ), "Interpolation should preserve span"


# ===========================
# EXTRA_ARGS TESTS
# ===========================


def test_deim_with_solve_kwargs(orthonormal_matrix):
    """Test DEIM with extra solve arguments."""
    Q = orthonormal_matrix

    # numpy.linalg.solve doesn't have extra kwargs, so just verify it doesn't break
    # Test with empty solve_kwargs
    p1, P_U1 = DEIM(Q, return_projector=True, solve_kwargs={})
    p2, P_U2 = DEIM(Q, return_projector=True)

    # Results should be identical for indices (greedy algorithm)
    assert p1 == p2, "Empty solve_kwargs shouldn't change greedy selection"
    # Projectors should be identical
    assert np.allclose(
        P_U1, P_U2, rtol=1e-10
    ), "Empty solve_kwargs shouldn't change projector"


# ===========================
# NUMERICAL STABILITY TESTS
# ===========================


def test_deim_condition_number(orthonormal_matrix):
    """Test that the condition number of U[p, :] is reasonable."""
    Q = orthonormal_matrix
    n, k = Q.shape

    p = DEIM(Q)
    U_p = Q[p, :]

    # Compute smallest singular value
    sigma_min = la.svdvals(U_p)[-1]

    # sigma_min should not be too small
    assert sigma_min > 1e-12, f"Smallest singular value {sigma_min} is too small"


def test_deim_reconstruction_accuracy(orthonormal_matrix):
    """Test reconstruction accuracy using DEIM interpolation."""
    Q = orthonormal_matrix
    k = Q.shape[1]

    p, P_U = DEIM(Q, return_projector=True)

    # Test reconstruction of each column
    for i in range(k):
        col = Q[:, i]
        reconstructed = P_U @ col[p]
        error = la.norm(col - reconstructed)

        # Since Q is orthonormal, reconstruction should be exact
        assert error < 1e-9, f"Reconstruction error {error} too large for column {i}"


def test_deim_matrix_inversion_accuracy(orthonormal_matrix):
    """Test accuracy of the matrix inversion in DEIM."""
    Q = orthonormal_matrix

    p, P_U, inv_U = DEIM(Q, return_projector=True, return_inverse=True)

    U_p = Q[p, :]

    # Check that inv_U * U_p = I
    identity_check = inv_U @ U_p
    eye_k = np.eye(Q.shape[1])

    assert np.allclose(
        identity_check, eye_k, rtol=1e-10
    ), "inv_U @ U[p,:] should be identity"


# ===========================
# ERROR HANDLING
# ===========================


def test_deim_non_orthonormal_input():
    """Test DEIM with non-orthonormal input (should still work but may not be optimal)."""
    np.random.seed(48)
    # Create a non-orthonormal matrix
    A = np.random.randn(20, 5)

    # DEIM should still run
    p = DEIM(A)

    assert len(p) == 5, "Should return k indices even for non-orthonormal input"


def test_deim_return_inverse_without_projector():
    """Test that return_inverse=True without return_projector=True returns only indices."""
    np.random.seed(49)
    A = np.random.randn(20, 5)
    Q, _ = la.qr(A, mode="economic")

    # return_inverse=True but return_projector=False
    result = DEIM(Q, return_inverse=True)

    # Should only return indices
    assert isinstance(result, list), "Should return only indices as a list"


# ===========================
# REPRODUCIBILITY TEST
# ===========================


def test_deim_reproducibility():
    """Test that DEIM gives consistent results for the same input."""
    np.random.seed(50)
    A = np.random.randn(25, 6)
    Q, _ = la.qr(A, mode="economic")

    p1 = DEIM(Q)
    p2 = DEIM(Q)

    assert p1 == p2, "DEIM should be deterministic"


# ===========================
# COMPARISON TESTS
# ===========================


def test_deim_indices_differ_from_first_k():
    """Test that DEIM doesn't just return [0, 1, 2, ..., k-1]."""
    np.random.seed(51)
    A = np.random.randn(30, 5)
    Q, _ = la.qr(A, mode="economic")

    p = DEIM(Q)

    # Should not just be [0, 1, 2, 3, 4]
    # (This could fail with very low probability for certain random seeds)
    assert p != list(
        range(5)
    ), "DEIM should select indices based on greedy algorithm, not sequential"


def test_deim_vs_qdeim_both_work():
    """Test that both DEIM and QDEIM work on the same input."""
    from lowrank import QDEIM

    np.random.seed(52)
    A = np.random.randn(40, 6)
    Q, _ = la.qr(A, mode="economic")

    p_deim = DEIM(Q)
    p_qdeim = QDEIM(Q)

    # Both should return 6 indices
    assert len(p_deim) == 6, "DEIM should return 6 indices"
    assert len(p_qdeim) == 6, "QDEIM should return 6 indices"

    # Both should produce invertible submatrices
    U_p_deim = Q[p_deim, :]
    U_p_qdeim = Q[p_qdeim, :]

    assert np.linalg.matrix_rank(U_p_deim) == 6, "DEIM submatrix should be full rank"
    assert np.linalg.matrix_rank(U_p_qdeim) == 6, "QDEIM submatrix should be full rank"


# ===========================
# DOCUMENTATION EXAMPLE TEST
# ===========================


def test_deim_documentation_example():
    """Test example that could be in documentation."""
    # Create a simple orthonormal basis
    np.random.seed(100)
    A = np.random.randn(50, 8)
    U, _ = la.qr(A, mode="economic")

    # Basic usage
    p = DEIM(U)
    assert len(p) == 8, "Should select 8 rows"
    assert isinstance(p, list), "Should return list of indices"

    # With projector
    p, P_U = DEIM(U, return_projector=True)
    assert P_U.shape == (50, 8), "Projector should be n x k"

    # Verify interpolation property
    test_vector = U @ np.random.randn(8)
    interpolated = P_U @ test_vector[p]
    assert np.allclose(
        test_vector, interpolated, rtol=1e-10
    ), "Should interpolate exactly"

    # With both outputs
    p, P_U, inv_U = DEIM(U, return_projector=True, return_inverse=True)
    assert inv_U.shape == (8, 8), "Inverse should be k x k"
    assert np.allclose(P_U, U @ inv_U, rtol=1e-10), "Relationship should hold"

    print("DEIM documentation example test passed!")


# ===========================
# ALGORITHM CORRECTNESS TEST
# ===========================


def test_deim_algorithm_steps():
    """Test the DEIM algorithm step by step to verify correctness."""
    np.random.seed(53)
    n, k = 15, 3
    A = np.random.randn(n, k)
    U, _ = la.qr(A, mode="economic")

    # Manual DEIM computation
    p_manual = []

    # Step 1: First index is max of first column
    p1 = np.argmax(np.abs(U[:, 0]))
    p_manual.append(p1)

    # Step 2: Second index
    c = np.linalg.solve(U[[p1], :1], U[[p1], 1])
    r = U[:, 1] - U[:, :1] @ c
    p2 = np.argmax(np.abs(r))
    p_manual.append(p2)

    # Step 3: Third index
    c = np.linalg.solve(U[p_manual, :2], U[p_manual, 2])
    r = U[:, 2] - U[:, :2] @ c
    p3 = np.argmax(np.abs(r))
    p_manual.append(p3)

    # Compare with DEIM function
    p_deim = DEIM(U)

    assert p_manual == p_deim, f"Manual DEIM {p_manual} != function DEIM {p_deim}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
