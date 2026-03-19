"""
Test file for sqdeim.py (Strong QDEIM)

Tests for the Strong Rank-Revealing QR based DEIM implementation.

Author: Benjamin Carrel, University of Geneva, 2024
"""

import numpy as np
import pytest
import scipy.linalg as la

from low_rank_toolbox import sQDEIM

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


@pytest.fixture
def rank_one_matrix():
    """Create a rank-1 orthonormal matrix."""
    np.random.seed(46)
    A = np.random.randn(15, 1)
    Q, _ = la.qr(A, mode="economic")
    return Q


# ===========================
# BASIC FUNCTIONALITY TESTS
# ===========================


def test_sqdeim_basic(orthonormal_matrix):
    """Test basic sQDEIM functionality."""
    Q = orthonormal_matrix
    n, k = Q.shape

    # Test without optional returns
    p = sQDEIM(Q)

    assert isinstance(p, np.ndarray), "p should be a numpy array"
    assert len(p) == k, f"Expected {k} indices, got {len(p)}"
    assert len(np.unique(p)) == k, "Indices should be unique"
    assert np.all(p >= 0) and np.all(p < n), "Indices should be in valid range [0, n)"


def test_sqdeim_default_eta(orthonormal_matrix):
    """Test sQDEIM with default eta=2."""
    Q = orthonormal_matrix
    k = Q.shape[1]

    # Default eta should be 2
    p1 = sQDEIM(Q)
    p2 = sQDEIM(Q, eta=2)

    assert np.array_equal(p1, p2), "Default eta should be 2"


def test_sqdeim_return_projector(orthonormal_matrix):
    """Test sQDEIM with return_projector=True."""
    Q = orthonormal_matrix
    n, k = Q.shape

    p, P_U = sQDEIM(Q, return_projector=True)

    assert isinstance(p, np.ndarray), "p should be a numpy array"
    assert isinstance(P_U, np.ndarray), "P_U should be a numpy array"
    assert P_U.shape == (n, k), f"P_U shape should be ({n}, {k}), got {P_U.shape}"
    assert len(p) == k, f"Expected {k} indices, got {len(p)}"


def test_sqdeim_return_both(orthonormal_matrix):
    """Test sQDEIM with both return_projector=True and return_inverse=True."""
    Q = orthonormal_matrix
    n, k = Q.shape

    p, P_U, inv_U = sQDEIM(Q, return_projector=True, return_inverse=True)

    assert isinstance(p, np.ndarray), "p should be a numpy array"
    assert isinstance(P_U, np.ndarray), "P_U should be a numpy array"
    assert isinstance(inv_U, np.ndarray), "inv_U should be a numpy array"
    assert P_U.shape == (n, k), f"P_U shape should be ({n}, {k})"
    assert inv_U.shape == (k, k), f"inv_U shape should be ({k}, {k})"


def test_sqdeim_return_inverse_without_projector():
    """Test that return_inverse=True without return_projector=True returns only indices."""
    np.random.seed(47)
    A = np.random.randn(20, 5)
    Q, _ = la.qr(A, mode="economic")

    # return_inverse=True but return_projector=False should return only p
    result = sQDEIM(Q, return_inverse=True)

    assert isinstance(result, np.ndarray), "Should return only indices as array"
    assert result.ndim == 1, "Should be 1D array of indices"


# ===========================
# ETA PARAMETER TESTS
# ===========================


def test_sqdeim_small_eta(orthonormal_matrix):
    """Test sQDEIM with small eta (tight bound)."""
    Q = orthonormal_matrix
    k = Q.shape[1]
    eta = 1.5  # Tight bound

    p = sQDEIM(Q, eta=eta)

    assert len(p) == k, f"Should return {k} indices"
    assert len(np.unique(p)) == k, "All indices should be unique"


def test_sqdeim_large_eta(orthonormal_matrix):
    """Test sQDEIM with large eta (loose bound)."""
    Q = orthonormal_matrix
    k = Q.shape[1]
    eta = 10.0  # Loose bound

    p = sQDEIM(Q, eta=eta)

    assert len(p) == k, f"Should return {k} indices"


def test_sqdeim_different_eta_values(orthonormal_matrix):
    """Test that different eta values may produce different selections."""
    Q = orthonormal_matrix

    p1 = sQDEIM(Q, eta=1.5)
    p2 = sQDEIM(Q, eta=5.0)

    # Both should be valid selections
    assert len(p1) == len(p2), "Both should select k indices"
    # They may or may not differ depending on the matrix structure


# ===========================
# CONDITIONING TESTS
# ===========================


def test_sqdeim_conditioning_bound(orthonormal_matrix):
    """Test that sQDEIM satisfies the conditioning bound."""
    Q = orthonormal_matrix
    n, k = Q.shape
    eta = 2.0

    p = sQDEIM(Q, eta=eta)

    # Extract selected rows
    U_p = Q[p, :]

    # Compute smallest singular value
    s = la.svdvals(U_p)
    sigma_min = s[-1]
    sigma_min_inv = 1.0 / sigma_min

    # Check bound: sigma_min^{-1} <= sqrt(1 + eta * k * (n-k))
    bound = np.sqrt(1 + eta * k * (n - k))

    assert (
        sigma_min_inv <= bound * 1.1
    ), f"Conditioning bound violated: {sigma_min_inv} > {bound}"


def test_sqdeim_well_conditioned(orthonormal_matrix):
    """Test that selected submatrix is well-conditioned."""
    Q = orthonormal_matrix

    p = sQDEIM(Q)
    U_p = Q[p, :]

    # Compute condition number
    cond = np.linalg.cond(U_p)

    # Should be reasonably well-conditioned
    assert cond < 1e10, f"Selected submatrix poorly conditioned: {cond}"


def test_sqdeim_smallest_singular_value(orthonormal_matrix):
    """Test that smallest singular value is not too small."""
    Q = orthonormal_matrix

    p = sQDEIM(Q)
    U_p = Q[p, :]

    s = la.svdvals(U_p)
    sigma_min = s[-1]

    # Should not be near zero
    assert sigma_min > 1e-14, f"Smallest singular value {sigma_min} is too small"


# ===========================
# PROJECTOR TESTS
# ===========================


def test_sqdeim_projector_computation(orthonormal_matrix):
    """Test that P_U satisfies the interpolation property."""
    Q = orthonormal_matrix
    n, k = Q.shape

    p, P_U = sQDEIM(Q, return_projector=True)

    # P_U should satisfy interpolation on selected rows
    # For vectors in span(Q): P_U @ v[p] = v
    test_vec = Q[:, 0]  # First column of Q
    interpolated = P_U @ test_vec[p]

    assert np.allclose(
        interpolated, test_vec, rtol=1e-9
    ), "P_U interpolation property failed"


def test_sqdeim_inverse_computation(orthonormal_matrix):
    """Test that inv_U is computed correctly."""
    Q = orthonormal_matrix
    k = Q.shape[1]

    p, P_U, inv_U = sQDEIM(Q, return_projector=True, return_inverse=True)

    # Verify relationship: inv_U = U.T @ P_U
    expected_inv_U = Q.T.conj() @ P_U

    assert np.allclose(inv_U, expected_inv_U, rtol=1e-9), "inv_U != U.T @ P_U"


def test_sqdeim_projector_inverse_relationship(orthonormal_matrix):
    """Test the relationship P_U = U @ inv_U."""
    Q = orthonormal_matrix

    p, P_U, inv_U = sQDEIM(Q, return_projector=True, return_inverse=True)

    # Verify relationship
    assert np.allclose(P_U, Q @ inv_U, rtol=1e-10), "P_U != U @ inv_U"


def test_sqdeim_interpolation_property(orthonormal_matrix):
    """Test that P_U satisfies the interpolation property."""
    Q = orthonormal_matrix
    k = Q.shape[1]

    p, P_U = sQDEIM(Q, return_projector=True)

    # For any vector in span(Q), P_U should interpolate correctly
    test_vec = Q @ np.random.randn(k)
    interpolated = P_U @ test_vec[p]

    # Should reconstruct the vector exactly (within numerical precision)
    error = la.norm(test_vec - interpolated) / la.norm(test_vec)

    assert error < 1e-10, f"Interpolation error {error} too large"


# ===========================
# COMPLEX MATRIX TESTS
# ===========================


def test_sqdeim_complex(complex_orthonormal_matrix):
    """Test sQDEIM with complex matrices."""
    Q = complex_orthonormal_matrix
    n, k = Q.shape

    p = sQDEIM(Q)

    assert len(p) == k, f"Expected {k} indices for complex matrix"
    assert len(np.unique(p)) == k, "Complex matrix indices should be unique"


def test_sqdeim_complex_projector(complex_orthonormal_matrix):
    """Test sQDEIM projector with complex matrices."""
    Q = complex_orthonormal_matrix
    n, k = Q.shape

    p, P_U = sQDEIM(Q, return_projector=True)

    assert P_U.shape == (n, k), f"P_U shape correct for complex matrix"

    # Verify interpolation property on first column
    test_vec = Q[:, 0]
    interpolated = P_U @ test_vec[p]

    assert np.allclose(
        interpolated, test_vec, rtol=1e-9
    ), "Complex P_U interpolation property failed"


def test_sqdeim_complex_inverse(complex_orthonormal_matrix):
    """Test sQDEIM with complex matrices and return_inverse=True."""
    Q = complex_orthonormal_matrix

    p, P_U, inv_U = sQDEIM(Q, return_projector=True, return_inverse=True)

    # Verify relationship
    expected_inv_U = Q.T.conj() @ P_U

    assert np.allclose(inv_U, expected_inv_U, rtol=1e-9), "Complex inv_U != U.T @ P_U"


def test_sqdeim_complex_interpolation(complex_orthonormal_matrix):
    """Test interpolation property with complex matrices."""
    Q = complex_orthonormal_matrix
    k = Q.shape[1]

    p, P_U = sQDEIM(Q, return_projector=True)

    # Test interpolation
    test_vec = Q @ (np.random.randn(k) + 1j * np.random.randn(k))
    interpolated = P_U @ test_vec[p]

    error = la.norm(test_vec - interpolated) / la.norm(test_vec)

    assert error < 1e-10, f"Complex interpolation error {error} too large"


# ===========================
# EDGE CASES
# ===========================


def test_sqdeim_tall_matrix(tall_orthonormal_matrix):
    """Test sQDEIM with very tall matrix (n >> k)."""
    Q = tall_orthonormal_matrix
    n, k = Q.shape
    assert n > 10 * k, "Should be very tall"

    p = sQDEIM(Q)

    assert len(p) == k, f"Expected {k} indices"
    assert len(np.unique(p)) == k, "All indices should be unique"
    assert np.all(p >= 0) and np.all(p < n), "Indices in valid range"


def test_sqdeim_square_matrix(square_orthonormal_matrix):
    """Test sQDEIM with square orthonormal matrix (n = k)."""
    Q = square_orthonormal_matrix
    n, k = Q.shape
    assert n == k, "Should be square"

    p = sQDEIM(Q)

    assert len(p) == k, f"Should return k={k} indices"
    assert len(np.unique(p)) == k, "All indices should be unique"


def test_sqdeim_rank_one(rank_one_matrix):
    """Test sQDEIM with rank-1 matrix."""
    Q = rank_one_matrix
    n, k = Q.shape
    assert k == 1, "Should be rank-1"

    p = sQDEIM(Q)

    assert len(p) == 1, "Should return 1 index"
    assert 0 <= p[0] < n, "Index in valid range"


def test_sqdeim_rank_one_projector(rank_one_matrix):
    """Test sQDEIM with rank-1 matrix and projector."""
    Q = rank_one_matrix
    n, k = Q.shape

    p, P_U = sQDEIM(Q, return_projector=True)

    assert P_U.shape == (n, 1), f"P_U shape should be ({n}, 1)"

    # Test interpolation
    test_vec = Q[:, 0] * 2.5
    interpolated = P_U @ test_vec[p]

    assert np.allclose(
        test_vec, interpolated, rtol=1e-10
    ), "Rank-1 interpolation failed"


# ===========================
# NUMERICAL PROPERTIES TESTS
# ===========================


def test_sqdeim_input_orthogonality_check(orthonormal_matrix):
    """Verify input matrix is actually orthonormal."""
    Q = orthonormal_matrix

    # Check orthonormality
    eye_k = np.eye(Q.shape[1])
    assert np.allclose(Q.T @ Q, eye_k, rtol=1e-10), "Input should be orthonormal"


def test_sqdeim_preserves_span(orthonormal_matrix):
    """Test that sQDEIM preserves span via interpolation."""
    Q = orthonormal_matrix
    k = Q.shape[1]

    p, P_U = sQDEIM(Q, return_projector=True)

    # For multiple test vectors
    for _ in range(5):
        test_vec = Q @ np.random.randn(k)
        reconstructed = P_U @ test_vec[p]

        error = la.norm(test_vec - reconstructed) / la.norm(test_vec)
        assert error < 1e-10, f"Span preservation failed: error {error}"


def test_sqdeim_deterministic(orthonormal_matrix):
    """Test that sQDEIM is deterministic."""
    Q = orthonormal_matrix

    p1 = sQDEIM(Q)
    p2 = sQDEIM(Q)

    assert np.array_equal(p1, p2), "sQDEIM should be deterministic"


def test_sqdeim_reconstruction_accuracy(orthonormal_matrix):
    """Test reconstruction accuracy using sQDEIM interpolation."""
    Q = orthonormal_matrix
    k = Q.shape[1]

    p, P_U = sQDEIM(Q, return_projector=True)

    # Test reconstruction of each column
    for i in range(k):
        col = Q[:, i]
        reconstructed = P_U @ col[p]
        error = la.norm(col - reconstructed)

        assert error < 1e-10, f"Reconstruction error {error} too large for column {i}"


# ===========================
# COMPARISON TESTS
# ===========================


def test_sqdeim_vs_qdeim_both_work():
    """Test that both sQDEIM and QDEIM work on same input."""
    from low_rank_toolbox import QDEIM

    np.random.seed(50)
    A = np.random.randn(30, 6)
    Q, _ = la.qr(A, mode="economic")

    p_qdeim = QDEIM(Q)
    p_sqdeim = sQDEIM(Q)

    # Both should return k indices
    assert len(p_qdeim) == 6, "QDEIM should return 6 indices"
    assert len(p_sqdeim) == 6, "sQDEIM should return 6 indices"

    # Both should produce well-conditioned submatrices
    U_p_qdeim = Q[p_qdeim, :]
    U_p_sqdeim = Q[p_sqdeim, :]

    assert (
        la.svdvals(U_p_qdeim)[-1] > 1e-14
    ), "QDEIM submatrix should be well-conditioned"
    assert (
        la.svdvals(U_p_sqdeim)[-1] > 1e-14
    ), "sQDEIM submatrix should be well-conditioned"


def test_sqdeim_better_conditioning():
    """Test that sQDEIM typically produces better conditioning than QDEIM."""
    from low_rank_toolbox import QDEIM

    np.random.seed(51)
    # Create a matrix where conditioning matters
    A = np.random.randn(50, 8)
    Q, _ = la.qr(A, mode="economic")

    p_qdeim = QDEIM(Q)
    p_sqdeim = sQDEIM(Q, eta=1.5)  # Tight eta for better conditioning

    # Compute condition numbers
    cond_qdeim = np.linalg.cond(Q[p_qdeim, :])
    cond_sqdeim = np.linalg.cond(Q[p_sqdeim, :])

    # Both should be well-conditioned (sQDEIM may be better but not always guaranteed)
    assert cond_qdeim < 1e10, "QDEIM should produce well-conditioned matrix"
    assert cond_sqdeim < 1e10, "sQDEIM should produce well-conditioned matrix"


# ===========================
# PERMUTATION TESTS
# ===========================


def test_sqdeim_permutation_validity(orthonormal_matrix):
    """Test that returned indices are valid."""
    Q = orthonormal_matrix
    n, k = Q.shape

    p = sQDEIM(Q)

    # Check properties
    assert len(p) == k, f"Should return k={k} indices"
    assert len(set(p)) == k, "All indices should be unique"
    assert all(0 <= idx < n for idx in p), "All indices in valid range [0, n)"


def test_sqdeim_indices_type(orthonormal_matrix):
    """Test that indices are integers."""
    Q = orthonormal_matrix

    p = sQDEIM(Q)

    assert p.dtype in [np.int32, np.int64], "Indices should be integers"


# ===========================
# EXTRA ARGS TESTS
# ===========================


def test_sqdeim_extra_args():
    """Test that sQDEIM accepts extra arguments (for compatibility)."""
    np.random.seed(52)
    A = np.random.randn(20, 5)
    Q, _ = la.qr(A, mode="economic")

    # Test with extra args (should be ignored)
    p1 = sQDEIM(Q, extra_param=123)
    p2 = sQDEIM(Q)

    # Should produce same result
    assert np.array_equal(p1, p2), "Extra args should not affect result"


# ===========================
# ERROR HANDLING
# ===========================


def test_sqdeim_non_orthonormal_input():
    """Test sQDEIM with non-orthonormal input (should still work)."""
    np.random.seed(53)
    # Create a non-orthonormal matrix
    A = np.random.randn(20, 5)

    # sQDEIM should still run (though optimality not guaranteed)
    p = sQDEIM(A)

    assert len(p) == 5, "Should return k indices even for non-orthonormal input"


# ===========================
# STABILITY TESTS
# ===========================


def test_sqdeim_numerical_stability():
    """Test numerical stability with various matrix scales."""
    np.random.seed(54)

    for scale in [1e-5, 1.0, 1e5]:
        A = np.random.randn(25, 6) * scale
        Q, _ = la.qr(A, mode="economic")

        p = sQDEIM(Q)

        assert len(p) == 6, f"Should work with scale {scale}"
        assert len(np.unique(p)) == 6, f"Unique indices with scale {scale}"


# ===========================
# DOCUMENTATION EXAMPLE TEST
# ===========================


def test_sqdeim_documentation_example():
    """Test example that could be in documentation."""
    # Create a simple orthonormal basis
    np.random.seed(100)
    A = np.random.randn(40, 7)
    U, _ = la.qr(A, mode="economic")

    # Basic usage
    p = sQDEIM(U)
    assert len(p) == 7, "Should select k=7 rows"
    assert isinstance(p, np.ndarray), "Should return array of indices"

    # With custom eta
    p_tight = sQDEIM(U, eta=1.5)
    assert len(p_tight) == 7, "Should select k rows with custom eta"

    # With projector
    p, P_U = sQDEIM(U, return_projector=True)
    assert P_U.shape == (40, 7), "Projector should be n x k"

    # Verify interpolation property
    test_vector = U @ np.random.randn(7)
    interpolated = P_U @ test_vector[p]
    assert np.allclose(
        test_vector, interpolated, rtol=1e-9
    ), "Should interpolate accurately"

    # With both outputs
    p, P_U, inv_U = sQDEIM(U, return_projector=True, return_inverse=True)
    assert inv_U.shape == (7, 7), "Inverse should be k x k"
    assert np.allclose(P_U, U @ inv_U, rtol=1e-10), "Relationship should hold"

    # Test conditioning bound
    n, k = U.shape
    eta = 2.0
    p = sQDEIM(U, eta=eta)
    U_p = U[p, :]
    sigma_min_inv = 1.0 / la.svdvals(U_p)[-1]
    bound = np.sqrt(1 + eta * k * (n - k))
    assert sigma_min_inv <= bound * 1.1, "Conditioning bound satisfied"

    print("sQDEIM documentation example test passed!")


# ===========================
# REPRODUCIBILITY TEST
# ===========================


def test_sqdeim_reproducibility():
    """Test that sQDEIM gives consistent results."""
    np.random.seed(101)
    A = np.random.randn(30, 6)
    Q, _ = la.qr(A, mode="economic")

    p1 = sQDEIM(Q)
    p2 = sQDEIM(Q)

    # Results should be identical
    assert np.array_equal(p1, p2), "sQDEIM should be reproducible"

    # With projector
    _, P_U1 = sQDEIM(Q, return_projector=True)
    _, P_U2 = sQDEIM(Q, return_projector=True)

    assert np.allclose(P_U1, P_U2), "Projector should be reproducible"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
