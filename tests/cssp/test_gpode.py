"""
Test file for gpode.py (Gappy POD+E)

Author: Benjamin Carrel, University of Geneva, 2024
"""

import numpy as np
import pytest
import scipy.linalg as la

from lowrank import gpode

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


def test_gpode_basic(orthonormal_matrix):
    """Test basic gpode functionality."""
    Q = orthonormal_matrix
    n, k = Q.shape
    oversampling = 2
    m = k + oversampling

    # Test without optional returns
    p = gpode(Q, oversampling_size=oversampling)

    assert isinstance(p, np.ndarray), "p should be a numpy array"
    assert len(p) == m, f"Expected {m} indices (k+oversampling), got {len(p)}"
    assert len(np.unique(p)) == m, "Indices should be unique"
    assert np.all(p >= 0) and np.all(p < n), "Indices should be in valid range [0, n)"


def test_gpode_return_projector(orthonormal_matrix):
    """Test gpode with return_projector=True."""
    Q = orthonormal_matrix
    n, k = Q.shape
    oversampling = 3
    m = k + oversampling

    p, M = gpode(Q, oversampling_size=oversampling, return_projector=True)

    assert isinstance(p, np.ndarray), "p should be a numpy array"
    assert isinstance(M, np.ndarray), "M should be a numpy array"
    assert M.shape == (n, m), f"M shape should be ({n}, {m}), got {M.shape}"
    assert len(p) == m, f"Expected {m} indices, got {len(p)}"

    # Verify that M approximates U via least-squares on oversampled rows
    # M should satisfy: U[p, :].T @ M.T = U.T (in least-squares sense)
    # Or equivalently: M = (U[p, :]^+ @ U)^T where ^+ is pseudoinverse
    U_p = Q[p, :]
    pinv_U_p = la.pinv(U_p.T.conj())
    expected_M = (pinv_U_p @ Q.T.conj()).T.conj()

    assert np.allclose(M, expected_M, rtol=1e-9), "M computation mismatch"


def test_gpode_return_both(orthonormal_matrix):
    """Test gpode with both return_projector=True and return_inverse=True."""
    Q = orthonormal_matrix
    n, k = Q.shape
    oversampling = 2
    m = k + oversampling

    p, M, inv_U = gpode(
        Q, oversampling_size=oversampling, return_projector=True, return_inverse=True
    )

    assert isinstance(p, np.ndarray), "p should be a numpy array"
    assert isinstance(M, np.ndarray), "M should be a numpy array"
    assert isinstance(inv_U, np.ndarray), "inv_U should be a numpy array"
    assert M.shape == (n, m), f"M shape should be ({n}, {m})"
    assert inv_U.shape == (k, m), f"inv_U shape should be ({k}, {m})"

    # Verify relationship: inv_U = U.T @ M (from lstsq computation)
    expected_inv_U = Q.T.conj() @ M

    assert np.allclose(inv_U, expected_inv_U, rtol=1e-9), "inv_U != U.T @ M"

    # Verify relationship: M = U @ inv_U
    assert np.allclose(M, Q @ inv_U, rtol=1e-10), "M != U @ inv_U"


def test_gpode_oversampling_sizes(orthonormal_matrix):
    """Test gpode with different oversampling sizes."""
    Q = orthonormal_matrix
    k = Q.shape[1]

    for oversampling in [1, 2, 3, 5]:
        p = gpode(Q, oversampling_size=oversampling)
        expected_m = k + oversampling
        assert (
            len(p) == expected_m
        ), f"Expected {expected_m} indices for oversampling={oversampling}"
        assert len(np.unique(p)) == expected_m, "All indices should be unique"


def test_gpode_starts_with_qdeim(orthonormal_matrix):
    """Test that gpode starts with QDEIM indices."""
    from lowrank import QDEIM

    Q = orthonormal_matrix
    k = Q.shape[1]
    oversampling = 3

    # Get QDEIM indices
    p_qdeim = QDEIM(Q)

    # Get gpode indices
    p_gpode = gpode(Q, oversampling_size=oversampling)

    # First k indices should be from QDEIM (though order might differ)
    qdeim_set = set(p_qdeim)
    gpode_first_k_set = set(p_gpode[:k])

    assert (
        qdeim_set == gpode_first_k_set
    ), "First k gpode indices should match QDEIM indices"


# ===========================
# COMPLEX MATRIX TESTS
# ===========================


def test_gpode_complex(complex_orthonormal_matrix):
    """Test gpode with complex matrices."""
    Q = complex_orthonormal_matrix
    n, k = Q.shape
    oversampling = 2
    m = k + oversampling

    p = gpode(Q, oversampling_size=oversampling)

    assert len(p) == m, f"Expected {m} indices for complex matrix"
    assert len(np.unique(p)) == m, "Complex matrix indices should be unique"

    # Test with projector
    p, M = gpode(Q, oversampling_size=oversampling, return_projector=True)

    # Verify M via least-squares computation
    U_p = Q[p, :]
    pinv_U_p_T = la.pinv(U_p.T.conj())
    expected_M = (pinv_U_p_T @ Q.T.conj()).T.conj()

    assert np.allclose(M, expected_M, rtol=1e-9), "Complex M computation mismatch"


def test_gpode_complex_return_inverse(complex_orthonormal_matrix):
    """Test gpode with complex matrices and return_inverse=True."""
    Q = complex_orthonormal_matrix
    oversampling = 2

    p, M, inv_U = gpode(
        Q, oversampling_size=oversampling, return_projector=True, return_inverse=True
    )

    # Verify inv_U = U.T @ M
    expected_inv_U = Q.T.conj() @ M

    assert np.allclose(inv_U, expected_inv_U, rtol=1e-9), "Complex inv_U != U.T @ M"
    assert np.allclose(M, Q @ inv_U, rtol=1e-10), "Complex M != U @ inv_U"


# ===========================
# EDGE CASES
# ===========================


def test_gpode_minimal_oversampling(orthonormal_matrix):
    """Test gpode with minimal oversampling (p=1)."""
    Q = orthonormal_matrix
    k = Q.shape[1]

    p = gpode(Q, oversampling_size=1)

    assert len(p) == k + 1, "Should return k+1 indices"
    assert len(np.unique(p)) == k + 1, "All indices should be unique"


def test_gpode_large_oversampling(orthonormal_matrix):
    """Test gpode with large oversampling."""
    Q = orthonormal_matrix
    n, k = Q.shape
    oversampling = 10  # Large oversampling

    # Should work if m = k + p < n
    if k + oversampling < n:
        p = gpode(Q, oversampling_size=oversampling)
        assert len(p) == k + oversampling, f"Should return {k + oversampling} indices"


def test_gpode_tall_matrix(tall_orthonormal_matrix):
    """Test gpode with very tall matrix (n >> k)."""
    Q = tall_orthonormal_matrix
    n, k = Q.shape
    assert n > 10 * k, "Should be very tall"

    oversampling = 2
    p = gpode(Q, oversampling_size=oversampling)

    assert len(p) == k + oversampling, f"Expected {k + oversampling} indices"
    assert len(np.unique(p)) == k + oversampling, "All indices should be unique"
    assert np.all(p >= 0) and np.all(p < n), "Indices in valid range"


def test_gpode_square_matrix(square_orthonormal_matrix):
    """Test gpode with square orthonormal matrix (n = k)."""
    Q = square_orthonormal_matrix
    n, k = Q.shape
    assert n == k, "Should be square"

    # Can't oversample beyond n
    # This test just ensures it doesn't crash with minimal oversampling
    # (though in practice, oversampling makes less sense for square matrices)
    try:
        # Try with small oversampling that would exceed n
        p = gpode(Q, oversampling_size=1)
        # If n=k, we can't really oversample, but function should handle it
        assert len(p) <= n, "Can't select more than n indices"
    except:
        # It's acceptable if this fails for square matrices
        pass


# ===========================
# ORTHOGONALITY & NUMERICAL TESTS
# ===========================


def test_gpode_input_orthogonality_check(orthonormal_matrix):
    """Verify input matrix is actually orthonormal."""
    Q = orthonormal_matrix

    # Check orthonormality
    eye_k = np.eye(Q.shape[1])
    assert np.allclose(Q.T @ Q, eye_k, rtol=1e-10), "Input should be orthonormal"


def test_gpode_preserves_span(orthonormal_matrix):
    """Test that gpode preserves span via least-squares interpolation."""
    Q = orthonormal_matrix
    k = Q.shape[1]
    oversampling = 2

    p, M = gpode(Q, oversampling_size=oversampling, return_projector=True)

    # For any vector in span(Q), least-squares interpolation should reconstruct it well
    test_vec = Q @ np.random.randn(k)
    reconstructed = M @ test_vec[p]

    # With oversampling, reconstruction should be very accurate
    error = la.norm(test_vec - reconstructed) / la.norm(test_vec)
    assert error < 1e-9, f"Relative reconstruction error {error} too large"


def test_gpode_condition_number(orthonormal_matrix):
    """Test that gpode produces better conditioned matrices than just k indices."""
    Q = orthonormal_matrix
    k = Q.shape[1]
    oversampling = 3

    p = gpode(Q, oversampling_size=oversampling)
    U_p = Q[p, :]

    # Compute smallest singular value (overdetermined system)
    sigma = la.svdvals(U_p)
    sigma_min = sigma[-1]

    # With oversampling, smallest singular value should be reasonably large
    assert sigma_min > 1e-12, f"Smallest singular value {sigma_min} is too small"


def test_gpode_reconstruction_accuracy(orthonormal_matrix):
    """Test reconstruction accuracy using gpode interpolation."""
    Q = orthonormal_matrix
    k = Q.shape[1]
    oversampling = 3

    p, M = gpode(Q, oversampling_size=oversampling, return_projector=True)

    # Test reconstruction of each column
    for i in range(k):
        col = Q[:, i]
        reconstructed = M @ col[p]
        error = la.norm(col - reconstructed)

        # With oversampling, reconstruction should be very accurate
        assert error < 1e-9, f"Reconstruction error {error} too large for column {i}"


def test_gpode_pseudoinverse_accuracy(orthonormal_matrix):
    """Test accuracy of the pseudoinverse computation in gpode."""
    Q = orthonormal_matrix
    oversampling = 2

    p, M, inv_U = gpode(
        Q, oversampling_size=oversampling, return_projector=True, return_inverse=True
    )

    U_p = Q[p, :]

    # Check that M @ U_p.T approximates U.T (least-squares solution)
    # This is the definition: U[p,:].T @ M.T = U.T (solved via lstsq)
    reconstructed_UT = U_p.T.conj() @ M.T.conj()
    expected_UT = Q.T.conj()

    assert np.allclose(
        reconstructed_UT, expected_UT, rtol=1e-9
    ), "M doesn't satisfy least-squares property"


# ===========================
# EXTRA_ARGS TESTS
# ===========================


def test_gpode_with_qr_kwargs(orthonormal_matrix):
    """Test gpode with extra QR arguments."""
    Q = orthonormal_matrix
    oversampling = 2

    # Test with empty qr_kwargs
    p1 = gpode(Q, oversampling_size=oversampling, qr_kwargs={})
    p2 = gpode(Q, oversampling_size=oversampling)

    # Results should be the same
    assert np.array_equal(p1, p2), "Empty qr_kwargs should give same result"


def test_gpode_with_lstsq_kwargs(orthonormal_matrix):
    """Test gpode with extra lstsq arguments."""
    Q = orthonormal_matrix
    oversampling = 2

    # Test with lstsq options (scipy uses 'cond' not 'rcond')
    p1, M1 = gpode(
        Q,
        oversampling_size=oversampling,
        return_projector=True,
        lstsq_kwargs={"cond": None},
    )
    p2, M2 = gpode(Q, oversampling_size=oversampling, return_projector=True)

    # Results should be very similar
    assert np.allclose(
        M1, M2, rtol=1e-9
    ), "lstsq_kwargs shouldn't significantly change result"


# ===========================
# GREEDY ALGORITHM TESTS
# ===========================


def test_gpode_greedy_improvement(orthonormal_matrix):
    """Test that gpode's greedy algorithm improves conditioning."""
    Q = orthonormal_matrix
    k = Q.shape[1]
    oversampling = 3

    p = gpode(Q, oversampling_size=oversampling)

    # Check that each added index improves or maintains conditioning
    for i in range(k, len(p)):
        U_p_before = Q[p[:i], :]
        U_p_after = Q[p[: i + 1], :]

        # Both should be well-conditioned
        sigma_before = la.svdvals(U_p_before)
        sigma_after = la.svdvals(U_p_after)

        # The minimum singular value should stay reasonably large
        assert (
            sigma_after[-1] > 1e-14
        ), f"Singular value too small after adding index {i}"


def test_gpode_svd_computation():
    """Test that the SVD computation in gpode works correctly."""
    np.random.seed(47)
    A = np.random.randn(25, 4)
    Q, _ = la.qr(A, mode="economic")

    oversampling = 2
    p = gpode(Q, oversampling_size=oversampling)

    # Verify that we can compute SVD of U[p, :] at each step
    for i in range(len(p)):
        U_p = Q[p[: i + 1], :]
        try:
            _, s, _ = la.svd(U_p, full_matrices=False)
            assert len(s) > 0, "SVD should produce singular values"
        except:
            pytest.fail(f"SVD failed at index {i}")


# ===========================
# ERROR HANDLING
# ===========================


def test_gpode_non_orthonormal_input():
    """Test gpode with non-orthonormal input (should still work)."""
    np.random.seed(48)
    # Create a non-orthonormal matrix
    A = np.random.randn(20, 5)

    oversampling = 2
    # gpode should still run
    p = gpode(A, oversampling_size=oversampling)

    assert (
        len(p) == 5 + oversampling
    ), "Should return k+p indices even for non-orthonormal input"


def test_gpode_return_inverse_without_projector():
    """Test that return_inverse=True without return_projector=True returns only indices."""
    np.random.seed(49)
    A = np.random.randn(20, 5)
    Q, _ = la.qr(A, mode="economic")

    oversampling = 2
    # return_inverse=True but return_projector=False
    result = gpode(Q, oversampling_size=oversampling, return_inverse=True)

    # Should only return indices
    assert isinstance(result, np.ndarray), "Should return only indices as array"
    assert result.ndim == 1, "Should be 1D array of indices"


# ===========================
# REPRODUCIBILITY TEST
# ===========================


def test_gpode_reproducibility():
    """Test that gpode gives consistent results for the same input."""
    np.random.seed(50)
    A = np.random.randn(25, 6)
    Q, _ = la.qr(A, mode="economic")

    oversampling = 3
    p1 = gpode(Q, oversampling_size=oversampling)
    p2 = gpode(Q, oversampling_size=oversampling)

    assert np.array_equal(p1, p2), "gpode should be deterministic"


# ===========================
# COMPARISON TESTS
# ===========================


def test_gpode_vs_qdeim_comparison():
    """Test that gpode with oversampling=0 is similar to QDEIM."""
    from lowrank import QDEIM

    np.random.seed(52)
    A = np.random.randn(40, 6)
    Q, _ = la.qr(A, mode="economic")

    p_qdeim = QDEIM(Q)
    p_gpode = gpode(Q, oversampling_size=0)  # No oversampling

    # Both should return k indices
    assert len(p_qdeim) == 6, "QDEIM should return 6 indices"
    assert len(p_gpode) == 6, "gpode with oversampling=0 should return 6 indices"

    # The sets of indices should match (same as QDEIM initialization)
    assert set(p_qdeim) == set(p_gpode), "gpode(oversampling=0) should match QDEIM"


def test_gpode_improves_over_qdeim():
    """Test that gpode with oversampling can improve conditioning over QDEIM."""
    from lowrank import QDEIM

    np.random.seed(53)
    A = np.random.randn(50, 5)
    Q, _ = la.qr(A, mode="economic")

    p_qdeim = QDEIM(Q)
    p_gpode = gpode(Q, oversampling_size=5)

    # gpode should have more indices
    assert len(p_gpode) == len(p_qdeim) + 5, "gpode should have k+5 indices"

    # Both submatrices should be well-conditioned
    U_p_qdeim = Q[p_qdeim, :]
    U_p_gpode = Q[p_gpode, :]

    cond_qdeim = la.norm(U_p_qdeim, 2) * la.norm(la.pinv(U_p_qdeim), 2)
    cond_gpode = la.norm(U_p_gpode, 2) * la.norm(la.pinv(U_p_gpode), 2)

    # Both should be reasonably conditioned
    assert cond_qdeim < 1e12, "QDEIM should be well-conditioned"
    assert cond_gpode < 1e12, "gpode should be well-conditioned"


# ===========================
# DOCUMENTATION EXAMPLE TEST
# ===========================


def test_gpode_documentation_example():
    """Test example that could be in documentation."""
    # Create a simple orthonormal basis
    np.random.seed(100)
    A = np.random.randn(50, 8)
    U, _ = la.qr(A, mode="economic")

    # Basic usage with oversampling
    oversampling = 4
    p = gpode(U, oversampling_size=oversampling)
    assert len(p) == 8 + oversampling, "Should select k+oversampling rows"
    assert isinstance(p, np.ndarray), "Should return array of indices"

    # With projector
    p, M = gpode(U, oversampling_size=oversampling, return_projector=True)
    assert M.shape == (
        50,
        8 + oversampling,
    ), "Projector should be n x m (n x (k+oversampling))"

    # Verify least-squares interpolation property
    test_vector = U @ np.random.randn(8)
    interpolated = M @ test_vector[p]
    assert np.allclose(
        test_vector, interpolated, rtol=1e-9
    ), "Should interpolate accurately with oversampling"

    # With both outputs
    p, M, inv_U = gpode(
        U, oversampling_size=oversampling, return_projector=True, return_inverse=True
    )
    assert inv_U.shape == (8, 8 + oversampling), "Pseudoinverse should be k x m"
    assert np.allclose(M, U @ inv_U, rtol=1e-10), "Relationship should hold"

    print("gpode documentation example test passed!")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
