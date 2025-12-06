"""
Test file for gpodr.py (Gappy POD+R - Randomized)

Author: Benjamin Carrel, University of Geneva, 2024
"""

import numpy as np
import scipy.linalg as la
from lowrank import gpodr
import pytest


# ===========================
# FIXTURES
# ===========================

@pytest.fixture
def orthonormal_matrix():
    """Create a random orthonormal matrix."""
    np.random.seed(42)
    A = np.random.randn(20, 5)
    Q, _ = la.qr(A, mode='economic')
    return Q


@pytest.fixture
def complex_orthonormal_matrix():
    """Create a random complex orthonormal matrix."""
    np.random.seed(43)
    A = np.random.randn(30, 7) + 1j * np.random.randn(30, 7)
    Q, _ = la.qr(A, mode='economic')
    return Q


@pytest.fixture
def tall_orthonormal_matrix():
    """Create a tall orthonormal matrix (n >> k)."""
    np.random.seed(44)
    A = np.random.randn(100, 3)
    Q, _ = la.qr(A, mode='economic')
    return Q


@pytest.fixture
def square_orthonormal_matrix():
    """Create a square orthonormal matrix."""
    np.random.seed(45)
    A = np.random.randn(10, 10)
    Q, _ = la.qr(A, mode='economic')
    return Q


# ===========================
# BASIC FUNCTIONALITY TESTS
# ===========================

def test_gpodr_basic(orthonormal_matrix):
    """Test basic gpodr functionality with oversampling_size."""
    Q = orthonormal_matrix
    n, k = Q.shape
    oversampling = 3
    m = k + oversampling
    
    # Test without optional returns
    p = gpodr(Q, oversampling_size=oversampling)
    
    assert isinstance(p, np.ndarray), "p should be a numpy array"
    assert len(p) == m, f"Expected {m} indices (k+oversampling), got {len(p)}"
    assert len(np.unique(p)) == m, "Indices should be unique"
    assert np.all(p >= 0) and np.all(p < n), "Indices should be in valid range [0, n)"


def test_gpodr_default_oversampling(orthonormal_matrix):
    """Test gpodr with default oversampling (should be k)."""
    Q = orthonormal_matrix
    n, k = Q.shape
    
    # Test with no oversampling specified (should default to k)
    p = gpodr(Q)
    
    assert len(p) == 2 * k, f"Default oversampling should give 2*k={2*k} indices"
    assert len(np.unique(p)) == 2 * k, "All indices should be unique"


def test_gpodr_return_projector(orthonormal_matrix):
    """Test gpodr with return_projector=True."""
    Q = orthonormal_matrix
    n, k = Q.shape
    oversampling = 3
    m = k + oversampling
    
    p, P_U = gpodr(Q, oversampling_size=oversampling, return_projector=True)
    
    assert isinstance(p, np.ndarray), "p should be a numpy array"
    assert isinstance(P_U, np.ndarray), "P_U should be a numpy array"
    assert P_U.shape == (n, m), f"P_U shape should be ({n}, {m}), got {P_U.shape}"
    assert len(p) == m, f"Expected {m} indices, got {len(p)}"
    
    # Verify P_U via least-squares computation
    U_p = Q[p, :]
    pinv_U_p_T = la.pinv(U_p.T.conj())
    expected_P_U = (pinv_U_p_T @ Q.T.conj()).T.conj()
    
    assert np.allclose(P_U, expected_P_U, rtol=1e-9), "P_U computation mismatch"


def test_gpodr_return_both(orthonormal_matrix):
    """Test gpodr with both return_projector=True and return_inverse=True."""
    Q = orthonormal_matrix
    n, k = Q.shape
    oversampling = 2
    m = k + oversampling
    
    p, P_U, inv_U = gpodr(Q, oversampling_size=oversampling, return_projector=True, return_inverse=True)
    
    assert isinstance(p, np.ndarray), "p should be a numpy array"
    assert isinstance(P_U, np.ndarray), "P_U should be a numpy array"
    assert isinstance(inv_U, np.ndarray), "inv_U should be a numpy array"
    assert P_U.shape == (n, m), f"P_U shape should be ({n}, {m})"
    assert inv_U.shape == (k, m), f"inv_U shape should be ({k}, {m})"
    
    # Verify relationship: inv_U = U.T @ P_U
    expected_inv_U = Q.T.conj() @ P_U
    
    assert np.allclose(inv_U, expected_inv_U, rtol=1e-9), "inv_U != U.T @ P_U"
    
    # Verify relationship: P_U = U @ inv_U
    assert np.allclose(P_U, Q @ inv_U, rtol=1e-10), "P_U != U @ inv_U"


# ===========================
# TOLERANCE-BASED TESTS
# ===========================

def test_gpodr_with_tolerance(orthonormal_matrix):
    """Test gpodr with tolerance parameter instead of oversampling_size."""
    Q = orthonormal_matrix
    n, k = Q.shape
    
    tol = 10.0  # Reasonable tolerance
    
    p = gpodr(Q, tol=tol)
    
    # Check that selected indices exist
    assert isinstance(p, np.ndarray), "p should be a numpy array"
    assert len(p) >= k, "Should select at least k indices"
    assert len(np.unique(p)) == len(p), "All indices should be unique"
    
    # Verify tolerance condition: sigma_min(U[p,:])^{-1} <= tol
    U_p = Q[p, :]
    s = la.svdvals(U_p)
    sigma_min_inv = 1.0 / s[-1]
    
    assert sigma_min_inv <= tol * 1.1, f"Tolerance condition violated: {sigma_min_inv} > {tol}"


def test_gpodr_with_tight_tolerance(orthonormal_matrix):
    """Test gpodr with tight tolerance (should select more rows)."""
    Q = orthonormal_matrix
    n, k = Q.shape
    
    tight_tol = 2.0  # Tight tolerance
    loose_tol = 100.0  # Loose tolerance
    
    p_tight = gpodr(Q, tol=tight_tol)
    p_loose = gpodr(Q, tol=loose_tol)
    
    # Tight tolerance should require more samples
    assert len(p_tight) >= len(p_loose), "Tight tolerance should require more samples"


def test_gpodr_with_max_iter(orthonormal_matrix):
    """Test gpodr with max_iter parameter."""
    Q = orthonormal_matrix
    
    # Very tight tolerance but limited iterations
    tol = 1.0
    max_iter = 2
    
    p = gpodr(Q, tol=tol, max_iter=max_iter)
    
    # Should not exceed k + max_iter samples
    k = Q.shape[1]
    assert len(p) <= k + max_iter, f"Should not exceed k+max_iter={k+max_iter} samples"


# ===========================
# OVERSAMPLING SIZE TESTS
# ===========================

def test_gpodr_oversampling_sizes(orthonormal_matrix):
    """Test gpodr with different oversampling sizes."""
    Q = orthonormal_matrix
    k = Q.shape[1]
    
    for oversampling in [1, 2, 3, 5]:
        p = gpodr(Q, oversampling_size=oversampling)
        expected_m = k + oversampling
        assert len(p) == expected_m, f"Expected {expected_m} indices for oversampling={oversampling}"
        assert len(np.unique(p)) == expected_m, "All indices should be unique"


def test_gpodr_zero_oversampling(orthonormal_matrix):
    """Test gpodr with zero oversampling (should return k indices)."""
    Q = orthonormal_matrix
    k = Q.shape[1]
    
    p = gpodr(Q, oversampling_size=0)
    
    assert len(p) == k, f"Zero oversampling should return exactly k={k} indices"


def test_gpodr_large_oversampling(orthonormal_matrix):
    """Test gpodr with large oversampling."""
    Q = orthonormal_matrix
    n, k = Q.shape
    oversampling = 10  # Large oversampling
    
    # Should work if m = k + p < n
    if k + oversampling < n:
        p = gpodr(Q, oversampling_size=oversampling)
        assert len(p) == k + oversampling, f"Should return {k + oversampling} indices"


# ===========================
# QDEIM INITIALIZATION TESTS
# ===========================

def test_gpodr_starts_with_qdeim(orthonormal_matrix):
    """Test that gpodr uses QDEIM ordering from QR pivoting."""
    from lowrank import QDEIM
    
    Q = orthonormal_matrix
    k = Q.shape[1]
    
    # Get QDEIM indices
    p_qdeim = QDEIM(Q)
    
    # Get gpodr indices with zero oversampling
    p_gpodr = gpodr(Q, oversampling_size=0)
    
    # Should be identical to QDEIM
    qdeim_set = set(p_qdeim)
    gpodr_set = set(p_gpodr)
    
    assert qdeim_set == gpodr_set, "gpodr with oversampling=0 should match QDEIM indices"


def test_gpodr_extends_qdeim_ordering(orthonormal_matrix):
    """Test that gpodr extends QDEIM with more indices from pivoting order."""
    from lowrank import QDEIM
    
    Q = orthonormal_matrix
    k = Q.shape[1]
    oversampling = 3
    
    # Get QDEIM indices
    p_qdeim = QDEIM(Q)
    
    # Get gpodr indices
    p_gpodr = gpodr(Q, oversampling_size=oversampling)
    
    # First k indices should be QDEIM indices (same set, possibly different order)
    qdeim_set = set(p_qdeim)
    gpodr_first_k_set = set(p_gpodr[:k])
    
    assert qdeim_set == gpodr_first_k_set, "First k gpodr indices should match QDEIM"


# ===========================
# COMPLEX MATRIX TESTS
# ===========================

def test_gpodr_complex(complex_orthonormal_matrix):
    """Test gpodr with complex matrices."""
    Q = complex_orthonormal_matrix
    n, k = Q.shape
    oversampling = 2
    m = k + oversampling
    
    p = gpodr(Q, oversampling_size=oversampling)
    
    assert len(p) == m, f"Expected {m} indices for complex matrix"
    assert len(np.unique(p)) == m, "Complex matrix indices should be unique"
    
    # Test with projector
    p, P_U = gpodr(Q, oversampling_size=oversampling, return_projector=True)
    
    # Verify P_U via least-squares computation
    U_p = Q[p, :]
    pinv_U_p_T = la.pinv(U_p.T.conj())
    expected_P_U = (pinv_U_p_T @ Q.T.conj()).T.conj()
    
    assert np.allclose(P_U, expected_P_U, rtol=1e-9), "Complex P_U computation mismatch"


def test_gpodr_complex_return_inverse(complex_orthonormal_matrix):
    """Test gpodr with complex matrices and return_inverse=True."""
    Q = complex_orthonormal_matrix
    oversampling = 2
    
    p, P_U, inv_U = gpodr(Q, oversampling_size=oversampling, return_projector=True, return_inverse=True)
    
    # Verify inv_U = U.T @ P_U
    expected_inv_U = Q.T.conj() @ P_U
    
    assert np.allclose(inv_U, expected_inv_U, rtol=1e-9), "Complex inv_U != U.T @ P_U"
    assert np.allclose(P_U, Q @ inv_U, rtol=1e-10), "Complex P_U != U @ inv_U"


def test_gpodr_complex_with_tolerance(complex_orthonormal_matrix):
    """Test gpodr with complex matrices and tolerance."""
    Q = complex_orthonormal_matrix
    
    tol = 10.0
    p = gpodr(Q, tol=tol)
    
    # Verify tolerance condition
    U_p = Q[p, :]
    s = la.svdvals(U_p)
    sigma_min_inv = 1.0 / s[-1]
    
    assert sigma_min_inv <= tol * 1.1, f"Complex tolerance condition violated"


# ===========================
# EDGE CASES
# ===========================

def test_gpodr_tall_matrix(tall_orthonormal_matrix):
    """Test gpodr with very tall matrix (n >> k)."""
    Q = tall_orthonormal_matrix
    n, k = Q.shape
    assert n > 10 * k, "Should be very tall"
    
    oversampling = 2
    p = gpodr(Q, oversampling_size=oversampling)
    
    assert len(p) == k + oversampling, f"Expected {k + oversampling} indices"
    assert len(np.unique(p)) == k + oversampling, "All indices should be unique"
    assert np.all(p >= 0) and np.all(p < n), "Indices in valid range"


def test_gpodr_square_matrix(square_orthonormal_matrix):
    """Test gpodr with square orthonormal matrix (n = k)."""
    Q = square_orthonormal_matrix
    n, k = Q.shape
    assert n == k, "Should be square"
    
    # With zero oversampling, should work
    p = gpodr(Q, oversampling_size=0)
    assert len(p) == k, f"Should return k={k} indices"


def test_gpodr_minimal_oversampling(orthonormal_matrix):
    """Test gpodr with minimal oversampling (p=1)."""
    Q = orthonormal_matrix
    k = Q.shape[1]
    
    p = gpodr(Q, oversampling_size=1)
    
    assert len(p) == k + 1, "Should return k+1 indices"
    assert len(np.unique(p)) == k + 1, "All indices should be unique"


# ===========================
# NUMERICAL PROPERTIES TESTS
# ===========================

def test_gpodr_input_orthogonality_check(orthonormal_matrix):
    """Verify input matrix is actually orthonormal."""
    Q = orthonormal_matrix
    
    # Check orthonormality
    eye_k = np.eye(Q.shape[1])
    assert np.allclose(Q.T @ Q, eye_k, rtol=1e-10), "Input should be orthonormal"


def test_gpodr_preserves_span(orthonormal_matrix):
    """Test that gpodr preserves span via least-squares interpolation."""
    Q = orthonormal_matrix
    k = Q.shape[1]
    oversampling = 3
    
    p, P_U = gpodr(Q, oversampling_size=oversampling, return_projector=True)
    
    # For any vector in span(Q), least-squares interpolation should reconstruct it well
    test_vec = Q @ np.random.randn(k)
    reconstructed = P_U @ test_vec[p]
    
    # With oversampling, reconstruction should be very accurate
    error = la.norm(test_vec - reconstructed) / la.norm(test_vec)
    assert error < 1e-9, f"Relative reconstruction error {error} too large"


def test_gpodr_condition_number(orthonormal_matrix):
    """Test that gpodr produces well-conditioned matrices."""
    Q = orthonormal_matrix
    k = Q.shape[1]
    oversampling = 3
    
    p = gpodr(Q, oversampling_size=oversampling)
    U_p = Q[p, :]
    
    # Compute smallest singular value (overdetermined system)
    sigma = la.svdvals(U_p)
    sigma_min = sigma[-1]
    
    # With oversampling, smallest singular value should be reasonably large
    assert sigma_min > 1e-12, f"Smallest singular value {sigma_min} is too small"


def test_gpodr_reconstruction_accuracy(orthonormal_matrix):
    """Test reconstruction accuracy using gpodr interpolation."""
    Q = orthonormal_matrix
    k = Q.shape[1]
    oversampling = 3
    
    p, P_U = gpodr(Q, oversampling_size=oversampling, return_projector=True)
    
    # Test reconstruction of each column
    for i in range(k):
        col = Q[:, i]
        reconstructed = P_U @ col[p]
        error = la.norm(col - reconstructed)
        
        # With oversampling, reconstruction should be very accurate
        assert error < 1e-9, f"Reconstruction error {error} too large for column {i}"


def test_gpodr_lstsq_accuracy(orthonormal_matrix):
    """Test accuracy of the least-squares computation in gpodr."""
    Q = orthonormal_matrix
    oversampling = 2
    
    p, P_U, inv_U = gpodr(Q, oversampling_size=oversampling, return_projector=True, return_inverse=True)
    
    U_p = Q[p, :]
    
    # Check that P_U @ U_p.T approximates U.T (least-squares solution)
    reconstructed_UT = U_p.T.conj() @ P_U.T.conj()
    expected_UT = Q.T.conj()
    
    assert np.allclose(reconstructed_UT, expected_UT, rtol=1e-9), "P_U doesn't satisfy least-squares property"


# ===========================
# EXTRA_ARGS TESTS
# ===========================

def test_gpodr_with_qr_kwargs(orthonormal_matrix):
    """Test gpodr with extra QR arguments."""
    Q = orthonormal_matrix
    oversampling = 2
    
    # Test with empty qr_kwargs
    p1 = gpodr(Q, oversampling_size=oversampling, qr_kwargs={})
    p2 = gpodr(Q, oversampling_size=oversampling)
    
    # Results should be the same
    assert np.array_equal(p1, p2), "Empty qr_kwargs should give same result"


def test_gpodr_with_lstsq_kwargs(orthonormal_matrix):
    """Test gpodr with extra lstsq arguments."""
    Q = orthonormal_matrix
    oversampling = 2
    
    # Test with lstsq options (scipy uses 'cond' not 'rcond')
    p1, P_U1 = gpodr(Q, oversampling_size=oversampling, return_projector=True, lstsq_kwargs={'cond': None})
    p2, P_U2 = gpodr(Q, oversampling_size=oversampling, return_projector=True)
    
    # Results should be very similar
    assert np.allclose(P_U1, P_U2, rtol=1e-9), "lstsq_kwargs shouldn't significantly change result"


# ===========================
# ERROR HANDLING
# ===========================

def test_gpodr_non_orthonormal_input():
    """Test gpodr with non-orthonormal input (should still work)."""
    np.random.seed(48)
    # Create a non-orthonormal matrix
    A = np.random.randn(20, 5)
    
    oversampling = 2
    # gpodr should still run
    p = gpodr(A, oversampling_size=oversampling)
    
    assert len(p) == 5 + oversampling, "Should return k+p indices even for non-orthonormal input"


def test_gpodr_return_inverse_without_projector():
    """Test that return_inverse=True without return_projector=True returns only indices."""
    np.random.seed(49)
    A = np.random.randn(20, 5)
    Q, _ = la.qr(A, mode='economic')
    
    oversampling = 2
    # return_inverse=True but return_projector=False
    result = gpodr(Q, oversampling_size=oversampling, return_inverse=True)
    
    # Should only return indices
    assert isinstance(result, np.ndarray), "Should return only indices as array"
    assert result.ndim == 1, "Should be 1D array of indices"


# ===========================
# REPRODUCIBILITY TEST
# ===========================

def test_gpodr_reproducibility():
    """Test that gpodr gives consistent results for the same input."""
    np.random.seed(50)
    A = np.random.randn(25, 6)
    Q, _ = la.qr(A, mode='economic')
    
    oversampling = 3
    p1 = gpodr(Q, oversampling_size=oversampling)
    p2 = gpodr(Q, oversampling_size=oversampling)
    
    assert np.array_equal(p1, p2), "gpodr should be deterministic"


# ===========================
# COMPARISON TESTS
# ===========================

def test_gpodr_vs_qdeim_comparison():
    """Test that gpodr with oversampling=0 matches QDEIM."""
    from lowrank import QDEIM
    
    np.random.seed(52)
    A = np.random.randn(40, 6)
    Q, _ = la.qr(A, mode='economic')
    
    p_qdeim = QDEIM(Q)
    p_gpodr = gpodr(Q, oversampling_size=0)
    
    # Both should return k indices
    assert len(p_qdeim) == 6, "QDEIM should return 6 indices"
    assert len(p_gpodr) == 6, "gpodr with oversampling=0 should return 6 indices"
    
    # The sets of indices should match
    assert set(p_qdeim) == set(p_gpodr), "gpodr(oversampling=0) should match QDEIM"


def test_gpodr_vs_gpode_both_work():
    """Test that both gpodr and gpode work on the same input."""
    from lowrank import gpode
    
    np.random.seed(53)
    A = np.random.randn(40, 5)
    Q, _ = la.qr(A, mode='economic')
    
    oversampling = 3
    p_gpodr = gpodr(Q, oversampling_size=oversampling)
    p_gpode = gpode(Q, oversampling_size=oversampling)
    
    # Both should return k+oversampling indices
    assert len(p_gpodr) == 5 + oversampling, "gpodr should return k+oversampling indices"
    assert len(p_gpode) == 5 + oversampling, "gpode should return k+oversampling indices"
    
    # Both should produce well-conditioned submatrices
    U_p_gpodr = Q[p_gpodr, :]
    U_p_gpode = Q[p_gpode, :]
    
    assert la.svdvals(U_p_gpodr)[-1] > 1e-14, "gpodr submatrix should be well-conditioned"
    assert la.svdvals(U_p_gpode)[-1] > 1e-14, "gpode submatrix should be well-conditioned"


# ===========================
# DOCUMENTATION EXAMPLE TEST
# ===========================

def test_gpodr_documentation_example():
    """Test example that could be in documentation."""
    # Create a simple orthonormal basis
    np.random.seed(100)
    A = np.random.randn(50, 8)
    U, _ = la.qr(A, mode='economic')
    
    # Basic usage with oversampling
    oversampling = 4
    p = gpodr(U, oversampling_size=oversampling)
    assert len(p) == 8 + oversampling, "Should select k+oversampling rows"
    assert isinstance(p, np.ndarray), "Should return array of indices"
    
    # With default oversampling (k)
    p_default = gpodr(U)
    assert len(p_default) == 2 * 8, "Default should select 2*k rows"
    
    # With tolerance instead of oversampling
    p_tol = gpodr(U, tol=5.0)
    assert len(p_tol) >= 8, "Should select at least k rows"
    
    # With projector
    p, P_U = gpodr(U, oversampling_size=oversampling, return_projector=True)
    assert P_U.shape == (50, 8 + oversampling), "Projector should be n x m"
    
    # Verify least-squares interpolation property
    test_vector = U @ np.random.randn(8)
    interpolated = P_U @ test_vector[p]
    assert np.allclose(test_vector, interpolated, rtol=1e-9), "Should interpolate accurately"
    
    # With both outputs
    p, P_U, inv_U = gpodr(U, oversampling_size=oversampling, return_projector=True, return_inverse=True)
    assert inv_U.shape == (8, 8 + oversampling), "Pseudoinverse should be k x m"
    assert np.allclose(P_U, U @ inv_U, rtol=1e-10), "Relationship should hold"
    
    print("gpodr documentation example test passed!")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
