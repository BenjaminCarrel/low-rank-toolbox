"""
Test file for oversampling_sqdeim.py (Oversampling sQDEIM)

Tests for the oversampled version of Strong QDEIM based on Park & Nakatsukasa.

Author: Benjamin Carrel, University of Geneva, 2024
"""

import numpy as np
import scipy.linalg as la
from lowrank.cssp import oversampling_sQDEIM
import pytest


# ===========================
# FIXTURES
# ===========================

@pytest.fixture
def orthonormal_matrix():
    """Create a random orthonormal matrix."""
    np.random.seed(42)
    A = np.random.randn(30, 6)
    Q, _ = la.qr(A, mode='economic')
    return Q


@pytest.fixture
def complex_orthonormal_matrix():
    """Create a random complex orthonormal matrix."""
    np.random.seed(43)
    A = np.random.randn(40, 8) + 1j * np.random.randn(40, 8)
    Q, _ = la.qr(A, mode='economic')
    return Q


@pytest.fixture
def tall_orthonormal_matrix():
    """Create a tall orthonormal matrix (n >> k)."""
    np.random.seed(44)
    A = np.random.randn(100, 5)
    Q, _ = la.qr(A, mode='economic')
    return Q


@pytest.fixture
def square_orthonormal_matrix():
    """Create a square orthonormal matrix."""
    np.random.seed(45)
    A = np.random.randn(15, 15)
    Q, _ = la.qr(A, mode='economic')
    return Q


@pytest.fixture
def rank_one_matrix():
    """Create a rank-1 orthonormal matrix."""
    np.random.seed(46)
    A = np.random.randn(20, 1)
    Q, _ = la.qr(A, mode='economic')
    return Q


# ===========================
# BASIC FUNCTIONALITY TESTS
# ===========================

def test_oversampling_sqdeim_basic(orthonormal_matrix):
    """Test basic oversampling_sQDEIM functionality."""
    Q = orthonormal_matrix
    n, k = Q.shape
    oversampling = 2
    m = k + oversampling
    
    # Test without optional returns
    p = oversampling_sQDEIM(Q, oversampling_size=oversampling)
    
    assert isinstance(p, np.ndarray), "p should be a numpy array"
    assert len(p) == m, f"Expected {m} indices (k+oversampling), got {len(p)}"
    assert len(np.unique(p)) == m, "Indices should be unique"
    assert np.all(p >= 0) and np.all(p < n), "Indices should be in valid range [0, n)"


def test_oversampling_sqdeim_zero_oversampling(orthonormal_matrix):
    """Test oversampling_sQDEIM with zero oversampling."""
    Q = orthonormal_matrix
    k = Q.shape[1]
    
    # Zero oversampling should return k indices
    p = oversampling_sQDEIM(Q, oversampling_size=0)
    
    assert len(p) == k, f"Zero oversampling should return k={k} indices"


def test_oversampling_sqdeim_return_projection(orthonormal_matrix):
    """Test oversampling_sQDEIM with return_projection=True."""
    Q = orthonormal_matrix
    n, k = Q.shape
    oversampling = 2
    m = k + oversampling
    
    p, P_U = oversampling_sQDEIM(Q, oversampling_size=oversampling, return_projection=True)
    
    assert isinstance(p, np.ndarray), "p should be a numpy array"
    assert isinstance(P_U, np.ndarray), "P_U should be a numpy array"
    assert P_U.shape == (n, m), f"P_U shape should be ({n}, {m}), got {P_U.shape}"
    assert len(p) == m, f"Expected {m} indices, got {len(p)}"


def test_oversampling_sqdeim_return_both(orthonormal_matrix):
    """Test oversampling_sQDEIM with both return_projection=True and return_inverse=True."""
    Q = orthonormal_matrix
    n, k = Q.shape
    oversampling = 2
    m = k + oversampling
    
    p, P_U, inv_U = oversampling_sQDEIM(Q, oversampling_size=oversampling, 
                                         return_projection=True, return_inverse=True)
    
    assert isinstance(p, np.ndarray), "p should be a numpy array"
    assert isinstance(P_U, np.ndarray), "P_U should be a numpy array"
    assert isinstance(inv_U, np.ndarray), "inv_U should be a numpy array"
    assert P_U.shape == (n, m), f"P_U shape should be ({n}, {m})"
    assert inv_U.shape == (k, m), f"inv_U shape should be ({k}, {m})"


def test_oversampling_sqdeim_return_inverse_without_projection():
    """Test that return_inverse=True without return_projection=True returns only indices."""
    np.random.seed(47)
    A = np.random.randn(25, 6)
    Q, _ = la.qr(A, mode='economic')
    
    oversampling = 2
    # return_inverse=True but return_projection=False
    result = oversampling_sQDEIM(Q, oversampling_size=oversampling, return_inverse=True)
    
    # Should only return indices
    assert isinstance(result, np.ndarray), "Should return only indices as array"
    assert result.ndim == 1, "Should be 1D array of indices"


# ===========================
# OVERSAMPLING SIZE TESTS
# ===========================

def test_oversampling_sqdeim_different_oversampling_sizes(orthonormal_matrix):
    """Test oversampling_sQDEIM with different oversampling sizes."""
    Q = orthonormal_matrix
    k = Q.shape[1]
    
    for oversampling in [1, 2, 3, 4]:
        p = oversampling_sQDEIM(Q, oversampling_size=oversampling)
        expected_m = k + oversampling
        assert len(p) == expected_m, f"Expected {expected_m} indices for oversampling={oversampling}"
        assert len(np.unique(p)) == expected_m, "All indices should be unique"


def test_oversampling_sqdeim_max_oversampling(orthonormal_matrix):
    """Test oversampling_sQDEIM with maximum oversampling (oversampling = k)."""
    Q = orthonormal_matrix
    k = Q.shape[1]
    
    # Maximum oversampling should be k
    p = oversampling_sQDEIM(Q, oversampling_size=k)
    
    assert len(p) == 2 * k, f"Max oversampling should return 2*k={2*k} indices"


# ===========================
# TOLERANCE MODE TESTS
# ===========================

def test_oversampling_sqdeim_with_tolerance(orthonormal_matrix):
    """Test oversampling_sQDEIM with tolerance parameter."""
    Q = orthonormal_matrix
    oversampling = 2
    tol = 1e-4
    
    p = oversampling_sQDEIM(Q, oversampling_size=oversampling, tol=tol)
    
    # Should still work with tolerance
    assert isinstance(p, np.ndarray), "p should be a numpy array"
    assert len(p) >= Q.shape[1], "Should select at least k indices"
    assert len(np.unique(p)) == len(p), "All indices should be unique"


def test_oversampling_sqdeim_tolerance_vs_no_tolerance(orthonormal_matrix):
    """Test that tolerance affects the selection."""
    Q = orthonormal_matrix
    oversampling = 2
    
    p1 = oversampling_sQDEIM(Q, oversampling_size=oversampling)
    p2 = oversampling_sQDEIM(Q, oversampling_size=oversampling, tol=1e-5)
    
    # Both should be valid selections
    assert len(p1) > 0, "Without tolerance should select indices"
    assert len(p2) > 0, "With tolerance should select indices"


# ===========================
# CONDITIONING TESTS
# ===========================

def test_oversampling_sqdeim_well_conditioned(orthonormal_matrix):
    """Test that oversampled selection produces well-conditioned submatrix."""
    Q = orthonormal_matrix
    oversampling = 3
    
    p = oversampling_sQDEIM(Q, oversampling_size=oversampling)
    U_p = Q[p, :]
    
    # Compute condition number (overdetermined system)
    s = la.svdvals(U_p)
    sigma_min = s[-1]
    
    # With oversampling, should be well-conditioned
    assert sigma_min > 1e-14, f"Smallest singular value {sigma_min} too small"


def test_oversampling_sqdeim_better_than_standard(orthonormal_matrix):
    """Test that oversampling improves conditioning."""
    from lowrank import sQDEIM
    
    Q = orthonormal_matrix
    
    # Standard sQDEIM
    p_standard = sQDEIM(Q)
    U_standard = Q[p_standard, :]
    cond_standard = np.linalg.cond(U_standard)
    
    # Oversampled sQDEIM
    p_oversampled = oversampling_sQDEIM(Q, oversampling_size=2)
    U_oversampled = Q[p_oversampled, :]
    
    # Oversampled should have more rows
    assert len(p_oversampled) > len(p_standard), "Oversampled should select more rows"
    
    # Both should be well-conditioned
    assert cond_standard < 1e10, "Standard should be well-conditioned"
    
    # Oversampled matrix smallest singular value should be reasonable
    s = la.svdvals(U_oversampled)
    assert s[-1] > 1e-14, "Oversampled should have good smallest singular value"


# ===========================
# PROJECTOR TESTS
# ===========================

def test_oversampling_sqdeim_projector_computation(orthonormal_matrix):
    """Test that P_U satisfies the interpolation property."""
    Q = orthonormal_matrix
    k = Q.shape[1]
    oversampling = 2
    
    p, P_U = oversampling_sQDEIM(Q, oversampling_size=oversampling, return_projection=True)
    
    # Test interpolation on columns of Q
    for i in range(k):
        test_vec = Q[:, i]
        interpolated = P_U @ test_vec[p]
        error = la.norm(test_vec - interpolated) / la.norm(test_vec)
        
        # With oversampling, should be very accurate
        assert error < 1e-9, f"Interpolation error {error} too large for column {i}"


def test_oversampling_sqdeim_inverse_computation(orthonormal_matrix):
    """Test that inv_U is computed correctly."""
    Q = orthonormal_matrix
    k = Q.shape[1]
    oversampling = 2
    
    p, P_U, inv_U = oversampling_sQDEIM(Q, oversampling_size=oversampling,
                                         return_projection=True, return_inverse=True)
    
    # Verify relationship: inv_U = U.T @ P_U
    expected_inv_U = Q.T.conj() @ P_U
    
    assert np.allclose(inv_U, expected_inv_U, rtol=1e-9), "inv_U != U.T @ P_U"


def test_oversampling_sqdeim_projector_inverse_relationship(orthonormal_matrix):
    """Test the relationship P_U = U @ inv_U."""
    Q = orthonormal_matrix
    oversampling = 2
    
    p, P_U, inv_U = oversampling_sQDEIM(Q, oversampling_size=oversampling,
                                         return_projection=True, return_inverse=True)
    
    # Verify relationship
    assert np.allclose(P_U, Q @ inv_U, rtol=1e-10), "P_U != U @ inv_U"


def test_oversampling_sqdeim_reconstruction_accuracy(orthonormal_matrix):
    """Test reconstruction accuracy with oversampling."""
    Q = orthonormal_matrix
    k = Q.shape[1]
    oversampling = 3
    
    p, P_U = oversampling_sQDEIM(Q, oversampling_size=oversampling, return_projection=True)
    
    # Test on random vectors in span(Q)
    for _ in range(5):
        coeffs = np.random.randn(k)
        test_vec = Q @ coeffs
        reconstructed = P_U @ test_vec[p]
        
        error = la.norm(test_vec - reconstructed) / la.norm(test_vec)
        assert error < 1e-9, f"Reconstruction error {error} too large"


# ===========================
# COMPLEX MATRIX TESTS
# ===========================

def test_oversampling_sqdeim_complex(complex_orthonormal_matrix):
    """Test oversampling_sQDEIM with complex matrices."""
    Q = complex_orthonormal_matrix
    n, k = Q.shape
    oversampling = 3
    m = k + oversampling
    
    p = oversampling_sQDEIM(Q, oversampling_size=oversampling)
    
    assert len(p) == m, f"Expected {m} indices for complex matrix"
    assert len(np.unique(p)) == m, "Complex matrix indices should be unique"


def test_oversampling_sqdeim_complex_projector(complex_orthonormal_matrix):
    """Test oversampling_sQDEIM projector with complex matrices."""
    Q = complex_orthonormal_matrix
    n, k = Q.shape
    oversampling = 2
    
    p, P_U = oversampling_sQDEIM(Q, oversampling_size=oversampling, return_projection=True)
    m = k + oversampling
    
    assert P_U.shape == (n, m), f"P_U shape correct for complex matrix"
    
    # Test interpolation
    test_vec = Q[:, 0]
    interpolated = P_U @ test_vec[p]
    
    error = la.norm(test_vec - interpolated) / la.norm(test_vec)
    assert error < 1e-9, f"Complex interpolation error {error} too large"


def test_oversampling_sqdeim_complex_inverse(complex_orthonormal_matrix):
    """Test oversampling_sQDEIM with complex matrices and return_inverse=True."""
    Q = complex_orthonormal_matrix
    oversampling = 2
    
    p, P_U, inv_U = oversampling_sQDEIM(Q, oversampling_size=oversampling,
                                         return_projection=True, return_inverse=True)
    
    # Verify relationship
    expected_inv_U = Q.T.conj() @ P_U
    
    assert np.allclose(inv_U, expected_inv_U, rtol=1e-9), "Complex inv_U != U.T @ P_U"


# ===========================
# EDGE CASES
# ===========================

def test_oversampling_sqdeim_tall_matrix(tall_orthonormal_matrix):
    """Test oversampling_sQDEIM with very tall matrix (n >> k)."""
    Q = tall_orthonormal_matrix
    n, k = Q.shape
    assert n > 10 * k, "Should be very tall"
    
    oversampling = 2
    p = oversampling_sQDEIM(Q, oversampling_size=oversampling)
    
    assert len(p) == k + oversampling, f"Expected {k + oversampling} indices"
    assert len(np.unique(p)) == k + oversampling, "All indices should be unique"
    assert np.all(p >= 0) and np.all(p < n), "Indices in valid range"


def test_oversampling_sqdeim_square_matrix(square_orthonormal_matrix):
    """Test oversampling_sQDEIM with square orthonormal matrix (n = k)."""
    Q = square_orthonormal_matrix
    n, k = Q.shape
    assert n == k, "Should be square"
    
    # With zero oversampling
    p = oversampling_sQDEIM(Q, oversampling_size=0)
    
    assert len(p) == k, f"Should return k={k} indices"


def test_oversampling_sqdeim_minimal_oversampling(orthonormal_matrix):
    """Test oversampling_sQDEIM with minimal oversampling (p=1)."""
    Q = orthonormal_matrix
    k = Q.shape[1]
    
    p = oversampling_sQDEIM(Q, oversampling_size=1)
    
    assert len(p) == k + 1, "Should return k+1 indices"
    assert len(np.unique(p)) == k + 1, "All indices should be unique"


# ===========================
# ERROR HANDLING TESTS
# ===========================

def test_oversampling_sqdeim_negative_oversampling():
    """Test that negative oversampling raises an error."""
    np.random.seed(48)
    A = np.random.randn(20, 5)
    Q, _ = la.qr(A, mode='economic')
    
    with pytest.raises(ValueError, match="must be positive"):
        oversampling_sQDEIM(Q, oversampling_size=-1)


def test_oversampling_sqdeim_oversampling_too_large():
    """Test that oversampling > k raises an error."""
    np.random.seed(49)
    A = np.random.randn(20, 5)
    Q, _ = la.qr(A, mode='economic')
    k = Q.shape[1]
    
    with pytest.raises(ValueError, match="must be smaller"):
        oversampling_sQDEIM(Q, oversampling_size=k + 1)


def test_oversampling_sqdeim_non_orthonormal_input():
    """Test oversampling_sQDEIM with non-orthonormal input (should still work)."""
    np.random.seed(50)
    # Create a non-orthonormal matrix
    A = np.random.randn(25, 6)
    
    oversampling = 2
    # Should still run (though optimality not guaranteed)
    p = oversampling_sQDEIM(A, oversampling_size=oversampling)
    
    assert len(p) == 6 + oversampling, "Should return k+oversampling indices"


# ===========================
# NUMERICAL PROPERTIES TESTS
# ===========================

def test_oversampling_sqdeim_input_orthogonality_check(orthonormal_matrix):
    """Verify input matrix is actually orthonormal."""
    Q = orthonormal_matrix
    
    # Check orthonormality
    eye_k = np.eye(Q.shape[1])
    assert np.allclose(Q.T @ Q, eye_k, rtol=1e-10), "Input should be orthonormal"


def test_oversampling_sqdeim_preserves_span(orthonormal_matrix):
    """Test that oversampling_sQDEIM preserves span via interpolation."""
    Q = orthonormal_matrix
    k = Q.shape[1]
    oversampling = 3
    
    p, P_U = oversampling_sQDEIM(Q, oversampling_size=oversampling, return_projection=True)
    
    # For multiple test vectors
    for _ in range(5):
        test_vec = Q @ np.random.randn(k)
        reconstructed = P_U @ test_vec[p]
        
        error = la.norm(test_vec - reconstructed) / la.norm(test_vec)
        assert error < 1e-9, f"Span preservation failed: error {error}"


def test_oversampling_sqdeim_deterministic(orthonormal_matrix):
    """Test that oversampling_sQDEIM is deterministic."""
    Q = orthonormal_matrix
    oversampling = 2
    
    p1 = oversampling_sQDEIM(Q, oversampling_size=oversampling)
    p2 = oversampling_sQDEIM(Q, oversampling_size=oversampling)
    
    assert np.array_equal(p1, p2), "oversampling_sQDEIM should be deterministic"


def test_oversampling_sqdeim_indices_validity(orthonormal_matrix):
    """Test that returned indices are valid."""
    Q = orthonormal_matrix
    n, k = Q.shape
    oversampling = 2
    
    p = oversampling_sQDEIM(Q, oversampling_size=oversampling)
    
    # Check properties
    assert len(p) == k + oversampling, f"Should return k+oversampling indices"
    assert len(set(p)) == k + oversampling, "All indices should be unique"
    assert all(0 <= idx < n for idx in p), "All indices in valid range [0, n)"


# ===========================
# COMPARISON TESTS
# ===========================

def test_oversampling_sqdeim_vs_sqdeim():
    """Test that oversampling extends sQDEIM selection."""
    from lowrank import sQDEIM
    
    np.random.seed(51)
    A = np.random.randn(35, 7)
    Q, _ = la.qr(A, mode='economic')
    
    # Standard sQDEIM
    p_sqdeim = sQDEIM(Q)
    
    # Oversampled with 0 oversampling
    p_oversampled_0 = oversampling_sQDEIM(Q, oversampling_size=0)
    
    # Both should return k indices
    assert len(p_sqdeim) == 7, "sQDEIM should return 7 indices"
    assert len(p_oversampled_0) == 7, "Oversampled with p=0 should return 7 indices"


def test_oversampling_sqdeim_extends_selection():
    """Test that oversampling adds more indices."""
    np.random.seed(52)
    A = np.random.randn(30, 6)
    Q, _ = la.qr(A, mode='economic')
    
    p0 = oversampling_sQDEIM(Q, oversampling_size=0)
    p2 = oversampling_sQDEIM(Q, oversampling_size=2)
    p4 = oversampling_sQDEIM(Q, oversampling_size=4)
    
    # Should select progressively more rows
    assert len(p0) == 6, "p=0 should select k=6 rows"
    assert len(p2) == 8, "p=2 should select k+2=8 rows"
    assert len(p4) == 10, "p=4 should select k+4=10 rows"


# ===========================
# NUMERICAL STABILITY TESTS
# ===========================

def test_oversampling_sqdeim_numerical_stability():
    """Test numerical stability with various matrix scales."""
    np.random.seed(53)
    
    for scale in [1e-5, 1.0, 1e5]:
        A = np.random.randn(28, 6) * scale
        Q, _ = la.qr(A, mode='economic')
        
        oversampling = 2
        p = oversampling_sQDEIM(Q, oversampling_size=oversampling)
        
        assert len(p) == 8, f"Should work with scale {scale}"
        assert len(np.unique(p)) == 8, f"Unique indices with scale {scale}"


# ===========================
# DOCUMENTATION EXAMPLE TEST
# ===========================

def test_oversampling_sqdeim_documentation_example():
    """Test example that could be in documentation."""
    # Create a simple orthonormal basis
    np.random.seed(100)
    A = np.random.randn(50, 8)
    U, _ = la.qr(A, mode='economic')
    
    # Basic usage with oversampling
    oversampling = 3
    p = oversampling_sQDEIM(U, oversampling_size=oversampling)
    assert len(p) == 8 + oversampling, "Should select k+oversampling rows"
    assert isinstance(p, np.ndarray), "Should return array of indices"
    
    # With projection
    p, P_U = oversampling_sQDEIM(U, oversampling_size=oversampling, return_projection=True)
    assert P_U.shape == (50, 11), "Projector should be n x m where m = k + oversampling"
    
    # Verify interpolation property
    test_vector = U @ np.random.randn(8)
    interpolated = P_U @ test_vector[p]
    assert np.allclose(test_vector, interpolated, rtol=1e-9), "Should interpolate accurately"
    
    # With both outputs
    p, P_U, inv_U = oversampling_sQDEIM(U, oversampling_size=oversampling,
                                         return_projection=True, return_inverse=True)
    assert inv_U.shape == (8, 11), "Inverse should be k x m where m = k + oversampling"
    assert np.allclose(P_U, U @ inv_U, rtol=1e-10), "Relationship should hold"
    
    # With tolerance
    p_tol = oversampling_sQDEIM(U, oversampling_size=oversampling, tol=1e-5)
    assert len(p_tol) >= 8, "Should select at least k indices with tolerance"
    
    print("oversampling_sQDEIM documentation example test passed!")


# ===========================
# REPRODUCIBILITY TEST
# ===========================

def test_oversampling_sqdeim_reproducibility():
    """Test that oversampling_sQDEIM gives consistent results."""
    np.random.seed(101)
    A = np.random.randn(32, 7)
    Q, _ = la.qr(A, mode='economic')
    
    oversampling = 3
    p1 = oversampling_sQDEIM(Q, oversampling_size=oversampling)
    p2 = oversampling_sQDEIM(Q, oversampling_size=oversampling)
    
    # Results should be identical
    assert np.array_equal(p1, p2), "oversampling_sQDEIM should be reproducible"
    
    # With projector
    _, P_U1 = oversampling_sQDEIM(Q, oversampling_size=oversampling, return_projection=True)
    _, P_U2 = oversampling_sQDEIM(Q, oversampling_size=oversampling, return_projection=True)
    
    assert np.allclose(P_U1, P_U2), "Projector should be reproducible"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
