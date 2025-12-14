"""
Test file for QDEIM.py

Author: Benjamin Carrel, University of Geneva, 2024
"""

import numpy as np
import scipy.linalg as la
from lowrank import QDEIM
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

def test_qdeim_basic(orthonormal_matrix):
    """Test basic QDEIM functionality."""
    Q = orthonormal_matrix
    n, k = Q.shape
    
    # Test without optional returns
    p = QDEIM(Q)
    
    assert isinstance(p, np.ndarray), "p should be a numpy array"
    assert len(p) == k, f"Expected {k} indices, got {len(p)}"
    assert len(np.unique(p)) == k, "Indices should be unique"
    assert np.all(p >= 0) and np.all(p < n), "Indices should be in valid range [0, n)"


def test_qdeim_return_projector(orthonormal_matrix):
    """Test QDEIM with return_projector=True."""
    Q = orthonormal_matrix
    n, k = Q.shape
    
    p, P_U = QDEIM(Q, return_projector=True)
    
    assert isinstance(p, np.ndarray), "p should be a numpy array"
    assert isinstance(P_U, np.ndarray), "P_U should be a numpy array"
    assert P_U.shape == (n, k), f"P_U shape should be ({n}, {k}), got {P_U.shape}"
    assert len(p) == k, f"Expected {k} indices, got {len(p)}"
    
    # Verify that P_U = U @ inv(U[p, :])
    U_p = Q[p, :]
    inv_U_p = la.inv(U_p)
    expected_P_U = Q @ inv_U_p
    
    assert np.allclose(P_U, expected_P_U), "P_U != U @ inv(U[p, :])"


def test_qdeim_return_both(orthonormal_matrix):
    """Test QDEIM with both return_projector=True and return_inverse=True."""
    Q = orthonormal_matrix
    n, k = Q.shape
    
    p, P_U, inv_U = QDEIM(Q, return_projector=True, return_inverse=True)
    
    assert isinstance(p, np.ndarray), "p should be a numpy array"
    assert isinstance(P_U, np.ndarray), "P_U should be a numpy array"
    assert isinstance(inv_U, np.ndarray), "inv_U should be a numpy array"
    assert P_U.shape == (n, k), f"P_U shape should be ({n}, {k})"
    assert inv_U.shape == (k, k), f"inv_U shape should be ({k}, {k})"
    
    # Verify inv_U = inv(U[p, :])
    U_p = Q[p, :]
    expected_inv_U = la.inv(U_p)
    
    assert np.allclose(inv_U, expected_inv_U), "inv_U != inv(U[p, :])"
    
    # Verify relationship: P_U = U @ inv_U
    assert np.allclose(P_U, Q @ inv_U), "P_U != U @ inv_U"


def test_qdeim_interpolation_property(orthonormal_matrix):
    """Test that QDEIM satisfies the interpolation property: U[p, :] is well-conditioned."""
    Q = orthonormal_matrix
    k = Q.shape[1]
    
    p = QDEIM(Q)
    U_p = Q[p, :]
    
    # Check that U[p, :] is invertible (well-conditioned)
    cond_number = np.linalg.cond(U_p)
    
    # The condition number should be bounded (QDEIM provides this guarantee)
    # According to the paper: norm(inv(U[p,:])) <= sqrt(n-k+1) * 2^k
    n = Q.shape[0]
    theoretical_bound = np.sqrt(n - k + 1) * (2 ** k)
    
    inv_U_p = la.inv(U_p)
    norm_inv = la.norm(inv_U_p, ord=2)
    
    assert norm_inv <= theoretical_bound * 10, f"norm(inv(U[p,:])) = {norm_inv} exceeds reasonable bound"
    assert cond_number < 1e10, f"Condition number {cond_number} is too large"


# ===========================
# COMPLEX MATRIX TESTS
# ===========================

def test_qdeim_complex(complex_orthonormal_matrix):
    """Test QDEIM with complex matrices."""
    Q = complex_orthonormal_matrix
    n, k = Q.shape
    
    p = QDEIM(Q)
    
    assert len(p) == k, f"Expected {k} indices for complex matrix"
    assert len(np.unique(p)) == k, "Complex matrix indices should be unique"
    
    # Test with projector
    p, P_U = QDEIM(Q, return_projector=True)
    
    U_p = Q[p, :]
    inv_U_p = la.inv(U_p)
    expected_P_U = Q @ inv_U_p
    
    assert np.allclose(P_U, expected_P_U), "Complex P_U != U @ inv(U[p, :])"


def test_qdeim_complex_return_inverse(complex_orthonormal_matrix):
    """Test QDEIM with complex matrices and return_inverse=True."""
    Q = complex_orthonormal_matrix
    
    p, P_U, inv_U = QDEIM(Q, return_projector=True, return_inverse=True)
    
    U_p = Q[p, :]
    expected_inv_U = la.inv(U_p)
    
    assert np.allclose(inv_U, expected_inv_U), "Complex inv_U != inv(U[p, :])"
    assert np.allclose(P_U, Q @ inv_U), "Complex P_U != U @ inv_U"


# ===========================
# EDGE CASES
# ===========================

def test_qdeim_rank_one():
    """Test QDEIM with rank-1 matrix."""
    np.random.seed(46)
    v = np.random.randn(15, 1)
    Q, _ = la.qr(v, mode='economic')
    
    p = QDEIM(Q)
    
    assert len(p) == 1, "Rank-1 matrix should return 1 index"
    assert 0 <= p[0] < 15, "Index should be valid"
    
    # The selected row should be non-zero
    assert np.abs(Q[p[0], 0]) > 0, "Selected row should be non-zero"


def test_qdeim_square_matrix(square_orthonormal_matrix):
    """Test QDEIM with square orthonormal matrix (n = k)."""
    Q = square_orthonormal_matrix
    n, k = Q.shape
    assert n == k, "Should be square"
    
    p = QDEIM(Q)
    
    assert len(p) == k, f"Expected {k} indices"
    assert len(np.unique(p)) == k, "All indices should be unique"
    
    # For a square orthonormal matrix, U[p, :] should still be well-conditioned
    U_p = Q[p, :]
    cond_number = np.linalg.cond(U_p)
    assert cond_number < 1e10, f"Square matrix condition number {cond_number} too large"


def test_qdeim_tall_matrix(tall_orthonormal_matrix):
    """Test QDEIM with very tall matrix (n >> k)."""
    Q = tall_orthonormal_matrix
    n, k = Q.shape
    assert n > 10 * k, "Should be very tall"
    
    p = QDEIM(Q)
    
    assert len(p) == k, f"Expected {k} indices"
    assert len(np.unique(p)) == k, "All indices should be unique"
    assert np.all(p >= 0) and np.all(p < n), "Indices in valid range"


def test_qdeim_wide_component():
    """Test QDEIM when the transpose is wide (this tests the QR on U.T.conj())."""
    np.random.seed(47)
    # Create a case where n = 15, k = 5
    A = np.random.randn(15, 5)
    Q, _ = la.qr(A, mode='economic')
    
    p = QDEIM(Q)
    
    assert len(p) == 5, "Should return k=5 indices"


# ===========================
# ORTHOGONALITY VERIFICATION
# ===========================

def test_qdeim_input_orthogonality_check(orthonormal_matrix):
    """Verify input matrix is actually orthonormal."""
    Q = orthonormal_matrix
    
    # Check orthonormality
    eye_k = np.eye(Q.shape[1])
    assert np.allclose(Q.T @ Q, eye_k), "Input should be orthonormal"


def test_qdeim_preserves_span(orthonormal_matrix):
    """Test that the selected rows preserve the span."""
    Q = orthonormal_matrix
    k = Q.shape[1]
    
    p, P_U = QDEIM(Q, return_projector=True)
    
    # For any vector in span(Q), interpolation at p should reconstruct it
    test_vec = Q @ np.random.randn(k)
    reconstructed = P_U @ test_vec[p]
    
    assert np.allclose(test_vec, reconstructed), "Interpolation should preserve span"


# ===========================
# EXTRA_ARGS TESTS
# ===========================

def test_qdeim_with_qr_kwargs(orthonormal_matrix):
    """Test QDEIM with extra QR arguments."""
    Q = orthonormal_matrix
    
    # Test with different QR modes - scipy doesn't have mode for qr with pivoting
    # but we can test that extra_args doesn't break things
    p1 = QDEIM(Q, qr_kwargs={})
    p2 = QDEIM(Q)
    
    # Results should be the same
    assert np.array_equal(p1, p2), "Empty qr_kwargs should give same result"


def test_qdeim_with_solve_kwargs(orthonormal_matrix):
    """Test QDEIM with extra solve arguments."""
    Q = orthonormal_matrix
    
    # Test with solve options
    p1, P_U1 = QDEIM(Q, return_projector=True, solve_kwargs={'assume_a': 'gen'})
    p2, P_U2 = QDEIM(Q, return_projector=True)
    
    # Results should be very similar
    assert np.allclose(P_U1, P_U2), "solve_kwargs shouldn't significantly change result"


# ===========================
# NUMERICAL STABILITY TESTS
# ===========================

def test_qdeim_condition_number_bound(orthonormal_matrix):
    """Test that the condition number of U[p, :] is bounded."""
    Q = orthonormal_matrix
    n, k = Q.shape
    
    p = QDEIM(Q)
    U_p = Q[p, :]
    
    # Compute smallest singular value
    sigma_min = la.svdvals(U_p)[-1]
    
    # sigma_min should not be too small (matrix should be well-conditioned)
    assert sigma_min > 1e-10, f"Smallest singular value {sigma_min} is too small"


def test_qdeim_reconstruction_accuracy(orthonormal_matrix):
    """Test reconstruction accuracy using QDEIM interpolation."""
    Q = orthonormal_matrix
    k = Q.shape[1]
    
    p, P_U = QDEIM(Q, return_projector=True)
    
    # Test reconstruction of each column
    for i in range(k):
        col = Q[:, i]
        reconstructed = P_U @ col[p]
        error = la.norm(col - reconstructed)
        
        # Since Q is orthonormal, reconstruction should be exact
        assert error < 1e-10, f"Reconstruction error {error} too large for column {i}"


# ===========================
# COMPARISON WITH STANDARD DEIM
# ===========================

def test_qdeim_vs_deim_comparison(orthonormal_matrix):
    """Compare QDEIM with standard DEIM (if available)."""
    from lowrank import DEIM
    
    Q = orthonormal_matrix
    k = Q.shape[1]
    
    # Get indices from both methods
    p_qdeim = QDEIM(Q)
    p_deim = DEIM(Q)
    
    # Both should return k indices
    assert len(p_qdeim) == k, "QDEIM should return k indices"
    assert len(p_deim) == k, "DEIM should return k indices"
    
    # Check condition numbers
    U_p_qdeim = Q[p_qdeim, :]
    U_p_deim = Q[p_deim, :]
    
    cond_qdeim = np.linalg.cond(U_p_qdeim)
    cond_deim = np.linalg.cond(U_p_deim)
    
    # QDEIM typically provides better (or at least comparable) conditioning
    # We just check both are reasonable
    assert cond_qdeim < 1e10, f"QDEIM condition number {cond_qdeim} too large"
    assert cond_deim < 1e10, f"DEIM condition number {cond_deim} too large"
    
    print(f"QDEIM condition number: {cond_qdeim:.2e}")
    print(f"DEIM condition number: {cond_deim:.2e}")


# ===========================
# ERROR HANDLING
# ===========================

def test_qdeim_non_orthonormal_warning():
    """Test QDEIM with non-orthonormal input (should still work but may not satisfy bounds)."""
    np.random.seed(48)
    # Create a non-orthonormal matrix
    A = np.random.randn(20, 5)
    
    # QDEIM should still run (it's the user's responsibility to provide orthonormal input)
    p = QDEIM(A)
    
    assert len(p) == 5, "Should return k indices even for non-orthonormal input"


def test_qdeim_return_inverse_without_projector():
    """Test that return_inverse=True without return_projector=True returns only indices."""
    np.random.seed(49)
    A = np.random.randn(20, 5)
    Q, _ = la.qr(A, mode='economic')
    
    # return_inverse=True but return_projector=False
    result = QDEIM(Q, return_inverse=True)
    
    # Should only return indices
    assert isinstance(result, np.ndarray), "Should return only indices"
    assert result.ndim == 1, "Should be 1D array of indices"


# ===========================
# REPRODUCIBILITY TEST
# ===========================

def test_qdeim_reproducibility():
    """Test that QDEIM gives consistent results for the same input."""
    np.random.seed(50)
    A = np.random.randn(25, 6)
    Q, _ = la.qr(A, mode='economic')
    
    p1 = QDEIM(Q)
    p2 = QDEIM(Q)
    
    assert np.array_equal(p1, p2), "QDEIM should be deterministic"


# ===========================
# DOCUMENTATION EXAMPLE TEST
# ===========================

def test_qdeim_documentation_example():
    """Test example that could be in documentation."""
    # Create a simple orthonormal basis
    np.random.seed(100)
    A = np.random.randn(50, 8)
    U, _ = la.qr(A, mode='economic')
    
    # Basic usage
    p = QDEIM(U)
    assert len(p) == 8, "Should select 8 rows"
    
    # With projector
    p, P_U = QDEIM(U, return_projector=True)
    assert P_U.shape == (50, 8), "Projector should be n x k"
    
    # Verify interpolation property
    test_vector = U @ np.random.randn(8)
    interpolated = P_U @ test_vector[p]
    assert np.allclose(test_vector, interpolated, rtol=1e-10), "Should interpolate exactly"
    
    # With both outputs
    p, P_U, inv_U = QDEIM(U, return_projector=True, return_inverse=True)
    assert inv_U.shape == (8, 8), "Inverse should be k x k"
    assert np.allclose(P_U, U @ inv_U), "Relationship should hold"
    
    print("Documentation example test passed!")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
