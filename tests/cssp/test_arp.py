import numpy as np
import scipy.linalg as la
import pytest
from lowrank.cssp import ARP


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def orthonormal_matrix():
    """Create a standard orthonormal matrix for testing."""
    np.random.seed(42)
    n, r = 30, 6
    A = np.random.randn(n, r)
    Q, _ = la.qr(A, mode='economic')
    return Q


@pytest.fixture
def complex_orthonormal_matrix():
    """Create a complex orthonormal matrix for testing."""
    np.random.seed(43)
    n, r = 40, 8
    A = np.random.randn(n, r) + 1j * np.random.randn(n, r)
    Q, _ = la.qr(A, mode='economic')
    return Q


@pytest.fixture
def tall_orthonormal_matrix():
    """Create a very tall orthonormal matrix."""
    np.random.seed(44)
    n, r = 100, 5
    A = np.random.randn(n, r)
    Q, _ = la.qr(A, mode='economic')
    return Q


@pytest.fixture
def square_orthonormal_matrix():
    """Create a square orthonormal matrix (n = r)."""
    np.random.seed(45)
    n = 10
    A = np.random.randn(n, n)
    Q, _ = la.qr(A, mode='economic')
    return Q


@pytest.fixture
def rank_one_matrix():
    """Create a rank-1 orthonormal matrix."""
    np.random.seed(46)
    n = 20
    v = np.random.randn(n, 1)
    Q, _ = la.qr(v, mode='economic')
    return Q


# ============================================================================
# Basic Functionality Tests
# ============================================================================

def test_arp_basic(orthonormal_matrix):
    """Test basic ARP functionality."""
    Q = orthonormal_matrix
    n, r = Q.shape
    
    J = ARP(Q, seed=42)
    
    assert isinstance(J, np.ndarray), "Should return ndarray"
    assert len(J) == r, f"Should select r={r} indices"
    assert len(np.unique(J)) == r, "Indices should be unique"
    assert np.all(J >= 0) and np.all(J < n), "Indices should be in valid range"


def test_arp_reproducible_with_seed(orthonormal_matrix):
    """Test that ARP is reproducible with same seed."""
    Q = orthonormal_matrix
    
    J1 = ARP(Q, seed=42)
    J2 = ARP(Q, seed=42)
    
    assert np.array_equal(J1, J2), "Same seed should give same results"


def test_arp_different_with_different_seed(orthonormal_matrix):
    """Test that ARP gives different results with different seeds."""
    Q = orthonormal_matrix
    
    J1 = ARP(Q, seed=42)
    J2 = ARP(Q, seed=123)
    
    # Should be different (with very high probability)
    assert not np.array_equal(J1, J2), "Different seeds should give different results"


def test_arp_return_projector(orthonormal_matrix):
    """Test ARP with return_projector=True."""
    Q = orthonormal_matrix
    n, r = Q.shape
    
    J, P_U = ARP(Q, seed=42, return_projector=True)
    
    assert isinstance(J, np.ndarray), "J should be ndarray"
    assert isinstance(P_U, np.ndarray), "P_U should be ndarray"
    assert P_U.shape == (n, r), f"P_U shape should be ({n}, {r})"
    assert len(J) == r, f"Should select r={r} indices"


def test_arp_return_both(orthonormal_matrix):
    """Test ARP with both return_projector=True and return_inverse=True."""
    Q = orthonormal_matrix
    n, r = Q.shape
    
    J, P_U, inv_U = ARP(Q, seed=42, return_projector=True, return_inverse=True)
    
    assert isinstance(J, np.ndarray), "J should be ndarray"
    assert isinstance(P_U, np.ndarray), "P_U should be ndarray"
    assert isinstance(inv_U, np.ndarray), "inv_U should be ndarray"
    assert P_U.shape == (n, r), f"P_U shape should be ({n}, {r})"
    assert inv_U.shape == (r, r), f"inv_U shape should be ({r}, {r})"


def test_arp_return_inverse_without_projector():
    """Test that return_inverse=True without return_projector=True returns only indices."""
    np.random.seed(47)
    A = np.random.randn(25, 6)
    Q, _ = la.qr(A, mode='economic')
    
    result = ARP(Q, seed=42, return_projector=False, return_inverse=True)
    
    # Should return only indices since return_projector=False
    assert isinstance(result, np.ndarray), "Should return ndarray"
    assert result.ndim == 1, "Should be 1D array of indices"
    assert len(result) == 6, "Should select r=6 indices"


# ============================================================================
# Randomization Tests
# ============================================================================

def test_arp_randomization_varies_results():
    """Test that without seed, ARP gives different results each run."""
    np.random.seed(48)
    n, r = 30, 6
    A = np.random.randn(n, r)
    Q, _ = la.qr(A, mode='economic')
    
    # Run multiple times without seed
    results = [ARP(Q, seed=None) for _ in range(5)]
    
    # At least some should be different (very high probability)
    unique_results = [tuple(J) for J in results]
    assert len(set(unique_results)) > 1, "Should have variation without seed"


def test_arp_sampling_probabilities():
    """Test that ARP samples according to row norms."""
    np.random.seed(49)
    n, r = 50, 5
    
    # Create matrix where certain rows have larger norms
    A = np.random.randn(n, r) * 0.1
    # Make rows 10, 20, 30, 40 have much larger values
    A[[10, 20, 30, 40], :] *= 10
    Q, _ = la.qr(A, mode='economic')
    
    # Run ARP multiple times
    selections = []
    for seed in range(20):
        J = ARP(Q, seed=seed)
        selections.append(J[0])  # First selected index
    
    # High-norm rows should be selected more often (in expectation)
    # At least one of the high-norm rows should appear frequently
    selection_counts = np.bincount(selections, minlength=n)
    # Check that high-norm indices are selected
    high_norm_count = sum(selection_counts[[10, 20, 30, 40]])
    assert high_norm_count > 5, "High-norm rows should be selected more frequently"


# ============================================================================
# Selection Quality Tests
# ============================================================================

def test_arp_selected_submatrix_invertible(orthonormal_matrix):
    """Test that U[J, :] is invertible (well-conditioned)."""
    Q = orthonormal_matrix
    r = Q.shape[1]
    
    J = ARP(Q, seed=42)
    U_J = Q[J, :]
    
    # Check that it's full rank
    assert np.linalg.matrix_rank(U_J) == r, "Selected submatrix should be full rank"
    
    # Check condition number is reasonable
    cond = np.linalg.cond(U_J)
    assert cond < 1e10, f"Condition number {cond} too large"


def test_arp_selection_quality():
    """Test that ARP provides good approximation quality."""
    np.random.seed(50)
    n, r = 50, 8
    A = np.random.randn(n, r)
    Q, _ = la.qr(A, mode='economic')
    
    J, P_U = ARP(Q, seed=42, return_projector=True)
    
    # Test reconstruction accuracy for columns of Q
    for i in range(r):
        test_vec = Q[:, i]
        reconstructed = P_U @ test_vec[J]
        error = la.norm(test_vec - reconstructed) / la.norm(test_vec)
        
        # Should have good relative error
        assert error < 1e-9, f"Reconstruction error {error} too large for column {i}"


# ============================================================================
# Projector and Inverse Tests
# ============================================================================

def test_arp_projector_computation(orthonormal_matrix):
    """Test that P_U satisfies the interpolation property."""
    Q = orthonormal_matrix
    r = Q.shape[1]
    
    J, P_U = ARP(Q, seed=42, return_projector=True)
    
    # Test interpolation: P_U @ v[J] ≈ v for v in span(Q)
    for i in range(r):
        test_vec = Q[:, i]
        interpolated = P_U @ test_vec[J]
        error = la.norm(test_vec - interpolated) / la.norm(test_vec)
        
        assert error < 1e-9, f"Interpolation error {error} too large for column {i}"


def test_arp_inverse_computation(orthonormal_matrix):
    """Test that inv_U is computed correctly."""
    Q = orthonormal_matrix
    
    J, P_U, inv_U = ARP(Q, seed=42, return_projector=True, return_inverse=True)
    
    # Verify relationship: inv_U = U.T @ P_U
    expected_inv_U = Q.T.conj() @ P_U
    
    assert np.allclose(inv_U, expected_inv_U, rtol=1e-9), "inv_U != U.T @ P_U"


def test_arp_projector_inverse_relationship(orthonormal_matrix):
    """Test relationship between projector and inverse."""
    Q = orthonormal_matrix
    
    J, P_U, inv_U = ARP(Q, seed=42, return_projector=True, return_inverse=True)
    
    # inv_U should be close to inv(U[J, :])
    U_J = Q[J, :]
    expected_inv = la.inv(U_J)
    
    assert np.allclose(inv_U, expected_inv, rtol=1e-9), "inv_U should equal inv(U[J, :])"


def test_arp_reconstruction_accuracy(orthonormal_matrix):
    """Test reconstruction accuracy of approximation."""
    Q = orthonormal_matrix
    n, r = Q.shape
    
    J, P_U = ARP(Q, seed=42, return_projector=True)
    
    # For a vector in the column space, reconstruction should be exact
    test_vec = Q @ np.random.randn(r)
    reconstructed = P_U @ test_vec[J]
    
    error = la.norm(test_vec - reconstructed) / la.norm(test_vec)
    assert error < 1e-9, f"Reconstruction error {error} for vector in span(Q)"


# ============================================================================
# Complex Matrix Tests
# ============================================================================

def test_arp_complex(complex_orthonormal_matrix):
    """Test ARP with complex matrices."""
    Q = complex_orthonormal_matrix
    n, r = Q.shape
    
    J = ARP(Q, seed=42)
    
    assert len(J) == r, f"Should select r={r} indices"
    assert len(np.unique(J)) == r, "Indices should be unique"
    
    # Check that selected submatrix is invertible
    U_J = Q[J, :]
    assert np.linalg.matrix_rank(U_J) == r, "Selected submatrix should be full rank"


def test_arp_complex_projector(complex_orthonormal_matrix):
    """Test ARP projector with complex matrices."""
    Q = complex_orthonormal_matrix
    n, r = Q.shape
    
    J, P_U = ARP(Q, seed=42, return_projector=True)
    
    assert P_U.shape == (n, r), f"P_U shape correct for complex matrix"
    assert P_U.dtype == np.complex128 or np.iscomplexobj(P_U), "P_U should be complex"
    
    # Test interpolation
    for i in range(r):
        test_vec = Q[:, i]
        interpolated = P_U @ test_vec[J]
        error = la.norm(test_vec - interpolated) / la.norm(test_vec)
        assert error < 1e-9, f"Complex interpolation error {error} too large"


def test_arp_complex_inverse(complex_orthonormal_matrix):
    """Test ARP with complex matrices and return_inverse=True."""
    Q = complex_orthonormal_matrix
    
    J, P_U, inv_U = ARP(Q, seed=42, return_projector=True, return_inverse=True)
    
    assert inv_U.dtype == np.complex128 or np.iscomplexobj(inv_U), "inv_U should be complex"
    
    # Verify relationship
    expected_inv_U = Q.T.conj() @ P_U
    assert np.allclose(inv_U, expected_inv_U, rtol=1e-9), "inv_U relationship incorrect"


def test_arp_complex_reproducible(complex_orthonormal_matrix):
    """Test that ARP with complex matrices is reproducible."""
    Q = complex_orthonormal_matrix
    
    J1 = ARP(Q, seed=42)
    J2 = ARP(Q, seed=42)
    
    assert np.array_equal(J1, J2), "Same seed should give same results for complex matrix"


# ============================================================================
# Edge Case Tests
# ============================================================================

def test_arp_tall_matrix(tall_orthonormal_matrix):
    """Test ARP with very tall matrix (n >> r)."""
    Q = tall_orthonormal_matrix
    n, r = Q.shape
    assert n > 10 * r, "Should be very tall"
    
    J = ARP(Q, seed=42)
    
    assert len(J) == r, f"Should select r={r} indices"
    assert len(np.unique(J)) == r, "All indices should be unique"
    
    # Check selected submatrix is well-conditioned
    U_J = Q[J, :]
    cond = np.linalg.cond(U_J)
    assert cond < 1e10, f"Condition number {cond} too large for tall matrix"


def test_arp_square_matrix(square_orthonormal_matrix):
    """Test ARP with square orthonormal matrix (n = r)."""
    Q = square_orthonormal_matrix
    n, r = Q.shape
    assert n == r, "Should be square"
    
    J = ARP(Q, seed=42)
    
    assert len(J) == r, f"Should select all r={r} indices"
    assert len(np.unique(J)) == r, "All indices should be unique"
    
    # For square orthonormal matrix, should be well-conditioned
    U_J = Q[J, :]
    cond = np.linalg.cond(U_J)
    assert cond < 1e10, f"Condition number {cond} too large"


def test_arp_rank_one(rank_one_matrix):
    """Test ARP with rank-1 matrix."""
    Q = rank_one_matrix
    n, r = Q.shape
    assert r == 1, "Should be rank-1"
    
    J = ARP(Q, seed=42)
    
    assert len(J) == 1, "Should select 1 index"
    
    # The selected row should have large norm
    selected_value = Q[J[0], 0]
    assert np.abs(selected_value) > 1e-10, "Selected value should be non-zero"


def test_arp_rank_deficient_input():
    """Test ARP behavior with rank-deficient input."""
    np.random.seed(51)
    n, r = 30, 5
    # Create rank-deficient matrix (rank 3 < r)
    A = np.random.randn(n, 3) @ np.random.randn(3, r)
    
    # Even though input is rank-deficient, algorithm should run
    J = ARP(A, seed=42)
    
    assert len(J) == r, f"Should still select r={r} indices"


# ============================================================================
# Error Handling Tests
# ============================================================================

def test_arp_r_greater_than_n():
    """Test error when r > n."""
    np.random.seed(52)
    n, r = 5, 10  # r > n
    A = np.random.randn(n, r)
    
    with pytest.raises(ValueError, match="Number of columns r must be less than or equal to number of rows n"):
        ARP(A, seed=42)


def test_arp_empty_matrix():
    """Test with empty matrix."""
    A = np.array([]).reshape(0, 0)
    
    # Should handle gracefully or raise error
    try:
        J = ARP(A, seed=42)
        assert len(J) == 0, "Empty matrix should return empty selection"
    except (ValueError, IndexError):
        # Also acceptable to raise error
        pass


def test_arp_single_column():
    """Test with single column matrix."""
    np.random.seed(53)
    n = 20
    v = np.random.randn(n, 1)
    Q, _ = la.qr(v, mode='economic')
    
    J = ARP(Q, seed=42)
    
    assert len(J) == 1, "Should select 1 index"
    assert Q[J[0], 0] != 0, "Selected element should be non-zero"


# ============================================================================
# Numerical Stability Tests
# ============================================================================

def test_arp_numerical_stability():
    """Test numerical stability with various matrix scales."""
    np.random.seed(54)
    
    for scale in [1e-5, 1.0, 1e5]:
        n, r = 28, 6
        A = np.random.randn(n, r) * scale
        Q, _ = la.qr(A, mode='economic')
        
        J = ARP(Q, seed=42)
        
        assert len(J) == r, f"Should select r={r} indices for scale {scale}"
        assert len(np.unique(J)) == r, f"Indices should be unique for scale {scale}"
        
        # Check condition number
        U_J = Q[J, :]
        cond = np.linalg.cond(U_J)
        assert cond < 1e10, f"Condition number {cond} too large for scale {scale}"


def test_arp_near_collinear_columns():
    """Test with nearly collinear columns."""
    np.random.seed(55)
    n = 30
    # Create base vector
    v1 = np.random.randn(n)
    v1 /= la.norm(v1)
    
    # Create nearly collinear vectors
    r = 4
    A = np.column_stack([v1 + 1e-6 * np.random.randn(n) for _ in range(r)])
    Q, _ = la.qr(A, mode='economic')
    
    J = ARP(Q, seed=42)
    
    assert len(J) == r, f"Should select r={r} indices"
    # May have higher condition number but should still work
    U_J = Q[J, :]
    assert np.linalg.matrix_rank(U_J) == r, "Should be full rank"


# ============================================================================
# Householder Helper Tests
# ============================================================================

def test_householder_vector_real():
    """Test _householder_vector with real input."""
    from lowrank.cssp.arp import _householder_vector
    
    x = np.array([3.0, 4.0, 0.0])
    v, beta = _householder_vector(x)
    
    assert v.shape == x.shape, "v should have same shape as x"
    assert beta >= 0, "beta should be non-negative"
    
    # Apply reflection: (I - beta * v * v^T) @ x should be [±||x||, 0, 0]
    H = np.eye(len(x)) - beta * np.outer(v, v)
    result = H @ x
    
    assert np.abs(result[1]) < 1e-10, "Second component should be zero"
    assert np.abs(result[2]) < 1e-10, "Third component should be zero"


def test_householder_vector_complex():
    """Test _householder_vector with complex input."""
    from lowrank.cssp.arp import _householder_vector
    
    x = np.array([1+2j, 3-1j, 2+2j])
    v, beta = _householder_vector(x)
    
    assert v.shape == x.shape, "v should have same shape as x"
    assert beta >= 0, "beta should be non-negative"
    
    # Apply reflection
    H = np.eye(len(x), dtype=complex) - beta * np.outer(v, v.conj())
    result = H @ x
    
    assert np.abs(result[1]) < 1e-10, "Second component should be zero"
    assert np.abs(result[2]) < 1e-10, "Third component should be zero"


def test_householder_vector_zero_input():
    """Test _householder_vector with zero input."""
    from lowrank.cssp.arp import _householder_vector
    
    x = np.array([0.0, 0.0, 0.0])
    v, beta = _householder_vector(x)
    
    assert np.allclose(v, 0), "v should be zero for zero input"
    assert beta == 0, "beta should be zero for zero input"


def test_apply_householder_right():
    """Test _apply_householder_right."""
    from lowrank.cssp.arp import _apply_householder_right, _householder_vector
    
    A = np.random.randn(5, 3)
    x = np.array([3.0, 4.0, 1.0])
    v, beta = _householder_vector(x)
    
    A_updated = _apply_householder_right(A, v, beta)
    
    # A_updated should equal A @ (I - beta * v * v^T)
    H = np.eye(len(x)) - beta * np.outer(v, v)
    expected = A @ H
    
    assert np.allclose(A_updated, expected, atol=1e-10), "Householder application incorrect"


# ============================================================================
# Documentation Example Tests
# ============================================================================

def test_arp_documentation_real_example():
    """Test the real example from the documentation."""
    np.random.seed(0)
    n, r = 10, 4
    U = np.random.randn(n, r)
    U, _ = la.qr(U, mode='economic')
    
    J = ARP(U, seed=42)
    
    assert len(J) == r, "Should select r indices"
    assert U[J, :].shape == (r, r), "Selected submatrix should be r x r"


def test_arp_documentation_complex_example():
    """Test the complex example from the documentation."""
    np.random.seed(0)
    n, r = 10, 4
    U_complex = np.random.randn(n, r) + 1j * np.random.randn(n, r)
    U_complex, _ = la.qr(U_complex, mode='economic')
    
    J_complex = ARP(U_complex, seed=42)
    
    assert len(J_complex) == r, "Should select r indices"
    assert U_complex[J_complex, :].shape == (r, r), "Selected submatrix should be r x r"


# ============================================================================
# Comparison Tests
# ============================================================================

def test_arp_orthonormality_preserved():
    """Test that orthonormality of input is utilized."""
    np.random.seed(56)
    n, r = 35, 7
    A = np.random.randn(n, r)
    Q, _ = la.qr(A, mode='economic')
    
    # Verify Q is orthonormal
    assert np.allclose(Q.T @ Q, np.eye(r), atol=1e-10), "Q should be orthonormal"
    
    J, P_U = ARP(Q, seed=42, return_projector=True)
    
    # For orthonormal Q, reconstruction should be very accurate
    for i in range(r):
        test_vec = Q[:, i]
        reconstructed = P_U @ test_vec[J]
        error = la.norm(test_vec - reconstructed) / la.norm(test_vec)
        assert error < 1e-9, f"Should have excellent reconstruction for orthonormal input"


def test_arp_indices_validity():
    """Test that returned indices are valid."""
    Q = np.random.randn(30, 6)
    Q, _ = la.qr(Q, mode='economic')
    n, r = Q.shape
    
    J = ARP(Q, seed=42)
    
    # All indices should be valid
    assert np.all(J >= 0), "All indices should be non-negative"
    assert np.all(J < n), f"All indices should be less than n={n}"
    
    # Should be integers
    assert J.dtype in [np.int32, np.int64, int], "Indices should be integers"
    
    # Should be unique
    assert len(J) == len(np.unique(J)), "Indices should be unique"
