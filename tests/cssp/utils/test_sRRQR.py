"""
Test file for sRRQR.py (Strong Rank-Revealing QR)

Tests for the Strong Rank-Revealing QR factorization implementation
based on Gu & Eisenstat (1996).

Author: Benjamin Carrel, University of Geneva, 2024
"""

import numpy as np
import scipy.linalg as la
from lowrank.cssp.utils import sRRQR, sRRQR_rank, sRRQR_tol
import pytest


# ===========================
# FIXTURES
# ===========================

@pytest.fixture
def low_rank_matrix():
    """Create a low-rank matrix with known structure."""
    np.random.seed(42)
    m, n, k = 50, 40, 5
    U = np.random.randn(m, k)
    V = np.random.randn(n, k)
    A = U @ V.T
    return A


@pytest.fixture
def tall_low_rank_matrix():
    """Create a tall low-rank matrix."""
    np.random.seed(43)
    m, n, k = 100, 20, 8
    U = np.random.randn(m, k)
    V = np.random.randn(n, k)
    A = U @ V.T
    return A


@pytest.fixture
def wide_low_rank_matrix():
    """Create a wide low-rank matrix."""
    np.random.seed(44)
    m, n, k = 20, 60, 6
    U = np.random.randn(m, k)
    V = np.random.randn(n, k)
    A = U @ V.T
    return A


@pytest.fixture
def complex_low_rank_matrix():
    """Create a complex low-rank matrix."""
    np.random.seed(45)
    m, n, k = 40, 30, 5
    U = np.random.randn(m, k) + 1j * np.random.randn(m, k)
    V = np.random.randn(n, k) + 1j * np.random.randn(n, k)
    A = U @ V.T.conj()
    return A


@pytest.fixture
def full_rank_matrix():
    """Create a full-rank matrix."""
    np.random.seed(46)
    n = 30
    A = np.random.randn(n, n)
    # Ensure full rank by adding identity scaled
    A = A + 0.1 * np.eye(n)
    return A


@pytest.fixture
def diagonal_matrix():
    """Create a diagonal matrix with decreasing values."""
    np.random.seed(47)
    n = 20
    diag_vals = np.logspace(0, -10, n)  # 1 to 1e-10
    A = np.diag(diag_vals)
    return A


# ===========================
# BASIC FUNCTIONALITY TESTS
# ===========================

def test_sRRQR_rank_basic(low_rank_matrix):
    """Test basic sRRQR_rank functionality."""
    A = low_rank_matrix
    m, n = A.shape
    k = 5
    eta = 2.0
    
    Q, R, p = sRRQR_rank(A, eta, k)
    
    # Check shapes
    assert Q.shape == (m, k), f"Q shape should be ({m}, {k}), got {Q.shape}"
    assert R.shape == (k, n), f"R shape should be ({k}, {n}), got {R.shape}"
    assert len(p) == n, f"Permutation should have {n} elements, got {len(p)}"
    
    # Check that p is a valid permutation
    assert set(p) == set(range(n)), "p should be a permutation of [0, n)"


def test_sRRQR_rank_orthogonality(low_rank_matrix):
    """Test that Q is orthonormal."""
    A = low_rank_matrix
    k = 5
    eta = 2.0
    
    Q, R, p = sRRQR_rank(A, eta, k)
    
    # Check Q is orthonormal
    QtQ = Q.T @ Q
    eye_k = np.eye(k)
    
    assert np.allclose(QtQ, eye_k, rtol=1e-10), "Q should be orthonormal"


def test_sRRQR_rank_triangular(low_rank_matrix):
    """Test that R is upper triangular."""
    A = low_rank_matrix
    k = 5
    eta = 2.0
    
    Q, R, p = sRRQR_rank(A, eta, k)
    
    # Check R is upper triangular (up to numerical errors)
    R_lower = np.tril(R, -1)
    
    assert np.allclose(R_lower, 0, atol=1e-12), "R should be upper triangular"


def test_sRRQR_rank_factorization(low_rank_matrix):
    """Test that A[:, p] = Q @ R."""
    A = low_rank_matrix
    k = 5
    eta = 2.0
    
    Q, R, p = sRRQR_rank(A, eta, k)
    
    # Reconstruct A with permutation
    A_permuted = A[:, p]
    reconstructed = Q @ R
    
    # Check that first k columns match well
    error = np.linalg.norm(A_permuted[:, :k] - reconstructed[:, :k], 'fro')
    
    assert error < 1e-10, f"Factorization error {error} too large"


def test_sRRQR_rank_eta_bound(low_rank_matrix):
    """Test that the eta bound on inv(R11) * R12 is satisfied."""
    A = low_rank_matrix
    k = 5
    eta = 2.0
    
    Q, R, p = sRRQR_rank(A, eta, k)
    
    # Extract R11 and R12
    R11 = R[:k, :k]
    R12 = R[:k, k:]
    
    # Compute inv(R11) * R12
    inv_R11_R12 = la.solve_triangular(R11, R12, lower=False)
    
    # Check eta bound
    max_entry = np.max(np.abs(inv_R11_R12))
    
    assert max_entry <= eta + 1e-10, f"Eta bound violated: {max_entry} > {eta}"


# ===========================
# SRRQR_TOL TESTS
# ===========================

def test_sRRQR_tol_basic(diagonal_matrix):
    """Test basic sRRQR_tol functionality."""
    A = diagonal_matrix
    m, n = A.shape
    eta = 2.0
    tol = 1e-5
    
    Q, R, p = sRRQR_tol(A, eta, tol)
    
    k = Q.shape[1]
    
    # Check shapes
    assert Q.shape == (m, k), f"Q shape should be ({m}, {k})"
    assert R.shape == (k, n), f"R shape should be ({k}, {n})"
    assert len(p) == n, f"Permutation should have {n} elements"


def test_sRRQR_tol_rank_detection(diagonal_matrix):
    """Test that sRRQR_tol correctly detects rank based on tolerance."""
    A = diagonal_matrix
    eta = 2.0
    tol = 1e-5
    
    Q, R, p = sRRQR_tol(A, eta, tol)
    
    # Count singular values above tolerance
    s = la.svdvals(A)
    expected_rank = np.sum(s > tol)
    
    k = Q.shape[1]
    
    # Rank should be close to expected (within a few due to algorithm specifics)
    assert abs(k - expected_rank) <= 2, f"Rank {k} far from expected {expected_rank}"


def test_sRRQR_tol_orthogonality(low_rank_matrix):
    """Test that Q from sRRQR_tol is orthonormal."""
    A = low_rank_matrix
    eta = 2.0
    tol = 1e-4
    
    Q, R, p = sRRQR_tol(A, eta, tol)
    
    k = Q.shape[1]
    if k > 0:
        QtQ = Q.T @ Q
        eye_k = np.eye(k)
        
        assert np.allclose(QtQ, eye_k, rtol=1e-10), "Q should be orthonormal"


def test_sRRQR_tol_triangular(low_rank_matrix):
    """Test that R from sRRQR_tol is upper triangular."""
    A = low_rank_matrix
    eta = 2.0
    tol = 1e-4
    
    Q, R, p = sRRQR_tol(A, eta, tol)
    
    if R.shape[0] > 0:
        k = R.shape[0]
        R_lower = np.tril(R[:, :k], -1)
        
        assert np.allclose(R_lower, 0, atol=1e-12), "R should be upper triangular"


def test_sRRQR_tol_eta_bound(low_rank_matrix):
    """Test that the eta bound is satisfied for sRRQR_tol."""
    A = low_rank_matrix
    eta = 2.0
    tol = 1e-4
    
    Q, R, p = sRRQR_tol(A, eta, tol)
    
    k = R.shape[0]
    if k > 0 and k < R.shape[1]:
        R11 = R[:k, :k]
        R12 = R[:k, k:]
        
        inv_R11_R12 = la.solve_triangular(R11, R12, lower=False)
        max_entry = np.max(np.abs(inv_R11_R12))
        
        assert max_entry <= eta + 1e-10, f"Eta bound violated: {max_entry} > {eta}"


# ===========================
# DISPATCHER TESTS
# ===========================

def test_sRRQR_dispatcher_rank_mode(low_rank_matrix):
    """Test sRRQR dispatcher with rank mode."""
    A = low_rank_matrix
    eta = 2.0
    k = 5
    
    Q1, R1, p1 = sRRQR(A, eta, 'rank', k)
    Q2, R2, p2 = sRRQR_rank(A, eta, k)
    
    # Results should be identical
    assert np.allclose(Q1, Q2), "Dispatcher should match sRRQR_rank for Q"
    assert np.allclose(R1, R2), "Dispatcher should match sRRQR_rank for R"
    assert np.array_equal(p1, p2), "Dispatcher should match sRRQR_rank for p"


def test_sRRQR_dispatcher_tol_mode(low_rank_matrix):
    """Test sRRQR dispatcher with tol mode."""
    A = low_rank_matrix
    eta = 2.0
    tol = 1e-4
    
    Q1, R1, p1 = sRRQR(A, eta, 'tol', tol)
    Q2, R2, p2 = sRRQR_tol(A, eta, tol)
    
    # Results should be identical
    assert np.allclose(Q1, Q2), "Dispatcher should match sRRQR_tol for Q"
    assert np.allclose(R1, R2), "Dispatcher should match sRRQR_tol for R"
    assert np.array_equal(p1, p2), "Dispatcher should match sRRQR_tol for p"


def test_sRRQR_dispatcher_invalid_mode():
    """Test that sRRQR raises error for invalid mode."""
    A = np.random.randn(10, 10)
    eta = 2.0
    
    with pytest.raises(ValueError, match="Unknown mode"):
        sRRQR(A, eta, 'invalid', 5)


# ===========================
# COMPLEX MATRIX TESTS
# ===========================

def test_sRRQR_rank_complex(complex_low_rank_matrix):
    """Test sRRQR_rank with complex matrices."""
    A = complex_low_rank_matrix
    m, n = A.shape
    k = 5
    eta = 2.0
    
    Q, R, p = sRRQR_rank(A, eta, k)
    
    # Check shapes
    assert Q.shape == (m, k), f"Q shape should be ({m}, {k})"
    assert R.shape == (k, n), f"R shape should be ({k}, {n})"
    
    # Check Q is unitary
    QtQ = Q.T.conj() @ Q
    eye_k = np.eye(k)
    
    assert np.allclose(QtQ, eye_k, rtol=1e-10), "Q should be unitary for complex matrices"


def test_sRRQR_rank_complex_triangular(complex_low_rank_matrix):
    """Test that R is upper triangular for complex matrices."""
    A = complex_low_rank_matrix
    k = 5
    eta = 2.0
    
    Q, R, p = sRRQR_rank(A, eta, k)
    
    # Check R is upper triangular
    R_lower = np.tril(R, -1)
    
    assert np.allclose(R_lower, 0, atol=1e-12), "R should be upper triangular for complex"


def test_sRRQR_rank_complex_factorization(complex_low_rank_matrix):
    """Test factorization accuracy for complex matrices."""
    A = complex_low_rank_matrix
    k = 5
    eta = 2.0
    
    Q, R, p = sRRQR_rank(A, eta, k)
    
    # Reconstruct A with permutation
    A_permuted = A[:, p]
    reconstructed = Q @ R
    
    error = np.linalg.norm(A_permuted[:, :k] - reconstructed[:, :k], 'fro')
    
    assert error < 1e-10, f"Complex factorization error {error} too large"


def test_sRRQR_tol_complex(complex_low_rank_matrix):
    """Test sRRQR_tol with complex matrices."""
    A = complex_low_rank_matrix
    m, n = A.shape
    eta = 2.0
    tol = 1e-4
    
    Q, R, p = sRRQR_tol(A, eta, tol)
    
    k = Q.shape[1]
    
    # Check Q is unitary
    if k > 0:
        QtQ = Q.T.conj() @ Q
        eye_k = np.eye(k)
        
        assert np.allclose(QtQ, eye_k, rtol=1e-10), "Q should be unitary"


def test_sRRQR_complex_eta_bound(complex_low_rank_matrix):
    """Test eta bound for complex matrices."""
    A = complex_low_rank_matrix
    k = 5
    eta = 2.0
    
    Q, R, p = sRRQR_rank(A, eta, k)
    
    R11 = R[:k, :k]
    R12 = R[:k, k:]
    
    inv_R11_R12 = la.solve_triangular(R11, R12, lower=False)
    max_entry = np.max(np.abs(inv_R11_R12))
    
    assert max_entry <= eta + 1e-10, f"Eta bound violated for complex: {max_entry} > {eta}"


# ===========================
# EDGE CASES
# ===========================

def test_sRRQR_rank_k_equals_n(low_rank_matrix):
    """Test sRRQR_rank when k equals n (full rank requested)."""
    A = low_rank_matrix
    m, n = A.shape
    k = n
    eta = 2.0
    
    Q, R, p = sRRQR_rank(A, eta, k)
    
    # k is limited by min(k, m, n)
    expected_k = min(k, m, n)
    
    assert Q.shape[1] == expected_k, f"Should return {expected_k} columns"


def test_sRRQR_rank_k_exceeds_dimensions(low_rank_matrix):
    """Test sRRQR_rank when k exceeds matrix dimensions."""
    A = low_rank_matrix
    m, n = A.shape
    k = max(m, n) + 10
    eta = 2.0
    
    Q, R, p = sRRQR_rank(A, eta, k)
    
    # k should be limited to min(m, n)
    expected_k = min(m, n)
    
    assert Q.shape[1] == expected_k, f"Should limit k to {expected_k}"


def test_sRRQR_rank_k_equals_1():
    """Test sRRQR_rank with k=1."""
    np.random.seed(50)
    A = np.random.randn(20, 15)
    k = 1
    eta = 2.0
    
    Q, R, p = sRRQR_rank(A, eta, k)
    
    assert Q.shape == (20, 1), "Should return single column"
    assert R.shape == (1, 15), "Should return single row"


def test_sRRQR_rank_tall_matrix(tall_low_rank_matrix):
    """Test sRRQR_rank with tall matrix (m >> n)."""
    A = tall_low_rank_matrix
    m, n = A.shape
    assert m > 2 * n, "Should be tall matrix"
    
    k = 8
    eta = 2.0
    
    Q, R, p = sRRQR_rank(A, eta, k)
    
    assert Q.shape == (m, k), f"Q shape correct for tall matrix"
    
    # Check orthogonality
    QtQ = Q.T @ Q
    assert np.allclose(QtQ, np.eye(k), rtol=1e-10), "Q orthonormal for tall matrix"


def test_sRRQR_rank_wide_matrix(wide_low_rank_matrix):
    """Test sRRQR_rank with wide matrix (n >> m)."""
    A = wide_low_rank_matrix
    m, n = A.shape
    assert n > 2 * m, "Should be wide matrix"
    
    k = 6
    eta = 2.0
    
    Q, R, p = sRRQR_rank(A, eta, k)
    
    assert Q.shape == (m, k), f"Q shape correct for wide matrix"
    assert R.shape == (k, n), f"R shape correct for wide matrix"


def test_sRRQR_tol_zero_tolerance():
    """Test sRRQR_tol with very small tolerance (near full rank)."""
    np.random.seed(51)
    A = np.random.randn(25, 20)
    eta = 2.0
    tol = 1e-14  # Very small
    
    Q, R, p = sRRQR_tol(A, eta, tol)
    
    k = Q.shape[1]
    
    # Should find approximately full rank
    expected_rank = min(A.shape)
    
    assert k >= expected_rank - 2, f"Should find near-full rank"


def test_sRRQR_tol_large_tolerance():
    """Test sRRQR_tol with large tolerance (low rank)."""
    A = np.diag([10.0, 5.0, 1.0, 0.1, 0.01, 0.001])
    eta = 2.0
    tol = 0.5
    
    Q, R, p = sRRQR_tol(A, eta, tol)
    
    k = Q.shape[1]
    
    # Should find rank around 3-4 (values above 0.5)
    assert k <= 4, f"Should find low rank with large tolerance"


def test_sRRQR_rank_square_matrix(full_rank_matrix):
    """Test sRRQR_rank with square full-rank matrix."""
    A = full_rank_matrix
    n = A.shape[0]
    k = n // 2
    eta = 2.0
    
    Q, R, p = sRRQR_rank(A, eta, k)
    
    assert Q.shape == (n, k), "Correct shape for square matrix"
    
    # Check orthogonality
    QtQ = Q.T @ Q
    assert np.allclose(QtQ, np.eye(k), rtol=1e-10), "Q orthonormal for square matrix"


# ===========================
# ETA PARAMETER TESTS
# ===========================

def test_sRRQR_rank_small_eta():
    """Test sRRQR_rank with small eta (tight bound)."""
    np.random.seed(52)
    A = np.random.randn(30, 25)
    k = 10
    eta = 1.001  # Very tight bound
    
    Q, R, p = sRRQR_rank(A, eta, k)
    
    # Check eta bound is satisfied
    R11 = R[:k, :k]
    R12 = R[:k, k:]
    
    inv_R11_R12 = la.solve_triangular(R11, R12, lower=False)
    max_entry = np.max(np.abs(inv_R11_R12))
    
    assert max_entry <= eta + 1e-10, f"Tight eta bound violated: {max_entry} > {eta}"


def test_sRRQR_rank_large_eta():
    """Test sRRQR_rank with large eta (loose bound)."""
    np.random.seed(53)
    A = np.random.randn(30, 25)
    k = 10
    eta = 100.0  # Very loose bound
    
    Q, R, p = sRRQR_rank(A, eta, k)
    
    # Should still produce valid factorization
    assert Q.shape == (30, k), "Valid factorization with large eta"
    
    # Check orthogonality
    QtQ = Q.T @ Q
    assert np.allclose(QtQ, np.eye(k), rtol=1e-10), "Q orthonormal with large eta"


def test_sRRQR_rank_eta_less_than_one():
    """Test that eta < 1 is automatically adjusted to 2."""
    np.random.seed(54)
    A = np.random.randn(20, 15)
    k = 5
    eta = 0.5  # Less than 1
    
    # Should not raise error, eta automatically set to 2
    Q, R, p = sRRQR_rank(A, eta, k)
    
    # Check eta bound with adjusted value (should be satisfied with eta=2)
    R11 = R[:k, :k]
    R12 = R[:k, k:]
    
    inv_R11_R12 = la.solve_triangular(R11, R12, lower=False)
    max_entry = np.max(np.abs(inv_R11_R12))
    
    assert max_entry <= 2.0 + 1e-10, "Should use eta=2 when eta<1 provided"


def test_sRRQR_tol_eta_less_than_one():
    """Test that eta < 1 is automatically adjusted in sRRQR_tol."""
    np.random.seed(55)
    A = np.random.randn(20, 15)
    tol = 1e-4
    eta = 0.8  # Less than 1
    
    # Should not raise error
    Q, R, p = sRRQR_tol(A, eta, tol)
    
    # Should produce valid factorization
    k = Q.shape[1]
    if k > 0:
        QtQ = Q.T @ Q
        assert np.allclose(QtQ, np.eye(k), rtol=1e-10), "Valid factorization despite eta<1"


# ===========================
# RECONSTRUCTION TESTS
# ===========================

def test_sRRQR_rank_reconstruction_accuracy(low_rank_matrix):
    """Test reconstruction accuracy of low-rank approximation."""
    A = low_rank_matrix
    k = 5
    eta = 2.0
    
    Q, R, p = sRRQR_rank(A, eta, k)
    
    # Reconstruct using first k columns
    A_permuted = A[:, p]
    A_approx = Q @ R[:, :k]
    
    error = np.linalg.norm(A_permuted[:, :k] - A_approx, 'fro') / np.linalg.norm(A, 'fro')
    
    assert error < 1e-10, f"Reconstruction error {error} too large"


def test_sRRQR_tol_reconstruction_accuracy(low_rank_matrix):
    """Test reconstruction accuracy for sRRQR_tol."""
    A = low_rank_matrix
    tol = 1e-4
    eta = 2.0
    
    Q, R, p = sRRQR_tol(A, eta, tol)
    
    k = Q.shape[1]
    if k > 0:
        A_permuted = A[:, p]
        A_approx = Q @ R[:, :k]
        
        error = np.linalg.norm(A_permuted[:, :k] - A_approx, 'fro') / np.linalg.norm(A, 'fro')
        
        # Error should be small for low-rank structure
        assert error < 0.1, f"Reconstruction error {error} too large"


def test_sRRQR_full_reconstruction():
    """Test that full QR decomposition reconstructs the matrix exactly."""
    np.random.seed(56)
    A = np.random.randn(20, 15)
    k = min(A.shape)
    eta = 2.0
    
    Q, R, p = sRRQR_rank(A, eta, k)
    
    # Full reconstruction
    A_permuted = A[:, p]
    A_reconstructed = Q @ R
    
    error = np.linalg.norm(A_permuted - A_reconstructed, 'fro') / np.linalg.norm(A, 'fro')
    
    assert error < 1e-12, f"Full reconstruction error {error} should be near zero"


# ===========================
# NUMERICAL STABILITY TESTS
# ===========================

def test_sRRQR_rank_diagonal_positive():
    """Test that diagonal elements of R have positive real part."""
    np.random.seed(57)
    A = np.random.randn(30, 20)
    k = 10
    eta = 2.0
    
    Q, R, p = sRRQR_rank(A, eta, k)
    
    # Check diagonal elements
    diag_R = np.diag(R[:k, :k])
    
    # Real part should be non-negative (phase normalization)
    assert np.all(np.real(diag_R) >= -1e-14), "Diagonal should have non-negative real part"


def test_sRRQR_tol_diagonal_positive():
    """Test diagonal positivity for sRRQR_tol."""
    np.random.seed(58)
    A = np.random.randn(30, 20)
    tol = 1e-4
    eta = 2.0
    
    Q, R, p = sRRQR_tol(A, eta, tol)
    
    k = Q.shape[1]
    if k > 0:
        diag_R = np.diag(R[:k, :k])
        
        assert np.all(np.real(diag_R) >= -1e-14), "Diagonal should have non-negative real part"


def test_sRRQR_rank_condition_number():
    """Test that R11 is well-conditioned."""
    np.random.seed(59)
    A = np.random.randn(40, 30)
    k = 15
    eta = 2.0
    
    Q, R, p = sRRQR_rank(A, eta, k)
    
    R11 = R[:k, :k]
    
    # Compute condition number
    cond = np.linalg.cond(R11)
    
    # Should be reasonably conditioned (not near singular)
    assert cond < 1e10, f"R11 poorly conditioned: {cond}"


# ===========================
# COMPARISON TESTS
# ===========================

def test_sRRQR_vs_standard_qr():
    """Compare sRRQR with standard QR pivoting."""
    np.random.seed(60)
    A = np.random.randn(30, 25)
    k = 10
    eta = 2.0
    
    # Standard QR with pivoting
    Q_std, R_std, p_std = la.qr(A, mode='economic', pivoting=True)
    
    # sRRQR
    Q_srrqr, R_srrqr, p_srrqr = sRRQR_rank(A, eta, k)
    
    # Both should produce valid factorizations
    # (permutations may differ, but both should be valid)
    
    # Check orthogonality
    assert np.allclose(Q_srrqr.T @ Q_srrqr, np.eye(k), rtol=1e-10), "sRRQR Q orthonormal"
    
    # Check triangular
    R_lower = np.tril(R_srrqr, -1)
    assert np.allclose(R_lower, 0, atol=1e-12), "sRRQR R triangular"


def test_sRRQR_different_eta_values():
    """Test that different eta values produce different results."""
    np.random.seed(61)
    A = np.random.randn(30, 25)
    k = 10
    
    Q1, R1, p1 = sRRQR_rank(A, 1.5, k)
    Q2, R2, p2 = sRRQR_rank(A, 5.0, k)
    
    # Both should be valid, but may differ
    # (tighter eta may require more column interchanges)
    
    # Check both satisfy their respective eta bounds
    R11_1 = R1[:k, :k]
    R12_1 = R1[:k, k:]
    max_entry_1 = np.max(np.abs(la.solve_triangular(R11_1, R12_1, lower=False)))
    
    R11_2 = R2[:k, :k]
    R12_2 = R2[:k, k:]
    max_entry_2 = np.max(np.abs(la.solve_triangular(R11_2, R12_2, lower=False)))
    
    assert max_entry_1 <= 1.5 + 1e-10, "eta=1.5 bound satisfied"
    assert max_entry_2 <= 5.0 + 1e-10, "eta=5.0 bound satisfied"


# ===========================
# PERMUTATION TESTS
# ===========================

def test_sRRQR_rank_permutation_validity(low_rank_matrix):
    """Test that permutation is valid."""
    A = low_rank_matrix
    n = A.shape[1]
    k = 5
    eta = 2.0
    
    Q, R, p = sRRQR_rank(A, eta, k)
    
    # Check permutation properties
    assert len(p) == n, "Permutation length matches n"
    assert len(set(p)) == n, "All indices unique"
    assert set(p) == set(range(n)), "Permutation of [0, n)"
    assert np.all(p >= 0) and np.all(p < n), "Indices in valid range"


def test_sRRQR_tol_permutation_validity(low_rank_matrix):
    """Test permutation validity for sRRQR_tol."""
    A = low_rank_matrix
    n = A.shape[1]
    tol = 1e-4
    eta = 2.0
    
    Q, R, p = sRRQR_tol(A, eta, tol)
    
    # Check permutation properties
    assert len(p) == n, "Permutation length matches n"
    assert len(set(p)) == n, "All indices unique"
    assert set(p) == set(range(n)), "Permutation of [0, n)"


# ===========================
# DOCUMENTATION EXAMPLE TEST
# ===========================

def test_sRRQR_documentation_example():
    """Test example similar to what would be in documentation."""
    from scipy.spatial.distance import cdist
    
    # Create test matrix similar to MATLAB example
    np.random.seed(100)
    X0 = 4 * (1 - 2 * np.random.rand(100, 3))
    Y0 = np.array([12, 0, 0]) + 4 * (1 - 2 * np.random.rand(100, 3))
    
    dist = cdist(X0, Y0)
    A = 1.0 / dist
    
    # Test with fixed rank
    k = 15
    eta = 1.5
    Q, R, p = sRRQR(A, eta, 'rank', k)
    
    assert Q.shape == (100, k), "Q shape correct"
    assert R.shape == (k, 100), "R shape correct"
    
    # Check factorization accuracy (low-rank approximation)
    A_permuted = A[:, p]
    error = np.linalg.norm(A_permuted - Q @ R, 'fro') / np.linalg.norm(A, 'fro')
    assert error < 0.01, "Low-rank factorization reasonable"
    
    # Check eta bound
    R11 = R[:k, :k]
    R12 = R[:k, k:]
    max_entry = np.max(np.abs(la.solve_triangular(R11, R12, lower=False)))
    assert max_entry <= eta + 1e-10, f"Eta bound satisfied"
    
    # Test with tolerance
    tol = 1e-3
    eta = 2.0
    Q, R, p = sRRQR(A, eta, 'tol', tol)
    
    k_found = Q.shape[1]
    assert k_found > 0, "Should find non-zero rank"
    
    # Check orthogonality
    QtQ = Q.T @ Q
    assert np.allclose(QtQ, np.eye(k_found), rtol=1e-10), "Q orthonormal"
    
    print("sRRQR documentation example test passed!")


# ===========================
# REPRODUCIBILITY TEST
# ===========================

def test_sRRQR_rank_reproducibility():
    """Test that sRRQR_rank gives consistent results."""
    np.random.seed(101)
    A = np.random.randn(30, 25)
    k = 10
    eta = 2.0
    
    Q1, R1, p1 = sRRQR_rank(A, eta, k)
    Q2, R2, p2 = sRRQR_rank(A, eta, k)
    
    # Results should be identical
    assert np.allclose(Q1, Q2), "Q should be reproducible"
    assert np.allclose(R1, R2), "R should be reproducible"
    assert np.array_equal(p1, p2), "Permutation should be reproducible"


def test_sRRQR_tol_reproducibility():
    """Test that sRRQR_tol gives consistent results."""
    np.random.seed(102)
    A = np.random.randn(30, 25)
    tol = 1e-4
    eta = 2.0
    
    Q1, R1, p1 = sRRQR_tol(A, eta, tol)
    Q2, R2, p2 = sRRQR_tol(A, eta, tol)
    
    # Results should be identical
    assert np.allclose(Q1, Q2), "Q should be reproducible"
    assert np.allclose(R1, R2), "R should be reproducible"
    assert np.array_equal(p1, p2), "Permutation should be reproducible"


# ===========================
# TESTS FOR COLUMN INTERCHANGE LOGIC
# ===========================

def test_sRRQR_rank_column_interchange():
    """Test sRRQR_rank with matrix that requires column interchanges."""
    np.random.seed(100)
    # Create a matrix where initial pivoting is not optimal
    # Use a matrix with columns of varying norms to trigger swaps
    m, n, k = 50, 40, 8
    A = np.random.randn(m, n)
    # Make some columns much larger to ensure interchange
    A[:, 5:10] *= 10.0
    A[:, 15:20] *= 0.1
    
    eta = 1.5  # Strict eta to force interchanges
    
    Q, R, p = sRRQR_rank(A, eta, k)
    
    # Check output dimensions
    assert Q.shape == (m, k)
    assert R.shape == (k, n)
    assert len(p) == n
    
    # Verify QR decomposition (first k columns should match well)
    A_perm = A[:, p]
    reconstructed = Q @ R
    error = np.linalg.norm(A_perm[:, :k] - reconstructed[:, :k], 'fro')
    assert error < 1e-9, f"QR decomposition error {error} too large"
    
    # Check that permutation was applied (not identity)
    assert not np.array_equal(p, np.arange(n)), "No column interchange occurred"


def test_sRRQR_rank_violates_eta_condition():
    """Test with matrix designed to violate eta condition."""
    np.random.seed(101)
    # Create matrix with specific structure to violate eta
    m, n = 30, 25
    k = 5
    
    # Create low-rank matrix with specific column ordering
    U = np.random.randn(m, k)
    V = np.random.randn(k, n)
    # Scale columns unevenly
    V[:, :k] *= 10.0
    V[:, k:] *= 0.01
    A = U @ V
    
    eta = 1.2  # Small eta to trigger violations
    
    Q, R, p = sRRQR_rank(A, eta, k)
    
    # Verify orthogonality
    assert np.allclose(Q.T @ Q, np.eye(k), atol=1e-10)
    
    # Verify factorization for first k columns
    A_perm = A[:, p]
    assert np.allclose(Q @ R[:, :k], A_perm[:, :k], atol=1e-9)
    
    # Check R11 and R12 blocks
    R11 = R[:k, :k]
    R12 = R[:k, k:]
    
    # R11 should be well-conditioned
    cond_R11 = np.linalg.cond(R11)
    assert cond_R11 < 1e10, f"R11 is ill-conditioned: {cond_R11}"


def test_sRRQR_rank_multiple_iterations():
    """Test sRRQR_rank requiring multiple column swaps."""
    np.random.seed(102)
    m, n, k = 60, 50, 10
    
    # Create matrix with deliberately bad initial ordering
    A = np.zeros((m, n))
    for i in range(n):
        # Reverse importance: first columns are weakest
        A[:, i] = np.random.randn(m) * (0.1 if i < k else 10.0)
    
    eta = 1.5
    Q, R, p = sRRQR_rank(A, eta, k)
    
    # Verify dimensions
    assert Q.shape == (m, k)
    assert R.shape == (k, n)
    
    # Verify orthogonality
    assert np.allclose(Q.T @ Q, np.eye(k), atol=1e-10)
    
    # Permutation should significantly differ from identity
    identity_perm = np.arange(n)
    changes = np.sum(p != identity_perm)
    assert changes >= k, f"Only {changes} columns swapped, expected at least {k}"


def test_sRRQR_tol_rank_reduction():
    """Test sRRQR_tol with rank reduction scenarios."""
    np.random.seed(103)
    # Create low-rank matrix with known structure
    m, n, true_rank = 50, 40, 5
    U = np.random.randn(m, true_rank)
    V = np.random.randn(n, true_rank)
    A = U @ V.T
    
    # Add small noise
    A += 1e-10 * np.random.randn(m, n)
    
    eta = 2.0
    tol = 1e-8  # Should detect rank ~5
    
    Q, R, p = sRRQR_tol(A, eta, tol)
    
    # Rank should be close to true_rank
    assert Q.shape[1] <= true_rank + 2, f"Detected rank {Q.shape[1]} too large"
    assert Q.shape[1] >= true_rank - 2, f"Detected rank {Q.shape[1]} too small"
    
    # Verify orthogonality
    k_detected = Q.shape[1]
    assert np.allclose(Q.T @ Q, np.eye(k_detected), atol=1e-10)


def test_sRRQR_tol_iterative_rank_reduction():
    """Test sRRQR_tol that reduces rank iteratively."""
    np.random.seed(104)
    # Create matrix with gradual singular value decay
    m, n = 40, 35
    U, _ = np.linalg.qr(np.random.randn(m, min(m, n)))
    V, _ = np.linalg.qr(np.random.randn(n, min(m, n)))
    
    # Exponentially decaying singular values
    s = np.exp(-np.arange(min(m, n)))
    A = U @ np.diag(s) @ V.T
    
    eta = 2.0
    tol = 0.1  # Should eliminate many singular values
    
    Q, R, p = sRRQR_tol(A, eta, tol)
    
    k = Q.shape[1]
    assert k < min(m, n), "No rank reduction occurred"
    assert k >= 1, "Rank reduced to zero"
    
    # Verify factorization quality
    A_perm = A[:, p]
    A_approx = Q @ R[:, :n]
    relative_error = np.linalg.norm(A_perm - A_approx) / np.linalg.norm(A)
    assert relative_error < 0.2, f"High approximation error: {relative_error}"


def test_sRRQR_rank_complex_interchange():
    """Test sRRQR_rank with complex matrices requiring interchange."""
    np.random.seed(105)
    m, n, k = 40, 30, 6
    
    # Complex matrix with varying column magnitudes
    A = np.random.randn(m, n) + 1j * np.random.randn(m, n)
    A[:, :5] *= 0.1
    A[:, 10:15] *= 10.0
    
    eta = 1.8
    Q, R, p = sRRQR_rank(A, eta, k)
    
    # Verify complex orthogonality
    assert np.allclose(Q.conj().T @ Q, np.eye(k), atol=1e-10)
    
    # Verify factorization for first k columns
    A_perm = A[:, p]
    assert np.allclose(Q @ R[:, :k], A_perm[:, :k], atol=1e-9)
    
    # Diagonal of R should have positive real parts
    for i in range(k):
        assert np.real(R[i, i]) > 0, f"R[{i},{i}] has negative real part"


def test_sRRQR_tol_near_singular():
    """Test sRRQR_tol with nearly singular matrix."""
    np.random.seed(106)
    m, n = 35, 30
    
    # Create matrix with very small singular values
    U, _ = np.linalg.qr(np.random.randn(m, min(m, n)))
    V, _ = np.linalg.qr(np.random.randn(n, min(m, n)))
    s = np.array([1, 0.1, 0.01, 1e-5, 1e-8] + [1e-10] * (min(m, n) - 5))
    A = U @ np.diag(s) @ V.T
    
    eta = 2.0
    tol = 1e-6
    
    Q, R, p = sRRQR_tol(A, eta, tol)
    
    # Should detect rank around 3-4
    k = Q.shape[1]
    assert 2 <= k <= 5, f"Detected rank {k} outside expected range [2, 5]"
    
    # Verify orthogonality
    assert np.allclose(Q.T @ Q, np.eye(k), atol=1e-10)


def test_sRRQR_dispatcher():
    """Test sRRQR dispatcher function with different modes."""
    np.random.seed(107)
    m, n = 30, 25
    A = np.random.randn(m, n)
    eta = 2.0
    
    # Test 'rank' mode
    k = 5
    Q1, R1, p1 = sRRQR(A, eta, mode='rank', param=k)
    Q2, R2, p2 = sRRQR_rank(A, eta, k)
    assert np.allclose(Q1, Q2), "Dispatcher 'rank' mode failed"
    assert np.allclose(R1, R2), "Dispatcher 'rank' mode failed"
    
    # Test 'tol' mode
    tol = 1e-6
    Q1, R1, p1 = sRRQR(A, eta, mode='tol', param=tol)
    Q2, R2, p2 = sRRQR_tol(A, eta, tol)
    assert np.allclose(Q1, Q2), "Dispatcher 'tol' mode failed"
    assert np.allclose(R1, R2), "Dispatcher 'tol' mode failed"


def test_sRRQR_edge_case_k_equals_1():
    """Test sRRQR_rank with k=1 (edge case)."""
    np.random.seed(108)
    m, n = 20, 15
    A = np.random.randn(m, n)
    eta = 2.0
    k = 1
    
    Q, R, p = sRRQR_rank(A, eta, k)
    
    assert Q.shape == (m, 1)
    assert R.shape == (1, n)
    assert len(p) == n
    
    # Verify orthogonality (single column should be unit norm)
    assert np.allclose(np.linalg.norm(Q), 1.0)


def test_sRRQR_edge_case_k_equals_min_dim():
    """Test sRRQR_rank with k equal to min dimension."""
    np.random.seed(109)
    m, n = 25, 20
    A = np.random.randn(m, n)
    eta = 2.0
    k = min(m, n)
    
    Q, R, p = sRRQR_rank(A, eta, k)
    
    assert Q.shape == (m, k)
    assert R.shape == (k, n)
    
    # Should produce full QR factorization
    assert np.allclose(Q.T @ Q, np.eye(k), atol=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
