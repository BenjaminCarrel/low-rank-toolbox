"""
Tests for generalized Nyström method

Author: Benjamin Carrel, Paul Scherrer Institute, 2025
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.sparse.linalg import aslinearoperator
from lowrank.randomized import generalized_nystrom
from lowrank import QuasiSVD, SVD, LowRankMatrix


@pytest.fixture
def low_rank_matrix():
    """Create a test low-rank matrix with exponential decay of singular values."""
    np.random.seed(1234)
    m, n, r = 100, 80, 10
    U = np.random.randn(m, r)
    U, _ = np.linalg.qr(U)
    V = np.random.randn(n, r)
    V, _ = np.linalg.qr(V)
    s = np.logspace(0, -10, r)  # Exponential decay
    A = U @ np.diag(s) @ V.T
    return A, m, n, r, s


@pytest.fixture
def large_low_rank_matrix():
    """Create a large test matrix with varying rank and singular value decay."""
    np.random.seed(5678)
    m, n, r = 200, 180, 30
    U = np.random.randn(m, r)
    U, _ = np.linalg.qr(U)
    V = np.random.randn(n, r)
    V, _ = np.linalg.qr(V)
    s = np.exp(-np.linspace(0, 8, r))  # Exponential decay
    A = U @ np.diag(s) @ V.T
    return A, m, n, r, s


@pytest.fixture
def symmetric_matrix():
    """Create a symmetric positive definite matrix."""
    np.random.seed(9999)
    n, r = 80, 12
    U = np.random.randn(n, r)
    U, _ = np.linalg.qr(U)
    s = np.logspace(0, -8, r)
    A = U @ np.diag(s) @ U.T
    return A, n, r, s


class TestGeneralizedNystrom:
    """Tests for the generalized_nystrom function."""
    
    def test_returns_quasi_svd_type(self, low_rank_matrix):
        """Test that function returns QuasiSVD object."""
        A, _, _, r, _ = low_rank_matrix
        result = generalized_nystrom(A, r, seed=1234)
        assert isinstance(result, QuasiSVD), "Should return QuasiSVD object"
    
    def test_output_rank(self, low_rank_matrix):
        """Test that output has correct rank."""
        A, _, _, r, _ = low_rank_matrix
        result = generalized_nystrom(A, r, seed=1234)
        assert result.rank == r, f"Expected rank {r}, got {result.rank}"
    
    def test_reconstruction_accuracy(self, low_rank_matrix):
        """Test reconstruction accuracy for low-rank matrix."""
        A, _, _, r, _ = low_rank_matrix
        result = generalized_nystrom(A, r, oversampling_params=(10, 15), seed=1234)
        
        A_approx = result.full()
        error = np.linalg.norm(A - A_approx, 'fro')
        
        # Should give good approximation for exact low-rank matrix
        assert error < 1e-6, f"Reconstruction error {error} too large"
    
    def test_oversampling_improves_accuracy(self, low_rank_matrix):
        """Test that more oversampling improves accuracy."""
        A, _, _, r, _ = low_rank_matrix
        
        result1 = generalized_nystrom(A, r, oversampling_params=(2, 3), seed=1234)
        result2 = generalized_nystrom(A, r, oversampling_params=(15, 20), seed=1234)
        
        error1 = np.linalg.norm(A - result1.full(), 'fro')
        error2 = np.linalg.norm(A - result2.full(), 'fro')
        
        # Both should give very good approximations for low-rank matrix
        assert error1 < 1e-4, f"Error {error1} too large"
        assert error2 < 1e-4, f"Error {error2} too large"
    
    def test_both_oversampling_parameters(self, low_rank_matrix):
        """Test effect of both oversampling parameters."""
        A, _, _, r, _ = low_rank_matrix
        
        # Vary p1
        result1 = generalized_nystrom(A, r, oversampling_params=(5, 10), seed=1234)
        result2 = generalized_nystrom(A, r, oversampling_params=(15, 10), seed=1234)
        
        error1 = np.linalg.norm(A - result1.full(), 'fro')
        error2 = np.linalg.norm(A - result2.full(), 'fro')
        
        assert error2 <= error1 * 1.5, "Increasing p1 should help"
        
        # Vary p2
        result3 = generalized_nystrom(A, r, oversampling_params=(10, 5), seed=1234)
        result4 = generalized_nystrom(A, r, oversampling_params=(10, 15), seed=1234)
        
        error3 = np.linalg.norm(A - result3.full(), 'fro')
        error4 = np.linalg.norm(A - result4.full(), 'fro')
        
        assert error4 <= error3 * 1.5, "Increasing p2 should help"
    
    def test_stable_version_with_epsilon(self, low_rank_matrix):
        """Test stable version with epsilon parameter."""
        A, _, _, r, _ = low_rank_matrix
        
        result = generalized_nystrom(A, r, epsilon=1e-10, seed=1234)
        
        assert isinstance(result, QuasiSVD)
        A_approx = result.full()
        error = np.linalg.norm(A - A_approx, 'fro')
        
        assert error < 1e-5, f"Stable version error {error} too large"
    
    def test_epsilon_affects_rank(self, low_rank_matrix):
        """Test that epsilon can affect effective rank."""
        A, _, _, r, _ = low_rank_matrix
        # Add small noise
        A_noisy = A + 1e-8 * np.random.RandomState(1234).randn(*A.shape)
        
        # Without epsilon, tries to capture all r components
        result1 = generalized_nystrom(A_noisy, r, seed=1234)
        
        # With epsilon, might reduce rank based on singular values
        result2 = generalized_nystrom(A_noisy, r, epsilon=1e-7, seed=1234)
        
        # Both should give good approximations
        error1 = np.linalg.norm(A - result1.full(), 'fro')
        error2 = np.linalg.norm(A - result2.full(), 'fro')
        
        assert error1 < 1e-5
        assert error2 < 1e-5
    
    def test_seed_reproducibility(self, low_rank_matrix):
        """Test that same seed produces same output."""
        A, _, _, r, _ = low_rank_matrix
        
        result1 = generalized_nystrom(A, r, seed=1234)
        result2 = generalized_nystrom(A, r, seed=1234)
        
        assert_allclose(result1.U, result2.U, atol=1e-14)
        assert_allclose(result1.S, result2.S, atol=1e-14)
        assert_allclose(result1.V, result2.V, atol=1e-14)
    
    def test_different_seeds_differ(self, low_rank_matrix):
        """Test that different seeds produce different output."""
        A, _, _, r, _ = low_rank_matrix
        
        result1 = generalized_nystrom(A, r, seed=1234)
        result2 = generalized_nystrom(A, r, seed=5678)
        
        # Should not be identical (both give good approximations though)
        assert not np.allclose(result1.U, result2.U, atol=1e-10)
    
    def test_conversion_to_svd(self, low_rank_matrix):
        """Test conversion from QuasiSVD to SVD."""
        A, _, _, r, _ = low_rank_matrix
        
        result = generalized_nystrom(A, r, seed=1234)
        svd_result = result.to_svd()
        
        assert isinstance(svd_result, SVD)
        
        # Reconstruction should be similar
        assert_allclose(result.full(), svd_result.full(), atol=1e-10)
    
    def test_symmetric_matrix(self, symmetric_matrix):
        """Test on symmetric positive definite matrix."""
        A, _, r, _ = symmetric_matrix
        
        result = generalized_nystrom(A, r, seed=1234)
        
        A_approx = result.full()
        error = np.linalg.norm(A - A_approx, 'fro')
        
        # Should give good approximation
        assert error < 1e-5, f"Error {error} too large for symmetric matrix"
    
    def test_ill_conditioned_matrix_stable_version(self):
        """Test stable version on ill-conditioned matrix."""
        np.random.seed(1234)
        m, n, r = 80, 70, 10
        U = np.random.randn(m, r)
        U, _ = np.linalg.qr(U)
        V = np.random.randn(n, r)
        V, _ = np.linalg.qr(V)
        # Very ill-conditioned
        s = np.logspace(0, -14, r)
        A = U @ np.diag(s) @ V.T
        
        # Stable version should handle this better
        result = generalized_nystrom(A, r, epsilon=1e-12, seed=1234)
        
        # Should still give reasonable approximation
        error = np.linalg.norm(A - result.full(), 'fro')
        assert error < 1e-8
    
    def test_large_matrix_various_ranks(self, large_low_rank_matrix):
        """Test on large matrix with various target ranks."""
        A, _, _, true_r, s = large_low_rank_matrix
        
        for r in [20, 30, 40]:
            result = generalized_nystrom(A, r, oversampling_params=(10, 15), seed=1234)
            error = np.linalg.norm(A - result.full(), 'fro')
            
            # Error should be related to truncated singular values
            if r < true_r:
                expected_error = np.linalg.norm(s[r:])
                assert error < expected_error * 5, \
                    f"Error {error} too large for rank {r}"
    
    def test_large_matrix_oversampling_sweep(self, large_low_rank_matrix):
        """Test convergence with varying oversampling on large matrix."""
        A, _, _, _, _ = large_low_rank_matrix
        target_r = 25
        
        errors = []
        oversampling_configs = [(5, 5), (10, 10), (15, 15), (20, 20)]
        
        for p1, p2 in oversampling_configs:
            result = generalized_nystrom(A, target_r, 
                                        oversampling_params=(p1, p2), 
                                        seed=1234)
            error = np.linalg.norm(A - result.full(), 'fro')
            errors.append(error)
        
        # Errors should generally decrease
        assert errors[-1] <= errors[0], \
            "Error should improve with more oversampling"
    
    def test_square_matrix(self, symmetric_matrix):
        """Test on square matrix."""
        A, n, r, _ = symmetric_matrix
        
        result = generalized_nystrom(A, r, seed=1234)
        
        assert result.shape == (n, n)
        error = np.linalg.norm(A - result.full(), 'fro')
        assert error < 1e-5
    
    def test_tall_matrix(self):
        """Test on tall matrix (m >> n)."""
        np.random.seed(1234)
        m, n, r = 150, 50, 10
        U = np.random.randn(m, r)
        V = np.random.randn(n, r)
        A = U @ V.T
        
        result = generalized_nystrom(A, r, seed=1234)
        
        assert result.shape == (m, n)
        error = np.linalg.norm(A - result.full(), 'fro')
        assert error < 1e-5
    
    def test_wide_matrix(self):
        """Test on wide matrix (m << n)."""
        np.random.seed(1234)
        m, n, r = 50, 150, 10
        U = np.random.randn(m, r)
        V = np.random.randn(n, r)
        A = U @ V.T
        
        result = generalized_nystrom(A, r, seed=1234)
        
        assert result.shape == (m, n)
        error = np.linalg.norm(A - result.full(), 'fro')
        assert error < 1e-5
    
    def test_invalid_rank(self, low_rank_matrix):
        """Test error handling for invalid rank."""
        A, _, _, _, _ = low_rank_matrix
        
        with pytest.raises(ValueError, match="Rank must be at least 1"):
            generalized_nystrom(A, r=0)
        
        with pytest.raises(ValueError, match="Rank must be at least 1"):
            generalized_nystrom(A, r=-1)
    
    def test_invalid_epsilon(self, low_rank_matrix):
        """Test error handling for invalid epsilon."""
        A, _, _, r, _ = low_rank_matrix
        
        with pytest.raises(ValueError, match="Epsilon must be positive"):
            generalized_nystrom(A, r=r, epsilon=0)
        
        with pytest.raises(ValueError, match="Epsilon must be positive"):
            generalized_nystrom(A, r=r, epsilon=-1e-6)
    
    def test_oversampling_exceeds_columns(self, low_rank_matrix):
        """Test error when r + p1 exceeds number of columns."""
        A, _, n, _, _ = low_rank_matrix
        
        with pytest.raises(ValueError, match="exceeds number of columns"):
            generalized_nystrom(A, r=n, oversampling_params=(5, 10))
    
    def test_identity_matrix(self):
        """Test on identity matrix (full rank, well-conditioned)."""
        n = 50
        r = 20
        A = np.eye(n)
        
        result = generalized_nystrom(A, r, seed=1234)
        
        # Should approximate leading r columns/rows
        A_approx = result.full()
        assert A_approx.shape == (n, n)
    
    def test_comparison_with_svd(self, low_rank_matrix):
        """Compare with standard SVD for accuracy."""
        A, _, _, r, _ = low_rank_matrix
        
        # Generalized Nyström
        result_gn = generalized_nystrom(A, r, oversampling_params=(15, 20), seed=1234)
        error_gn = np.linalg.norm(A - result_gn.full(), 'fro')
        
        # Standard truncated SVD
        U, s, Vt = np.linalg.svd(A, full_matrices=False)
        A_svd = U[:, :r] @ np.diag(s[:r]) @ Vt[:r, :]
        error_svd = np.linalg.norm(A - A_svd, 'fro')
        
        # Both should be at machine precision for exact low-rank matrix
        assert error_gn < 1e-10, f"GN error {error_gn} too large"
        assert error_svd < 1e-10, f"SVD error {error_svd} too large"
