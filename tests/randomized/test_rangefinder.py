"""
Tests for rangefinder functions

Author: Benjamin Carrel, Paul Scherrer Institute, 2025
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_less
from scipy.sparse.linalg import aslinearoperator

from lowrank.randomized import adaptive_rangefinder, rangefinder


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
    m, n, r = 200, 200, 30
    U = np.random.randn(m, r)
    U, _ = np.linalg.qr(U)
    V = np.random.randn(n, r)
    V, _ = np.linalg.qr(V)
    s = np.exp(-np.linspace(0, 10, r))  # Exponential decay
    A = U @ np.diag(s) @ V.T
    return A, m, n, r, s


class TestRangefinder:
    """Tests for the rangefinder function."""

    def test_output_shape(self, low_rank_matrix):
        """Test that output has correct shape."""
        A, m, _, r, _ = low_rank_matrix
        p = 5
        Q = rangefinder(A, r, p=p, seed=1234)
        assert Q.shape == (m, r + p), f"Expected shape {(m, r+p)}, got {Q.shape}"

    def test_orthogonality(self, low_rank_matrix):
        """Test that Q has orthonormal columns."""
        A, _, _, r, _ = low_rank_matrix
        Q = rangefinder(A, r, p=5, seed=1234)
        identity = np.eye(r + 5)
        assert_allclose(
            Q.T @ Q, identity, atol=1e-10, err_msg="Q columns are not orthonormal"
        )

    def test_approximation_error(self, low_rank_matrix):
        """Test approximation quality for low-rank matrix."""
        A, _, _, r, _ = low_rank_matrix
        Q = rangefinder(A, r, p=10, seed=1234)
        # Compute approximation error
        A_approx = Q @ (Q.T @ A)
        error = np.linalg.norm(A - A_approx, "fro")
        # Should be very small for low-rank matrix
        assert error < 1e-8, f"Approximation error {error} too large"

    def test_power_iteration_improves_accuracy(self, low_rank_matrix):
        """Test that power iterations improve accuracy."""
        A, _, _, r, _ = low_rank_matrix
        # Add noise to make it harder
        A_noisy = A + 0.1 * np.random.randn(*A.shape)

        Q0 = rangefinder(A_noisy, r, p=5, q=0, seed=1234)
        Q1 = rangefinder(A_noisy, r, p=5, q=1, seed=1234)
        Q2 = rangefinder(A_noisy, r, p=5, q=2, seed=1234)

        error0 = np.linalg.norm(A_noisy - Q0 @ (Q0.T @ A_noisy), "fro")
        error1 = np.linalg.norm(A_noisy - Q1 @ (Q1.T @ A_noisy), "fro")
        error2 = np.linalg.norm(A_noisy - Q2 @ (Q2.T @ A_noisy), "fro")

        # Higher q should give better or equal accuracy
        assert error1 <= error0 * 1.1, "Power iteration q=1 didn't improve"
        assert error2 <= error1 * 1.1, "Power iteration q=2 didn't improve"

    def test_oversampling_improves_accuracy(self, low_rank_matrix):
        """Test that more oversampling improves accuracy."""
        A, _, _, r, _ = low_rank_matrix

        Q1 = rangefinder(A, r, p=2, seed=1234)
        Q2 = rangefinder(A, r, p=10, seed=1234)

        error1 = np.linalg.norm(A - Q1 @ (Q1.T @ A), "fro")
        error2 = np.linalg.norm(A - Q2 @ (Q2.T @ A), "fro")

        assert error2 <= error1, "More oversampling should reduce error"

    def test_seed_reproducibility(self, low_rank_matrix):
        """Test that same seed produces same output."""
        A, _, _, r, _ = low_rank_matrix

        Q1 = rangefinder(A, r, p=5, seed=1234)
        Q2 = rangefinder(A, r, p=5, seed=1234)

        assert_allclose(
            Q1, Q2, atol=1e-14, err_msg="Same seed should produce identical output"
        )

    def test_different_seeds_differ(self, low_rank_matrix):
        """Test that different seeds produce different output."""
        A, _, _, r, _ = low_rank_matrix

        Q1 = rangefinder(A, r, p=5, seed=1234)
        Q2 = rangefinder(A, r, p=5, seed=5678)

        # Should not be identical (with very high probability)
        assert not np.allclose(
            Q1, Q2, atol=1e-10
        ), "Different seeds should produce different output"

    def test_complex_matrix(self):
        """Test with complex-valued matrix."""
        np.random.seed(1234)
        m, n, r = 50, 40, 8
        A = np.random.randn(m, r) + 1j * np.random.randn(m, r)
        A = A @ (np.random.randn(r, n) + 1j * np.random.randn(r, n))

        Q = rangefinder(A, r, p=5, seed=1234)

        # Check orthogonality (Hermitian inner product)
        assert_allclose(Q.T.conj() @ Q, np.eye(r + 5), atol=1e-10)

        # Check approximation
        error = np.linalg.norm(A - Q @ (Q.T.conj() @ A), "fro")
        assert error < 1e-8

    def test_custom_omega(self, low_rank_matrix):
        """Test with custom sketching matrix."""
        A, _, n, r, _ = low_rank_matrix
        p = 5

        # Create custom Omega
        np.random.seed(9999)
        Omega = np.random.randn(n, r + p)

        Q = rangefinder(A, r, p=p, seed=1234, Omega=Omega)

        # Should still be orthonormal
        assert_allclose(Q.T @ Q, np.eye(r + p), atol=1e-10)

    def test_linear_operator_input(self, low_rank_matrix):
        """Test with LinearOperator input."""
        A, _, _, r, _ = low_rank_matrix
        A_op = aslinearoperator(A)

        Q = rangefinder(A_op, r, p=5, seed=1234)

        # Check orthogonality
        assert_allclose(Q.T @ Q, np.eye(r + 5), atol=1e-10)

        # Check approximation
        A_approx = Q @ (Q.T @ A)
        error = np.linalg.norm(A - A_approx, "fro")
        assert error < 1e-8

    def test_large_matrix_convergence(self, large_low_rank_matrix):
        """Test convergence on large matrix with varying ranks."""
        A, _, _, r, s = large_low_rank_matrix

        # Test different target ranks
        for target_r in [20, 30, 40]:
            Q = rangefinder(A, target_r, p=10, q=1, seed=1234)
            error = np.linalg.norm(A - Q @ (Q.T @ A), "fro")

            # Error should be related to truncated singular values
            if target_r < r:
                expected_error = np.linalg.norm(s[target_r:])
                assert (
                    error < expected_error * 10
                ), f"Error {error} too large for rank {target_r}"

    def test_invalid_rank(self, low_rank_matrix):
        """Test error handling for invalid rank."""
        A, _, _, _, _ = low_rank_matrix

        with pytest.raises(ValueError, match="Target rank must be at least 1"):
            rangefinder(A, r=0, p=5)

        with pytest.raises(ValueError, match="Target rank must be at least 1"):
            rangefinder(A, r=-1, p=5)

    def test_invalid_oversampling(self, low_rank_matrix):
        """Test error handling for invalid oversampling."""
        A, _, _, _, _ = low_rank_matrix

        with pytest.raises(
            ValueError, match="Oversampling parameter must be non-negative"
        ):
            rangefinder(A, r=5, p=-1)

    def test_invalid_power_iterations(self, low_rank_matrix):
        """Test error handling for invalid power iterations."""
        A, _, _, _, _ = low_rank_matrix

        with pytest.raises(
            ValueError, match="Number of power iterations must be non-negative"
        ):
            rangefinder(A, r=5, p=5, q=-1)

    def test_rank_exceeds_dimension(self, low_rank_matrix):
        """Test error handling when rank + oversampling exceeds matrix dimension."""
        A, m, _, _, _ = low_rank_matrix

        with pytest.raises(ValueError, match="exceeds minimum matrix dimension"):
            rangefinder(A, r=m, p=5)


class TestAdaptiveRangefinder:
    """Tests for the adaptive_rangefinder function."""

    def test_output_orthogonality(self, low_rank_matrix):
        """Test that output has orthonormal columns."""
        A, _, _, _, _ = low_rank_matrix
        Q = adaptive_rangefinder(A, tol=1e-8, failure_prob=1e-6, seed=1234)

        identity = np.eye(Q.shape[1])
        assert_allclose(
            Q.T @ Q, identity, atol=1e-10, err_msg="Q columns are not orthonormal"
        )

    def test_tolerance_satisfied(self, low_rank_matrix):
        """Test that approximation error is below tolerance."""
        A, _, _, _, _ = low_rank_matrix
        tol = 1e-6
        Q = adaptive_rangefinder(A, tol=tol, failure_prob=1e-6, seed=1234)

        A_approx = Q @ (Q.T @ A)
        error = np.linalg.norm(A - A_approx, "fro")

        # Should satisfy tolerance (with high probability)
        assert error < tol, f"Error {error} exceeds tolerance {tol}"

    def test_rank_adapts_to_matrix(self, low_rank_matrix):
        """Test that rank adapts to the matrix rank."""
        A, _, _, r, _ = low_rank_matrix

        Q = adaptive_rangefinder(A, tol=1e-8, failure_prob=1e-6, seed=1234)

        # Rank should be close to true rank (within oversampling)
        assert Q.shape[1] >= r, "Rank too small"
        assert Q.shape[1] <= r + 30, "Rank unnecessarily large"

    def test_tighter_tolerance_increases_rank(self, low_rank_matrix):
        """Test that tighter tolerance leads to higher rank."""
        A, _, _, _, _ = low_rank_matrix
        # Add some noise
        A_noisy = A + 1e-6 * np.random.RandomState(1234).randn(*A.shape)

        Q1 = adaptive_rangefinder(A_noisy, tol=1e-4, failure_prob=1e-6, seed=1234)
        Q2 = adaptive_rangefinder(A_noisy, tol=1e-6, failure_prob=1e-6, seed=1234)

        # Tighter tolerance should need more columns
        assert Q2.shape[1] >= Q1.shape[1], "Tighter tolerance should increase rank"

    def test_seed_reproducibility(self, low_rank_matrix):
        """Test that same seed produces same output."""
        A, _, _, _, _ = low_rank_matrix

        Q1 = adaptive_rangefinder(A, tol=1e-6, failure_prob=1e-6, seed=1234)
        Q2 = adaptive_rangefinder(A, tol=1e-6, failure_prob=1e-6, seed=1234)

        assert_allclose(Q1, Q2, atol=1e-14)

    def test_complex_matrix(self):
        """Test with complex-valued matrix."""
        np.random.seed(1234)
        m, n, r = 80, 60, 10
        A = np.random.randn(m, r) + 1j * np.random.randn(m, r)
        A = A @ (np.random.randn(r, n) + 1j * np.random.randn(r, n))

        tol = 1e-6
        Q = adaptive_rangefinder(A, tol=tol, failure_prob=1e-6, seed=1234)

        # Check orthogonality
        assert_allclose(Q.T.conj() @ Q, np.eye(Q.shape[1]), atol=1e-10)

        # Check approximation
        error = np.linalg.norm(A - Q @ (Q.T.conj() @ A), "fro")
        assert error < tol

    def test_large_matrix_various_ranks(self, large_low_rank_matrix):
        """Test on large matrix with various singular value decay rates."""
        A, _, _, r, _ = large_low_rank_matrix

        tolerances = [1e-4, 1e-6, 1e-8]
        prev_rank = 0

        for tol in tolerances:
            Q = adaptive_rangefinder(A, tol=tol, failure_prob=1e-6, seed=1234)
            error = np.linalg.norm(A - Q @ (Q.T @ A), "fro")

            # Check tolerance satisfied
            assert error < tol * 1.5, f"Error {error} exceeds tolerance {tol}"

            # Rank should increase with tighter tolerance
            assert Q.shape[1] >= prev_rank
            prev_rank = Q.shape[1]

    def test_invalid_tolerance(self, low_rank_matrix):
        """Test error handling for invalid tolerance."""
        A, _, _, _, _ = low_rank_matrix

        with pytest.raises(ValueError, match="Tolerance must be positive"):
            adaptive_rangefinder(A, tol=0)

        with pytest.raises(ValueError, match="Tolerance must be positive"):
            adaptive_rangefinder(A, tol=-1e-6)

    def test_invalid_failure_prob(self, low_rank_matrix):
        """Test error handling for invalid failure probability."""
        A, _, _, _, _ = low_rank_matrix

        with pytest.raises(ValueError, match="Failure probability must be in"):
            adaptive_rangefinder(A, tol=1e-6, failure_prob=0)

        with pytest.raises(ValueError, match="Failure probability must be in"):
            adaptive_rangefinder(A, tol=1e-6, failure_prob=1)

        with pytest.raises(ValueError, match="Failure probability must be in"):
            adaptive_rangefinder(A, tol=1e-6, failure_prob=-0.1)

        with pytest.raises(ValueError, match="Failure probability must be in"):
            adaptive_rangefinder(A, tol=1e-6, failure_prob=1.1)
