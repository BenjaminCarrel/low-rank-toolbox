"""
Tests for randomized SVD functions

Author: Benjamin Carrel, Paul Scherrer Institute, 2025
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_less
from scipy.sparse.linalg import aslinearoperator

from low_rank_toolbox import SVD, LowRankMatrix
from low_rank_toolbox.randomized import adaptive_randomized_svd, randomized_svd


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
    s = np.exp(-np.linspace(0, 8, r))  # Exponential decay
    A = U @ np.diag(s) @ V.T
    return A, m, n, r, s


class TestRandomizedSVD:
    """Tests for the randomized_svd function."""

    def test_returns_svd_type(self, low_rank_matrix):
        """Test that function returns SVD object."""
        A, _, _, r, _ = low_rank_matrix
        result = randomized_svd(A, r, p=10, seed=1234)
        assert isinstance(result, SVD), "Should return SVD object"

    def test_output_rank(self, low_rank_matrix):
        """Test that output has correct rank."""
        A, _, _, r, _ = low_rank_matrix
        result = randomized_svd(A, r, p=10, truncate=True, seed=1234)
        assert result.rank == r, f"Expected rank {r}, got {result.rank}"

    def test_output_rank_without_truncation(self, low_rank_matrix):
        """Test output rank when truncate=False."""
        A, _, _, r, _ = low_rank_matrix
        p = 10
        result = randomized_svd(A, r, p=p, truncate=False, seed=1234)
        assert result.rank == r + p, f"Expected rank {r+p}, got {result.rank}"

    def test_orthogonality_u(self, low_rank_matrix):
        """Test that U has orthonormal columns."""
        A, _, _, r, _ = low_rank_matrix
        result = randomized_svd(A, r, p=10, seed=1234)

        identity = np.eye(r)
        assert_allclose(
            result.U.T @ result.U,
            identity,
            atol=1e-10,
            err_msg="U columns are not orthonormal",
        )

    def test_orthogonality_v(self, low_rank_matrix):
        """Test that V has orthonormal columns."""
        A, _, _, r, _ = low_rank_matrix
        result = randomized_svd(A, r, p=10, seed=1234)

        identity = np.eye(r)
        assert_allclose(
            result.V.T @ result.V,
            identity,
            atol=1e-10,
            err_msg="V columns are not orthonormal",
        )

    def test_reconstruction_accuracy(self, low_rank_matrix):
        """Test reconstruction accuracy for low-rank matrix."""
        A, _, _, r, _ = low_rank_matrix
        result = randomized_svd(A, r, p=10, seed=1234)

        A_approx = result.full()
        error = np.linalg.norm(A - A_approx, "fro")

        assert error < 1e-8, f"Reconstruction error {error} too large"

    def test_singular_values_close_to_true(self, low_rank_matrix):
        """Test that computed singular values are close to true values."""
        A, _, _, r, s = low_rank_matrix
        result = randomized_svd(A, r, p=10, seed=1234)

        # Compare largest singular values
        assert_allclose(
            result.s[:r],
            s,
            rtol=1e-6,
            atol=1e-10,
            err_msg="Singular values differ from true values",
        )

    def test_oversampling_improves_accuracy(self, low_rank_matrix):
        """Test that more oversampling improves accuracy."""
        A, _, _, r, _ = low_rank_matrix

        result1 = randomized_svd(A, r, p=2, seed=1234)
        result2 = randomized_svd(A, r, p=15, seed=1234)

        error1 = np.linalg.norm(A - result1.full(), "fro")
        error2 = np.linalg.norm(A - result2.full(), "fro")

        assert error2 <= error1, "More oversampling should reduce error"

    def test_power_iteration_improves_accuracy(self, low_rank_matrix):
        """Test that power iterations improve accuracy for difficult matrices."""
        A, _, _, r, _ = low_rank_matrix
        # Add noise to make problem harder
        A_noisy = A + 0.1 * np.random.RandomState(1234).randn(*A.shape)

        result0 = randomized_svd(A_noisy, r, p=10, q=0, seed=1234)
        result1 = randomized_svd(A_noisy, r, p=10, q=1, seed=1234)
        result2 = randomized_svd(A_noisy, r, p=10, q=2, seed=1234)

        # Measure error in approximating original clean matrix A
        error0 = np.linalg.norm(A - result0.full(), "fro")
        error1 = np.linalg.norm(A - result1.full(), "fro")
        error2 = np.linalg.norm(A - result2.full(), "fro")

        # More power iterations should help (within tolerance)
        assert error1 <= error0 * 1.1, "Power iteration q=1 should help"
        assert error2 <= error1 * 1.1, "Power iteration q=2 should help"

    def test_seed_reproducibility(self, low_rank_matrix):
        """Test that same seed produces same output."""
        A, _, _, r, _ = low_rank_matrix

        result1 = randomized_svd(A, r, p=10, seed=1234)
        result2 = randomized_svd(A, r, p=10, seed=1234)

        assert_allclose(result1.U, result2.U, atol=1e-14)
        assert_allclose(result1.s, result2.s, atol=1e-14)
        assert_allclose(result1.V, result2.V, atol=1e-14)

    def test_different_seeds_differ(self, low_rank_matrix):
        """Test that different seeds produce different output."""
        A, _, _, r, _ = low_rank_matrix

        result1 = randomized_svd(A, r, p=10, seed=1234)
        result2 = randomized_svd(A, r, p=10, seed=5678)

        # Should not be identical (both give good approximations though)
        assert not np.allclose(result1.U, result2.U, atol=1e-10)

    def test_complex_matrix(self):
        """Test with complex-valued matrix."""
        np.random.seed(1234)
        m, n, r = 60, 50, 8
        A = np.random.randn(m, r) + 1j * np.random.randn(m, r)
        A = A @ (np.random.randn(r, n) + 1j * np.random.randn(r, n))

        result = randomized_svd(A, r, p=5, seed=1234)

        # Check orthogonality (Hermitian)
        assert_allclose(result.U.T.conj() @ result.U, np.eye(r), atol=1e-10)
        assert_allclose(result.V.T.conj() @ result.V, np.eye(r), atol=1e-10)

        # Check reconstruction
        error = np.linalg.norm(A - result.full(), "fro")
        assert error < 1e-6

    def test_large_matrix_various_ranks(self, large_low_rank_matrix):
        """Test on large matrix with various target ranks."""
        A, _, _, true_r, s = large_low_rank_matrix

        for r in [20, 30, 40]:
            result = randomized_svd(A, r, p=10, q=1, seed=1234)
            error = np.linalg.norm(A - result.full(), "fro")

            # Error should be comparable to truncated singular values
            if r < true_r:
                expected_error = np.linalg.norm(s[r:])
                assert (
                    error < expected_error * 3
                ), f"Error {error} too large for rank {r}"

    def test_large_matrix_oversampling_sweep(self, large_low_rank_matrix):
        """Test convergence with varying oversampling on large matrix."""
        A, _, _, r, _ = large_low_rank_matrix
        target_r = 25

        errors = []
        for p in [5, 10, 15, 20]:
            result = randomized_svd(A, target_r, p=p, seed=1234)
            error = np.linalg.norm(A - result.full(), "fro")
            errors.append(error)

        # Errors should generally decrease or stabilize (allow small numerical tolerance)
        # The improvement may be negligible for already well-conditioned matrices
        assert errors[-1] <= errors[0] * (
            1 + 1e-10
        ), f"Error should not increase significantly with more oversampling: {errors[-1]} vs {errors[0]}"

    def test_invalid_rank(self, low_rank_matrix):
        """Test error handling for invalid rank."""
        A, _, _, _, _ = low_rank_matrix

        with pytest.raises(ValueError, match="Rank must be at least 1"):
            randomized_svd(A, r=0)

        with pytest.raises(ValueError, match="Rank must be at least 1"):
            randomized_svd(A, r=-1)

    def test_invalid_oversampling(self, low_rank_matrix):
        """Test error handling for invalid oversampling."""
        A, _, _, _, _ = low_rank_matrix

        with pytest.raises(
            ValueError, match="Oversampling parameter must be non-negative"
        ):
            randomized_svd(A, r=5, p=-1)

    def test_invalid_power_iterations(self, low_rank_matrix):
        """Test error handling for invalid power iterations."""
        A, _, _, _, _ = low_rank_matrix

        with pytest.raises(
            ValueError, match="Number of power iterations must be non-negative"
        ):
            randomized_svd(A, r=5, p=5, q=-1)

    def test_rank_exceeds_dimension(self, low_rank_matrix):
        """Test error handling when rank + oversampling exceeds dimension."""
        A, m, _, _, _ = low_rank_matrix

        with pytest.raises(ValueError, match="exceeds minimum matrix dimension"):
            randomized_svd(A, r=m, p=5)


class TestAdaptiveRandomizedSVD:
    """Tests for the adaptive_randomized_svd function."""

    def test_returns_svd_type(self, low_rank_matrix):
        """Test that function returns SVD object."""
        A, _, _, _, _ = low_rank_matrix
        result = adaptive_randomized_svd(A, tol=1e-6, failure_prob=1e-6, seed=1234)
        assert isinstance(result, SVD), "Should return SVD object"

    def test_orthogonality(self, low_rank_matrix):
        """Test that U and V have orthonormal columns."""
        A, _, _, _, _ = low_rank_matrix
        result = adaptive_randomized_svd(A, tol=1e-6, failure_prob=1e-6, seed=1234)

        r = result.rank
        assert_allclose(result.U.T @ result.U, np.eye(r), atol=1e-10)
        assert_allclose(result.V.T @ result.V, np.eye(r), atol=1e-10)

    def test_tolerance_satisfied(self, low_rank_matrix):
        """Test that approximation error is below tolerance."""
        A, _, _, _, _ = low_rank_matrix
        tol = 1e-5
        result = adaptive_randomized_svd(A, tol=tol, failure_prob=1e-6, seed=1234)

        A_approx = result.full()
        error = np.linalg.norm(A - A_approx, "fro")

        assert error < tol * 2, f"Error {error} exceeds tolerance {tol}"

    def test_rank_adapts_to_matrix(self, low_rank_matrix):
        """Test that rank adapts to matrix structure."""
        A, _, _, r, _ = low_rank_matrix

        result = adaptive_randomized_svd(A, tol=1e-8, failure_prob=1e-6, seed=1234)

        # Rank should be close to true rank
        assert result.rank >= r, "Rank too small"
        assert result.rank <= r + 50, "Rank unnecessarily large"

    def test_tighter_tolerance_increases_rank(self, low_rank_matrix):
        """Test that tighter tolerance leads to higher rank."""
        A, _, _, _, _ = low_rank_matrix
        # Add noise
        A_noisy = A + 1e-5 * np.random.RandomState(1234).randn(*A.shape)

        result1 = adaptive_randomized_svd(
            A_noisy, tol=1e-3, failure_prob=1e-6, seed=1234
        )
        result2 = adaptive_randomized_svd(
            A_noisy, tol=1e-5, failure_prob=1e-6, seed=1234
        )

        assert result2.rank >= result1.rank, "Tighter tolerance should increase rank"

    def test_max_rank_constraint(self, low_rank_matrix):
        """Test that max_rank parameter is respected."""
        A, _, _, _, _ = low_rank_matrix
        max_r = 8

        result = adaptive_randomized_svd(
            A, tol=1e-10, failure_prob=1e-6, max_rank=max_r, seed=1234
        )

        assert result.rank <= max_r, f"Rank {result.rank} exceeds max_rank {max_r}"

    def test_seed_reproducibility(self, low_rank_matrix):
        """Test that same seed produces same output."""
        A, _, _, _, _ = low_rank_matrix

        result1 = adaptive_randomized_svd(A, tol=1e-6, failure_prob=1e-6, seed=1234)
        result2 = adaptive_randomized_svd(A, tol=1e-6, failure_prob=1e-6, seed=1234)

        assert result1.rank == result2.rank
        assert_allclose(result1.U, result2.U, atol=1e-14)
        assert_allclose(result1.s, result2.s, atol=1e-14)
        assert_allclose(result1.V, result2.V, atol=1e-14)

    def test_complex_matrix(self):
        """Test with complex-valued matrix."""
        np.random.seed(1234)
        m, n, r = 80, 70, 10
        A = np.random.randn(m, r) + 1j * np.random.randn(m, r)
        A = A @ (np.random.randn(r, n) + 1j * np.random.randn(r, n))

        tol = 1e-6
        result = adaptive_randomized_svd(A, tol=tol, failure_prob=1e-6, seed=1234)

        # Check orthogonality
        assert_allclose(result.U.T.conj() @ result.U, np.eye(result.rank), atol=1e-10)
        assert_allclose(result.V.T.conj() @ result.V, np.eye(result.rank), atol=1e-10)

        # Check approximation
        error = np.linalg.norm(A - result.full(), "fro")
        assert error < tol * 2

    def test_large_matrix_various_tolerances(self, large_low_rank_matrix):
        """Test on large matrix with various tolerances."""
        A, _, _, _, _ = large_low_rank_matrix

        tolerances = [1e-3, 1e-5, 1e-7]
        prev_rank = 0

        for tol in tolerances:
            result = adaptive_randomized_svd(A, tol=tol, failure_prob=1e-6, seed=1234)
            error = np.linalg.norm(A - result.full(), "fro")

            # Check tolerance satisfied (with some margin)
            assert error < tol * 3, f"Error {error} exceeds tolerance {tol}"

            # Rank should increase with tighter tolerance
            assert result.rank >= prev_rank
            prev_rank = result.rank

    def test_invalid_tolerance(self, low_rank_matrix):
        """Test error handling for invalid tolerance."""
        A, _, _, _, _ = low_rank_matrix

        with pytest.raises(ValueError, match="Tolerance must be positive"):
            adaptive_randomized_svd(A, tol=0)

        with pytest.raises(ValueError, match="Tolerance must be positive"):
            adaptive_randomized_svd(A, tol=-1e-6)

    def test_invalid_failure_prob(self, low_rank_matrix):
        """Test error handling for invalid failure probability."""
        A, _, _, _, _ = low_rank_matrix

        with pytest.raises(ValueError, match="Failure probability must be in"):
            adaptive_randomized_svd(A, tol=1e-6, failure_prob=0)

        with pytest.raises(ValueError, match="Failure probability must be in"):
            adaptive_randomized_svd(A, tol=1e-6, failure_prob=1)

        with pytest.raises(ValueError, match="Failure probability must be in"):
            adaptive_randomized_svd(A, tol=1e-6, failure_prob=-0.1)

    def test_invalid_max_rank(self, low_rank_matrix):
        """Test error handling for invalid max_rank."""
        A, _, _, _, _ = low_rank_matrix

        with pytest.raises(ValueError, match="Maximum rank must be at least 1"):
            adaptive_randomized_svd(A, tol=1e-6, max_rank=0)

        with pytest.raises(ValueError, match="Maximum rank must be at least 1"):
            adaptive_randomized_svd(A, tol=1e-6, max_rank=-1)
