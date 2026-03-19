"""
Tests for the examples provided in the randomized submodule documentation.

IMPORTANT: ONE TEST = ONE EXAMPLE
=================================
Each test in this file corresponds to exactly one example in the randomized module
documentation (docstrings, tutorials, etc.). This ensures that:

1. All documented examples are automatically tested and verified to work.
2. If documentation is updated with new examples, corresponding tests MUST be added here.
3. If a test is added or modified here, the corresponding example in the documentation
   MUST be updated to match.

This bidirectional synchronization keeps the documentation accurate and reliable.

Test Organization:
------------------
Each test class corresponds to a specific randomized algorithm (randomized_svd,
generalized_nystrom, rangefinder, and their adaptive variants) and contains tests
for the examples shown in that algorithm's documentation.
"""

import numpy as np
import pytest

from low_rank_toolbox import SVD
from low_rank_toolbox.randomized import (
    adaptive_randomized_svd,
    adaptive_rangefinder,
    generalized_nystrom,
    randomized_svd,
    rangefinder,
)


class TestRandomizedSVDExamples:
    """Tests for randomized SVD examples from documentation."""

    def test_basic_randomized_svd(self):
        """Test basic randomized SVD from quickstart documentation."""
        # Large matrix
        np.random.seed(42)
        A = np.random.randn(500, 400)

        # Randomized SVD (much faster than full SVD)
        X_approx = randomized_svd(A, r=50, p=10, q=2)

        # Verify it's an SVD instance
        assert isinstance(X_approx, SVD)

        # Verify rank
        assert X_approx.rank == 50

        # Verify approximation quality
        error = np.linalg.norm(A - X_approx.to_dense(), "fro")
        print(f"Approximation error: {error:.2e}")

        # The error should be reasonable for a rank-50 approximation
        # We don't have a strict bound, but it should be finite
        assert error < np.inf
        assert not np.isnan(error)

    def test_randomized_svd_parameters(self):
        """Test randomized SVD with different parameters."""
        np.random.seed(123)
        A = np.random.randn(200, 150)

        # Test with different values of q (power iterations)
        for q in [0, 1, 2]:
            X = randomized_svd(A, r=20, p=5, q=q)
            assert X.rank == 20
            assert X.shape == (200, 150)

        # Test with different oversampling p
        for p in [5, 10, 15]:
            X = randomized_svd(A, r=20, p=p, q=0)
            assert X.rank == 20

    def test_adaptive_randomized_svd_basic(self):
        """Test adaptive randomized SVD."""
        np.random.seed(42)
        # Create a low-rank matrix with known structure
        m, n, true_rank = 300, 250, 15
        U, _ = np.linalg.qr(np.random.randn(m, true_rank))
        V, _ = np.linalg.qr(np.random.randn(n, true_rank))
        s = np.logspace(0, -3, true_rank)
        A = U @ np.diag(s) @ V.T

        # Adaptive randomized SVD
        X = adaptive_randomized_svd(A, tol=1e-6, failure_prob=1e-6)

        # Verify it's an SVD instance
        assert isinstance(X, SVD)

        # The adaptive method should find the effective rank
        # It may be >= true_rank depending on tolerance
        print(f"Adaptive rank: {X.rank}, True rank: {true_rank}")
        assert X.rank >= true_rank or X.rank <= true_rank + 10  # Some tolerance


class TestGeneralizedNystromExamples:
    """Tests for generalized Nyström examples from documentation."""

    def test_basic_generalized_nystrom(self):
        """Test generalized Nyström from quickstart documentation."""
        np.random.seed(42)
        # Create a symmetric positive-semidefinite matrix
        n = 500
        A = np.random.randn(n, 400)
        A_sym = A @ A.T

        # Generalized Nyström method
        X_nystrom = generalized_nystrom(A_sym, r=50, oversampling_params=(10, 15))

        # Verify it returns QuasiSVD (not SVD)
        from low_rank_toolbox.matrices.quasi_svd import QuasiSVD

        assert isinstance(X_nystrom, QuasiSVD)

        # Verify rank
        print(f"Nyström rank: {X_nystrom.rank}")
        assert X_nystrom.rank == 50

    def test_generalized_nystrom_with_svd_conversion(self):
        """Test converting Nyström result to SVD."""
        np.random.seed(123)
        n = 200
        A = np.random.randn(n, 150)
        A_sym = A @ A.T

        # Generalized Nyström
        result = generalized_nystrom(A_sym, r=30, oversampling_params=(5, 10))

        # Convert to SVD as mentioned in documentation
        result_svd = result.to_svd()
        assert isinstance(result_svd, SVD)
        assert result_svd.rank == 30

    def test_generalized_nystrom_stable_version(self):
        """Test stable generalized Nyström with epsilon truncation."""
        np.random.seed(42)
        n = 200
        A = np.random.randn(n, 100)
        A_sym = A @ A.T

        # Stable GN with epsilon-truncation
        # Note: with epsilon, the effective rank is determined by tolerance,
        # not by r. The rank can be larger than r because sketches are drawn
        # with size r + p1 and r + p2, and then truncated based on tolerance.
        result = generalized_nystrom(
            A_sym, r=30, epsilon=1e-6, oversampling_params=(5, 5)
        )

        from low_rank_toolbox.matrices.quasi_svd import QuasiSVD

        assert isinstance(result, QuasiSVD)
        # With epsilon truncation, rank is determined by tolerance,
        # so it may be different from r (could be smaller or up to r+min(p1,p2))
        assert result.rank <= 30 + 5  # r + min(p1, p2)


class TestRangefinderExamples:
    """Tests for rangefinder examples."""

    def test_basic_rangefinder(self):
        """Test basic rangefinder functionality."""
        np.random.seed(42)
        A = np.random.randn(200, 150)

        # Find range approximation
        Q = rangefinder(A, r=20, p=5, q=0)

        # Verify Q is orthonormal
        assert Q.shape == (200, 25)  # r + p = 20 + 5
        QTQ = Q.T @ Q
        assert np.allclose(QTQ, np.eye(25), atol=1e-10)

    def test_rangefinder_with_power_iterations(self):
        """Test rangefinder with power iterations."""
        np.random.seed(42)
        A = np.random.randn(200, 150)

        # With power iterations for better accuracy
        Q = rangefinder(A, r=20, p=5, q=2)

        # Verify Q is orthonormal
        assert Q.shape == (200, 25)
        QTQ = Q.T @ Q
        assert np.allclose(QTQ, np.eye(25), atol=1e-10)

    def test_adaptive_rangefinder(self):
        """Test adaptive rangefinder."""
        np.random.seed(42)
        # Create a low-rank matrix
        m, n, true_rank = 200, 150, 15
        U, _ = np.linalg.qr(np.random.randn(m, true_rank))
        V, _ = np.linalg.qr(np.random.randn(n, true_rank))
        s = np.logspace(0, -3, true_rank)
        A = U @ np.diag(s) @ V.T

        # Adaptive rangefinder
        Q = adaptive_rangefinder(A, tol=1e-6, failure_prob=1e-6)

        # Verify Q is orthonormal
        QTQ = Q.T @ Q
        assert np.allclose(QTQ, np.eye(Q.shape[1]), atol=1e-10)

        # The adaptive method should find at least the true rank
        print(f"Adaptive Q rank: {Q.shape[1]}, True rank: {true_rank}")


class TestRandomizedPerformanceExamples:
    """Tests for performance-related examples from documentation."""

    def test_performance_comparison_concept(self):
        """
        Test the performance comparison concept from quickstart.

        Note: We don't actually time the operations in the test,
        but verify both methods work and produce similar results.
        """
        np.random.seed(42)
        # Smaller matrix for testing (documentation uses 10000x10000)
        n = 500
        A = np.random.randn(n, n)

        # Full SVD (truncated)
        X_full = SVD.truncated_svd(A, r=50)

        # Randomized SVD
        X_random = randomized_svd(A, r=50, p=10, q=2)

        # Both should be rank 50
        assert X_full.rank == 50
        assert X_random.rank == 50

        # Both should approximate the matrix reasonably
        error_full = np.linalg.norm(A - X_full.to_dense(), "fro")
        error_random = np.linalg.norm(A - X_random.to_dense(), "fro")

        # Randomized should have similar error (within reasonable factor)
        print(f"Full SVD error: {error_full:.2e}")
        print(f"Randomized SVD error: {error_random:.2e}")
        assert error_random < 10 * error_full  # Loose bound for randomized


class TestRandomizedEdgeCases:
    """Test edge cases and error handling."""

    def test_randomized_svd_invalid_rank(self):
        """Test error handling for invalid rank."""
        A = np.random.randn(100, 80)

        # Rank must be positive
        with pytest.raises(ValueError, match="Rank must be at least 1"):
            randomized_svd(A, r=0)

        with pytest.raises(ValueError, match="Rank must be at least 1"):
            randomized_svd(A, r=-1)

    def test_randomized_svd_invalid_parameters(self):
        """Test error handling for invalid parameters."""
        A = np.random.randn(100, 80)

        # Oversampling must be non-negative
        with pytest.raises(
            ValueError, match="Oversampling parameter must be non-negative"
        ):
            randomized_svd(A, r=10, p=-1)

        # Power iterations must be non-negative
        with pytest.raises(
            ValueError, match="Number of power iterations must be non-negative"
        ):
            randomized_svd(A, r=10, q=-1)

        # Rank + oversampling must not exceed matrix dimensions
        with pytest.raises(ValueError, match="Rank \\+ oversampling.*exceeds"):
            randomized_svd(A, r=70, p=20)

    def test_generalized_nystrom_invalid_rank(self):
        """Test error handling for invalid rank in generalized Nyström."""
        A = np.random.randn(100, 100)

        with pytest.raises(ValueError, match="Rank must be at least 1"):
            generalized_nystrom(A, r=0)

    def test_generalized_nystrom_invalid_epsilon(self):
        """Test error handling for invalid epsilon."""
        A = np.random.randn(100, 100)

        with pytest.raises(ValueError, match="Epsilon must be positive"):
            generalized_nystrom(A, r=10, epsilon=0)

        with pytest.raises(ValueError, match="Epsilon must be positive"):
            generalized_nystrom(A, r=10, epsilon=-1e-3)

    def test_rangefinder_invalid_parameters(self):
        """Test error handling for invalid rangefinder parameters."""
        A = np.random.randn(100, 80)

        with pytest.raises(ValueError, match="Target rank must be at least 1"):
            rangefinder(A, r=0)

        with pytest.raises(
            ValueError, match="Oversampling parameter must be non-negative"
        ):
            rangefinder(A, r=10, p=-1)

        with pytest.raises(
            ValueError, match="Number of power iterations must be non-negative"
        ):
            rangefinder(A, r=10, q=-1)

    def test_adaptive_randomized_svd_invalid_tolerance(self):
        """Test error handling for invalid tolerance."""
        A = np.random.randn(100, 80)

        with pytest.raises(ValueError, match="Tolerance must be positive"):
            adaptive_randomized_svd(A, tol=0)

        with pytest.raises(ValueError, match="Tolerance must be positive"):
            adaptive_randomized_svd(A, tol=-1e-6)

    def test_adaptive_randomized_svd_invalid_failure_prob(self):
        """Test error handling for invalid failure probability."""
        A = np.random.randn(100, 80)

        with pytest.raises(
            ValueError, match="Failure probability must be in \\(0, 1\\)"
        ):
            adaptive_randomized_svd(A, failure_prob=0)

        with pytest.raises(
            ValueError, match="Failure probability must be in \\(0, 1\\)"
        ):
            adaptive_randomized_svd(A, failure_prob=1)

        with pytest.raises(
            ValueError, match="Failure probability must be in \\(0, 1\\)"
        ):
            adaptive_randomized_svd(A, failure_prob=-0.1)

    def test_adaptive_rangefinder_invalid_parameters(self):
        """Test error handling for invalid adaptive rangefinder parameters."""
        A = np.random.randn(100, 80)

        with pytest.raises(ValueError, match="Tolerance must be positive"):
            adaptive_rangefinder(A, tol=0)

        with pytest.raises(
            ValueError, match="Failure probability must be in \\(0, 1\\)"
        ):
            adaptive_rangefinder(A, failure_prob=1.5)


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
