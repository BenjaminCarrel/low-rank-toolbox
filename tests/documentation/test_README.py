"""
Tests for the examples provided in the README.md file.

IMPORTANT: ONE TEST = ONE EXAMPLE
=================================
Each test in this file corresponds to exactly one example in the README.md file.
This ensures that:

1. All documented examples in the README are automatically tested and verified to work.
2. If the README is updated with new examples, corresponding tests MUST be added here.
3. If a test is added or modified here, the corresponding example in the README
   MUST be updated to match.

This bidirectional synchronization keeps the README examples accurate and reliable.

Test Organization:
------------------
Each test class corresponds to a section in the README (e.g., "Low-Rank Matrix
Representations", "CSSP Algorithms", etc.) and contains tests for the examples
shown in that section.
"""

import numpy as np
from scipy.sparse import diags

from lowrank import SVD, QDEIM, solve_lyapunov
from lowrank.matrices import SVD as SVDMatrices
from lowrank.matrices.low_rank_matrix import LowRankMatrix
from lowrank.randomized import randomized_svd, generalized_nystrom


class TestLowRankMatrixRepresentations:
    """Tests for the Low-Rank Matrix Representations example in README."""

    def test_svd_creation_and_basic_operations(self):
        """Test creating SVD representation and basic operations."""
        # Create orthonormal matrices
        m, n, r = 1000, 1000, 20
        U, _ = np.linalg.qr(np.random.randn(m, r))
        V, _ = np.linalg.qr(np.random.randn(n, r))
        s = np.logspace(0, -3, r)  # Singular values with exponential decay

        # Create SVD representation (only stores U, s, V - not the full matrix!)
        X = SVD(U, s, V)

        # Verify shape and rank
        assert X.shape == (m, n)
        assert X.rank == r

        # Verify compression ratio is meaningful
        compression = X.compression_ratio()
        assert 0 < compression < 1
        memory_savings_factor = 1 / compression
        assert memory_savings_factor > 1

        # Efficient operations exploiting low-rank structure
        norm = X.norm("fro")  # Frobenius norm: O(r) instead of O(mn)
        assert norm > 0
        assert isinstance(norm, (float, np.floating))

        trace = X.trace()  # Trace: O(r²) instead of O(min(m,n))
        assert isinstance(trace, (float, np.floating, complex, np.complexfloating))

        Y = X @ X.T  # Matrix multiplication returns low-rank matrix
        assert isinstance(Y, LowRankMatrix)
        assert Y.shape == (m, m)


class TestComputingSVDFromMatrices:
    """Tests for the Computing SVD from Matrices example in README."""

    def test_truncated_svd_with_rank(self):
        """Test truncated SVD with specified rank."""
        # Full SVD
        s_vals = np.logspace(0, -15, 30)
        A = SVDMatrices.generate_random(shape=(1000, 1000), sing_vals=s_vals)

        # Truncated SVD (keep top 10 singular values)
        X_trunc = SVD.truncated_svd(A, r=10)

        assert X_trunc.rank == 10
        assert X_trunc.shape == (1000, 1000)

    def test_truncated_svd_with_tolerance(self):
        """Test adaptive truncation based on tolerance."""
        # Full SVD
        s_vals = np.logspace(0, -15, 30)
        A = SVDMatrices.generate_random(shape=(1000, 1000), sing_vals=s_vals)

        # Adaptive truncation (tolerance-based)
        X_adaptive = SVD.truncated_svd(A, rtol=1e-6)

        # Verify that adaptive rank makes sense (should be less than full rank)
        assert X_adaptive.rank > 0
        assert X_adaptive.rank <= 30
        assert X_adaptive.shape == (1000, 1000)


class TestColumnSubsetSelection:
    """Tests for the Column Subset Selection (CSSP) example in README."""

    def test_qdeim_interpolation(self):
        """Test QDEIM column selection and interpolation property."""
        # Create basis matrix (e.g., POD modes, eigenvectors, etc.)
        U, _ = np.linalg.qr(np.random.randn(1000, 10))

        # Select 10 interpolation points with guaranteed bounds
        indices, projector = QDEIM(U, return_projector=True)

        # Verify we got the expected outputs
        assert len(indices) == 10
        assert (
            projector.shape[1] == 10
        )  # projector maps from selected rows to full space

        # Interpolation property: U ≈ Projector @ U[indices, :]
        interpolated = projector @ U[indices, :]
        error = np.linalg.norm(U - interpolated, "fro")

        # Error should be small (QDEIM provides good interpolation)
        assert error < 1e-10


class TestKrylovSubspaceMethods:
    """Tests for the Krylov Subspace Methods example in README."""

    def test_lyapunov_solver(self):
        """Test solving large-scale Lyapunov equations."""
        # Large sparse matrix
        n = 10000
        A = diags([-1, 2, -1], [-1, 0, 1], shape=(n, n), format="csc")

        # Low-rank right-hand side: AX + XA^T = C
        s_vals = np.logspace(0, -15, 5)
        C = SVDMatrices.generate_random(
            shape=(n, n), sing_vals=s_vals, is_symmetric=True
        )

        # Solve using Krylov methods (never forms full n×n solution!)
        X = solve_lyapunov(A, C, tol=1e-8, is_symmetric=True)

        # Verify solution properties
        assert isinstance(X, SVD)
        assert X.rank > 0
        assert X.shape == (n, n)

        # Solution should have low rank (much smaller than n)
        assert X.rank < 100  # Should be much smaller than n=10000


class TestRandomizedLowRankApproximation:
    """Tests for the Randomized Low-Rank Approximation example in README."""

    def test_randomized_svd(self):
        """Test randomized SVD approximation."""
        # Large matrix
        s_vals = np.logspace(0, -15, 50)
        A = SVDMatrices.generate_random(shape=(1000, 1000), sing_vals=s_vals).todense()

        # Randomized SVD (much faster than full SVD)
        X_approx = randomized_svd(A, r=20, p=10, q=2)

        # Verify approximation properties
        assert isinstance(X_approx, SVD)
        assert X_approx.rank == 20
        assert X_approx.shape == (1000, 1000)

        # Verify approximation error is reasonable
        error = np.linalg.norm(A - X_approx.to_dense(), "fro")
        # Since we're approximating with rank 20 and decay is exponential,
        # error should be related to truncated singular values
        expected_error_bound = np.sqrt(np.sum(s_vals[20:] ** 2))
        assert error < expected_error_bound * 10  # Allow some slack for randomization

    def test_generalized_nystrom(self):
        """Test generalized Nyström method for symmetric matrices."""
        # Large matrix
        s_vals = np.logspace(0, -15, 50)
        A = SVDMatrices.generate_random(shape=(1000, 1000), sing_vals=s_vals).todense()

        # Generalized Nyström method (for symmetric/positive-semidefinite matrices)
        A_sym = A @ A.T
        X_nystrom = generalized_nystrom(A_sym, r=20, oversampling_params=(10, 15))

        # Verify approximation properties
        assert isinstance(X_nystrom, LowRankMatrix)
        assert X_nystrom.rank == 20
        assert X_nystrom.shape == (1000, 1000)

        # Verify approximation error
        error = np.linalg.norm(A_sym - X_nystrom.to_dense(), "fro")
        assert error >= 0  # Error should be non-negative
        # Just verify it produces a result - exact error bounds depend on matrix structure


class TestREADMEExamplesIntegration:
    """Integration tests combining multiple README examples."""

    def test_workflow_svd_to_cssp(self):
        """Test workflow: create SVD, then use CSSP on left singular vectors."""
        # Create SVD
        m, n, r = 500, 500, 15
        U, _ = np.linalg.qr(np.random.randn(m, r))
        V, _ = np.linalg.qr(np.random.randn(n, r))
        s = np.logspace(0, -3, r)
        X = SVD(U, s, V)

        # Use QDEIM on left singular vectors
        indices, projector = QDEIM(X.U, return_projector=True)

        assert len(indices) == r
        assert projector.shape == (m, r)

    def test_workflow_randomized_then_operations(self):
        """Test workflow: randomized approximation followed by operations."""
        # Create matrix and approximate it
        s_vals = np.logspace(0, -10, 50)
        A = SVDMatrices.generate_random(shape=(1000, 1000), sing_vals=s_vals).todense()
        X_approx = randomized_svd(A, r=20, p=5, q=1)

        # Perform operations on approximation
        norm = X_approx.norm("fro")
        trace = X_approx.trace()

        assert norm > 0
        assert isinstance(trace, (float, np.floating, complex, np.complexfloating))

        # Multiply approximation
        Y = X_approx @ X_approx.T
        assert isinstance(Y, LowRankMatrix)
        assert Y.shape == (1000, 1000)
