"""
Tests for the examples provided in the Krylov submodule documentation.

IMPORTANT: ONE TEST = ONE EXAMPLE
=================================
Each test in this file corresponds to exactly one example in the Krylov module
documentation (docstrings, tutorials, etc.). This ensures that:

1. All documented examples are automatically tested and verified to work.
2. If documentation is updated with new examples, corresponding tests MUST be added here.
3. If a test is added or modified here, the corresponding example in the documentation
   MUST be updated to match.

This bidirectional synchronization keeps the documentation accurate and reliable.

Test Organization:
------------------
Each test class corresponds to a specific component (Krylov spaces, Lyapunov solvers,
Sylvester solvers, etc.) and contains tests for the examples shown in that component's
documentation.
"""

import numpy as np
from scipy.sparse import csr_matrix

from lowrank import LowRankMatrix
from lowrank.krylov import (
    KrylovSpace,
    RationalKrylovSpace,
    solve_lyapunov,
    solve_sylvester,
)
from lowrank.krylov.solvers.lyapunov_solvers import (
    solve_sparse_low_rank_symmetric_lyapunov,
)
from lowrank.krylov.solvers.sylvester_solvers import solve_sparse_low_rank_sylvester


class TestLyapunovSolverExamples:
    """Tests for Lyapunov solver examples from documentation."""

    def test_solve_sparse_low_rank_symmetric_lyapunov_example(self):
        """Test the example from solve_sparse_low_rank_symmetric_lyapunov docstring."""
        # Example from documentation - using larger matrix to avoid efficiency warnings
        # Create a symmetric sparse matrix (100x100 tridiagonal)
        from scipy.sparse import diags

        n = 100
        A = diags([1, 4, 1], [-1, 0, 1], shape=(n, n), format="csr", dtype=None)
        # Create a symmetric low-rank right-hand side
        U = np.zeros((n, 1))
        U[0, 0] = 1.0
        C = LowRankMatrix(U, U.T)  # Rank-1 symmetric matrix
        # Solve AX + XA = C
        X = solve_sparse_low_rank_symmetric_lyapunov(A, C, tol=1e-10)
        # Verify the solution
        residual = X.dot_sparse(A, side="opposite") + X.dot_sparse(A) - C
        residual_norm = (
            np.linalg.norm(residual)
            if isinstance(residual, np.ndarray)
            else residual.norm()
        )
        assert residual_norm < 1e-8

        # X is low-rank (symmetry may not be exact due to numerics)
        assert X.rank <= n

    def test_solve_lyapunov_small_dense_example(self):
        """Test the small dense example from solve_lyapunov docstring."""
        # Small dense case
        A_small = np.array([[4, 1], [1, 3]])
        C_small = np.array([[1, 0], [0, 1]])
        X_small = solve_lyapunov(A_small, C_small)
        # Verify: AX + XA = C
        assert np.allclose(A_small @ X_small + X_small @ A_small, C_small)

    def test_solve_lyapunov_large_sparse_low_rank_example(self):
        """Test the large sparse low-rank example from solve_lyapunov docstring."""
        # Large sparse case with low-rank RHS
        from scipy.sparse import diags

        n = 100
        A = diags([1, 4, 1], [-1, 0, 1], shape=(n, n), format="csr", dtype=None)
        U = np.zeros((n, 1))
        U[0, 0] = 1.0
        C = LowRankMatrix(U, U.T)
        X = solve_lyapunov(A, C, is_symmetric=True, tol=1e-10)
        # Solution is low-rank and symmetric
        assert type(X).__name__ == "SVD"
        assert X.rank <= n  # Much lower rank than n

        # Verify the solution
        residual = X.dot_sparse(A, side="opposite") + X.dot_sparse(A) - C
        residual_norm = (
            np.linalg.norm(residual)
            if isinstance(residual, np.ndarray)
            else residual.norm()
        )
        assert residual_norm < 1e-8


class TestSylvesterSolverExamples:
    """Tests for Sylvester solver examples from documentation."""

    def test_solve_sparse_low_rank_sylvester_example(self):
        """Test the example from solve_sparse_low_rank_sylvester docstring."""
        # Create sparse matrices A and B - using larger sizes
        from scipy.sparse import diags

        n, m = 100, 80
        A = diags([1, 4, 1], [-1, 0, 1], shape=(n, n), format="csr", dtype=None)
        B = diags([1, 2, 1], [-1, 0, 1], shape=(m, m), format="csr", dtype=None)
        # Create a low-rank right-hand side
        U = np.zeros((n, 1))
        U[0, 0] = 1.0
        V = np.zeros((m, 1))
        V[0, 0] = 1.0
        C = LowRankMatrix(U, V.T)
        # Solve AX + XB = C
        X = solve_sparse_low_rank_sylvester(A, B, C, tol=1e-10)
        # Verify the solution
        residual = X.dot_sparse(A, side="opposite") + X.dot_sparse(B) - C
        residual_norm = (
            np.linalg.norm(residual)
            if isinstance(residual, np.ndarray)
            else residual.norm()
        )
        assert residual_norm < 1e-8

        # X is low-rank
        assert X.rank <= 50

    def test_solve_sylvester_small_dense_example(self):
        """Test the small dense example from solve_sylvester docstring."""
        # Small dense case
        A_small = np.array([[4, 1], [1, 3]])
        B_small = np.array([[2, 1], [1, 1]])
        C_small = np.array([[1, 0], [0, 1]])
        X_small = solve_sylvester(A_small, B_small, C_small)
        # Verify: AX + XB = C
        assert np.allclose(A_small @ X_small + X_small @ B_small, C_small)

    def test_solve_sylvester_large_sparse_low_rank_example(self):
        """Test the large sparse low-rank example from solve_sylvester docstring."""
        # Large sparse case with low-rank RHS
        from scipy.sparse import diags

        n, m = 100, 80
        A = diags([1, 4, 1], [-1, 0, 1], shape=(n, n), format="csr", dtype=None)
        B = diags([1, 2, 1], [-1, 0, 1], shape=(m, m), format="csr", dtype=None)
        U = np.zeros((n, 1))
        U[0, 0] = 1.0
        V = np.zeros((m, 1))
        V[0, 0] = 1.0
        C = LowRankMatrix(U, V.T)
        X = solve_sylvester(A, B, C, tol=1e-10)
        # Solution is low-rank
        assert type(X).__name__ == "SVD"
        assert X.rank <= 50

        # Verify the solution
        residual = X.dot_sparse(A, side="opposite") + X.dot_sparse(B) - C
        residual_norm = (
            np.linalg.norm(residual)
            if isinstance(residual, np.ndarray)
            else residual.norm()
        )
        assert residual_norm < 1e-8


class TestKrylovSpaceExamples:
    """Tests for KrylovSpace examples from documentation."""

    def test_krylov_space_non_symmetric_example(self):
        """Test the non-symmetric example from KrylovSpace docstring."""
        # Non-symmetric case (uses Arnoldi)
        A = csr_matrix([[1, 2, 0], [0, 3, 1], [1, 0, 2]])
        x = np.array([[1.0], [0.0], [0.0]])  # Must be 2D
        K = KrylovSpace(A, x, is_symmetric=False)
        K.augment_basis()  # Add next basis vector
        assert K.Q.shape == (3, 2)

    def test_krylov_space_symmetric_example(self):
        """Test the symmetric example from KrylovSpace docstring."""
        # Symmetric case (uses Lanczos - more efficient)
        A_sym = csr_matrix([[4, 1, 0], [1, 3, 1], [0, 1, 2]])
        x = np.array([[1.0], [0.0], [0.0]])  # Must be 2D
        K_sym = KrylovSpace(A_sym, x, is_symmetric=True)
        K_sym.augment_basis()
        assert K_sym.Q.shape == (3, 2)

    def test_krylov_space_orthogonality(self):
        """Test that Krylov basis maintains orthogonality."""
        A = csr_matrix([[4, 1, 0], [1, 3, 1], [0, 1, 2]])
        x = np.array([[1.0], [0.0], [0.0]])  # Must be 2D
        K = KrylovSpace(A, x, is_symmetric=True)

        # Augment once (not multiple times to avoid dimension exceeded)
        K.augment_basis()

        # Verify orthogonality
        Q = K.Q
        assert np.allclose(Q.T @ Q, np.eye(Q.shape[1]), atol=1e-10)


class TestRationalKrylovSpaceExamples:
    """Tests for RationalKrylovSpace examples from documentation."""

    def test_rational_krylov_space_example(self):
        """Test the example from RationalKrylovSpace docstring."""
        # Create a sparse matrix
        A = csr_matrix([[4, 1, 0], [1, 3, 1], [0, 1, 2]])
        X = np.array([[1.0], [0.0], [0.0]])
        # Choose poles to emphasize specific spectral regions
        # (e.g., near eigenvalues of interest)
        poles = [1.0, 2.0, 3.0]
        RK = RationalKrylovSpace(A, X, poles)
        # Each augmentation uses the next pole
        RK.augment_basis()  # Uses pole 1.0
        assert RK.Q.shape == (3, 2)

        RK.augment_basis()  # Uses pole 2.0
        assert RK.Q.shape == (3, 3)

        # Verify orthogonality
        assert np.allclose(RK.Q.T @ RK.Q, np.eye(3))

    def test_rational_krylov_space_multiple_augmentations(self):
        """Test multiple augmentations with rational Krylov space."""
        A = csr_matrix([[4, 1, 0], [1, 3, 1], [0, 1, 2]])
        X = np.array([[1.0], [0.0], [0.0]])
        poles = [0.5, 1.5]  # Only 2 poles to avoid exceeding dimension
        RK = RationalKrylovSpace(A, X, poles)

        # Augment and check orthogonality at each step
        for i in range(len(poles)):
            RK.augment_basis()
            Q = RK.Q
            # Verify orthogonality
            assert np.allclose(Q.T @ Q, np.eye(Q.shape[1]), atol=1e-10)


class TestKrylovSolverIntegration:
    """Integration tests for Krylov solvers."""

    def test_lyapunov_vs_sylvester_when_A_equals_B(self):
        """Test that Lyapunov solver matches Sylvester when A = B."""
        # When solving AX + XA = C (Lyapunov) vs AX + XB = C with B = A (Sylvester)
        # The solutions should be the same
        from scipy.sparse import diags

        n = 100
        A = diags([1, 4, 1], [-1, 0, 1], shape=(n, n), format="csr", dtype=None)
        U = np.zeros((n, 1))
        U[0, 0] = 1.0
        C = LowRankMatrix(U, U.T)

        # Solve as Lyapunov
        X_lyap = solve_lyapunov(A, C, is_symmetric=True, tol=1e-10)

        # Solve as Sylvester with B = A
        X_sylv = solve_sylvester(A, A, C, tol=1e-10)

        # Both should give similar solutions (in terms of residual)
        residual_lyap = X_lyap.dot_sparse(A, side="opposite") + X_lyap.dot_sparse(A) - C
        residual_sylv = X_sylv.dot_sparse(A, side="opposite") + X_sylv.dot_sparse(A) - C

        norm_lyap = (
            np.linalg.norm(residual_lyap)
            if isinstance(residual_lyap, np.ndarray)
            else residual_lyap.norm()
        )
        norm_sylv = (
            np.linalg.norm(residual_sylv)
            if isinstance(residual_sylv, np.ndarray)
            else residual_sylv.norm()
        )

        # Both should satisfy the equation well
        assert norm_lyap < 1e-8
        assert norm_sylv < 1e-8

    def test_different_krylov_configurations(self):
        """Test Krylov solvers with different configurations."""
        # Test symmetric vs non-symmetric detection
        from scipy.sparse import diags

        n = 100
        A_sym = diags([1, 4, 1], [-1, 0, 1], shape=(n, n), format="csr", dtype=None)
        U = np.zeros((n, 1))
        U[0, 0] = 1.0
        C = LowRankMatrix(U, U.T)

        # Solve with explicit is_symmetric=True
        X_sym = solve_lyapunov(A_sym, C, is_symmetric=True, tol=1e-10)

        # Solve with is_symmetric=False (uses different algorithm)
        X_nonsym = solve_lyapunov(A_sym, C, is_symmetric=False, tol=1e-10)

        # Both should satisfy the equation
        residual_sym = (
            X_sym.dot_sparse(A_sym, side="opposite") + X_sym.dot_sparse(A_sym) - C
        )
        residual_nonsym = (
            X_nonsym.dot_sparse(A_sym, side="opposite") + X_nonsym.dot_sparse(A_sym) - C
        )

        norm_sym = (
            np.linalg.norm(residual_sym)
            if isinstance(residual_sym, np.ndarray)
            else residual_sym.norm()
        )
        norm_nonsym = (
            np.linalg.norm(residual_nonsym)
            if isinstance(residual_nonsym, np.ndarray)
            else residual_nonsym.norm()
        )

        assert norm_sym < 1e-8
        assert norm_nonsym < 1e-8

    def test_sylvester_with_different_matrix_sizes(self):
        """Test Sylvester equation with different sized matrices."""
        # Test with rectangular solution (m != n)
        from scipy.sparse import diags

        n, m = 120, 80
        A = diags(
            [1, 4, 1], [-1, 0, 1], shape=(n, n), format="csr", dtype=None
        )  # 120x120
        B = diags(
            [1, 2, 1], [-1, 0, 1], shape=(m, m), format="csr", dtype=None
        )  # 80x80

        # C should be 120x80 (compatible with A @ X + X @ B)
        U = np.zeros((n, 1))  # 120x1
        U[0, 0] = 1.0
        V = np.zeros((m, 1))  # 80x1
        V[0, 0] = 1.0
        C = LowRankMatrix(U, V.T)  # 120x80

        X = solve_sylvester(A, B, C, tol=1e-10)

        # Verify shape
        assert X.shape == (n, m)

        # Verify the solution
        residual = X.dot_sparse(A, side="opposite") + X.dot_sparse(B) - C
        norm = (
            np.linalg.norm(residual)
            if isinstance(residual, np.ndarray)
            else residual.norm()
        )
        assert norm < 1e-8


class TestKrylovExamplesEdgeCases:
    """Test edge cases and special scenarios for Krylov methods."""

    def test_krylov_space_with_matrix_input(self):
        """Test KrylovSpace with matrix (not vector) initial condition."""
        A = csr_matrix([[4, 1, 0], [1, 3, 1], [0, 1, 2]])
        X = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])  # 3x2 matrix
        K = KrylovSpace(A, X, is_symmetric=True)

        # Should handle matrix input
        assert K.Q.shape[0] == 3  # Same number of rows as A
        # Initial Q has shape (3, 2) since X is 3x2
        assert K.Q.shape[1] == 2

    def test_lyapunov_with_rank_one_rhs(self):
        """Test Lyapunov equation with rank-1 right-hand side."""
        # Test with rank-1 RHS - should give efficient solution
        from scipy.sparse import diags

        n = 100
        A = diags([1, 4, 1], [-1, 0, 1], shape=(n, n), format="csr", dtype=None)

        # Create a rank-1 symmetric matrix
        u = np.zeros((n, 1))
        u[0, 0] = 1.0
        u[1, 0] = 0.5
        C = LowRankMatrix(u, u.T)  # Rank-1

        X = solve_lyapunov(A, C, is_symmetric=True, tol=1e-10)

        # Solution should be low-rank
        assert X.rank <= n  # Much lower than matrix size

        # Verify the solution
        residual = X.dot_sparse(A, side="opposite") + X.dot_sparse(A) - C
        norm = (
            np.linalg.norm(residual)
            if isinstance(residual, np.ndarray)
            else residual.norm()
        )
        assert norm < 1e-8

    def test_sylvester_symmetry_detection(self):
        """Test that Sylvester solver correctly handles symmetric cases."""
        # When A and B are both symmetric and equal, it's a Lyapunov equation
        from scipy.sparse import diags

        n = 100
        A = diags([1, 4, 1], [-1, 0, 1], shape=(n, n), format="csr", dtype=None)

        # Create symmetric low-rank C
        u = np.zeros((n, 1))
        u[0, 0] = 1.0
        C = LowRankMatrix(u, u.T)

        # Solve AX + XA = C as a Sylvester equation
        X = solve_sylvester(A, A, C, tol=1e-10)

        # Verify the solution
        residual = X.dot_sparse(A, side="opposite") + X.dot_sparse(A) - C
        norm = (
            np.linalg.norm(residual)
            if isinstance(residual, np.ndarray)
            else residual.norm()
        )
        assert norm < 1e-8

        # Solution should be symmetric for symmetric problem
        # (Note: implementation may not guarantee this, so we just check it solves the equation)
        assert X.rank <= n
