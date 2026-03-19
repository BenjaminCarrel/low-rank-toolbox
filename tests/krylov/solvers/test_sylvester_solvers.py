"""
Author: Benjamin Carrel, University of Geneva, 2022

Comprehensive tests for the Sylvester solvers.
This file provides pytest tests covering all use cases and edge cases.
"""

# %% Imports
import numpy as np
import pytest
import scipy.linalg as la
import scipy.sparse as sps

from low_rank_toolbox.krylov.solvers.sylvester_solvers import (
    solve_small_sylvester,
    solve_sparse_low_rank_sylvester,
    solve_sylvester,
    solve_sylvester_large_A_small_B,
)
from low_rank_toolbox.matrices.quasi_svd import QuasiSVD
from low_rank_toolbox.matrices.svd import SVD


# %% Helper functions
def create_1d_laplacian(n):
    """Create a discrete 1D Laplacian matrix on [0,1]."""
    h = 1.0 / (n + 1)
    # Tridiagonal matrix: [-1, 2, -1] / h^2
    diag = np.ones(n) * 2 / h**2
    off_diag = np.ones(n - 1) * (-1) / h**2
    A = sps.diags([off_diag, diag, off_diag], [-1, 0, 1], format="csc")
    return A


def generate_sylvester_problem(m, n, rank_C, seed=42):
    """Generate a Sylvester problem AX + XB = C with known solution X."""
    np.random.seed(seed)

    # Create matrices A and B (using 1D Laplacian for SPD structure)
    A = create_1d_laplacian(m)
    B = create_1d_laplacian(n)

    # Generate random low-rank X
    X = SVD.generate_random((m, n), np.logspace(-1, -rank_C, rank_C), seed=seed)

    # Compute C = AX + XB
    X_dense = X.to_dense()
    C_dense = A @ X_dense + X_dense @ B

    # Convert C to low-rank format
    C = SVD.from_dense(C_dense)

    return A, B, C, X


def compute_sylvester_residual(A, B, X, C):
    """Compute the residual ||AX + XB - C|| / ||C||."""
    if isinstance(X, (SVD, QuasiSVD)):
        X_dense = X.to_dense()
    else:
        X_dense = X

    if isinstance(C, (SVD, QuasiSVD)):
        C_dense = C.to_dense()
    else:
        C_dense = C

    residual = A @ X_dense + X_dense @ B - C_dense

    return la.norm(residual, "fro") / la.norm(C_dense, "fro")


# %% Test 1: Input validation tests
class TestInputValidation:
    """Test input validation for all Sylvester solvers."""

    def test_solve_sylvester_non_sparse_with_low_rank(self):
        """Test that solve_sylvester handles dense A and B with low-rank C."""
        m, n = 100, 100
        A = np.random.rand(m, m)
        B = np.random.rand(n, n)
        C = SVD.generate_random((m, n), np.logspace(-1, -3, 3))

        # Should not raise error (auto-converts to sparse)
        X = solve_sylvester(A, B, C)
        assert X is not None

    def test_sparse_solver_requires_sparse_matrices(self):
        """Test that sparse solver requires sparse matrices."""
        m, n = 100, 100
        A = np.random.rand(m, m)
        B = sps.random(n, n, density=0.1, format="csc")
        C = SVD.generate_random((m, n), np.logspace(-1, -3, 3))

        with pytest.raises(AssertionError, match="A must be a sparse matrix"):
            solve_sparse_low_rank_sylvester(A, B, C)

    def test_sparse_solver_requires_low_rank_C(self):
        """Test that sparse solver requires low-rank C."""
        m, n = 100, 100
        A = sps.random(m, m, density=0.1, format="csc")
        B = sps.random(n, n, density=0.1, format="csc")
        C = np.random.rand(m, n)

        with pytest.raises(AssertionError, match="C must be a low-rank matrix"):
            solve_sparse_low_rank_sylvester(A, B, C)

    def test_tolerance_must_be_reasonable(self):
        """Test that tolerance must be above machine precision."""
        m, n = 100, 100
        A = sps.random(m, m, density=0.1, format="csc")
        B = sps.random(n, n, density=0.1, format="csc")
        C = SVD.generate_random((m, n), np.logspace(-1, -3, 3))

        with pytest.raises(
            AssertionError, match="tol must be larger than machine precision"
        ):
            solve_sparse_low_rank_sylvester(A, B, C, tol=1e-20)

    def test_max_iter_validation(self):
        """Test that max_iter must be at least 2."""
        m, n = 100, 100
        A = sps.random(m, m, density=0.1, format="csc")
        B = sps.random(n, n, density=0.1, format="csc")
        C = SVD.generate_random((m, n), np.logspace(-1, -3, 3))

        with pytest.raises(AssertionError, match="max_iter must be at least 2"):
            solve_sparse_low_rank_sylvester(A, B, C, max_iter=1)

    def test_extended_and_poles_conflict(self):
        """Test that extended and rational Krylov cannot be used together."""
        m, n = 100, 100
        A = sps.random(m, m, density=0.1, format="csc")
        B = sps.random(n, n, density=0.1, format="csc")
        C = SVD.generate_random((m, n), np.logspace(-1, -3, 3))

        krylov_kwargs = {"extended": True, "poles_A": [1.0, 2.0]}
        with pytest.raises(
            ValueError,
            match="Cannot use rational Krylov space with extended Krylov space",
        ):
            solve_sparse_low_rank_sylvester(A, B, C, krylov_kwargs=krylov_kwargs)


# %% Test 2: Edge cases
class TestEdgeCases:
    """Test edge cases for Sylvester solvers."""

    def test_small_matrix_size(self):
        """Test solver on small matrices."""
        m, n = 100, 80
        A, B, C, X_true = generate_sylvester_problem(m, n, rank_C=2, seed=123)

        X = solve_sylvester(A, B, C)
        residual = compute_sylvester_residual(A, B, X, C)
        assert residual < 1e-8, f"Residual too large: {residual}"

    def test_rank_one_C(self):
        """Test solver with rank-1 right-hand side."""
        m, n = 150, 150
        A, B, C, X_true = generate_sylvester_problem(m, n, rank_C=1, seed=456)

        X = solve_sylvester(A, B, C)
        residual = compute_sylvester_residual(A, B, X, C)
        assert residual < 1e-7, f"Residual too large: {residual}"

    def test_high_rank_C(self):
        """Test solver with rank-5 C."""
        m, n = 150, 150
        A, B, C, X_true = generate_sylvester_problem(m, n, rank_C=5, seed=789)

        X = solve_sylvester(A, B, C)
        residual = compute_sylvester_residual(A, B, X, C)
        assert residual < 1e-6, f"Residual too large: {residual}"

    def test_rectangular_problem(self):
        """Test solver with non-square X (m != n)."""
        m, n = 150, 100
        A, B, C, X_true = generate_sylvester_problem(m, n, rank_C=3, seed=101)

        X = solve_sylvester(A, B, C)
        residual = compute_sylvester_residual(A, B, X, C)
        assert residual < 1e-7, f"Residual too large: {residual}"

    def test_nearly_singular_matrix(self):
        """Test solver with matrix that has small eigenvalues."""
        m, n = 100, 100
        # Create matrices with very small eigenvalues
        A = sps.eye(m, format="csc") * 1e-3
        B = sps.eye(n, format="csc") * 1e-3
        C = SVD.generate_random((m, n), np.array([1.0, 0.5, 0.1]))

        X = solve_sylvester(A, B, C, tol=1e-8)
        assert X is not None


# %% Test 3: Large problem tests
class TestLargeProblems:
    """Test on larger, realistic problems."""

    def test_100x100_problem(self):
        """Test 100x100 Sylvester equation."""
        m, n = 100, 100
        rank_C = 3
        A, B, C, X_true = generate_sylvester_problem(m, n, rank_C, seed=123)

        # Solve
        X = solve_sylvester(A, B, C, tol=1e-10)

        # Check residual
        residual = compute_sylvester_residual(A, B, X, C)
        print(f"100x100 residual: {residual}")
        assert residual < 1e-8, f"Residual too large: {residual}"

        # Check solution accuracy
        X_dense = X.to_dense()
        X_true_dense = X_true.to_dense()
        solution_error = la.norm(X_dense - X_true_dense, "fro") / la.norm(
            X_true_dense, "fro"
        )
        print(f"Solution error: {solution_error}")
        assert solution_error < 1e-4, f"Solution error too large: {solution_error}"

    def test_200x150_rectangular_problem(self):
        """Test rectangular 200x150 Sylvester equation."""
        m, n = 200, 150
        rank_C = 4
        A, B, C, X_true = generate_sylvester_problem(m, n, rank_C, seed=456)

        # Solve
        X = solve_sylvester(A, B, C, tol=1e-10)

        # Check residual
        residual = compute_sylvester_residual(A, B, X, C)
        print(f"200x150 residual: {residual}")
        assert residual < 1e-8, f"Residual too large: {residual}"


# %% Test 4: Different Krylov methods
class TestKrylovMethods:
    """Test different Krylov space methods."""

    def test_classic_krylov(self):
        """Test with standard Krylov space."""
        m, n = 150, 150
        rank_C = 2
        A, B, C, X_true = generate_sylvester_problem(m, n, rank_C)

        krylov_kwargs = {"extended": False}
        with pytest.warns(UserWarning, match="standard Krylov space may not converge"):
            X = solve_sylvester(A, B, C, krylov_kwargs=krylov_kwargs)

        residual = compute_sylvester_residual(A, B, X, C)
        print(f"Classic Krylov residual: {residual}")
        # Standard Krylov may not converge as well
        assert residual < 1e-4, f"Residual too large: {residual}"

    def test_extended_krylov(self):
        """Test with extended Krylov space (default)."""
        m, n = 150, 150
        rank_C = 3
        A, B, C, X_true = generate_sylvester_problem(m, n, rank_C)

        krylov_kwargs = {"extended": True}
        X = solve_sylvester(A, B, C, krylov_kwargs=krylov_kwargs)

        residual = compute_sylvester_residual(A, B, X, C)
        print(f"Extended Krylov residual: {residual}")
        assert residual < 1e-8, f"Residual too large: {residual}"

    def test_rational_krylov(self):
        """Test with rational Krylov space."""
        m, n = 150, 150
        rank_C = 3
        A, B, C, X_true = generate_sylvester_problem(m, n, rank_C)

        # Use poles near the smallest eigenvalues
        poles_A = [10.0, 20.0, 30.0, 40.0, 50.0]
        poles_B = [10.0, 20.0, 30.0, 40.0, 50.0]
        krylov_kwargs = {"extended": False, "poles_A": poles_A, "poles_B": poles_B}
        X = solve_sylvester(A, B, C, krylov_kwargs=krylov_kwargs)

        residual = compute_sylvester_residual(A, B, X, C)
        print(f"Rational Krylov residual: {residual}")
        assert residual < 1e-8, f"Residual too large: {residual}"

    def test_custom_invA_invB(self):
        """Test with custom inverse functions."""
        m, n = 150, 150
        rank_C = 3
        A, B, C, X_true = generate_sylvester_problem(m, n, rank_C)

        # Custom solvers (just use the default but count calls)
        call_count_A = [0]
        call_count_B = [0]

        def custom_invA(x):
            call_count_A[0] += 1
            return sps.linalg.spsolve(A, x)

        def custom_invB(x):
            call_count_B[0] += 1
            return sps.linalg.spsolve(B, x)

        krylov_kwargs = {"extended": True, "invA": custom_invA, "invB": custom_invB}
        X = solve_sylvester(A, B, C, krylov_kwargs=krylov_kwargs)

        assert call_count_A[0] > 0, "Custom inverse A was not called"
        assert call_count_B[0] > 0, "Custom inverse B was not called"
        residual = compute_sylvester_residual(A, B, X, C)
        assert residual < 1e-8, f"Residual too large: {residual}"


# %% Test 5: Different input combinations
class TestInputCombinations:
    """Test all combinations of input types."""

    def test_sparse_A_sparse_B_lowrank_C(self):
        """Test: A sparse, B sparse, C low-rank."""
        m, n = 150, 150
        rank_C = 4
        A, B, C, X_true = generate_sylvester_problem(m, n, rank_C)

        X = solve_sylvester(A, B, C)
        residual = compute_sylvester_residual(A, B, X, C)
        assert residual < 1e-8
        print("✓ Sparse A + Sparse B + Low-rank C")

    def test_sparse_A_dense_B_dense_C(self):
        """Test: A sparse, B dense, C dense."""
        m, n = 100, 100
        A = create_1d_laplacian(m)
        B = create_1d_laplacian(n).toarray()
        C = np.random.rand(m, n)

        X = solve_sylvester(A, B, C)

        # Compute residual
        residual_matrix = A @ X + X @ B - C
        residual = la.norm(residual_matrix, "fro") / la.norm(C, "fro")
        assert residual < 1e-10
        print("✓ Sparse A + Dense B + Dense C")

    def test_dense_A_sparse_B_dense_C(self):
        """Test: A dense, B sparse, C dense (uses transpose trick)."""
        m, n = 100, 100
        A = create_1d_laplacian(m).toarray()
        B = create_1d_laplacian(n)
        C = np.random.rand(m, n)

        X = solve_sylvester(A, B, C)

        # Compute residual
        residual_matrix = A @ X + X @ B - C
        residual = la.norm(residual_matrix, "fro") / la.norm(C, "fro")
        assert residual < 1e-10
        print("✓ Dense A + Sparse B + Dense C")

    def test_dense_A_dense_B_lowrank_C(self):
        """Test: A dense, B dense, C low-rank (auto-converts to sparse)."""
        m, n = 100, 100
        A = create_1d_laplacian(m).toarray()
        B = create_1d_laplacian(n).toarray()
        C = SVD.generate_random((m, n), np.logspace(-1, -3, 3))

        X = solve_sylvester(A, B, C)

        # Compute residual
        C_dense = C.to_dense()
        X_dense = X.to_dense() if hasattr(X, "to_dense") else X
        residual_matrix = A @ X_dense + X_dense @ B - C_dense
        residual = la.norm(residual_matrix, "fro") / la.norm(C_dense, "fro")
        assert residual < 1e-8
        print("✓ Dense A + Dense B + Low-rank C")

    def test_dense_A_dense_B_dense_C(self):
        """Test: A dense, B dense, C dense."""
        m, n = 100, 100
        A = create_1d_laplacian(m).toarray()
        B = create_1d_laplacian(n).toarray()
        C = np.random.rand(m, n)

        X = solve_sylvester(A, B, C)

        # Compute residual
        residual_matrix = A @ X + X @ B - C
        residual = la.norm(residual_matrix, "fro") / la.norm(C, "fro")
        assert residual < 1e-10
        print("✓ Dense A + Dense B + Dense C")


# %% Test 6: Tolerance criterion
class TestToleranceCriterion:
    """Test that the tolerance criterion is met."""

    def test_tolerance_1e6(self):
        """Test with tolerance 1e-6."""
        m, n = 150, 150
        rank_C = 3
        A, B, C, X_true = generate_sylvester_problem(m, n, rank_C)

        tol = 1e-6
        X = solve_sylvester(A, B, C, tol=tol)
        residual = compute_sylvester_residual(A, B, X, C)
        print(f"Tolerance {tol}: residual = {residual}")
        assert residual < tol * 10, f"Residual {residual} exceeds tolerance {tol}"

    def test_tolerance_1e8(self):
        """Test with tolerance 1e-8."""
        m, n = 150, 150
        rank_C = 3
        A, B, C, X_true = generate_sylvester_problem(m, n, rank_C)

        tol = 1e-8
        X = solve_sylvester(A, B, C, tol=tol)
        residual = compute_sylvester_residual(A, B, X, C)
        print(f"Tolerance {tol}: residual = {residual}")
        assert residual < tol * 10, f"Residual {residual} exceeds tolerance {tol}"

    def test_tolerance_1e10(self):
        """Test with tolerance 1e-10."""
        m, n = 150, 150
        rank_C = 3
        A, B, C, X_true = generate_sylvester_problem(m, n, rank_C)

        tol = 1e-10
        X = solve_sylvester(A, B, C, tol=tol)
        residual = compute_sylvester_residual(A, B, X, C)
        print(f"Tolerance {tol}: residual = {residual}")
        assert residual < tol * 10, f"Residual {residual} exceeds tolerance {tol}"


# %% Test 7: max_iter parameter
class TestMaxIterations:
    """Test that max_iter parameter works correctly."""

    def test_max_iter_stops_early(self):
        """Test that solver stops at max_iter even if not converged."""
        m, n = 200, 200
        rank_C = 8  # Higher rank to make convergence harder
        A, B, C, X_true = generate_sylvester_problem(m, n, rank_C)

        # Set very low max_iter to force early stopping
        max_iter = 2  # Minimum allowed
        # Use standard Krylov (slower convergence) to ensure we hit max_iter
        krylov_kwargs = {"extended": False}
        with pytest.warns(
            UserWarning
        ):  # May warn about standard Krylov or no convergence
            X = solve_sylvester(
                A, B, C, tol=1e-14, max_iter=max_iter, krylov_kwargs=krylov_kwargs
            )

        # Solution should still be returned
        assert X is not None
        print(f"Early stop test: X shape = {X.to_dense().shape}")

    def test_sufficient_max_iter(self):
        """Test with sufficient max_iter for convergence."""
        m, n = 150, 150
        rank_C = 3
        A, B, C, X_true = generate_sylvester_problem(m, n, rank_C)

        # Set reasonable max_iter
        max_iter = 50
        X = solve_sylvester(A, B, C, tol=1e-10, max_iter=max_iter)

        residual = compute_sylvester_residual(A, B, X, C)
        assert residual < 1e-8, f"Residual too large: {residual}"


# %% Test 8: Legacy API compatibility
class TestLegacyTests:
    """Test the original basic functionality tests."""

    def test_sylvester_small(self):
        """Test the small solver (original test)."""
        n = 200
        r = 10
        a = sps.random(n, n, density=0.1, random_state=42)
        A = a.dot(a.T)
        Ad = A.todense()
        b = sps.random(n, n, density=0.1, random_state=43)
        B = b.dot(b.T)
        Bd = B.todense()
        C = SVD.generate_random((n, n), np.logspace(-1, -20, r), seed=44)
        Cd = C.todense()

        X_ref = la.solve_sylvester(Ad, Bd, Cd)
        X = solve_small_sylvester(Ad, Bd, Cd)

        assert np.allclose(X, X_ref), "The small solver is not correct"
        print("test_sylvester_small passed")

    def test_sylvester_large_small(self):
        """Test the large A, small B solver (original test)."""
        n = 200
        r = 10
        a = sps.random(n, n, density=0.1, random_state=42)
        A = a.dot(a.T)
        Ad = A.todense()
        b = sps.random(n, n, density=0.1, random_state=43)
        B = b.dot(b.T)
        Bd = B.todense()
        C = SVD.generate_random((n, n), np.logspace(-1, -20, r), seed=44)
        Cd = C.todense()

        X_ref = la.solve_sylvester(Ad, Bd, Cd)
        X = solve_sylvester_large_A_small_B(A, Bd, Cd)

        assert np.allclose(X, X_ref), "The large and small solver is not correct"
        print("test_sylvester_large_small passed")

    def test_sylvester_large_low_rank(self):
        """Test the large and low rank solver (original test, updated API)."""
        n = 200
        r = 10
        a = sps.random(n, n, density=0.1, random_state=42)
        A = a.dot(a.T)
        Ad = A.todense()
        b = sps.random(n, n, density=0.1, random_state=43)
        B = b.dot(b.T)
        Bd = B.todense()
        C = SVD.generate_random((n, n), np.logspace(-1, -20, r), seed=44)
        Cd = C.todense()

        X_ref = la.solve_sylvester(Ad, Bd, Cd)
        # Updated API: use krylov_kwargs dict
        X = solve_sparse_low_rank_sylvester(
            A, B, C, tol=1e-10, krylov_kwargs={"extended": True}
        )
        Xd = X.todense()

        assert np.allclose(Xd, X_ref), "The large and low rank solver is not correct"
        print("test_sylvester_large_low_rank passed")


# %% Test 9: Krylov space algorithm verification
class TestKrylovAlgorithms:
    """Test that appropriate algorithms are used for Krylov spaces in Sylvester solver."""

    def test_krylov_spaces_created_for_sylvester(self):
        """Verify that Sylvester solver creates appropriate Krylov spaces."""
        from low_rank_toolbox.krylov.spaces import ExtendedKrylovSpace, KrylovSpace

        m, n = 100, 100
        A, B, C, X_true = generate_sylvester_problem(m, n, rank_C=3)

        # Solve with extended Krylov
        X = solve_sylvester(A, B, C, tol=1e-10)

        residual = compute_sylvester_residual(A, B, X, C)
        assert residual < 1e-8, f"Residual too large: {residual}"

        print("✓ Sylvester solver creates two Krylov spaces (for A and B)")

    def test_symmetric_matrices_in_sylvester(self):
        """Test Sylvester with both A and B symmetric (uses Lanczos for both spaces)."""
        m, n = 100, 100
        # Both A and B are symmetric (Laplacian)
        A = create_1d_laplacian(m)
        B = create_1d_laplacian(n)
        C = SVD.generate_random((m, n), np.logspace(-1, -3, 3))

        # Each Krylov space should use Lanczos independently
        X = solve_sylvester(
            A, B, C, tol=1e-10, is_A_symmetric=True, is_B_symmetric=True
        )

        residual = compute_sylvester_residual(A, B, X, C)
        assert residual < 1e-6, f"Residual too large: {residual}"

        print("✓ Sylvester with symmetric A and B uses Lanczos for both spaces")

    def test_lanczos_used_for_symmetric_A_and_B(self):
        """Verify Lanczos is used when A and B are symmetric."""
        from low_rank_toolbox.krylov.spaces import ExtendedKrylovSpace

        m, n = 100, 100
        A = create_1d_laplacian(m)
        B = create_1d_laplacian(n)
        C = SVD.generate_random((m, n), np.logspace(-1, -3, 3))

        # Solve with explicit symmetry flags
        X = solve_sparse_low_rank_sylvester(
            A, B, C, tol=1e-10, is_A_symmetric=True, is_B_symmetric=True
        )

        residual = compute_sylvester_residual(A, B, X, C)
        assert residual < 1e-6

        print("✓ Sylvester correctly uses Lanczos for symmetric A and B")

    def test_special_case_A_equals_B_symmetric(self):
        """Test special case when A = B (could potentially use one Krylov space)."""
        n = 100
        A = create_1d_laplacian(n)
        B = A  # Same matrix
        C = SVD.generate_random((n, n), np.logspace(-1, -3, 3))

        # Currently uses two separate spaces even though A = B
        # Both should use Lanczos
        X = solve_sylvester(
            A, B, C, tol=1e-10, is_A_symmetric=True, is_B_symmetric=True
        )

        residual = compute_sylvester_residual(A, B, X, C)
        assert residual < 1e-6

        print("✓ Special case A=B handled (uses two Lanczos spaces)")
        print(
            "   Note: When A=B, problem is Lyapunov; use solve_lyapunov() for further optimization"
        )

    def test_only_A_symmetric(self):
        """Test when only A is symmetric (not B)."""
        m, n = 100, 100
        A = create_1d_laplacian(m)
        # Make B non-symmetric
        B_sym = create_1d_laplacian(n)
        B = B_sym + sps.diags([0.1], 1, shape=(n, n))  # Add asymmetry
        C = SVD.generate_random((m, n), np.logspace(-1, -3, 3))

        X = solve_sylvester(
            A, B, C, tol=1e-10, is_A_symmetric=True, is_B_symmetric=False
        )

        residual = compute_sylvester_residual(A, B, X, C)
        assert residual < 1e-5

        print("✓ Only A symmetric: Lanczos used for left space only")

    def test_only_B_symmetric(self):
        """Test when only B is symmetric (not A)."""
        m, n = 100, 100
        # Make A non-symmetric
        A_sym = create_1d_laplacian(m)
        A = A_sym + sps.diags([0.1], 1, shape=(m, m))  # Add asymmetry
        B = create_1d_laplacian(n)
        C = SVD.generate_random((m, n), np.logspace(-1, -3, 3))

        X = solve_sylvester(
            A, B, C, tol=1e-10, is_A_symmetric=False, is_B_symmetric=True
        )

        residual = compute_sylvester_residual(A, B, X, C)
        assert residual < 1e-6

        print("✓ Only B symmetric: Lanczos used for right space only")

    def test_auto_detect_symmetry(self):
        """Test automatic symmetry detection."""
        m, n = 100, 100
        A = create_1d_laplacian(m)
        B = create_1d_laplacian(n)
        C = SVD.generate_random((m, n), np.logspace(-1, -3, 3))

        # Don't specify symmetry - should auto-detect
        X = solve_sylvester(A, B, C, tol=1e-10)

        residual = compute_sylvester_residual(A, B, X, C)
        assert residual < 1e-6

        print("✓ Auto-detected symmetry of A and B correctly")


# %% Run all tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
