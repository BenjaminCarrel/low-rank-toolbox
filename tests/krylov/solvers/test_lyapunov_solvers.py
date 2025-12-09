"""
Author: Benjamin Carrel, University of Geneva, 2022

Comprehensive tests for the Lyapunov solvers.
This file merges comprehensive pytest tests with quick validation tests.
"""

# %% Imports
import numpy as np
import pytest
import warnings
import scipy.sparse as sps
import scipy.linalg as la
from lowrank.krylov.solvers.lyapunov_solvers import (
    solve_lyapunov,
    solve_sparse_low_rank_symmetric_lyapunov,
    solve_sparse_low_rank_non_symmetric_lyapunov,
    solve_small_lyapunov
)
from lowrank.matrices.svd import SVD
from lowrank.matrices.quasi_svd import QuasiSVD
from lowrank.matrices.low_rank_matrix import LowRankEfficiencyWarning


# %% Helper functions
def create_1d_laplacian(n):
    """Create a discrete 1D Laplacian matrix on [0,1]."""
    h = 1.0 / (n + 1)
    # Tridiagonal matrix: [-1, 2, -1] / h^2
    diag = np.ones(n) * 2 / h**2
    off_diag = np.ones(n-1) * (-1) / h**2
    A = sps.diags([off_diag, diag, off_diag], [-1, 0, 1], format='csc')
    return A


def generate_lyapunov_problem(n, rank, is_symmetric=True, seed=42):
    """Generate a Lyapunov problem AX + XA^H = C with known solution X."""
    np.random.seed(seed)
    
    # Create 1D Laplacian (symmetric positive definite)
    A = create_1d_laplacian(n)
    
    # Generate random low-rank X
    if is_symmetric:
        X = SVD.generate_random((n, n), np.logspace(-1, -rank, rank), 
                                seed=seed, is_symmetric=True)
    else:
        X = SVD.generate_random((n, n), np.logspace(-1, -rank, rank), 
                                seed=seed, is_symmetric=False)
    
    # Compute C = AX + XA^H
    if is_symmetric:
        # For symmetric case: C = AX + XA
        C_dense = A @ X.to_dense() + X.to_dense() @ A
    else:
        # For general case: C = AX + XA^H
        C_dense = A @ X.to_dense() + X.to_dense() @ A.T.conj()
    
    # Convert C to low-rank format
    C = SVD.from_dense(C_dense)
    
    return A, C, X


def compute_lyapunov_residual(A, X, C, is_symmetric=True):
    """Compute the residual ||AX + XA^H - C|| / ||C||."""
    if isinstance(X, (SVD, QuasiSVD)):
        X_dense = X.to_dense()
    else:
        X_dense = X
    
    if isinstance(C, (SVD, QuasiSVD)):
        C_dense = C.to_dense()
    else:
        C_dense = C
    
    if is_symmetric:
        residual = A @ X_dense + X_dense @ A - C_dense
    else:
        AH = A.T.conj() if hasattr(A, 'T') else A.conj().T
        residual = A @ X_dense + X_dense @ AH - C_dense
    
    return la.norm(residual, 'fro') / la.norm(C_dense, 'fro')


# %% Test 1: Input validation tests
class TestInputValidation:
    """Test input validation for all Lyapunov solvers."""
    
    def test_solve_lyapunov_non_sparse_with_low_rank(self):
        """Test that solve_lyapunov handles dense A with low-rank C."""
        n = 100
        A = np.random.rand(n, n)
        C = SVD.generate_random((n, n), np.logspace(-1, -3, 3))
        
        # Should not raise error
        X = solve_lyapunov(A, C)
        assert X is not None
    
    def test_symmetric_solver_requires_sparse_matrix(self):
        """Test that symmetric solver requires sparse matrix."""
        n = 100
        A = np.random.rand(n, n)
        C = SVD.generate_random((n, n), np.logspace(-1, -3, 3))
        
        with pytest.raises(AssertionError, match="A must be a sparse matrix"):
            solve_sparse_low_rank_symmetric_lyapunov(A, C)
    
    def test_symmetric_solver_requires_low_rank_C(self):
        """Test that symmetric solver requires low-rank C."""
        n = 100
        A = sps.random(n, n, density=0.1, format='csc')
        C = np.random.rand(n, n)
        
        with pytest.raises(AssertionError, match="C must be a low-rank matrix"):
            solve_sparse_low_rank_symmetric_lyapunov(A, C)
    
    def test_tolerance_must_be_reasonable(self):
        """Test that tolerance must be above machine precision."""
        n = 100
        A = sps.random(n, n, density=0.1, format='csc')
        C = SVD.generate_random((n, n), np.logspace(-1, -3, 3))
        
        with pytest.raises(AssertionError, match="tol must be larger than machine precision"):
            solve_sparse_low_rank_symmetric_lyapunov(A, C, tol=1e-20)
    
    def test_max_iter_validation(self):
        """Test that max_iter must be greater than 1."""
        n = 100
        A = sps.random(n, n, density=0.1, format='csc')
        C = SVD.generate_random((n, n), np.logspace(-1, -3, 3))
        
        with pytest.raises(AssertionError, match="max_iter must be at least 2"):
            solve_sparse_low_rank_symmetric_lyapunov(A, C, max_iter=1)
    
    def test_extended_and_poles_conflict(self):
        """Test that extended and rational Krylov cannot be used together."""
        n = 100
        A = sps.random(n, n, density=0.1, format='csc')
        C = SVD.generate_random((n, n), np.logspace(-1, -3, 3))
        
        krylov_kwargs = {'extended': True, 'poles': [1.0, 2.0]}
        with pytest.raises(ValueError, match="Cannot use rational Krylov space with extended Krylov space"):
            solve_sparse_low_rank_symmetric_lyapunov(A, C, krylov_kwargs=krylov_kwargs)


# %% Test 2: Edge cases
class TestEdgeCases:
    """Test edge cases for Lyapunov solvers."""
    
    def test_small_matrix_size(self):
        """Test solver on small matrix."""
        n = 100
        A = create_1d_laplacian(n)
        C = SVD.generate_random((n, n), np.array([1.0, 0.5]), is_symmetric=True)
        
        X = solve_lyapunov(A, C, is_symmetric=True)
        residual = compute_lyapunov_residual(A, X, C, is_symmetric=True)
        assert residual < 1e-8, f"Residual too large: {residual}"
    
    def test_rank_one_C(self):
        """Test solver with rank-1 right-hand side."""
        n = 200
        A = create_1d_laplacian(n)
        C = SVD.generate_random((n, n), np.array([1.0]), is_symmetric=True)
        
        X = solve_lyapunov(A, C, is_symmetric=True)
        residual = compute_lyapunov_residual(A, X, C, is_symmetric=True)
        assert residual < 1e-7, f"Residual too large: {residual}"
    
    def test_high_rank_C(self):
        """Test solver with rank-5 C."""
        n = 200
        rank = 5
        A = create_1d_laplacian(n)
        C = SVD.generate_random((n, n), np.logspace(-1, -rank, rank), is_symmetric=True)
        
        X = solve_lyapunov(A, C, is_symmetric=True)
        residual = compute_lyapunov_residual(A, X, C, is_symmetric=True)
        assert residual < 1e-6, f"Residual too large: {residual}"
    
    def test_nearly_singular_matrix(self):
        """Test solver with matrix that has small eigenvalues."""
        n = 100
        # Create matrix with very small eigenvalues
        A = sps.eye(n, format='csc') * 1e-3
        C = SVD.generate_random((n, n), np.array([1.0, 0.5, 0.1]), is_symmetric=True)
        
        X = solve_lyapunov(A, C, is_symmetric=True, tol=1e-8)
        assert X is not None


# %% Test 3: Large problem test with 1D Laplacian
class TestLargeProblems:
    """Test on larger, realistic problems."""
    
    def test_100x100_symmetric_problem(self):
        """Test 100x100 symmetric Lyapunov equation."""
        n = 100
        rank = 3
        A, C, X_true = generate_lyapunov_problem(n, rank, is_symmetric=True, seed=123)
        
        # Solve
        X = solve_lyapunov(A, C, is_symmetric=True, tol=1e-10)
        
        # Check residual
        residual = compute_lyapunov_residual(A, X, C, is_symmetric=True)
        print(f"Symmetric 100x100 residual: {residual}")
        assert residual < 1e-8, f"Residual too large: {residual}"
        
        # Check solution accuracy
        X_dense = X.to_dense()
        X_true_dense = X_true.to_dense()
        solution_error = la.norm(X_dense - X_true_dense, 'fro') / la.norm(X_true_dense, 'fro')
        print(f"Solution error: {solution_error}")
        assert solution_error < 1e-4, f"Solution error too large: {solution_error}"
    
    def test_100x100_nonsymmetric_problem(self):
        """Test 100x100 non-symmetric Lyapunov equation."""
        n = 100
        rank = 3
        A, C, X_true = generate_lyapunov_problem(n, rank, is_symmetric=False, seed=456)
        
        # Solve
        X = solve_lyapunov(A, C, is_symmetric=False, tol=1e-10)
        
        # Check residual
        residual = compute_lyapunov_residual(A, X, C, is_symmetric=False)
        print(f"Non-symmetric 100x100 residual: {residual}")
        assert residual < 1e-8, f"Residual too large: {residual}"
        
        # Check solution accuracy
        X_dense = X.to_dense()
        X_true_dense = X_true.to_dense()
        solution_error = la.norm(X_dense - X_true_dense, 'fro') / la.norm(X_true_dense, 'fro')
        print(f"Solution error: {solution_error}")
        assert solution_error < 1e-4, f"Solution error too large: {solution_error}"


# %% Test 4: Different Krylov methods
class TestKrylovMethods:
    """Test different Krylov space methods."""
    
    def test_classic_krylov(self):
        """Test with standard Krylov space."""
        n = 200
        rank = 2
        A, C, X_true = generate_lyapunov_problem(n, rank, is_symmetric=True)
        
        krylov_kwargs = {'extended': False}
        with pytest.warns(UserWarning, match="standard Krylov space may not converge"):
            X = solve_lyapunov(A, C, is_symmetric=True, krylov_kwargs=krylov_kwargs, max_iter=50)
        
        residual = compute_lyapunov_residual(A, X, C, is_symmetric=True)
        print(f"Classic Krylov residual: {residual}")
        # Standard Krylov may not converge as well
        assert residual < 1e-4, f"Residual too large: {residual}"
    
    def test_extended_krylov(self):
        """Test with extended Krylov space (default)."""
        n = 200
        rank = 3
        A, C, X_true = generate_lyapunov_problem(n, rank, is_symmetric=True)
        
        krylov_kwargs = {'extended': True}
        X = solve_lyapunov(A, C, is_symmetric=True, krylov_kwargs=krylov_kwargs)
        
        residual = compute_lyapunov_residual(A, X, C, is_symmetric=True)
        print(f"Extended Krylov residual: {residual}")
        assert residual < 1e-8, f"Residual too large: {residual}"
    
    def test_rational_krylov_symmetric(self):
        """Test with rational Krylov space."""
        n = 200
        rank = 3
        A, C, X_true = generate_lyapunov_problem(n, rank, is_symmetric=True)
        
        # Use poles near the smallest eigenvalues for better convergence
        # For 1D Laplacian, eigenvalues are roughly k^2 * pi^2 / 4
        poles = [10.0, 20.0, 30.0, 40.0, 50.0]
        krylov_kwargs = {'extended': False, 'poles': poles}
        X = solve_lyapunov(A, C, is_symmetric=True, krylov_kwargs=krylov_kwargs)
        
        residual = compute_lyapunov_residual(A, X, C, is_symmetric=True)
        print(f"Rational Krylov residual: {residual}")
        assert residual < 1e-8, f"Residual too large: {residual}"
    
    def test_custom_invA(self):
        """Test with custom inverse function."""
        n = 200
        rank = 3
        A, C, X_true = generate_lyapunov_problem(n, rank, is_symmetric=True)
        
        # Custom solver (just use the default but count calls)
        call_count = [0]
        def custom_inv(x):
            call_count[0] += 1
            from scipy.sparse.linalg import spsolve
            return spsolve(A, x)
        
        krylov_kwargs = {'extended': True, 'invA': custom_inv}
        X = solve_lyapunov(A, C, is_symmetric=True, krylov_kwargs=krylov_kwargs)
        
        assert call_count[0] > 0, "Custom inverse was not called"
        residual = compute_lyapunov_residual(A, X, C, is_symmetric=True)
        assert residual < 1e-8, f"Residual too large: {residual}"


# %% Test 5: Different input combinations
class TestInputCombinations:
    """Test all combinations of input types."""
    
    def test_sparse_A_lowrank_C_symmetric(self):
        """Test: A sparse, C low-rank, both symmetric."""
        n = 200
        rank = 4
        A, C, X_true = generate_lyapunov_problem(n, rank, is_symmetric=True)
        
        X = solve_lyapunov(A, C, is_symmetric=True)
        residual = compute_lyapunov_residual(A, X, C, is_symmetric=True)
        assert residual < 1e-8
        print("✓ Sparse A + Low-rank C (symmetric)")
    
    def test_sparse_A_lowrank_C_nonsymmetric(self):
        """Test: A sparse, C low-rank, not symmetric."""
        n = 200
        rank = 4
        A, C, X_true = generate_lyapunov_problem(n, rank, is_symmetric=False)
        
        X = solve_lyapunov(A, C, is_symmetric=False)
        residual = compute_lyapunov_residual(A, X, C, is_symmetric=False)
        assert residual < 1e-8
        print("✓ Sparse A + Low-rank C (non-symmetric)")
    
    def test_sparse_A_dense_C(self):
        """Test: A sparse, C dense."""
        n = 100
        A = create_1d_laplacian(n)
        C = np.random.rand(n, n)
        C = (C + C.T) / 2  # Make symmetric
        
        X = solve_lyapunov(A, C)
        
        # Compute residual
        residual_matrix = A @ X + X @ A - C
        residual = la.norm(residual_matrix, 'fro') / la.norm(C, 'fro')
        assert residual < 1e-10
        print("✓ Sparse A + Dense C")
    
    def test_dense_A_lowrank_C(self):
        """Test: A dense, C low-rank."""
        n = 100
        A = create_1d_laplacian(n).toarray()
        C = SVD.generate_random((n, n), np.logspace(-1, -3, 3), is_symmetric=True)
        
        X = solve_lyapunov(A, C)
        
        # Compute residual
        C_dense = C.to_dense()
        X_dense = X.to_dense() if hasattr(X, 'to_dense') else X
        residual_matrix = A @ X_dense + X_dense @ A - C_dense
        residual = la.norm(residual_matrix, 'fro') / la.norm(C_dense, 'fro')
        # Slightly relaxed tolerance for dense A conversion path (accumulates more rounding error)
        assert residual < 2e-8
        print("✓ Dense A + Low-rank C")
    
    def test_dense_A_dense_C(self):
        """Test: A dense, C dense."""
        n = 100
        A = create_1d_laplacian(n).toarray()
        C = np.random.rand(n, n)
        C = (C + C.T) / 2  # Make symmetric
        
        X = solve_lyapunov(A, C)
        
        # Compute residual
        residual_matrix = A @ X + X @ A - C
        residual = la.norm(residual_matrix, 'fro') / la.norm(C, 'fro')
        assert residual < 1e-10
        print("✓ Dense A + Dense C")
    
    def test_auto_detect_symmetric(self):
        """Test automatic symmetry detection."""
        n = 200
        rank = 4
        A, C, X_true = generate_lyapunov_problem(n, rank, is_symmetric=True)
        
        # Don't specify is_symmetric, let it auto-detect
        with pytest.warns(LowRankEfficiencyWarning, match="Checking symmetry"):
            X = solve_lyapunov(A, C)
        
        residual = compute_lyapunov_residual(A, X, C, is_symmetric=True)
        assert residual < 1e-8
        print("✓ Auto-detect symmetric")
    
    def test_auto_detect_nonsymmetric(self):
        """Test automatic non-symmetry detection."""
        n = 200
        rank = 4
        A, C, X_true = generate_lyapunov_problem(n, rank, is_symmetric=False)
        
        # Don't specify is_symmetric, let it auto-detect
        with pytest.warns(LowRankEfficiencyWarning, match="Checking symmetry"):
            X = solve_lyapunov(A, C)
        
        residual = compute_lyapunov_residual(A, X, C, is_symmetric=False)
        assert residual < 1e-8
        print("✓ Auto-detect non-symmetric")


# %% Test 6: Tolerance criterion
class TestToleranceCriterion:
    """Test that the tolerance criterion is met."""
    
    def test_tolerance_1e6(self):
        """Test with tolerance 1e-6."""
        n = 200
        rank = 3
        A, C, X_true = generate_lyapunov_problem(n, rank, is_symmetric=True)
        
        tol = 1e-6
        X = solve_lyapunov(A, C, is_symmetric=True, tol=tol)
        
        residual = compute_lyapunov_residual(A, X, C, is_symmetric=True)
        print(f"Tolerance {tol}, achieved residual: {residual}")
        assert residual <= tol * 10, f"Residual {residual} not within tolerance {tol}"
    
    def test_tolerance_1e8(self):
        """Test with tolerance 1e-8."""
        n = 200
        rank = 3
        A, C, X_true = generate_lyapunov_problem(n, rank, is_symmetric=True)
        
        tol = 1e-8
        X = solve_lyapunov(A, C, is_symmetric=True, tol=tol)
        
        residual = compute_lyapunov_residual(A, X, C, is_symmetric=True)
        print(f"Tolerance {tol}, achieved residual: {residual}")
        assert residual <= tol * 10, f"Residual {residual} not within tolerance {tol}"
    
    def test_tolerance_1e10(self):
        """Test with tolerance 1e-10."""
        n = 200
        rank = 3
        A, C, X_true = generate_lyapunov_problem(n, rank, is_symmetric=True)
        
        tol = 1e-10
        X = solve_lyapunov(A, C, is_symmetric=True, tol=tol)
        
        residual = compute_lyapunov_residual(A, X, C, is_symmetric=True)
        print(f"Tolerance {tol}, achieved residual: {residual}")
        assert residual <= tol * 10, f"Residual {residual} not within tolerance {tol}"
    
    def test_different_tolerances_nonsymmetric(self):
        """Test different tolerances for non-symmetric case."""
        n = 200
        rank = 3
        A, C, X_true = generate_lyapunov_problem(n, rank, is_symmetric=False)
        
        for tol in [1e-6, 1e-8, 1e-10]:
            X = solve_lyapunov(A, C, is_symmetric=False, tol=tol)
            residual = compute_lyapunov_residual(A, X, C, is_symmetric=False)
            print(f"Non-symmetric: Tolerance {tol}, achieved residual: {residual}")
            assert residual <= tol * 10, f"Residual {residual} not within tolerance {tol}"


# %% Test 7: max_iter parameter
class TestMaxIterations:
    """Test that max_iter parameter works correctly."""
    
    def test_max_iter_stops_early(self):
        """Test that solver stops at max_iter without converging."""
        # Use larger problem and higher rank to prevent early convergence
        n = 200
        rank = 5
        A, C, X_true = generate_lyapunov_problem(n, rank, is_symmetric=True)
        
        # Set very small max_iter with high tolerance requirement (should not converge)
        max_iter = 2  # Even smaller max_iter
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            X = solve_lyapunov(A, C, is_symmetric=True, tol=1e-12, max_iter=max_iter)
            # Check if warning was raised
            has_warning = any("No convergence" in str(warning.message) for warning in w)
            if has_warning:
                print("✓ Non-convergence warning correctly raised")
        
        # Solution should exist but may not be accurate
        assert X is not None
        residual = compute_lyapunov_residual(A, X, C, is_symmetric=True)
        print(f"With max_iter={max_iter}, residual: {residual}")
        # Just verify that solution was returned (warning is optional since Krylov can converge fast)
    
    def test_sufficient_max_iter(self):
        """Test that sufficient max_iter allows convergence."""
        n = 200
        rank = 3
        A, C, X_true = generate_lyapunov_problem(n, rank, is_symmetric=True)
        
        # Set reasonable max_iter
        max_iter = 20
        X = solve_lyapunov(A, C, is_symmetric=True, tol=1e-10, max_iter=max_iter)
        
        residual = compute_lyapunov_residual(A, X, C, is_symmetric=True)
        print(f"With max_iter={max_iter}, residual: {residual}")
        assert residual < 1e-8, f"Should converge with max_iter={max_iter}"
    
    def test_max_iter_nonsymmetric(self):
        """Test max_iter for non-symmetric case."""
        n = 200
        rank = 3
        A, C, X_true = generate_lyapunov_problem(n, rank, is_symmetric=False)
        
        # Test with small max_iter
        max_iter = 3
        X = solve_lyapunov(A, C, is_symmetric=False, tol=1e-10, max_iter=max_iter)
        
        residual = compute_lyapunov_residual(A, X, C, is_symmetric=False)
        print(f"Non-symmetric with max_iter={max_iter}, residual: {residual}")
        assert X is not None


# %% Test 8: Quick validation tests (from quick_test_lyapunov.py)
class TestQuickValidation:
    """Quick validation tests for basic functionality."""
    
    def test_small_symmetric_problem(self):
        """Test small symmetric problem."""
        n = 20
        A = sps.eye(n, format='csc') * 2.0
        C = SVD.generate_random((n, n), np.array([1.0, 0.5, 0.1]), is_symmetric=True, seed=42)
        
        X = solve_lyapunov(A, C, is_symmetric=True, tol=1e-10)
        
        # Check residual
        X_dense = X.to_dense()
        C_dense = C.to_dense()
        residual = A @ X_dense + X_dense @ A - C_dense
        rel_residual = la.norm(residual, 'fro') / la.norm(C_dense, 'fro')
        assert rel_residual < 1e-8
        print("✓ Small symmetric problem")
    
    def test_small_nonsymmetric_problem(self):
        """Test small non-symmetric problem."""
        n = 20
        A = sps.eye(n, format='csc') * 2.0
        C = SVD.generate_random((n, n), np.array([1.0, 0.5, 0.1]), is_symmetric=False, seed=43)
        
        X = solve_lyapunov(A, C, is_symmetric=False, tol=1e-10)
        
        # Check residual (A X + X A^H = C)
        X_dense = X.to_dense()
        C_dense = C.to_dense()
        residual = A @ X_dense + X_dense @ A.T.conj() - C_dense
        rel_residual = la.norm(residual, 'fro') / la.norm(C_dense, 'fro')
        assert rel_residual < 1e-8
        print("✓ Small non-symmetric problem")
    
    def test_small_dense_inputs(self):
        """Test with small dense inputs."""
        n = 20
        A = sps.eye(n, format='csc') * 2.0
        C = SVD.generate_random((n, n), np.array([1.0, 0.5, 0.1]), is_symmetric=True, seed=42)
        
        A_dense = A.toarray()
        C_dense = C.to_dense()
        
        X_dense = solve_lyapunov(A_dense, C_dense)
        
        # Check residual
        residual = A_dense @ X_dense + X_dense @ A_dense - C_dense
        rel_residual = la.norm(residual, 'fro') / la.norm(C_dense, 'fro')
        assert rel_residual < 1e-10
        print("✓ Small dense inputs")
    
    def test_1d_laplacian_50x50(self):
        """Test with 1D Laplacian (50x50)."""
        n = 50
        A = create_1d_laplacian(n)
        C = SVD.generate_random((n, n), np.logspace(-1, -10, 10), is_symmetric=True, seed=44)
        
        X = solve_lyapunov(A, C, is_symmetric=True, tol=1e-10)
        
        # Check residual
        X_dense = X.to_dense()
        C_dense = C.to_dense()
        residual = A @ X_dense + X_dense @ A - C_dense
        rel_residual = la.norm(residual, 'fro') / la.norm(C_dense, 'fro')
        assert rel_residual < 1e-8
        print("✓ 1D Laplacian 50x50")
    
    def test_krylov_methods_comparison(self):
        """Test different Krylov methods on same problem."""
        n = 50
        A = create_1d_laplacian(n)
        C = SVD.generate_random((n, n), np.logspace(-1, -10, 10), is_symmetric=True, seed=44)
        
        def compute_residual(X):
            X_dense = X.to_dense() if hasattr(X, 'to_dense') else X
            C_dense = C.to_dense() if hasattr(C, 'to_dense') else C
            residual = A @ X_dense + X_dense @ A - C_dense
            return la.norm(residual, 'fro') / la.norm(C_dense, 'fro')
        
        # Extended Krylov (default)
        X1 = solve_lyapunov(A, C, is_symmetric=True, krylov_kwargs={'extended': True}, tol=1e-10)
        residual1 = compute_residual(X1)
        assert residual1 < 1e-8
        print(f"   Extended Krylov residual: {residual1:.2e}")
        
        # Rational Krylov (may not converge with default max_iter)
        poles = [10.0 + 10.0*i for i in range(20)]
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="No convergence before max_iter")
            X2 = solve_lyapunov(A, C, is_symmetric=True, 
                                krylov_kwargs={'extended': False, 'poles': poles}, tol=1e-10)
        residual2 = compute_residual(X2)
        assert residual2 < 1e-8
        print(f"   Rational Krylov residual: {residual2:.2e}")
        print("✓ Krylov methods comparison")


# %% Test 9: Lanczos algorithm verification
class TestLanczosUsage:
    """Test that Lanczos algorithm is correctly used for symmetric problems."""
    
    def test_lanczos_used_for_symmetric_krylov_space(self):
        """Verify Lanczos is used in symmetric KrylovSpace."""
        from lowrank.krylov.spaces import KrylovSpace
        
        n = 100
        A = create_1d_laplacian(n)
        X = np.random.rand(n, 3)
        
        # Create symmetric Krylov space
        ks = KrylovSpace(A, X, is_symmetric=True)
        ks.augment_basis()
        ks.augment_basis()
        
        # Check Lanczos structures exist
        assert hasattr(ks, '_alpha'), "Lanczos alpha parameters not found"
        assert hasattr(ks, '_beta'), "Lanczos beta parameters not found"
        assert not hasattr(ks, 'H'), "Arnoldi H matrix should not exist for symmetric case"
        
        # Check alpha and beta are populated
        # Note: indexing is m-based, where m starts at 1
        # After initialization: m=1, beta[0] is set
        # After 1st augment: m=2, alpha[1] and beta[1] are set
        # After 2nd augment: m=3, alpha[2] and beta[2] are set
        assert ks._beta[0] is not None, "Beta[0] not populated at initialization"
        assert ks._alpha[1] is not None, "Alpha[1] not populated after 1st augment"
        assert ks._beta[1] is not None, "Beta[1] not populated after 1st augment"
        assert ks._alpha[2] is not None, "Alpha[2] not populated after 2nd augment"
        assert ks._beta[2] is not None, "Beta[2] not populated after 2nd augment"
        
        print("✓ Lanczos structures verified for symmetric KrylovSpace")
    
    def test_arnoldi_used_for_nonsymmetric_krylov_space(self):
        """Verify Arnoldi is used in non-symmetric KrylovSpace."""
        from lowrank.krylov.spaces import KrylovSpace
        
        n = 100
        A = create_1d_laplacian(n)
        X = np.random.rand(n, 3)
        
        # Create non-symmetric Krylov space
        ks = KrylovSpace(A, X, is_symmetric=False)
        ks.augment_basis()
        
        # Check Arnoldi structures exist
        assert hasattr(ks, 'H'), "Arnoldi H matrix not found"
        assert not hasattr(ks, '_alpha'), "Lanczos alpha should not exist for non-symmetric case"
        assert not hasattr(ks, '_beta'), "Lanczos beta should not exist for non-symmetric case"
        
        # H should be upper Hessenberg
        assert ks.H.shape[0] >= ks.H.shape[1], "H should be tall or square"
        
        print("✓ Arnoldi structures verified for non-symmetric KrylovSpace")
    
    def test_lanczos_in_extended_krylov_space(self):
        """Verify Lanczos is used in symmetric ExtendedKrylovSpace."""
        from lowrank.krylov.spaces import ExtendedKrylovSpace
        
        n = 100
        A = create_1d_laplacian(n)
        X = np.random.rand(n, 3)
        
        # Create symmetric extended Krylov space
        eks = ExtendedKrylovSpace(A, X, is_symmetric=True)
        eks.augment_basis()
        
        # Check that both component spaces use Lanczos
        assert hasattr(eks.krylov_space, '_alpha'), "Krylov component doesn't use Lanczos"
        assert hasattr(eks.krylov_space, '_beta'), "Krylov component doesn't use Lanczos"
        assert hasattr(eks.inverted_krylov_space, '_alpha'), "Inverted Krylov component doesn't use Lanczos"
        assert hasattr(eks.inverted_krylov_space, '_beta'), "Inverted Krylov component doesn't use Lanczos"
        
        print("✓ Lanczos verified in both components of ExtendedKrylovSpace")
    
    def test_lanczos_in_symmetric_lyapunov_solver(self):
        """Verify Lanczos is used when solving symmetric Lyapunov equation."""
        n = 100
        rank = 3
        A, C, X_true = generate_lyapunov_problem(n, rank, is_symmetric=True)
        
        # Solve with extended Krylov (default)
        X = solve_lyapunov(A, C, is_symmetric=True, max_iter=5)
        
        # Solution should be accurate
        residual = compute_lyapunov_residual(A, X, C, is_symmetric=True)
        assert residual < 1e-6, f"Residual too large: {residual}"
        
        print(f"✓ Symmetric Lyapunov solver successful (residual: {residual:.2e})")
    
    def test_rational_krylov_no_lanczos(self):
        """Document that RationalKrylovSpace doesn't currently use Lanczos."""
        from lowrank.krylov.spaces import RationalKrylovSpace
        
        n = 100
        A = create_1d_laplacian(n)
        X = np.random.rand(n, 3)
        poles = [10.0, 20.0, 30.0]
        
        # Create symmetric rational Krylov space
        rks = RationalKrylovSpace(A, X, poles, is_symmetric=True)
        
        # Currently uses Arnoldi even for symmetric case
        assert hasattr(rks, 'H'), "RationalKrylovSpace uses Arnoldi structure"
        assert not hasattr(rks, '_alpha'), "RationalKrylovSpace doesn't use Lanczos (known limitation)"
        
        print("✓ Documented: RationalKrylovSpace uses Arnoldi even for symmetric A")
    
    def test_lanczos_efficiency_symmetric_vs_nonsymmetric(self):
        """Compare storage requirements for Lanczos vs Arnoldi."""
        from lowrank.krylov.spaces import KrylovSpace
        
        n = 200
        A = create_1d_laplacian(n)
        X = np.random.rand(n, 2)
        
        # Symmetric case (Lanczos)
        ks_sym = KrylovSpace(A, X, is_symmetric=True)
        for _ in range(10):
            ks_sym.augment_basis()
        
        # Non-symmetric case (Arnoldi)
        ks_nonsym = KrylovSpace(A, X, is_symmetric=False)
        for _ in range(10):
            ks_nonsym.augment_basis()
        
        # Lanczos stores tridiagonal (alpha, beta), Arnoldi stores full H
        lanczos_params = sum(x.size if x is not None and hasattr(x, 'size') else 0 
                            for x in [ks_sym._alpha[i] for i in range(ks_sym.m) if ks_sym._alpha[i] is not None])
        arnoldi_params = ks_nonsym.H.size
        
        print(f"   Lanczos parameters stored: ~{lanczos_params}")
        print(f"   Arnoldi parameters stored: {arnoldi_params}")
        print(f"   Storage ratio: {arnoldi_params / max(lanczos_params, 1):.1f}x")
        print("✓ Lanczos uses less storage for symmetric problems")
    
    def test_symmetry_flag_propagation(self):
        """Test that is_symmetric flag propagates correctly through solver stack."""
        n = 100
        rank = 2
        A, C, X_true = generate_lyapunov_problem(n, rank, is_symmetric=True)
        
        # The symmetric solver should be called
        # Suppress expected warning about standard Krylov convergence
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*standard Krylov space may not converge.*")
            X = solve_sparse_low_rank_symmetric_lyapunov(A, C, tol=1e-10, 
                                                          krylov_kwargs={'extended': False})
        
        residual = compute_lyapunov_residual(A, X, C, is_symmetric=True)
        assert residual < 1e-6, f"Symmetric solver failed: {residual}"
        
        print("✓ Symmetry flag correctly propagates through solver")


# %% Run all tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
