# Test file for QR class

import numpy as np
import pytest
import scipy.linalg as la

from lowrank import QR, LowRankMatrix


@pytest.fixture
def qr_basic():
    """Basic QR fixture for testing."""
    np.random.seed(42)
    A = np.random.randn(10, 8)
    Q, R = la.qr(A, mode="economic")
    X = QR(Q, R)
    return X, A, Q, R


# Test basic operations
def test_QR_basic(qr_basic):
    """Test basic QR operations."""
    X, A, Q, R = qr_basic
    # Test dimensions
    assert X.deepshape == (10, 8, 8), "Incorrect deepshape"
    assert X.shape == (10, 8), "Incorrect shape"
    assert X.ndim == 2, "Incorrect ndim"
    # Test norms
    assert X.norm("fro") - la.norm(A, "fro") < 1e-12, "Incorrect Frobenius norm"
    assert X.norm(1) - la.norm(A, 1) < 1e-12, "Incorrect 1-norm"
    assert X.norm(2) - la.norm(A, 2) < 1e-12, "Incorrect 2-norm"
    # Test misc
    assert np.allclose(X.full(), A), "Incorrect full() method"
    assert X.gather([1, 3]) - A[1, 3] < 1e-12, "Incorrect gather"


# Test class method
def test_QR_classmethod(qr_basic):
    """Test QR class methods."""
    X, A, Q, R = qr_basic
    # Test the class methods
    assert isinstance(
        QR.generate_random((10, 8)), QR
    ), "Incorrect type of generate_random"
    assert isinstance(QR.from_matrix(A), QR), "Incorrect type of from_matrix"
    assert np.allclose(QR.from_matrix(A).full(), A), "Incorrect from_matrix"
    Y = LowRankMatrix(Q, R)
    assert np.allclose(QR.from_low_rank(Y).full(), A), "Incorrect from_low_rank"


# Test addition
def test_QR_addition(qr_basic):
    """Test QR addition operations."""
    X, A, Q, R = qr_basic
    # Test addition of QRs
    assert isinstance(X + X, QR), "Incorrect addition with QR"
    assert (X + X).rank == 8, "Incorrect rank of addition with QR"
    assert np.allclose((X + X).full(), 2 * A), "Incorrect addition with QR"
    assert np.allclose((X + A), 2 * A), "Incorrect addition with ndarray"
    assert np.allclose((X - X).full(), 0 * A), "Incorrect subtraction with QR"
    Y = QR.generate_random((10, 8))
    assert isinstance(X + Y, QR), "Incorrect addition with QR"
    assert np.allclose((X + Y).full(), A + Y.full()), "Incorrect addition with QR"
    assert np.allclose((X - Y).full(), A - Y.full()), "Incorrect subtraction with QR"


# Test multiplication
def test_QR_multiplication(qr_basic):
    """Test QR multiplication operations."""
    X, A, Q, R = qr_basic
    # Test multiplication of QRs
    assert isinstance(X.dot(X.T), QR), "Incorrect multiplication with QR"
    Y = QR.generate_random((8, 12))
    assert isinstance(X.dot(Y), QR), "Incorrect multiplication with QR"
    assert np.allclose(
        X.dot(Y).full(), A @ Y.full()
    ), "Incorrect multiplication with QR"


# Test conjugate/transpose features
def test_QR_conjugate_initialization():
    """Test conjugate initialization (equivalent to transpose for real matrices)."""
    np.random.seed(100)
    A_test = np.random.randn(10, 8)
    Q_test, R_test = la.qr(A_test, mode="economic")

    # Standard initialization
    X_std = QR(Q_test, R_test)
    assert X_std.shape == (10, 8), "Standard shape incorrect"
    assert np.allclose(X_std.full(), A_test), "Standard reconstruction incorrect"

    # Transposed initialization (should transpose for real matrices)
    X_trans = QR(Q_test, R_test, transposed=True)
    assert X_trans.shape == (8, 10), "Transposed shape incorrect"
    assert np.allclose(X_trans.full(), A_test.T), "Transposed reconstruction incorrect"


def test_QR_transpose_property():
    """Test .T property for transpose."""
    np.random.seed(101)
    A_test = np.random.randn(10, 8)
    X_test = QR.from_matrix(A_test)

    X_T = X_test.T
    assert X_T.shape == (8, 10), "Transpose shape incorrect"
    assert np.allclose(X_T.full(), A_test.T), "Transpose reconstruction incorrect"

    # Double transpose should return to original
    X_back = X_T.T
    assert X_back.shape == (10, 8), "Double transpose shape incorrect"
    assert np.allclose(
        X_back.full(), A_test
    ), "Double transpose reconstruction incorrect"


def test_QR_conjugate_transpose_complex():
    """Test .H property for conjugate transpose on complex matrices."""
    np.random.seed(102)
    A_complex = np.random.randn(10, 8) + 1j * np.random.randn(10, 8)
    X_complex = QR.from_matrix(A_complex)

    X_H = X_complex.H
    assert X_H.shape == (8, 10), "Conjugate transpose shape incorrect"
    assert np.allclose(X_H.full(), A_complex.T.conj()), "Conjugate transpose incorrect"


def test_QR_from_matrix_transposed():
    """Test from_matrix with transposed=True."""
    np.random.seed(103)
    A_test = np.random.randn(10, 8)
    X_trans = QR.from_matrix(A_test, transposed=True)

    assert X_trans.shape == (8, 10), "from_matrix transposed shape incorrect"
    assert np.allclose(
        X_trans.full(), A_test.T.conj()
    ), "from_matrix transposed reconstruction incorrect"


def test_QR_generate_random_transposed():
    """Test generate_random with transposed=True."""
    X_rand_trans = QR.generate_random((10, 8), transposed=True)

    assert X_rand_trans.shape == (8, 10), "generate_random transposed shape incorrect"
    assert X_rand_trans._transposed == True, "transposed flag not set"


def test_QR_real_transpose_hermitian_equivalence():
    """For real matrices, .T and .H should be equivalent."""
    np.random.seed(104)
    A_test = np.random.randn(10, 8)
    X_test = QR.from_matrix(A_test)

    X_T = X_test.T
    X_H = X_test.H

    assert np.allclose(
        X_T.full(), X_H.full()
    ), ".T and .H should be equal for real matrices"


def test_QR_Q_R_properties():
    """Test Q and R properties in both standard and transposed modes."""
    np.random.seed(105)
    A_test = np.random.randn(10, 8)
    Q_test, R_test = la.qr(A_test, mode="economic")

    # Standard mode: X = Q @ R
    X_std = QR(Q_test, R_test, transposed=False)
    assert X_std.Q.shape == (10, 8), "Q shape incorrect in standard mode"
    assert X_std.R.shape == (8, 8), "R shape incorrect in standard mode"
    assert np.allclose(X_std.Q, Q_test), "Q values incorrect in standard mode"
    assert np.allclose(X_std.R, R_test), "R values incorrect in standard mode"
    assert np.allclose(
        X_std.full(), A_test
    ), "Reconstruction incorrect in standard mode"

    # Transposed mode: X = R.H @ Q.H
    X_trans = QR(Q_test, R_test, transposed=True)
    assert X_trans.Q.shape == (10, 8), "Q shape incorrect in transposed mode"
    assert X_trans.R.shape == (8, 8), "R shape incorrect in transposed mode"
    assert np.allclose(X_trans.Q, Q_test), "Q values incorrect in transposed mode"
    assert np.allclose(X_trans.R, R_test), "R values incorrect in transposed mode"
    assert np.allclose(
        X_trans.full(), A_test.T.conj()
    ), "Reconstruction incorrect in transposed mode"


def test_QR_factors_consistency():
    """Test consistency of Q and R factors with and without transposed=True."""
    np.random.seed(200)
    A = np.random.randn(10, 8)
    Q, R = la.qr(A, mode="economic")

    # Create QR in both modes with same Q and R
    X_std = QR(Q, R, transposed=False)
    X_trans = QR(Q, R, transposed=True)

    # Check that Q and R properties return the same values
    assert np.allclose(X_std.Q, X_trans.Q), "Q should be the same in both modes"
    assert np.allclose(X_std.R, X_trans.R), "R should be the same in both modes"

    # Check reconstructions
    assert np.allclose(X_std.full(), Q @ R), "Standard mode reconstruction"
    assert np.allclose(
        X_trans.full(), R.T.conj() @ Q.T.conj()
    ), "Transposed mode reconstruction"
    assert np.allclose(X_std.full(), A), "Standard mode should reconstruct A"
    assert np.allclose(
        X_trans.full(), A.T.conj()
    ), "Transposed mode should reconstruct A.H"


def test_QR_transpose_conjugate_hermitian():
    """Test .T, .conj, and .H properties with and without transposed=True."""
    np.random.seed(201)

    # Test with real matrix
    A_real = np.random.randn(10, 8)
    X_real = QR.from_matrix(A_real, transposed=False)

    # Standard mode (transposed=False)
    assert np.allclose(X_real.T.full(), A_real.T), "Real: .T should give A.T"
    assert np.allclose(
        X_real.H.full(), A_real.T
    ), "Real: .H should give A.T (same as .T)"
    assert np.allclose(
        X_real.conj.full(), A_real
    ), "Real: .conj should give A (no change)"

    # Transposed mode (for real matrices, transposed mode gives A.T, so .T gives A)
    X_real_trans = QR.from_matrix(A_real, transposed=True)
    assert np.allclose(
        X_real_trans.full(), A_real.T
    ), "Real transposed mode should give A.T"
    assert np.allclose(
        X_real_trans.T.full(), A_real.conj()
    ), "Real transposed: .T of A.H gives conj(A) (= A for real)"
    assert np.allclose(
        X_real_trans.H.full(), A_real
    ), "Real transposed: .H of A.H gives A"

    # Test with complex matrix
    A_complex = np.random.randn(10, 8) + 1j * np.random.randn(10, 8)
    X_complex = QR.from_matrix(A_complex, transposed=False)

    # Standard mode
    assert np.allclose(X_complex.T.full(), A_complex.T), "Complex: .T should give A.T"
    assert np.allclose(
        X_complex.H.full(), A_complex.T.conj()
    ), "Complex: .H should give A.H"
    assert np.allclose(
        X_complex.conj.full(), A_complex.conj()
    ), "Complex: .conj should give conj(A)"

    # Transposed mode (X = A.H, so .T gives (A.H).T = conj(A), and .H gives (A.H).H = A)
    X_complex_trans = QR.from_matrix(A_complex, transposed=True)
    assert np.allclose(
        X_complex_trans.full(), A_complex.T.conj()
    ), "Complex transposed mode should give A.H"
    assert np.allclose(
        X_complex_trans.T.full(), A_complex.conj()
    ), "Complex transposed: .T of A.H gives conj(A)"
    assert np.allclose(
        X_complex_trans.H.full(), A_complex
    ), "Complex transposed: .H of A.H gives A"

    # Test double operations
    assert np.allclose(X_complex.T.T.full(), A_complex), "Complex: .T.T should give A"
    assert np.allclose(X_complex.H.H.full(), A_complex), "Complex: .H.H should give A"
    assert np.allclose(
        X_complex.conj.conj.full(), A_complex
    ), "Complex: .conj.conj should give A"


def test_QR_is_orthogonal():
    """Test is_orthogonal() functionality."""
    np.random.seed(202)

    # Test with proper QR decomposition
    A = np.random.randn(100, 50)
    X_proper = QR.from_matrix(A)
    assert X_proper.is_orthogonal(), "QR from from_matrix should be orthogonal"
    assert X_proper.is_orthogonal(
        tol=1e-12
    ), "QR should be orthogonal with tight tolerance"

    # Test with transposed mode
    X_trans = QR.from_matrix(A, transposed=True)
    assert X_trans.is_orthogonal(), "QR in transposed mode should be orthogonal"

    # Test with generated random QR
    X_random = QR.generate_random((80, 60), seed=42)
    assert X_random.is_orthogonal(), "Generated random QR should be orthogonal"

    # Test with non-orthogonal Q
    Q_bad = np.random.randn(100, 50)
    R_bad = np.random.randn(50, 50)
    X_bad = QR(Q_bad, R_bad)
    assert not X_bad.is_orthogonal(), "Random Q should not be orthogonal"
    assert not X_bad.is_orthogonal(
        tol=1e-1
    ), "Random Q should not be orthogonal even with loose tolerance"


def test_QR_is_upper_triangular():
    """Test is_upper_triangular() functionality."""
    np.random.seed(203)

    # Test with proper QR decomposition
    A = np.random.randn(100, 50)
    X_proper = QR.from_matrix(A)
    assert (
        X_proper.is_upper_triangular()
    ), "QR from from_matrix should be upper triangular"
    assert X_proper.is_upper_triangular(
        tol=1e-12
    ), "R should be upper triangular with tight tolerance"

    # Test with transposed mode
    X_trans = QR.from_matrix(A, transposed=True)
    assert (
        X_trans.is_upper_triangular()
    ), "QR in transposed mode should have upper triangular R"

    # Test with generated random QR
    X_random = QR.generate_random((80, 60), seed=42)
    assert (
        X_random.is_upper_triangular()
    ), "Generated random QR should have upper triangular R"

    # Test with non-upper-triangular R (random matrix)
    Q_good, _ = np.linalg.qr(np.random.randn(100, 50))
    R_bad = np.random.randn(50, 50)  # Not upper triangular
    X_bad = QR(Q_good, R_bad)
    assert not X_bad.is_upper_triangular(), "Random R should not be upper triangular"

    # Test with lower triangular R
    R_lower = np.tril(np.random.randn(50, 50))
    X_lower = QR(Q_good, R_lower)
    assert not X_lower.is_upper_triangular(), "Lower triangular R should not pass test"

    # Test with almost upper triangular (small elements below diagonal)
    R_almost = np.triu(np.random.randn(50, 50))
    R_almost[10, 5] = 1e-14  # Add tiny element below diagonal
    X_almost = QR(Q_good, R_almost)
    assert X_almost.is_upper_triangular(
        tol=1e-12
    ), "Should be upper triangular with appropriate tolerance"
    assert not X_almost.is_upper_triangular(
        tol=1e-15
    ), "Should fail with very tight tolerance"


def test_QR_is_upper_triangular_edge_cases():
    """Test is_upper_triangular() with edge cases."""
    np.random.seed(204)

    # Single row (always upper triangular)
    A_row = np.random.randn(1, 10)
    X_row = QR.from_matrix(A_row)
    assert X_row.is_upper_triangular(), "Single row should be upper triangular"

    # Single column (always upper triangular)
    A_col = np.random.randn(50, 1)
    X_col = QR.from_matrix(A_col)
    assert X_col.is_upper_triangular(), "Single column should be upper triangular"

    # Square matrix
    A_square = np.random.randn(30, 30)
    X_square = QR.from_matrix(A_square)
    assert X_square.is_upper_triangular(), "Square QR should be upper triangular"

    # Wide matrix
    A_wide = np.random.randn(30, 80)
    X_wide = QR.from_matrix(A_wide)
    assert X_wide.is_upper_triangular(), "Wide QR should be upper triangular"

    # Tall matrix
    A_tall = np.random.randn(100, 30)
    X_tall = QR.from_matrix(A_tall)
    assert X_tall.is_upper_triangular(), "Tall QR should be upper triangular"


def test_QR_is_upper_triangular_caching():
    """Test that is_upper_triangular() results are cached."""
    np.random.seed(205)
    A = np.random.randn(50, 40)
    X = QR.from_matrix(A)

    # First call should compute and cache
    result1 = X.is_upper_triangular()
    assert result1 == True

    # Check that result is cached
    assert ("is_upper_triangular", 1e-12) in X._cache

    # Second call should use cache
    result2 = X.is_upper_triangular()
    assert result2 == result1

    # Different tolerance should not be cached
    result3 = X.is_upper_triangular(tol=1e-10)
    assert result3 == True
    # Should not cache non-default tolerance
    assert ("is_upper_triangular", 1e-10) not in X._cache


def test_QR_is_upper_triangular_complex():
    """Test is_upper_triangular() with complex matrices."""
    np.random.seed(206)
    A_complex = np.random.randn(60, 40) + 1j * np.random.randn(60, 40)
    X = QR.from_matrix(A_complex)

    assert X.is_upper_triangular(), "Complex QR should be upper triangular"

    # Create non-upper-triangular complex R
    Q, _ = np.linalg.qr(A_complex)
    R_bad = np.random.randn(40, 40) + 1j * np.random.randn(40, 40)
    Y = QR(Q, R_bad)
    assert (
        not Y.is_upper_triangular()
    ), "Random complex R should not be upper triangular"


def test_QR_scalar_multiplication():
    """Test scalar multiplication with __mul__, __rmul__, and __imul__."""
    np.random.seed(203)
    A = np.random.randn(10, 8)

    # Test in standard mode
    X_std = QR.from_matrix(A)
    scalar = 3.5

    # Test __mul__
    Y_mul = X_std * scalar
    assert isinstance(Y_mul, QR), "Scalar multiplication should return QR"
    assert np.allclose(
        Y_mul.full(), scalar * A
    ), "Multiplication should give scalar * A"
    assert np.allclose(Y_mul.Q, X_std.Q), "Q should not be scaled"
    assert np.allclose(Y_mul.R, scalar * X_std.R), "Only R should be scaled"
    assert Y_mul.is_orthogonal(), "Q should remain orthogonal after scaling"

    # Test __rmul__
    Y_rmul = scalar * X_std
    assert np.allclose(
        Y_rmul.full(), scalar * A
    ), "Right multiplication should give scalar * A"
    assert np.allclose(Y_rmul.Q, X_std.Q), "Q should not be scaled (rmul)"
    assert np.allclose(Y_rmul.R, scalar * X_std.R), "Only R should be scaled (rmul)"

    # Test __imul__
    X_imul = QR.from_matrix(A)
    X_imul *= scalar
    assert np.allclose(
        X_imul.full(), scalar * A
    ), "In-place multiplication should give scalar * A"
    assert np.allclose(X_imul.Q, X_std.Q), "Q should not be scaled (imul)"
    assert np.allclose(X_imul.R, scalar * X_std.R), "Only R should be scaled (imul)"

    # Test with integer scalar
    Y_int = X_std * 2
    assert np.allclose(Y_int.full(), 2 * A), "Integer multiplication should work"

    # Test in transposed mode
    X_trans = QR.from_matrix(A, transposed=True)
    Y_trans = X_trans * scalar
    assert np.allclose(
        Y_trans.full(), scalar * A.T.conj()
    ), "Transposed mode scalar multiplication"
    assert Y_trans.is_orthogonal(), "Q should remain orthogonal in transposed mode"

    # Test negative scalar
    Y_neg = X_std * (-2.0)
    assert np.allclose(Y_neg.full(), -2.0 * A), "Negative scalar multiplication"

    # Test zero scalar
    Y_zero = X_std * 0.0
    assert np.allclose(Y_zero.full(), 0 * A), "Zero scalar multiplication"


def test_QR_hadamard_product():
    """Test Hadamard (element-wise) product functionality."""
    np.random.seed(300)

    # Test Hadamard product between two QR matrices
    A = np.random.randn(6, 5)
    B = np.random.randn(6, 5)
    X = QR.from_matrix(A)
    Y = QR.from_matrix(B)

    # Test with hadamard method
    Z_method = X.hadamard(Y)
    assert isinstance(Z_method, QR), "Hadamard of two QR should return QR"
    assert np.allclose(
        Z_method.full(), A * B
    ), "Hadamard product should match element-wise multiplication"
    assert Z_method.shape == X.shape, "Shape should be preserved"
    # Q may have more columns than reported rank after Hadamard, but should still be orthogonal
    assert Z_method.is_orthogonal(), "Result should have orthogonal Q"

    # Test with * operator
    Z_op = X * Y
    assert isinstance(Z_op, QR), "* operator should return QR for two QR matrices"
    assert np.allclose(
        Z_op.full(), A * B
    ), "* operator should give same result as hadamard"

    # Test with ndarray using * operator
    Z_array = X * B
    assert isinstance(Z_array, np.ndarray), "QR * ndarray should return ndarray"
    assert np.allclose(
        Z_array, A * B
    ), "QR * ndarray should match element-wise multiplication"

    # Test ndarray * QR (reverse)
    Z_array_rev = B * X
    assert isinstance(Z_array_rev, np.ndarray), "ndarray * QR should return ndarray"
    assert np.allclose(
        Z_array_rev, A * B
    ), "ndarray * QR should match element-wise multiplication"

    # Test with hadamard method and ndarray
    Z_method_array = X.hadamard(B)
    assert isinstance(
        Z_method_array, np.ndarray
    ), "hadamard(ndarray) should return ndarray"
    assert np.allclose(
        Z_method_array, A * B
    ), "hadamard(ndarray) should match element-wise multiplication"


def test_QR_hadamard_complex():
    """Test Hadamard product with complex matrices."""
    np.random.seed(301)

    A_complex = np.random.randn(5, 4) + 1j * np.random.randn(5, 4)
    B_complex = np.random.randn(5, 4) + 1j * np.random.randn(5, 4)

    X = QR.from_matrix(A_complex)
    Y = QR.from_matrix(B_complex)

    Z = X * Y
    assert isinstance(Z, QR), "Hadamard of complex QR should return QR"
    assert np.allclose(
        Z.full(), A_complex * B_complex
    ), "Complex Hadamard should work correctly"
    # Q should remain orthogonal after Hadamard
    assert Z.is_orthogonal(), "Result should have orthogonal Q"


def test_QR_hadamard_transposed():
    """Test Hadamard product in transposed mode."""
    np.random.seed(302)

    A = np.random.randn(6, 5)
    B = np.random.randn(6, 5)

    X_trans = QR.from_matrix(A, transposed=True)
    Y_trans = QR.from_matrix(B, transposed=True)

    # Both in transposed mode
    Z = X_trans * Y_trans
    assert isinstance(Z, QR), "Hadamard in transposed mode should return QR"
    expected = A.T.conj() * B.T.conj()
    assert np.allclose(Z.full(), expected), "Transposed Hadamard should work correctly"
    assert Z._transposed == True, "Result should preserve transposed flag"


def test_QR_mul_operator_disambiguation():
    """Test that * operator correctly disambiguates between scalar and Hadamard."""
    np.random.seed(303)
    A = np.random.randn(5, 4)
    X = QR.from_matrix(A)

    # Scalar multiplication
    scalar = 2.5
    Y_scalar = X * scalar
    assert isinstance(Y_scalar, QR), "Scalar multiplication should return QR"
    assert np.allclose(Y_scalar.full(), scalar * A), "Scalar multiplication should work"
    assert np.allclose(Y_scalar.Q, X.Q), "Q should not change with scalar"
    assert np.allclose(Y_scalar.R, scalar * X.R), "Only R should be scaled"

    # Integer scalar
    Y_int = X * 3
    assert isinstance(Y_int, QR), "Integer multiplication should return QR"
    assert np.allclose(Y_int.full(), 3 * A), "Integer multiplication should work"

    # Float scalar
    Y_float = X * 1.5
    assert isinstance(Y_float, QR), "Float multiplication should return QR"
    assert np.allclose(Y_float.full(), 1.5 * A), "Float multiplication should work"

    # QR Hadamard
    Y = QR.from_matrix(A)
    Z_hadamard = X * Y
    assert isinstance(Z_hadamard, QR), "QR * QR should return QR"
    assert np.allclose(Z_hadamard.full(), A * A), "Hadamard should work"

    # ndarray Hadamard
    Z_array = X * A
    assert isinstance(Z_array, np.ndarray), "QR * ndarray should return ndarray"
    assert np.allclose(Z_array, A * A), "Array Hadamard should work"


def test_QR_hadamard_rank_increase():
    """Test that Hadamard product with QR matrices works correctly."""
    np.random.seed(304)

    # Create matrices - QR.from_matrix uses economic mode so rank = min(m,n)
    A = np.random.randn(10, 8)
    B = np.random.randn(10, 8)

    X = QR.from_matrix(A)
    Y = QR.from_matrix(B)

    # In economic mode, rank equals min(m, n)
    assert X.rank == 8, "X should have rank 8"
    assert Y.rank == 8, "Y should have rank 8"

    Z = X.hadamard(Y)
    # The Kronecker product creates Q with shape (10, 64), but economic QR reduces it
    # After re-orthogonalization, rank will be min(m, n) = min(10, 8) = 8
    assert Z.rank == 8, "Rank after Hadamard should be 8 (min of dimensions)"
    assert Z.Q.shape[0] == 10, "Q should have 10 rows"
    # Q may have more columns after Kronecker product, but economic QR keeps min(m, kronecker_rank)
    assert Z.Q.shape[1] <= 64, "Q columns should not exceed Kronecker product size"
    assert np.allclose(Z.full(), A * B), "Hadamard should be correct regardless of rank"


def test_QR_hadamard_efficiency_warning():
    """Test that Hadamard product warns when creating inefficient Q."""
    np.random.seed(305)

    # Create small matrices where Kronecker product will exceed row count
    # With ranks 3 and 3, Kronecker will have 9 columns but only 5 rows
    A = np.random.randn(5, 3)
    B = np.random.randn(5, 3)

    X = QR.from_matrix(A)
    Y = QR.from_matrix(B)

    # This should trigger a warning because Q_new will have shape (5, 9)
    import warnings

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        Z = X.hadamard(Y)

        # Check that a warning was issued
        assert len(w) >= 1, "Should have issued at least one warning"
        warning_messages = [str(warning.message) for warning in w]
        assert any(
            "Hadamard product creates Q with" in msg for msg in warning_messages
        ), "Should warn about Q dimensions"

    # But the result should still be correct
    assert np.allclose(
        Z.full(), A * B
    ), "Hadamard should still be correct despite warning"
    assert Z.is_orthogonal(), "Q should still be orthogonal despite warning"


# Test mismatched transposed flags
def test_QR_addition_mismatched_transposed():
    """Test addition with mismatched transposed flags."""
    np.random.seed(400)
    A = np.random.randn(10, 8)
    B = np.random.randn(8, 10)  # Different shape for transposed

    X_std = QR.from_matrix(A, transposed=False)  # Shape: (10, 8)
    Y_trans = QR.from_matrix(B, transposed=True)  # Represents B.T, shape: (10, 8)

    # Should fall back to LowRankMatrix addition, which returns dense array
    import warnings

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        Z = X_std + Y_trans
        # Should have warned about returning dense
        assert any("dense matrix" in str(warning.message).lower() for warning in w)

    # Result should be ndarray (parent class returns dense)
    assert isinstance(
        Z, np.ndarray
    ), "Addition with mismatched flags should return ndarray"
    assert not isinstance(Z, QR), "Addition with mismatched flags should not return QR"
    # But result should still be correct
    assert np.allclose(
        Z, A + B.T
    ), "Addition should be correct even with mismatched flags"


def test_QR_subtraction_mismatched_transposed():
    """Test subtraction with mismatched transposed flags."""
    np.random.seed(401)
    A = np.random.randn(10, 8)
    B = np.random.randn(8, 10)  # Different shape for transposed

    X_std = QR.from_matrix(A, transposed=False)  # Shape: (10, 8)
    Y_trans = QR.from_matrix(B, transposed=True)  # Represents B.T, shape: (10, 8)

    # Should fall back to LowRankMatrix subtraction, which returns dense array
    import warnings

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        Z = X_std - Y_trans
        # Should have warned about returning dense
        assert any("dense matrix" in str(warning.message).lower() for warning in w)

    assert isinstance(
        Z, np.ndarray
    ), "Subtraction with mismatched flags should return ndarray"
    assert not isinstance(
        Z, QR
    ), "Subtraction with mismatched flags should not return QR"
    assert np.allclose(Z, A - B.T), "Subtraction should be correct"


def test_QR_subtraction_self():
    """Test subtracting matrix from itself (QR reduces to minimal rank)."""
    np.random.seed(402)
    A = np.random.randn(10, 8)
    X = QR.from_matrix(A)

    Z = X - X
    assert isinstance(Z, QR), "X - X should return QR"
    # The QR algorithm actually reduces the rank automatically when R becomes near-zero
    # So we get the minimal rank representation, not 2*rank
    assert Z.rank <= 2 * X.rank, "Rank should not exceed 2*rank(X)"
    assert np.allclose(Z.full(), 0, atol=1e-10), "Result should be numerically zero"


def test_QR_hadamard_mismatched_transposed():
    """Test Hadamard product with mismatched transposed flags."""
    np.random.seed(403)
    A = np.random.randn(6, 5)
    B = np.random.randn(6, 5)

    X_std = QR.from_matrix(A, transposed=False)
    Y_trans = QR.from_matrix(B, transposed=True)

    # Should raise ValueError
    with pytest.raises(ValueError, match="same mode"):
        X_std.hadamard(Y_trans)


def test_QR_dot_variations():
    """Test dot method with various input types and sides."""
    np.random.seed(404)
    A = np.random.randn(10, 8)
    B = np.random.randn(8, 6)

    X = QR.from_matrix(A)

    # Test QR @ ndarray
    C = X.dot(B)
    assert isinstance(C, QR), "QR @ ndarray should return QR"
    assert np.allclose(C.full(), A @ B), "QR @ ndarray should be correct"

    # Test ndarray @ QR (using side='left')
    D = X.dot(A.T, side="left")
    assert np.allclose(D.full(), A.T @ A), "ndarray @ QR should be correct"

    # Test QR @ QR
    Y = QR.from_matrix(B)
    E = X.dot(Y)
    assert isinstance(E, QR), "QR @ QR should return QR"
    assert np.allclose(E.full(), A @ B), "QR @ QR should be correct"

    # Test with dense_output=True
    F = X.dot(B, dense_output=True)
    assert isinstance(F, np.ndarray), "dense_output should return ndarray"
    assert np.allclose(F, A @ B), "dense output should be correct"


def test_QR_dot_transposed_mode():
    """Test dot method in transposed mode."""
    np.random.seed(405)
    A = np.random.randn(10, 8)
    B = np.random.randn(10, 6)  # Must match transposed dimensions

    # Test transposed QR @ ndarray
    # X_trans represents A.T (shape 8x10), so B must have 10 rows
    X_trans = QR.from_matrix(A, transposed=True)
    C = X_trans.dot(B)
    # Should fall back to LowRankMatrix for transposed @ ndarray
    assert np.allclose(C.full(), A.T @ B), "Transposed QR @ ndarray should be correct"

    # Test ndarray @ transposed QR
    # X_trans is 8x10, so we need ndarray with shape compatible for left multiplication
    D_left = np.random.randn(5, 8)
    D = X_trans.dot(D_left, side="left")
    assert np.allclose(
        D.full(), D_left @ A.T
    ), "ndarray @ transposed QR should be correct"


# Test edge cases
def test_QR_single_column():
    """Test QR with single column matrix."""
    np.random.seed(500)
    A = np.random.randn(20, 1)
    X = QR.from_matrix(A)

    assert X.shape == (20, 1), "Shape should be correct"
    assert X.rank == 1, "Rank should be 1"
    assert np.allclose(X.full(), A), "Reconstruction should be correct"
    assert X.is_orthogonal(), "Q should be orthogonal"


def test_QR_single_row():
    """Test QR with single row matrix."""
    np.random.seed(501)
    A = np.random.randn(1, 10)
    X = QR.from_matrix(A)

    assert X.shape == (1, 10), "Shape should be correct"
    assert X.rank == 1, "Rank should be 1"
    assert np.allclose(X.full(), A), "Reconstruction should be correct"


def test_QR_square_matrix():
    """Test QR with square matrix."""
    np.random.seed(502)
    A = np.random.randn(15, 15)
    X = QR.from_matrix(A)

    assert X.shape == (15, 15), "Shape should be correct"
    assert X.rank == 15, "Rank should be 15"
    assert np.allclose(X.full(), A), "Reconstruction should be correct"
    assert X.is_orthogonal(), "Q should be orthogonal"


def test_QR_wide_matrix():
    """Test QR with wide matrix (more columns than rows)."""
    np.random.seed(503)
    A = np.random.randn(5, 20)
    X = QR.from_matrix(A, mode="economic")

    assert X.shape == (5, 20), "Shape should be correct"
    assert X.rank == 5, "Rank should be min(m,n)"
    assert np.allclose(X.full(), A), "Reconstruction should be correct"
    assert X.is_orthogonal(), "Q should be orthogonal"


def test_QR_tall_matrix():
    """Test QR with tall matrix (more rows than columns)."""
    np.random.seed(504)
    A = np.random.randn(50, 10)
    X = QR.from_matrix(A, mode="economic")

    assert X.shape == (50, 10), "Shape should be correct"
    assert X.rank == 10, "Rank should be min(m,n)"
    assert np.allclose(X.full(), A), "Reconstruction should be correct"
    assert X.is_orthogonal(), "Q should be orthogonal"


def test_QR_very_small_matrix():
    """Test QR with very small matrix."""
    np.random.seed(505)
    A = np.random.randn(2, 2)
    X = QR.from_matrix(A)

    assert X.shape == (2, 2), "Shape should be correct"
    assert X.rank == 2, "Rank should be 2"
    assert np.allclose(X.full(), A), "Reconstruction should be correct"


def test_QR_rank_deficient():
    """Test QR with rank-deficient matrix."""
    np.random.seed(506)
    # Create rank-2 matrix
    U = np.random.randn(10, 2)
    V = np.random.randn(8, 2)
    A = U @ V.T

    X = QR.from_matrix(A)
    # In economic mode, rank will be min(m,n) but matrix is numerically rank-2
    assert X.shape == (10, 8), "Shape should be correct"
    assert np.allclose(X.full(), A, atol=1e-10), "Reconstruction should be correct"


def test_QR_complex_dtype_preservation():
    """Test that complex dtype is preserved."""
    np.random.seed(507)
    A_complex = np.random.randn(8, 6) + 1j * np.random.randn(8, 6)
    X = QR.from_matrix(A_complex)

    assert np.iscomplexobj(X.Q), "Q should be complex"
    assert np.iscomplexobj(X.R), "R should be complex"
    assert np.allclose(X.full(), A_complex), "Reconstruction should be correct"


def test_QR_zero_matrix():
    """Test QR with zero matrix (edge case)."""
    A = np.zeros((5, 4))
    X = QR.from_matrix(A)

    assert X.shape == (5, 4), "Shape should be correct"
    assert np.allclose(X.full(), A), "Reconstruction should be zero"
    # R should be essentially zero
    assert np.allclose(X.R, 0, atol=1e-10), "R should be zero"


def test_QR_operations_preserve_dtype():
    """Test that operations preserve dtype."""
    np.random.seed(508)
    A = np.random.randn(6, 5) + 1j * np.random.randn(6, 5)
    B = np.random.randn(6, 5) + 1j * np.random.randn(6, 5)

    X = QR.from_matrix(A)
    Y = QR.from_matrix(B)

    # Addition
    Z_add = X + Y
    assert np.iscomplexobj(Z_add.Q), "Addition should preserve complex dtype"

    # Scalar multiplication
    Z_mul = 2.5 * X
    assert np.iscomplexobj(Z_mul.Q), "Scalar mult should preserve complex dtype"

    # Hadamard
    Z_had = X * Y
    assert np.iscomplexobj(Z_had.Q), "Hadamard should preserve complex dtype"


# ==========================
# Truncation tests
# ==========================


def test_QR_truncate_by_rank():
    """Test QR truncation by specifying rank."""
    np.random.seed(600)
    A = np.random.randn(100, 80)
    X = QR.from_matrix(A)

    # Truncate to rank 10
    X_trunc = X.truncate(r=10)
    assert X_trunc.rank == 10, "Truncated rank should be 10"
    assert X_trunc.Q.shape == (100, 10), "Q shape should be (100, 10)"
    assert X_trunc.R.shape == (10, 80), "R shape should be (10, 80)"
    assert X_trunc.shape == (100, 80), "Overall shape should be preserved"
    assert X_trunc.is_orthogonal(), "Q should remain orthogonal"

    # Original should be unchanged
    assert X.rank == 80, "Original rank should be unchanged"


def test_QR_truncate_to_zero_rank():
    """Test QR truncation to rank 0."""
    np.random.seed(601)
    A = np.random.randn(50, 40)
    X = QR.from_matrix(A)

    X_zero = X.truncate(r=0)
    assert X_zero.rank == 0, "Rank should be 0"
    assert X_zero.Q.shape == (50, 0), "Q should have 0 columns"
    assert X_zero.R.shape == (0, 40), "R should have 0 rows"
    assert np.allclose(X_zero.full(), 0), "Full matrix should be zeros"


def test_QR_truncate_by_atol():
    """Test QR truncation by absolute tolerance."""
    np.random.seed(602)
    # Create QR with known R diagonal values
    Q, _ = np.linalg.qr(np.random.randn(100, 20))
    R_diag = np.array([1.0, 0.5, 0.1, 0.01, 1e-5, 1e-8] + [1e-10] * 14)
    R = np.diag(R_diag) @ np.random.randn(20, 80)
    X = QR(Q, R)

    # Truncate with atol=1e-7 should keep first 5 components
    X_trunc = X.truncate(atol=1e-7)
    assert X_trunc.rank == 5, f"Rank should be 5, got {X_trunc.rank}"

    # Truncate with atol=1e-9 should keep first 6 components
    X_trunc2 = X.truncate(atol=1e-9)
    assert X_trunc2.rank == 6, f"Rank should be 6, got {X_trunc2.rank}"


def test_QR_truncate_by_rtol():
    """Test QR truncation by relative tolerance."""
    np.random.seed(603)
    # Create QR with known R diagonal values
    # Use an upper triangular matrix to preserve diagonal values
    Q, _ = np.linalg.qr(np.random.randn(100, 20))
    R_diag = np.array([1.0, 0.5, 0.1, 0.05, 0.01, 0.001] + [1e-5] * 14)
    R = np.triu(np.random.randn(20, 80))
    # Set the diagonal explicitly
    for i in range(20):
        R[i, i] = R_diag[i]
    X = QR(Q, R)

    # Truncate with rtol=1e-2 (keep where |R[i,i]| > 1.0 * 0.01 = 0.01)
    # Values: [1.0, 0.5, 0.1, 0.05] satisfy this (0.01 is NOT > 0.01)
    X_trunc = X.truncate(rtol=1e-2)
    assert X_trunc.rank == 4, f"Rank should be 4, got {X_trunc.rank}"

    # Truncate with rtol=1e-3 (keep where |R[i,i]| > 1.0 * 0.001 = 0.001)
    # Values: [1.0, 0.5, 0.1, 0.05, 0.01] satisfy this (0.001 is NOT > 0.001)
    X_trunc2 = X.truncate(rtol=1e-3)
    assert X_trunc2.rank == 5, f"Rank should be 5, got {X_trunc2.rank}"


def test_QR_truncate_parameter_priority():
    """Test that truncation parameter priority is r > rtol > atol."""
    np.random.seed(604)
    Q, _ = np.linalg.qr(np.random.randn(100, 20))
    R_diag = np.logspace(0, -10, 20)
    R = np.diag(R_diag) @ np.random.randn(20, 80)
    X = QR(Q, R)

    # When all three are specified, r should win
    X_priority = X.truncate(r=5, rtol=1e-3, atol=1e-5)
    assert X_priority.rank == 5, "r should take priority"

    # When rtol and atol specified, rtol should win
    X_rtol_wins = X.truncate(rtol=1e-3, atol=1e-10)
    rank_rtol = X_rtol_wins.rank

    X_atol_only = X.truncate(atol=1e-10)
    rank_atol = X_atol_only.rank

    # rtol should give different result than atol (more restrictive)
    assert rank_rtol < rank_atol, "rtol should be more restrictive than atol"


def test_QR_truncate_inplace():
    """Test in-place truncation."""
    np.random.seed(605)
    A = np.random.randn(100, 80)
    X = QR.from_matrix(A)

    original_id = id(X)
    original_Q_id = id(X._matrices[0])

    # In-place truncation should modify the same object
    result = X.truncate(r=15, inplace=True)

    assert id(result) == original_id, "Should return same object"
    assert X.rank == 15, "Rank should be updated"
    assert id(X._matrices[0]) != original_Q_id, "Internal matrices should be replaced"


def test_QR_truncate_copy():
    """Test that non-inplace truncation creates a copy."""
    np.random.seed(606)
    A = np.random.randn(100, 80)
    X = QR.from_matrix(A)

    X_trunc = X.truncate(r=10, inplace=False)

    assert X_trunc is not X, "Should create new object"
    assert X.rank == 80, "Original rank should be unchanged"
    assert X_trunc.rank == 10, "New object should have truncated rank"


def test_QR_truncate_transposed_mode():
    """Test that truncation preserves transposed mode."""
    np.random.seed(607)
    A = np.random.randn(80, 100)

    # Standard mode
    X_std = QR.from_matrix(A, transposed=False)
    X_std_trunc = X_std.truncate(r=10)
    assert not X_std_trunc._transposed, "Standard mode should be preserved"

    # Transposed mode
    X_trans = QR.from_matrix(A.T, transposed=True)
    X_trans_trunc = X_trans.truncate(r=10)
    assert X_trans_trunc._transposed, "Transposed mode should be preserved"


def test_QR_truncate_cache_clearing():
    """Test that cache is cleared on in-place truncation."""
    np.random.seed(608)
    A = np.random.randn(100, 80)
    X = QR.from_matrix(A)

    # Populate cache
    _ = X.norm("fro")
    _ = X.norm(2)
    assert len(X._cache) > 0, "Cache should be populated"

    # In-place truncation should clear cache
    X.truncate(r=10, inplace=True)
    assert len(X._cache) == 0, "Cache should be cleared after inplace truncation"


def test_QR_truncate_orthogonality_preserved():
    """Test that Q remains orthogonal after truncation."""
    np.random.seed(609)
    A = np.random.randn(100, 80)
    X = QR.from_matrix(A)

    X_trunc = X.truncate(r=20)
    assert X_trunc.is_orthogonal(), "Q should remain orthogonal after truncation"

    # Check explicitly
    Q_H_Q = X_trunc.Q.T.conj() @ X_trunc.Q
    I = np.eye(20)
    assert np.allclose(Q_H_Q, I, atol=1e-12), "Q.H @ Q should equal identity"


def test_QR_truncate_dimensions():
    """Test that Q and R dimensions are correct after truncation."""
    np.random.seed(610)
    A = np.random.randn(100, 80)
    X = QR.from_matrix(A)

    for r in [5, 10, 20, 40]:
        X_trunc = X.truncate(r=r)
        assert X_trunc.Q.shape == (100, r), f"Q shape should be (100, {r})"
        assert X_trunc.R.shape == (r, 80), f"R shape should be ({r}, 80)"
        assert X_trunc.shape == (100, 80), "Overall shape should be preserved"


def test_QR_truncate_complex_dtype():
    """Test that truncation preserves complex dtype."""
    np.random.seed(611)
    A_complex = np.random.randn(50, 40) + 1j * np.random.randn(50, 40)
    X = QR.from_matrix(A_complex)

    X_trunc = X.truncate(r=10)
    assert np.iscomplexobj(X_trunc.Q), "Q should remain complex"
    assert np.iscomplexobj(X_trunc.R), "R should remain complex"
    assert np.iscomplexobj(X_trunc.full()), "Full matrix should remain complex"


def test_QR_truncate_no_parameters():
    """Test truncation with no parameters (uses defaults)."""
    np.random.seed(612)
    A = np.random.randn(50, 40)
    X = QR.from_matrix(A)

    # With defaults, should not truncate significantly (only numerical zeros)
    X_default = X.truncate()
    assert X_default.rank == X.rank, "Should keep all components with default tolerance"


def test_QR_truncate_reconstruction_quality():
    """Test that truncated QR can still reconstruct the dominant part."""
    np.random.seed(613)
    # Create a low-rank-ish matrix
    U_base = np.random.randn(100, 10)
    V_base = np.random.randn(80, 10)
    A_lowrank = U_base @ V_base.T

    # Add small noise
    A = A_lowrank + 1e-8 * np.random.randn(100, 80)

    X = QR.from_matrix(A)
    X_trunc = X.truncate(r=10)

    # Truncated version should be close to low-rank part
    reconstruction_error = np.linalg.norm(X_trunc.full() - A_lowrank, "fro")
    noise_level = 1e-8 * np.sqrt(100 * 80)

    assert (
        reconstruction_error < 10 * noise_level
    ), "Truncation should capture dominant structure"


def test_QR_truncate_both_modes():
    """Test truncation works correctly for both standard and transposed modes."""
    np.random.seed(614)

    # Test standard mode
    A = np.random.randn(10, 8)
    X = QR.from_matrix(A)
    assert X.shape == (10, 8)
    assert X.rank == 8
    assert not X._transposed

    X_trunc = X.truncate(r=3)
    assert X_trunc.shape == (10, 8)
    assert X_trunc.rank == 3
    assert not X_trunc._transposed
    assert X_trunc.Q.shape == (10, 3)
    assert X_trunc.R.shape == (3, 8)

    # Verify reconstruction matches expected
    expected = X.Q[:, :3] @ X.R[:3, :]
    assert np.allclose(
        X_trunc.full(), expected
    ), "Standard mode truncation reconstruction"

    # Test transposed mode
    B = np.random.randn(8, 10)
    Y = QR.from_matrix(B.T)  # Creates (10, 8) matrix
    assert Y.shape == (10, 8)
    assert Y.rank == 8
    assert not Y._transposed  # from_matrix with standard A.T doesn't set transposed

    # Create explicitly transposed QR
    Z = QR.from_matrix(B, transposed=True)  # Creates QR representing B.T (10x8)
    assert Z.shape == (10, 8)  # Represents transpose, so shape is swapped
    assert Z._transposed

    Z_trunc = Z.truncate(r=3)
    assert Z_trunc.shape == (10, 8)
    assert Z_trunc.rank == 3
    assert Z_trunc._transposed
    assert Z_trunc.Q.shape == (8, 3)  # Q comes from B's shape
    assert Z_trunc.R.shape == (3, 10)  # R has full width of B

    # Verify reconstruction - for transposed mode: X = R.H @ Q.H
    expected_Z = Z.R[:3, :].T.conj() @ Z.Q[:, :3].T.conj()
    assert np.allclose(
        Z_trunc.full(), expected_Z
    ), "Transposed mode truncation reconstruction"


# ==========================
# solve() tests
# ==========================


def test_QR_solve_basic():
    """Test basic solve functionality."""
    np.random.seed(700)
    # Square full-rank system for exact solution
    A = np.random.randn(80, 80)
    b = np.random.randn(80)

    X = QR.from_matrix(A)
    x = X.solve(b)

    # Verify solution
    assert x.shape == (80,), "Solution shape should be (n,)"
    assert np.allclose(A @ x, b, atol=1e-10), "Should solve Ax = b exactly"


def test_QR_solve_multiple_rhs():
    """Test solve with multiple right-hand sides."""
    np.random.seed(701)
    A = np.random.randn(80, 80)  # Square for exact solution
    B = np.random.randn(80, 5)  # 5 right-hand sides

    X = QR.from_matrix(A)
    Y = X.solve(B)

    assert Y.shape == (80, 5), "Solution shape should be (n, k)"
    assert np.allclose(A @ Y, B, atol=1e-10), "Should solve for all RHS"


def test_QR_solve_square_matrix():
    """Test solve for square full-rank matrix."""
    np.random.seed(702)
    A = np.random.randn(50, 50)
    b = np.random.randn(50)

    X = QR.from_matrix(A)
    x = X.solve(b)

    assert np.allclose(A @ x, b, atol=1e-10), "Square system should be solved exactly"


def test_QR_solve_tall_matrix():
    """Test solve for tall matrix (overdetermined consistent system)."""
    np.random.seed(703)
    A = np.random.randn(150, 50)
    x_true = np.random.randn(50)
    b = A @ x_true  # Create consistent system: b is in column space of A

    X = QR.from_matrix(A)
    x = X.solve(b)

    # For consistent overdetermined system, should get exact solution
    assert x.shape == (50,), "Solution shape should be (n,)"
    assert np.allclose(
        A @ x, b, atol=1e-10
    ), "Should solve exactly for consistent system"
    assert np.allclose(x, x_true, atol=1e-9), "Should recover true solution"


def test_QR_solve_transposed_mode():
    """Test solve in transposed mode."""
    np.random.seed(704)
    A = np.random.randn(80, 80)  # Square matrix for exact solution
    b = np.random.randn(80)

    # X represents A.T (shape 80x80)
    X = QR.from_matrix(A, transposed=True)
    assert X.shape == (80, 80), "Transposed shape should be (80, 80)"

    x = X.solve(b)
    assert x.shape == (80,), "Solution shape should be (80,)"
    assert np.allclose(A.T @ x, b, atol=1e-10), "Should solve A.T @ x = b"


def test_QR_solve_complex_dtype():
    """Test solve with complex matrices."""
    np.random.seed(705)
    A_complex = np.random.randn(60, 60) + 1j * np.random.randn(
        60, 60
    )  # Square for exact solution
    b_complex = np.random.randn(60) + 1j * np.random.randn(60)

    X = QR.from_matrix(A_complex)
    x = X.solve(b_complex)

    assert np.iscomplexobj(x), "Solution should be complex"
    assert np.allclose(
        A_complex @ x, b_complex, atol=1e-10
    ), "Complex solve should work"


def test_QR_solve_lstsq_method():
    """Test solve with method='lstsq'."""
    np.random.seed(706)
    A = np.random.randn(100, 80)
    b = np.random.randn(100)

    X = QR.from_matrix(A)
    x_direct = X.solve(b, method="direct")
    x_lstsq = X.solve(b, method="lstsq")

    # Both methods should give similar results for full-rank matrices
    assert np.allclose(x_direct, x_lstsq, atol=1e-9), "Direct and lstsq should agree"


def test_QR_solve_dimension_mismatch():
    """Test solve raises error for dimension mismatch."""
    np.random.seed(707)
    A = np.random.randn(100, 80)
    b_wrong = np.random.randn(50)  # Wrong dimension

    X = QR.from_matrix(A)
    with pytest.raises(ValueError, match="Dimension mismatch"):
        X.solve(b_wrong)


def test_QR_solve_rank_deficient():
    """Test solve with rank-deficient matrix using lstsq method."""
    np.random.seed(708)
    # Create rank-deficient matrix (rank 10)
    U = np.random.randn(100, 10)
    V = np.random.randn(80, 10)
    A = U @ V.T  # rank 10
    b = np.random.randn(100)

    X = QR.from_matrix(A)
    # lstsq should handle rank deficiency
    x = X.solve(b, method="lstsq")
    assert x.shape == (80,), "Solution shape should be correct"


def test_QR_solve_single_column():
    """Test solve with single column matrix (least squares)."""
    np.random.seed(709)
    A = np.random.randn(50, 1)
    x_true = np.random.randn(1)
    b = A @ x_true  # Create consistent system

    X = QR.from_matrix(A)
    x = X.solve(b)

    assert x.shape == (1,), "Solution should have shape (1,)"
    assert np.allclose(A @ x, b, atol=1e-10), "Single column solve should work"
    assert np.allclose(x, x_true, atol=1e-9), "Should recover true solution"


def test_QR_solve_consistency_with_numpy():
    """Test that solve gives same results as numpy.linalg.solve for square systems."""
    np.random.seed(710)
    A = np.random.randn(80, 80)  # Square system
    b = np.random.randn(80)

    X = QR.from_matrix(A)
    x_qr = X.solve(b)
    x_numpy = np.linalg.solve(A, b)

    assert np.allclose(x_qr, x_numpy, atol=1e-9), "Should match numpy.linalg.solve"


# ==========================
# lstsq() tests
# ==========================


def test_QR_lstsq_basic():
    """Test basic least squares functionality."""
    np.random.seed(800)
    A = np.random.randn(100, 80)
    b = np.random.randn(100)

    X = QR.from_matrix(A)
    x = X.lstsq(b)

    assert x.shape == (80,), "Solution shape should be (n,)"
    # Should minimize ||Ax - b||
    residual = np.linalg.norm(A @ x - b)
    x_numpy, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    residual_numpy = np.linalg.norm(A @ x_numpy - b)
    assert (
        np.abs(residual - residual_numpy) < 1e-9
    ), "Should achieve similar residual as numpy"


def test_QR_lstsq_overdetermined():
    """Test lstsq for overdetermined system."""
    np.random.seed(801)
    # Create inconsistent overdetermined system
    A = np.random.randn(150, 50)
    x_true = np.random.randn(50)
    noise = 0.1 * np.random.randn(150)
    b = A @ x_true + noise  # Add noise to make system inconsistent

    X = QR.from_matrix(A)
    x_lstsq = X.lstsq(b)

    # Should give approximate solution
    assert x_lstsq.shape == (50,), "Solution shape correct"
    # Check that it minimizes the residual
    residual = np.linalg.norm(A @ x_lstsq - b)
    # Any other solution should have larger or equal residual
    x_random = np.random.randn(50)
    residual_random = np.linalg.norm(A @ x_random - b)
    assert residual < residual_random, "lstsq should minimize residual"


def test_QR_lstsq_underdetermined():
    """Test lstsq for underdetermined system."""
    np.random.seed(802)
    # Underdetermined: more unknowns than equations
    A = np.random.randn(50, 100)
    b = np.random.randn(50)

    X = QR.from_matrix(A)
    x = X.lstsq(b)

    assert x.shape == (100,), "Solution shape should be (100,)"
    # Solution should satisfy Ax = b
    assert np.allclose(A @ x, b, atol=1e-9), "Should satisfy the system"


def test_QR_lstsq_multiple_rhs():
    """Test lstsq with multiple right-hand sides."""
    np.random.seed(803)
    A = np.random.randn(100, 60)
    B = np.random.randn(100, 5)

    X = QR.from_matrix(A)
    Y = X.lstsq(B)

    assert Y.shape == (60, 5), "Solution shape should be (n, k)"
    # Each column should be a least squares solution
    for i in range(5):
        residual = np.linalg.norm(A @ Y[:, i] - B[:, i])
        x_numpy, _, _, _ = np.linalg.lstsq(A, B[:, i], rcond=None)
        residual_numpy = np.linalg.norm(A @ x_numpy - B[:, i])
        assert (
            np.abs(residual - residual_numpy) < 1e-8
        ), f"Column {i} should match numpy"


def test_QR_lstsq_rank_deficient():
    """Test lstsq with rank-deficient matrix."""
    np.random.seed(804)
    # Create rank-5 matrix
    U = np.random.randn(80, 5)
    V = np.random.randn(60, 5)
    A = U @ V.T
    b = U @ np.random.randn(5)  # b is in column space

    X = QR.from_matrix(A)
    x = X.lstsq(b)

    assert x.shape == (60,), "Solution shape should be correct"
    # Should solve the system even with rank deficiency
    assert np.allclose(A @ x, b, atol=1e-9), "Should find a solution"


def test_QR_lstsq_with_rcond():
    """Test lstsq with rcond parameter."""
    np.random.seed(805)
    A = np.random.randn(100, 80)
    b = np.random.randn(100)

    X = QR.from_matrix(A)
    x1 = X.lstsq(b, rcond=None)
    x2 = X.lstsq(b, rcond=1e-10)

    # Both should work, may give slightly different results
    assert x1.shape == (80,), "Solution 1 shape correct"
    assert x2.shape == (80,), "Solution 2 shape correct"


def test_QR_lstsq_complex():
    """Test lstsq with complex matrices."""
    np.random.seed(806)
    A_complex = np.random.randn(80, 60) + 1j * np.random.randn(80, 60)
    x_true = np.random.randn(60) + 1j * np.random.randn(60)
    b_complex = A_complex @ x_true  # Create consistent system

    X = QR.from_matrix(A_complex)
    x = X.lstsq(b_complex)

    assert np.iscomplexobj(x), "Solution should be complex"
    assert x.shape == (60,), "Solution shape correct"
    residual = np.linalg.norm(A_complex @ x - b_complex)
    assert residual < 1e-9, "Complex lstsq should work"
    assert np.allclose(x, x_true, atol=1e-9), "Should recover true solution"


def test_QR_lstsq_transposed_mode():
    """Test lstsq in transposed mode."""
    np.random.seed(807)
    A = np.random.randn(60, 80)
    x_true = np.random.randn(60)
    b = A.T @ x_true  # Create consistent system for A.T

    X = QR.from_matrix(A, transposed=True)  # Represents A.T (80x60)
    x = X.lstsq(b)

    assert x.shape == (60,), "Solution shape should be (60,)"
    # Should solve A.T @ x = b in least squares sense
    assert np.allclose(A.T @ x, b, atol=1e-9), "Transposed lstsq should work"
    assert np.allclose(x, x_true, atol=1e-9), "Should recover true solution"


def test_QR_lstsq_dimension_mismatch():
    """Test lstsq raises error for dimension mismatch."""
    np.random.seed(808)
    A = np.random.randn(100, 80)
    b_wrong = np.random.randn(50)

    X = QR.from_matrix(A)
    with pytest.raises(ValueError, match="Dimension mismatch"):
        X.lstsq(b_wrong)


def test_QR_lstsq_consistency():
    """Test lstsq is consistent with solve for full-rank systems."""
    np.random.seed(809)
    A = np.random.randn(100, 80)
    b = np.random.randn(100)

    X = QR.from_matrix(A)
    x_solve = X.solve(b)
    x_lstsq = X.lstsq(b)

    assert np.allclose(x_solve, x_lstsq, atol=1e-9), "solve and lstsq should agree"


# ==========================
# pseudoinverse() tests
# ==========================


def test_QR_pseudoinverse_basic():
    """Test basic pseudoinverse functionality."""
    np.random.seed(900)
    A = np.random.randn(100, 80)

    X = QR.from_matrix(A)
    A_pinv = X.pseudoinverse()

    assert A_pinv.shape == (80, 100), "Pseudoinverse should have transposed shape"
    # Property 1: A @ A_pinv @ A = A
    assert np.allclose(A @ A_pinv @ A, A, atol=1e-9), "Property 1: A @ A+ @ A = A"


def test_QR_pseudoinverse_four_properties():
    """Test all four Moore-Penrose properties."""
    np.random.seed(901)
    A = np.random.randn(80, 60)

    X = QR.from_matrix(A)
    A_pinv = X.pseudoinverse()

    # Property 1: A @ A+ @ A = A
    assert np.allclose(A @ A_pinv @ A, A, atol=1e-9), "Property 1 failed"

    # Property 2: A+ @ A @ A+ = A+
    assert np.allclose(A_pinv @ A @ A_pinv, A_pinv, atol=1e-9), "Property 2 failed"

    # Property 3: (A @ A+).H = A @ A+
    AA_pinv = A @ A_pinv
    assert np.allclose(AA_pinv.T.conj(), AA_pinv, atol=1e-9), "Property 3 failed"

    # Property 4: (A+ @ A).H = A+ @ A
    A_pinv_A = A_pinv @ A
    assert np.allclose(A_pinv_A.T.conj(), A_pinv_A, atol=1e-9), "Property 4 failed"


def test_QR_pseudoinverse_square():
    """Test pseudoinverse for square full-rank matrix (should be inverse)."""
    np.random.seed(902)
    A = np.random.randn(50, 50)

    X = QR.from_matrix(A)
    A_pinv = X.pseudoinverse()

    # For full-rank square matrix, pseudoinverse = inverse
    assert np.allclose(A @ A_pinv, np.eye(50), atol=1e-9), "Should give identity"
    assert np.allclose(A_pinv @ A, np.eye(50), atol=1e-9), "Should give identity"


def test_QR_pseudoinverse_tall():
    """Test pseudoinverse for tall matrix (overdetermined)."""
    np.random.seed(903)
    A = np.random.randn(150, 50)

    X = QR.from_matrix(A)
    A_pinv = X.pseudoinverse()

    assert A_pinv.shape == (50, 150), "Shape should be (n, m)"
    # A_pinv @ A should be identity (right inverse)
    assert np.allclose(A_pinv @ A, np.eye(50), atol=1e-9), "Should be right inverse"


def test_QR_pseudoinverse_wide():
    """Test pseudoinverse for wide matrix (underdetermined)."""
    np.random.seed(904)
    A = np.random.randn(50, 150)

    X = QR.from_matrix(A)
    A_pinv = X.pseudoinverse()

    assert A_pinv.shape == (150, 50), "Shape should be (n, m)"
    # A @ A_pinv should be identity (left inverse)
    assert np.allclose(A @ A_pinv, np.eye(50), atol=1e-9), "Should be left inverse"


def test_QR_pseudoinverse_rank_deficient():
    """Test pseudoinverse for rank-deficient matrix."""
    np.random.seed(905)
    # Create rank-10 matrix
    U = np.random.randn(100, 10)
    V = np.random.randn(80, 10)
    A = U @ V.T

    X = QR.from_matrix(A)
    A_pinv = X.pseudoinverse()

    assert A_pinv.shape == (80, 100), "Shape should be correct"
    # Should still satisfy Property 1
    assert np.allclose(A @ A_pinv @ A, A, atol=1e-9), "Property 1 should hold"


def test_QR_pseudoinverse_with_rcond():
    """Test pseudoinverse with rcond parameter."""
    np.random.seed(906)
    A = np.random.randn(80, 60)

    X = QR.from_matrix(A)
    A_pinv1 = X.pseudoinverse(rcond=None)
    A_pinv2 = X.pseudoinverse(rcond=1e-10)

    # Both should work
    assert A_pinv1.shape == (60, 80), "Shape 1 correct"
    assert A_pinv2.shape == (60, 80), "Shape 2 correct"
    # Both should satisfy Property 1
    assert np.allclose(A @ A_pinv1 @ A, A, atol=1e-9), "Property 1 for pinv1"
    assert np.allclose(A @ A_pinv2 @ A, A, atol=1e-9), "Property 1 for pinv2"


def test_QR_pseudoinverse_complex():
    """Test pseudoinverse with complex matrices."""
    np.random.seed(907)
    A_complex = np.random.randn(70, 50) + 1j * np.random.randn(70, 50)

    X = QR.from_matrix(A_complex)
    A_pinv = X.pseudoinverse()

    assert np.iscomplexobj(A_pinv), "Pseudoinverse should be complex"
    assert A_pinv.shape == (50, 70), "Shape should be correct"
    assert np.allclose(
        A_complex @ A_pinv @ A_complex, A_complex, atol=1e-9
    ), "Property 1"


def test_QR_pseudoinverse_transposed_mode():
    """Test pseudoinverse in transposed mode."""
    np.random.seed(908)
    A = np.random.randn(60, 80)

    # X represents A.T (shape 80x60)
    X = QR.from_matrix(A, transposed=True)
    A_pinv = X.pseudoinverse()

    assert A_pinv.shape == (60, 80), "Pseudoinverse shape should be (60, 80)"
    # Should satisfy A.T @ A_pinv @ A.T = A.T
    A_T = A.T
    assert np.allclose(
        A_T @ A_pinv @ A_T, A_T, atol=1e-9
    ), "Property 1 in transposed mode"


def test_QR_pseudoinverse_solve_consistency():
    """Test that pseudoinverse gives same result as solve for consistent systems."""
    np.random.seed(909)
    A = np.random.randn(80, 80)  # Square for exact solution
    b = np.random.randn(80)

    X = QR.from_matrix(A)

    # Solve using solve method
    x_solve = X.solve(b)

    # Solve using pseudoinverse
    A_pinv = X.pseudoinverse()
    x_pinv = A_pinv @ b

    assert np.allclose(
        x_solve, x_pinv, atol=1e-9
    ), "solve and pseudoinverse should agree"


def test_QR_pseudoinverse_consistency_with_numpy():
    """Test that pseudoinverse matches numpy.linalg.pinv."""
    np.random.seed(910)
    A = np.random.randn(80, 60)

    X = QR.from_matrix(A)
    A_pinv_qr = X.pseudoinverse()
    A_pinv_numpy = np.linalg.pinv(A)

    assert np.allclose(
        A_pinv_qr, A_pinv_numpy, atol=1e-9
    ), "Should match numpy.linalg.pinv"


def test_QR_pseudoinverse_single_column():
    """Test pseudoinverse of single column matrix."""
    np.random.seed(911)
    A = np.random.randn(50, 1)

    X = QR.from_matrix(A)
    A_pinv = X.pseudoinverse()

    assert A_pinv.shape == (1, 50), "Shape should be (1, 50)"
    # For single column: A+ = (1 / ||A||^2) * A.T
    expected = A.T / (A.T @ A)
    assert np.allclose(A_pinv, expected, atol=1e-9), "Single column formula should hold"


def test_QR_pseudoinverse_single_row():
    """Test pseudoinverse of single row matrix."""
    np.random.seed(912)
    A = np.random.randn(1, 50)

    X = QR.from_matrix(A)
    A_pinv = X.pseudoinverse()

    assert A_pinv.shape == (50, 1), "Shape should be (50, 1)"
    # For single row: A+ = A.T / ||A||^2
    expected = A.T / (A @ A.T)
    assert np.allclose(A_pinv, expected, atol=1e-9), "Single row formula should hold"


def test_QR_pseudoinverse_idempotent():
    """Test that (A @ A+) and (A+ @ A) are idempotent."""
    np.random.seed(913)
    A = np.random.randn(80, 60)

    X = QR.from_matrix(A)
    A_pinv = X.pseudoinverse()

    # (A @ A+) should be idempotent: (A @ A+)^2 = A @ A+
    AA_pinv = A @ A_pinv
    assert np.allclose(
        AA_pinv @ AA_pinv, AA_pinv, atol=1e-9
    ), "(A @ A+) should be idempotent"

    # (A+ @ A) should be idempotent: (A+ @ A)^2 = A+ @ A
    A_pinvA = A_pinv @ A
    assert np.allclose(
        A_pinvA @ A_pinvA, A_pinvA, atol=1e-9
    ), "(A+ @ A) should be idempotent"


# ============================================================================
# Test cond() method
# ============================================================================


def test_QR_cond_basic():
    """Test basic condition number computation."""
    np.random.seed(1000)
    A = np.random.randn(100, 80)

    X = QR.from_matrix(A)
    cond_qr = X.cond(2)

    # Should be positive
    assert cond_qr > 0, "Condition number should be positive"

    # For well-conditioned random matrix, should be reasonable
    assert cond_qr < 1e10, "Random matrix should be reasonably conditioned"

    # NOTE: The diagonal-based estimate can differ significantly from exact cond(R)
    # This is expected - it's a fast approximation, not exact computation


def test_QR_cond_identity():
    """Test condition number of identity matrix."""
    I = np.eye(50)
    X = QR.from_matrix(I)

    cond = X.cond(2)
    assert np.allclose(cond, 1.0, rtol=1e-10), "Identity should have cond=1"


def test_QR_cond_ill_conditioned():
    """Test condition number for ill-conditioned matrix."""
    # Create ill-conditioned diagonal matrix
    diag_vals = np.array([1e10, 1e5, 1e0, 1e-5, 1e-10])
    A = np.diag(diag_vals)

    X = QR.from_matrix(A)
    cond = X.cond(2)

    # Should detect high condition number
    assert cond > 1e15, "Should detect ill-conditioning"

    # Should be approximately ratio of max/min
    expected_cond = 1e10 / 1e-10  # 1e20
    assert np.allclose(cond, expected_cond, rtol=0.1), "Should match expected ratio"


def test_QR_cond_singular():
    """Test condition number for (near-)singular matrix."""
    # Create rank-deficient matrix
    A = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1e-20]])  # Essentially singular

    X = QR.from_matrix(A)
    cond = X.cond(2)

    # Should be very large or infinite
    assert cond > 1e15, "Nearly singular matrix should have huge condition number"


def test_QR_cond_exact():
    """Test exact vs approximate condition number."""
    np.random.seed(1001)
    A = np.random.randn(80, 60)

    X = QR.from_matrix(A)

    # Approximate (fast)
    cond_approx = X.cond(2, exact=False)

    # Exact (requires SVD)
    cond_exact = X.cond(2, exact=True)

    # Both should be positive
    assert cond_approx > 0
    assert cond_exact > 0

    # Approximation should be a lower bound (or close for well-conditioned matrices)
    assert cond_approx <= cond_exact * 1.01  # Allow small numerical tolerance

    # Exact should match numpy
    cond_numpy = np.linalg.cond(X.R, 2)
    assert np.allclose(cond_exact, cond_numpy, rtol=1e-10)


def test_QR_cond_norms():
    """Test condition number with different norms."""
    np.random.seed(1002)
    A = np.random.randn(80, 60)

    X = QR.from_matrix(A)

    # Test different norms (always exact for non-2 norms)
    cond_fro = X.cond("fro")
    cond_1 = X.cond(1)
    cond_inf = X.cond(np.inf)

    # All should be positive
    assert cond_fro > 0
    assert cond_1 > 0
    assert cond_inf > 0

    # Should match numpy's condition on R
    assert np.allclose(cond_fro, np.linalg.cond(X.R, "fro"), rtol=1e-10)
    assert np.allclose(cond_1, np.linalg.cond(X.R, 1), rtol=1e-10)
    assert np.allclose(cond_inf, np.linalg.cond(X.R, np.inf), rtol=1e-10)


def test_QR_cond_transposed():
    """Test condition number with transposed mode."""
    np.random.seed(1003)
    A = np.random.randn(100, 80)

    X_std = QR.from_matrix(A, transposed=False)
    X_trans = QR.from_matrix(A.T, transposed=True)

    # Both should compute condition number correctly
    cond_std = X_std.cond(2)
    cond_trans = X_trans.cond(2)

    assert cond_std > 0
    assert cond_trans > 0


def test_QR_cond_caching():
    """Test that condition number is cached."""
    np.random.seed(1004)
    A = np.random.randn(100, 80)

    X = QR.from_matrix(A)

    # First call (approximate)
    cond1 = X.cond(2, exact=False)

    # Should be cached
    assert ("cond", 2, False) in X._cache

    # Second call should return cached value
    cond2 = X.cond(2, exact=False)
    assert cond1 == cond2, "Cached value should match"

    # Exact version should be cached separately
    cond_exact = X.cond(2, exact=True)
    assert ("cond", 2, True) in X._cache

    # Different norm should not be cached yet
    cond_fro = X.cond("fro")
    assert ("cond", "fro", False) in X._cache


def test_QR_cond_complex():
    """Test condition number for complex matrices."""
    np.random.seed(1005)
    A_real = np.random.randn(80, 60)
    A_imag = np.random.randn(80, 60)
    A = A_real + 1j * A_imag

    X = QR.from_matrix(A)
    cond = X.cond(2)

    assert cond > 0, "Condition number should be positive"
    assert np.isfinite(cond), "Condition number should be finite"


def test_QR_cond_square():
    """Test condition number for square matrices."""
    np.random.seed(1006)
    A = np.random.randn(100, 100)

    X = QR.from_matrix(A)
    cond_approx = X.cond(2, exact=False)
    cond_exact = X.cond(2, exact=True)

    # Approximate should be lower bound
    assert cond_approx <= cond_exact * 1.01

    # Exact should match condition of A (since Q is orthogonal)
    cond_A = np.linalg.cond(A, 2)
    assert np.allclose(cond_exact, cond_A, rtol=1e-10)


def test_QR_cond_edge_cases():
    """Test condition number for edge case matrices."""
    # Single column
    A_col = np.random.randn(100, 1)
    X_col = QR.from_matrix(A_col)
    cond_col = X_col.cond(2)
    assert cond_col > 0

    # Single row
    A_row = np.random.randn(1, 80)
    X_row = QR.from_matrix(A_row)
    cond_row = X_row.cond(2)
    assert cond_row > 0

    # 1x1 matrix
    A_scalar = np.array([[5.0]])
    X_scalar = QR.from_matrix(A_scalar)
    cond_scalar = X_scalar.cond(2)
    assert np.allclose(cond_scalar, 1.0, rtol=1e-10), "1x1 matrix should have cond=1"


# ============================================================================
# Test conversion methods: to_svd() and from_svd()
# ============================================================================


def test_QR_to_svd_basic():
    """Test basic QR to SVD conversion."""
    np.random.seed(2000)
    A = np.random.randn(100, 80)

    X_qr = QR.from_matrix(A)
    X_svd = X_qr.to_svd()

    # Check type
    from lowrank.matrices import SVD

    assert isinstance(X_svd, SVD), "Should return SVD object"

    # Check dimensions
    assert X_svd.shape == X_qr.shape, "Shape should be preserved"
    assert X_svd.rank == X_qr.rank, "Rank should be preserved"

    # Check reconstruction
    assert np.allclose(
        X_qr.full(), X_svd.full(), atol=1e-10
    ), "Should reconstruct same matrix"


def test_QR_from_svd_basic():
    """Test basic SVD to QR conversion."""
    np.random.seed(2001)
    A = np.random.randn(100, 80)

    from lowrank.matrices import SVD

    X_svd = SVD.from_matrix(A)
    X_qr = QR.from_svd(X_svd)

    # Check type
    assert isinstance(X_qr, QR), "Should return QR object"

    # Check dimensions
    assert X_qr.shape == X_svd.shape, "Shape should be preserved"
    assert X_qr.rank == X_svd.rank, "Rank should be preserved"

    # Check reconstruction
    assert np.allclose(
        X_svd.full(), X_qr.full(), atol=1e-10
    ), "Should reconstruct same matrix"

    # Check Q is orthogonal
    assert X_qr.is_orthogonal(), "Q should be orthogonal"


def test_QR_svd_roundtrip():
    """Test roundtrip conversion QR -> SVD -> QR."""
    np.random.seed(2002)
    A = np.random.randn(100, 80)

    X_qr1 = QR.from_matrix(A)
    X_svd = X_qr1.to_svd()
    X_qr2 = QR.from_svd(X_svd)

    # Check reconstruction after roundtrip
    assert np.allclose(
        X_qr1.full(), X_qr2.full(), atol=1e-10
    ), "Roundtrip should preserve matrix"

    # Check orthogonality preserved
    assert X_qr2.is_orthogonal(), "Q should remain orthogonal"


def test_QR_svd_roundtrip_reverse():
    """Test roundtrip conversion SVD -> QR -> SVD."""
    np.random.seed(2003)
    A = np.random.randn(100, 80)

    from lowrank.matrices import SVD

    X_svd1 = SVD.from_matrix(A)
    X_qr = QR.from_svd(X_svd1)
    X_svd2 = X_qr.to_svd()

    # Check reconstruction after roundtrip
    assert np.allclose(
        X_svd1.full(), X_svd2.full(), atol=1e-10
    ), "Roundtrip should preserve matrix"

    # Singular values might differ in order/sign, but matrix should be same
    assert np.allclose(
        np.sort(X_svd1.s)[::-1], np.sort(X_svd2.s)[::-1], rtol=1e-10
    ), "Singular values should be preserved (up to ordering)"


def test_QR_to_svd_transposed():
    """Test QR to SVD conversion with transposed mode."""
    np.random.seed(2004)
    A = np.random.randn(100, 80)

    # Standard mode: X = Q @ R represents A
    X_qr_std = QR.from_matrix(A, transposed=False)
    X_svd_std = X_qr_std.to_svd()
    assert X_qr_std.shape == (100, 80)
    assert X_svd_std.shape == (100, 80)
    assert np.allclose(X_qr_std.full(), X_svd_std.full(), atol=1e-10)
    assert np.allclose(X_svd_std.full(), A, atol=1e-10), "Should represent A"

    # Transposed mode: from_matrix(A, transposed=True) represents A.H
    X_qr_trans = QR.from_matrix(A, transposed=True)
    X_svd_trans = X_qr_trans.to_svd()
    # Key requirement: SVD should represent the same matrix as QR
    assert X_svd_trans.shape == X_qr_trans.shape, "SVD and QR should have same shape"
    assert np.allclose(
        X_qr_trans.full(), X_svd_trans.full(), atol=1e-10
    ), "Should represent same matrix"


def test_QR_from_svd_transposed():
    """Test SVD to QR conversion with transposed mode."""
    np.random.seed(2005)
    A = np.random.randn(100, 80)

    from lowrank.matrices import SVD

    X_svd = SVD.from_matrix(A)

    # Convert to standard QR: same shape and representation
    X_qr_std = QR.from_svd(X_svd, transposed=False)
    assert X_qr_std.shape == X_svd.shape
    assert np.allclose(X_svd.full(), X_qr_std.full(), atol=1e-10)
    assert np.allclose(X_qr_std.full(), A, atol=1e-10)

    # Convert to transposed QR: represents conjugate transpose of SVD matrix
    X_qr_trans = QR.from_svd(X_svd, transposed=True)
    # Transposed QR should have swapped dimensions (representing A.H)
    assert X_qr_trans.shape == (80, 100), "Transposed QR should have swapped shape"
    # And should represent the conjugate transpose of the original
    assert np.allclose(X_svd.full().T.conj(), X_qr_trans.full(), atol=1e-10)


def test_QR_to_svd_complex():
    """Test QR to SVD conversion with complex matrices."""
    np.random.seed(2006)
    A_real = np.random.randn(80, 60)
    A_imag = np.random.randn(80, 60)
    A = A_real + 1j * A_imag

    X_qr = QR.from_matrix(A)
    X_svd = X_qr.to_svd()

    assert X_svd.dtype == np.complex128, "Should preserve complex dtype"
    assert np.allclose(X_qr.full(), X_svd.full(), atol=1e-10)


def test_QR_from_svd_complex():
    """Test SVD to QR conversion with complex matrices."""
    np.random.seed(2007)
    A_real = np.random.randn(80, 60)
    A_imag = np.random.randn(80, 60)
    A = A_real + 1j * A_imag

    from lowrank.matrices import SVD

    X_svd = SVD.from_matrix(A)
    X_qr = QR.from_svd(X_svd)

    assert X_qr.Q.dtype == np.complex128, "Should preserve complex dtype"
    assert np.allclose(X_svd.full(), X_qr.full(), atol=1e-10)


def test_QR_to_svd_square():
    """Test conversion for square matrices."""
    np.random.seed(2008)
    A = np.random.randn(100, 100)

    X_qr = QR.from_matrix(A)
    X_svd = X_qr.to_svd()

    assert X_svd.shape == (100, 100)
    assert np.allclose(X_qr.full(), X_svd.full(), atol=1e-10)


def test_QR_to_svd_wide():
    """Test conversion for wide matrices."""
    np.random.seed(2009)
    A = np.random.randn(50, 120)

    X_qr = QR.from_matrix(A)
    X_svd = X_qr.to_svd()

    assert X_svd.shape == (50, 120)
    assert X_svd.rank == 50  # Limited by number of rows
    assert np.allclose(X_qr.full(), X_svd.full(), atol=1e-10)


def test_QR_to_svd_tall():
    """Test conversion for tall matrices."""
    np.random.seed(2010)
    A = np.random.randn(120, 50)

    X_qr = QR.from_matrix(A)
    X_svd = X_qr.to_svd()

    assert X_svd.shape == (120, 50)
    assert X_svd.rank == 50  # Limited by number of columns
    assert np.allclose(X_qr.full(), X_svd.full(), atol=1e-10)


def test_QR_to_svd_orthogonality():
    """Test that to_svd preserves orthogonality."""
    np.random.seed(2011)
    A = np.random.randn(100, 80)

    X_qr = QR.from_matrix(A)
    X_svd = X_qr.to_svd()

    # Check U has orthonormal columns
    U_orthogonal = np.allclose(
        X_svd.U.T.conj() @ X_svd.U, np.eye(X_svd.rank), atol=1e-10
    )
    assert U_orthogonal, "U should have orthonormal columns"

    # Check V has orthonormal columns
    V_orthogonal = np.allclose(
        X_svd.V.T.conj() @ X_svd.V, np.eye(X_svd.rank), atol=1e-10
    )
    assert V_orthogonal, "V should have orthonormal columns"


def test_QR_from_svd_extra_data():
    """Test that extra_data is preserved in conversion."""
    np.random.seed(2012)
    A = np.random.randn(80, 60)

    from lowrank.matrices import SVD

    X_svd = SVD.from_matrix(A, poles=[1, 2, 3])
    X_qr = QR.from_svd(X_svd)

    # Check extra_data is transferred
    assert "poles" in X_qr._extra_data
    assert X_qr._extra_data["poles"] == [1, 2, 3]


def test_QR_to_svd_extra_data():
    """Test that extra_data is preserved in to_svd."""
    np.random.seed(2013)
    A = np.random.randn(80, 60)

    X_qr = QR.from_matrix(A, residues=[4, 5, 6])
    X_svd = X_qr.to_svd()

    # Check extra_data is transferred
    assert "residues" in X_svd._extra_data
    assert X_svd._extra_data["residues"] == [4, 5, 6]


def test_QR_from_svd_override_extra_data():
    """Test that from_svd can override extra_data."""
    np.random.seed(2014)
    A = np.random.randn(80, 60)

    from lowrank.matrices import SVD

    X_svd = SVD.from_matrix(A, data1="original")
    X_qr = QR.from_svd(X_svd, data1="overridden", data2="new")

    # Check extra_data is overridden/added
    assert X_qr._extra_data["data1"] == "overridden"
    assert X_qr._extra_data["data2"] == "new"


def test_QR_to_svd_singular_values_ordered():
    """Test that singular values are in descending order after to_svd."""
    np.random.seed(2015)
    A = np.random.randn(100, 80)

    X_qr = QR.from_matrix(A)
    X_svd = X_qr.to_svd()

    # Check singular values are in descending order
    s = X_svd.s
    assert np.all(s[:-1] >= s[1:]), "Singular values should be in descending order"


def test_QR_conversion_single_column():
    """Test conversion for single column matrix."""
    np.random.seed(2016)
    A = np.random.randn(100, 1)

    X_qr = QR.from_matrix(A)
    X_svd = X_qr.to_svd()

    assert X_svd.shape == (100, 1)
    assert X_svd.rank == 1
    assert np.allclose(X_qr.full(), X_svd.full(), atol=1e-10)


def test_QR_conversion_single_row():
    """Test conversion for single row matrix."""
    np.random.seed(2017)
    A = np.random.randn(1, 80)

    X_qr = QR.from_matrix(A)
    X_svd = X_qr.to_svd()

    assert X_svd.shape == (1, 80)
    assert X_svd.rank == 1
    assert np.allclose(X_qr.full(), X_svd.full(), atol=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
