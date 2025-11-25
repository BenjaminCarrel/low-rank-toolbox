# Test file for QR class

import numpy as np
import scipy.linalg as la
from lowrank import LowRankMatrix, QR
import pytest


@pytest.fixture
def qr_basic():
    """Basic QR fixture for testing."""
    np.random.seed(42)
    A = np.random.randn(10, 8)
    Q, R = la.qr(A, mode='economic')
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
    assert X.norm('fro') - la.norm(A, 'fro') < 1e-12, "Incorrect Frobenius norm"
    assert X.norm(1) - la.norm(A, 1) < 1e-12, "Incorrect 1-norm"
    assert X.norm(2) - la.norm(A, 2) < 1e-12, "Incorrect 2-norm"
    # Test misc
    assert np.allclose(X.full(), A), "Incorrect full() method"
    assert X.gather([1,3]) - A[1,3] < 1e-12, "Incorrect gather"


# Test class method
def test_QR_classmethod(qr_basic):
    """Test QR class methods."""
    X, A, Q, R = qr_basic
    # Test the class methods
    assert isinstance(QR.generate_random((10, 8)), QR), "Incorrect type of generate_random"
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
    assert (X+X).rank == 8, "Incorrect rank of addition with QR"
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
    assert np.allclose(X.dot(Y).full(), A @ Y.full()), "Incorrect multiplication with QR"


# Test conjugate/transpose features
def test_QR_conjugate_initialization():
    """Test conjugate initialization (equivalent to transpose for real matrices)."""
    np.random.seed(100)
    A_test = np.random.randn(10, 8)
    Q_test, R_test = la.qr(A_test, mode='economic')
    
    # Standard initialization
    X_std = QR(Q_test, R_test)
    assert X_std.shape == (10, 8), "Standard shape incorrect"
    assert np.allclose(X_std.full(), A_test), "Standard reconstruction incorrect"
    
    # Conjugate initialization (should transpose for real matrices)
    X_conj = QR(Q_test, R_test, conjugate=True)
    assert X_conj.shape == (8, 10), "Conjugate shape incorrect"
    assert np.allclose(X_conj.full(), A_test.T), "Conjugate reconstruction incorrect"
    

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
    assert np.allclose(X_back.full(), A_test), "Double transpose reconstruction incorrect"


def test_QR_conjugate_transpose_complex():
    """Test .H property for conjugate transpose on complex matrices."""
    np.random.seed(102)
    A_complex = np.random.randn(10, 8) + 1j * np.random.randn(10, 8)
    X_complex = QR.from_matrix(A_complex)
    
    X_H = X_complex.H
    assert X_H.shape == (8, 10), "Conjugate transpose shape incorrect"
    assert np.allclose(X_H.full(), A_complex.T.conj()), "Conjugate transpose incorrect"


def test_QR_from_matrix_conjugate():
    """Test from_matrix with conjugate=True."""
    np.random.seed(103)
    A_test = np.random.randn(10, 8)
    X_conj = QR.from_matrix(A_test, conjugate=True)
    
    assert X_conj.shape == (8, 10), "from_matrix conjugate shape incorrect"
    assert np.allclose(X_conj.full(), A_test.T), "from_matrix conjugate reconstruction incorrect"


def test_QR_generate_random_conjugate():
    """Test generate_random with conjugate=True."""
    X_rand_conj = QR.generate_random((10, 8), conjugate=True)
    
    assert X_rand_conj.shape == (8, 10), "generate_random conjugate shape incorrect"
    assert X_rand_conj._conjugate == True, "conjugate flag not set"


def test_QR_real_transpose_hermitian_equivalence():
    """For real matrices, .T and .H should be equivalent."""
    np.random.seed(104)
    A_test = np.random.randn(10, 8)
    X_test = QR.from_matrix(A_test)
    
    X_T = X_test.T
    X_H = X_test.H
    
    assert np.allclose(X_T.full(), X_H.full()), ".T and .H should be equal for real matrices"


def test_QR_Q_R_properties():
    """Test Q and R properties in both standard and conjugate modes."""
    np.random.seed(105)
    A_test = np.random.randn(10, 8)
    Q_test, R_test = la.qr(A_test, mode='economic')
    
    # Standard mode
    X_std = QR(Q_test, R_test)
    assert X_std.Q.shape == (10, 8), "Q shape incorrect in standard mode"
    assert X_std.R.shape == (8, 8), "R shape incorrect in standard mode"
    assert np.allclose(X_std.Q, Q_test), "Q values incorrect"
    assert np.allclose(X_std.R, R_test), "R values incorrect"
    
    # Conjugate mode
    X_conj = QR(Q_test, R_test, conjugate=True)
    assert X_conj.Q.shape == (10, 8), "Q shape incorrect in conjugate mode"
    assert X_conj.R.shape == (8, 8), "R shape incorrect in conjugate mode"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
