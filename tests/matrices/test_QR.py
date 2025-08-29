# Test file for QR class

#%% Imports
import numpy as np
import scipy.linalg as la
from lowrank import LowRankMatrix, QR

#%% Setup
A = np.random.randn(10, 8)
Q, R = la.qr(A, mode='economic')
X = QR(Q, R)

#%% Test basic operations
def test_QR_basic():
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
    print('Basic operations passed')


#%% Test class method
def test_QR_classmethod():
    # Test the class methods
    assert isinstance(QR.generate_random((10, 8)), QR), "Incorrect type of generate_random"
    assert isinstance(QR.from_matrix(A), QR), "Incorrect type of from_matrix"
    assert np.allclose(QR.from_matrix(A).full(), A), "Incorrect from_matrix"
    Y = LowRankMatrix(Q, R)
    assert np.allclose(QR.from_low_rank(Y).full(), A), "Incorrect from_low_rank"
    print('Class methods passed')


#%% Test addition
def test_QR_addition():
    # Test addition of QRs
    assert isinstance(X + X, QR), "Incorrect addition with QR"
    assert (X+X).rank == X.rank, "Incorrect rank of addition with QR"
    assert np.allclose((X + X).full(), 2 * A), "Incorrect addition with QR"
    assert np.allclose((X + A), 2 * A), "Incorrect addition with ndarray"
    assert np.allclose((X - X).full(), 0 * A), "Incorrect subtraction with QR"
    Y = QR.generate_random((10, 8))
    assert isinstance(X + Y, QR), "Incorrect addition with QR"
    assert np.allclose((X + Y).full(), A + Y.full()), "Incorrect addition with QR"
    assert np.allclose((X - Y).full(), A - Y.full()), "Incorrect subtraction with QR"
    print('Addition of QRs passed')


#%% Test multiplication
def test_QR_multiplication():
    # Test multiplication of QRs
    assert isinstance(X.dot(X.T), QR), "Incorrect multiplication with QR"
    Y = QR.generate_random((8, 12))
    assert isinstance(X.dot(Y), QR), "Incorrect multiplication with QR"
    assert np.allclose(X.dot(Y).full(), A @ Y.full()), "Incorrect multiplication with QR"
    print('Multiplication of QRs passed')

# %%
