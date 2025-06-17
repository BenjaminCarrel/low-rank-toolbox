# Test file for LowRankMatrix class

#%% Imports
import numpy as np
import scipy.linalg as la
from lowrank import LowRankMatrix


#%% Setup
np.random.seed(1234)
A = np.random.randn(10, 5)
B = np.random.randn(5, 6)
C = np.random.randn(6, 4)
D = np.random.randn(4, 8)
X = LowRankMatrix(A, B, C, D)
X_full = np.linalg.multi_dot([A, B, C, D])

#%% Test Basic operations
def test_LowRankMatrix_basic():
    # Test shapes and rank
    assert X.deepshape == (10, 5, 6, 4, 8), "Incorrect deepshape"
    assert X.shape == (10, 8), "Incorrect shape"
    assert X.ndim == 2, "Incorrect ndim"
    assert X.rank == 4, "Incorrect rank"
    # Test norms
    assert X.norm('fro') - la.norm(X_full, 'fro') < 1e-12, "Incorrect Frobenius norm"
    assert X.norm('nuc') - la.norm(X_full, 'nuc') < 1e-12, "Incorrect nuclear norm"
    assert X.norm(1) - la.norm(X_full, 1) < 1e-12, "Incorrect 1-norm"
    assert X.norm(2) - la.norm(X_full, 2) < 1e-12, "Incorrect 2-norm" 
    # Test transpose
    assert X.T.shape == (8, 10), "Incorrect shape of transpose"
    assert X.T.deepshape == (8, 4, 6, 5, 10), "Incorrect deepshape of transpose"
    assert np.allclose(X.T.full(), X_full.T), "Incorrect transpose"
    # Test misc
    assert np.allclose(X.full(), X_full), "Incorrect full() method"
    assert X.gather([1,3]) - X_full[1,3] < 1e-12, "Incorrect gather"
    assert X.is_symmetric() == False, "Incorrect is_symmetric"
    print('Basic operations passed')

test_LowRankMatrix_basic()

#%% Test addition
def test_LowRankMatrix_addition():
    # Test addition
    assert np.allclose((X + X), 2 * X_full), "Incorrect addition with LowRankMatrix"
    assert np.allclose((X + X_full), 2 * X_full), "Incorrect addition with ndarray"
    assert np.allclose((X - X), 0 * X_full), "Incorrect subtraction with LowRankMatrix"
    assert np.allclose((X - X_full), 0 * X_full), "Incorrect subtraction with ndarray"  
    assert np.allclose((10 * X).full(), 10 * X_full), "Incorrect scalar multiplication"
    print('Addition passed')

test_LowRankMatrix_addition()

#%% Test multiplication
def test_LowRankMatrix_multiplication():
    # Test matrix-vector multiplication
    v = np.random.randn(8)
    assert (X.dot(v)).shape == (10,), "Incorrect shape of matrix-vector product"
    assert np.allclose(X.dot(v, dense_output=True), A @ (B @ (C @ (D @ v)))), "Incorrect matrix-vector multiplication"
    print('Matrix-vector multiplication passed')

    # Test matrix-matrix multiplication
    Y = np.random.randn(8, 7)
    assert (X.dot(Y)).shape == (10, 7), "Incorrect shape of matrix-matrix product"
    assert (X.dot(Y)).deepshape == (10, 5, 6, 4, 8, 7), "Incorrect deepshape of matrix-matrix product"
    assert np.allclose(X.dot(Y, dense_output=True), A @ (B @ (C @ (D @ Y)))), "Incorrect matrix-matrix multiplication"
    Y = LowRankMatrix(np.random.randn(8, 5), np.random.randn(5, 6), np.random.randn(6, 4), np.random.randn(4, 7))
    assert (X.dot(Y)).shape == (10, 7), "Incorrect shape of matrix-matrix product"
    assert (X.dot(Y)).deepshape == (10, 5, 6, 4, 8, 5, 6, 4, 7), "Incorrect deepshape of matrix-matrix product"
    print('Matrix-matrix multiplication passed')

test_LowRankMatrix_multiplication()

# %%
