"""
Test file for QuasiSVD class defined in low_rank_toolbox/matrices/quasi_svd.py

Author: Benjamin Carrel, University of Geneva, 2023
"""

#%% Imports
import numpy as np
import scipy.linalg as la
from lowrank import LowRankMatrix, QuasiSVD, SVD
from numpy import ndarray

#%% Setup Quasi-SVD
np.random.seed(1234)
A = np.random.randn(20, 4)
B = np.random.randn(4, 18)
Q1, _ = la.qr(A, mode='economic')
Q2, _ = la.qr(B.T, mode='economic')
S = np.diag(np.random.rand(4))
X = QuasiSVD(Q1, S, Q2)
X_full = Q1 @ S @ Q2.T

#%% Test Basic operations
def test_QuasiSVD_basic():
    # Test dimensions
    assert X.deepshape == (20, 4, 4, 18), "Incorrect deepshape"
    assert X.shape == (20, 18), "Incorrect shape"
    assert X.ndim == 2, "Incorrect ndim"
    # Test norms
    assert X.norm('fro') - la.norm(X_full, 'fro') < 1e-12, "Incorrect Frobenius norm"
    assert X.norm('nuc') - la.norm(X_full, 'nuc') < 1e-12, "Incorrect nuclear norm"
    assert X.norm(1) - la.norm(X_full, 1) < 1e-12, "Incorrect 1-norm"
    assert X.norm(2) - la.norm(X_full, 2) < 1e-12, "Incorrect 2-norm"
    # Test misc
    assert np.allclose(X.full(), X_full), "Incorrect full() method"
    assert X.check_orthogonality, "Incorrect check_orthogonality"
    print('Basic QuasiSVD operations passed')


#%% Test addition
def test_QuasiSVD_addition():
    # Test addition
    assert isinstance(X + X, SVD), "Incorrect addition with QuasiSVD"
    assert (X+X).rank == X.rank, "Incorrect rank of addition of QuasiSVD"
    assert (X+X).check_orthogonality, "Addition of QuasiSVD loses orthogonality"
    assert np.allclose((X + X).full(), 2 * X_full), "Incorrect addition with QuasiSVD"
    assert np.allclose((X + X_full), 2 * X_full), "Incorrect addition with ndarray"
    assert np.allclose((X - X).full(), 0 * X_full), "Incorrect subtraction with QuasiSVD"
    assert (X-X).rank == 0, "Incorrect rank of subtraction of QuasiSVD"
    assert np.allclose((X - X_full), 0 * X_full), "Incorrect subtraction with ndarray"
    assert np.allclose((10 * X).full(), 10 * X_full), "Incorrect scalar multiplication"
    Y = SVD.generate_random((20, 18), np.asarray([4, 3, 2, 1]))
    assert np.allclose((X+Y).full(), (Y+X).full()), "Addition of QuasiSVD and SVD is not commutative"
    assert isinstance(X+Y, SVD), "Addition of QuasiSVD and SVD is not SVD"
    assert isinstance(Y+X, SVD), "Addition of QuasiSVD and SVD is not SVD"
    print('QuasiSVD Addition passed')


#%% Test multiplication
def test_QuasiSVD_multiplication():
        # Test matrix-vector multiplication
    v = np.random.randn(18)
    assert (X.dot(v)).shape == (20,), "Incorrect shape of matrix-vector product"
    assert np.allclose(X.dot(v, dense_output=True), X_full @ v), "Incorrect matrix-vector multiplication"
    print('Matrix-vector multiplication passed')

    # Test QuasiSVD-matrix multiplication
    Y = np.random.randn(18, 17)
    assert (X.dot(Y)).shape == (20, 17), "Incorrect shape of matrix-matrix product"
    assert (X.dot(Y)).deepshape == (20, 4, 4, 18, 17), "Incorrect deepshape of matrix-matrix product"
    assert np.allclose(X.dot(Y, dense_output=True), X_full @ Y), "Incorrect matrix-matrix multiplication"
    assert not isinstance(X.dot(Y), QuasiSVD), "Incorrect type of QuasiSVD-matrix product"
    assert isinstance(X.dot(Y), LowRankMatrix), "Incorrect type of QuasiSVD-matrix product"
    print('QuasiSVD-matrix multiplication passed')

    # Test QuasiSVD-QuasiSVD multiplication
    assert isinstance(X.dot(X.T), SVD), "Incorrect type of QuasiSVD-QuasiSVD product"
    assert np.allclose(X.dot(X.T).full(), X_full @ X_full.T), "Incorrect QuasiSVD-QuasiSVD product"
    assert X.dot(X.T).rank == X.rank, "Incorrect rank of QuasiSVD-QuasiSVD product"
    Y = SVD.generate_random((18, 17), np.asarray([3, 2, 1]))
    assert isinstance(X.dot(Y), SVD), "Incorrect type of QuasiSVD-SVD product"
    assert np.allclose(X.dot(Y).full(), X_full @ Y.full()), "Incorrect QuasiSVD-SVD product"
    assert X.dot(Y).rank == min(X.rank, Y.rank), "Incorrect rank of QuasiSVD-SVD product"
    print('QuasiSVD-QuasiSVD multiplication passed')

    # Test QuasiSVD-LowRankMatrix multiplication
    Y = LowRankMatrix(np.random.randn(18, 5), np.random.randn(5, 6), np.random.randn(6, 4), np.random.randn(4, 17))
    assert (X.dot(Y)).shape == (20, 17), "Incorrect shape of SVD-LowRankMatrix product"
    assert (X.dot(Y)).deepshape == (20, 4, 4, 18, 5, 6, 4, 17), "Incorrect deepshape of SVD-LowRankMatrix product"
    assert np.allclose(X.dot(Y, dense_output=True), X_full @ Y.full()), "Incorrect SVD-LowRankMatrix multiplication"
    assert not isinstance(X.dot(Y), QuasiSVD), "Incorrect type of SVD-LowRankMatrix product"
    assert isinstance(X.dot(Y), LowRankMatrix), "Incorrect type of SVD-LowRankMatrix product"
    print('QuasiSVD-LowRankMatrix multiplication passed')


#%% Test projection
def test_QuasiSVD_projection():
    # Test projection routines
    assert np.allclose(X.project_onto_tangent_space(X).full(), X_full), "Incorrect projection onto tangent space of QuasiSVD"
    assert np.allclose(X.project_onto_tangent_space(X_full).full(), X_full), "Incorrect projection onto tangent space of ndarray"

    print('QuasiSVD Projection routines passed')


#%% Test Hadamard product
def test_QuasiSVD_hadamard():
    # Test Hadamard product
    X_hadamard_ref = X_full * X_full
    # QuasiSVD-QuasiSVD Hadamard product
    assert np.allclose(X.hadamard(X).full(), X_hadamard_ref), "Incorrect Hadamard product"
    assert isinstance(X.hadamard(X), QuasiSVD), "Incorrect type of Hadamard product"
    print('QuasiSVD-QuasiSVD Hadamard product passed')
    # QuasiSVD-ndarray Hadamard product
    assert np.allclose(X.hadamard(X_full), X_hadamard_ref), "Incorrect Hadamard product"
    assert isinstance(X.hadamard(X_full), ndarray), "Incorrect type of Hadamard product"
    print('QuasiSVD-dense Hadamard product passed')
    # QuasiSVD-LowRankMatrix Hadamard product
    Y = LowRankMatrix(np.random.randn(20, 5), np.random.randn(5, 6), np.random.randn(6, 4), np.random.randn(4, 18))
    assert isinstance(X.hadamard(Y), ndarray), "Incorrect type of Hadamard product"
    assert np.allclose(X.hadamard(Y), X_full * Y.full()), "Incorrect Hadamard product"
    print('QuasiSVD-LowRankMatrix Hadamard product passed')