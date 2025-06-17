"""
Test file for svd.py

Author: Benjamin Carrel, University of Geneva, 2023
"""


#%% Imports
import numpy as np
import scipy.linalg as la
from lowrank import LowRankMatrix, QuasiSVD, SVD
from numpy import ndarray


#%% QuasiSVD class
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

test_QuasiSVD_basic()

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

test_QuasiSVD_addition()

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

test_QuasiSVD_multiplication()

#%% Test projection
def test_QuasiSVD_projection():
    # Test projection routines
    assert np.allclose(X.project_onto_tangent_space(X).full(), X_full), "Incorrect projection onto tangent space of QuasiSVD"
    assert np.allclose(X.project_onto_tangent_space(X_full).full(), X_full), "Incorrect projection onto tangent space of ndarray"

    print('QuasiSVD Projection routines passed')

test_QuasiSVD_projection()

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

test_QuasiSVD_hadamard()

#%% SVD class
#%% Setup SVD
np.random.seed(1234)
A = np.random.randn(10, 5)
B = np.random.randn(5, 8)
X_full = A @ B
X = SVD.truncated_svd(X_full)

#%% Test Basic operations
def test_SVD_basic():
    # Test dimensions
    assert X.deepshape == (10, 5, 5, 8), "Incorrect deepshape"
    assert X.shape == (10, 8), "Incorrect shape"
    assert X.ndim == 2, "Incorrect ndim"
    assert X.rank == 5, "Incorrect rank"
    # Test norms
    assert X.norm('fro') - la.norm(X_full, 'fro') < 1e-12, "Incorrect Frobenius norm"
    assert X.norm('nuc') - la.norm(X_full, 'nuc') < 1e-12, "Incorrect nuclear norm"
    assert X.norm(1) - la.norm(X_full, 1) < 1e-12, "Incorrect 1-norm"
    assert X.norm(2) - la.norm(X_full, 2) < 1e-12, "Incorrect 2-norm"
    # Test misc
    assert np.allclose(X.full(), X_full), "Incorrect full() method"
    assert X.gather([1,3]) - X_full[1,3] < 1e-12, "Incorrect gather"
    assert X.is_symmetric() == False, "Incorrect is_symmetric"
    print('SVD Basic operations passed')

test_SVD_basic()

# %% Test class methods
def test_SVD_class_methods():
    # Test the class methods
    assert isinstance(SVD.generate_random((10, 8), np.asarray([4, 3, 2, 1])), SVD), "Incorrect type of generate_random"
    assert isinstance(SVD.truncated_svd(X_full), SVD), "Incorrect type of truncated_svd"
    assert SVD.truncated_svd(X_full).rank == 5, "Incorrect rank of truncated_svd"
    assert SVD.reduced_svd(X_full).rank == 8, "Incorrect rank of reduced_svd"
    assert SVD.full_svd(X_full).rank == 8, "Incorrect rank of full_svd"
    assert np.allclose(SVD.from_matrix(X_full).full(), X_full), "Incorrect from_matrix"
    Y = LowRankMatrix(A, B)
    assert np.allclose(SVD.from_low_rank(Y).full(), X_full), "Incorrect from_low_rank"
    # Randomized SVD requires an exponential decay
    Y = SVD.generate_random((50, 40), np.logspace(0,-10, 20))
    assert np.allclose(SVD.randomized_svd(Y.full(), 10, 10).full(), Y.truncate(10).full()), "Incorrect randomized_svd"
    # Randomized SVD supports LowRankMatrix input
    Y_bis = LowRankMatrix(Y.U, Y.S, Y.Vt)
    assert np.allclose(SVD.randomized_svd(Y_bis, 10, 10).full(), Y.truncate(10).full()), "Incorrect randomized_svd and LowRankMatrix"
    print('SVD class methods passed')

test_SVD_class_methods()

#%% Test addition
def test_SVD_addition():
    # Test addition of SVDs
    assert isinstance(X + X, SVD), "Incorrect addition with SVD"
    assert np.allclose((X + X).full(), 2 * X_full), "Incorrect addition with SVD"
    assert np.allclose((X + X_full), 2 * X_full), "Incorrect addition with ndarray"
    assert (X-X).rank == 0, "Incorrect subtraction with SVD"
    assert np.allclose((X - X).full(), 0 * X_full), "Incorrect subtraction with SVD"
    Y = SVD.generate_random((10, 8), np.asarray([4, 3, 2, 1])) # Try with a different rank
    assert isinstance(X + Y, SVD), "Incorrect addition with SVD"
    assert np.allclose((X + Y).full(), X_full + Y.full()), "Incorrect addition with SVD"
    assert np.allclose((X - Y).full(), X_full - Y.full()), "Incorrect subtraction with SVD"
    assert (X + Y).rank == 8, "Incorrect rank of addition with SVD"
    print('Addition of SVDs passed')

test_SVD_addition()

#%% Test multiplication
def test_SVD_multiplication():
    # Test multiplication of SVDs
    assert isinstance(X.dot(X.T), SVD), "Incorrect multiplication with SVD"
    assert X.dot(X.T).rank == 5, "Incorrect rank of multiplication with SVD"
    Y = SVD.generate_random((8, 12), np.asarray([4, 3, 2, 1])) # Try with a different rank
    assert isinstance(X.dot(Y), SVD), "Incorrect multiplication with SVD"
    assert X.dot(Y).rank == min(X.rank, Y.rank), "Incorrect rank of multiplication with SVD"
    assert np.allclose(X.dot(Y).full(), X_full @ Y.full()), "Incorrect multiplication with SVD"
    print('Multiplication of SVDs passed')

test_SVD_multiplication()

#%% Test truncated SVD
def test_truncated_SVD():
    # Specific tests for the truncated SVD
    X = SVD.generate_random((50, 50), np.logspace(0,-10, 20))
    X1 = X.truncate(r=10)
    assert X1.rank == 10, "Incorrect rank of truncated SVD"
    X2 = X.truncate_perpendicular(r=10)
    assert X2.rank == 10, "Incorrect rank of truncated SVD"
    assert np.allclose((X1 + X2).full(), X.full()), "Incorrect addition of truncated SVDs"
    X1_bis = X.truncate(rtol=1e-5)
    assert min(X1_bis.sing_vals) > 1e-5, "Incorrect rtol truncation"
    X2_bis = X.truncate_perpendicular(rtol=1e-5)
    assert max(X2_bis.sing_vals) < 1e-5, "Incorrect rtol truncation"
    assert np.allclose((X1_bis + X2_bis).full(), X.full()), "Incorrect addition of truncated SVDs"
    print('Truncation of SVDs passed')

test_truncated_SVD()

# %% Test SVD Hadamard product
def test_SVD_hadamard():
    np.random.seed(0)
    rank = 3
    A = np.random.randn(20, 4)
    B = np.random.randn(4, 18)
    Q1, _ = la.qr(A, mode='economic')
    Q2, _ = la.qr(B.T, mode='economic')
    S = np.diag(np.random.rand(4))
    X = QuasiSVD(Q1, S, Q2)
    Y = SVD.generate_random((20, 18), np.logspace(0,-10, rank))
    Y_full = Y.full()
    # SVD-SVD Hadamard product
    assert isinstance(Y.hadamard(Y), QuasiSVD), "Incorrect Hadamard product with SVD"
    assert Y.hadamard(Y).rank == 2*rank, "Incorrect rank of Hadamard product with SVD"
    assert np.allclose(Y.hadamard(Y).full(), Y_full**2), "Incorrect Hadamard product with SVD"
    print('SVD-SVD Hadamard product passed')
    # SVD-ndarray Hadamard product
    assert isinstance(Y.hadamard(Y_full), ndarray), "Incorrect Hadamard product with ndarray"
    assert np.allclose(Y.hadamard(Y_full), Y_full**2), "Incorrect Hadamard product with ndarray"
    print('SVD-ndarray Hadamard product passed')
    # SVD-QuasiSVD Hadamard product
    assert isinstance(Y.hadamard(X), QuasiSVD), "Incorrect Hadamard product with QuasiSVD"
    assert Y.hadamard(X).rank == Y.rank * X.rank, "Incorrect rank of Hadamard product with QuasiSVD"
    assert np.allclose(Y.hadamard(X).full(), Y_full * X.full()), "Incorrect Hadamard product with QuasiSVD"
    print('SVD-QuasiSVD Hadamard product passed')
    # QuasiSVD-SVD Hadamard product
    assert isinstance(X.hadamard(Y), QuasiSVD), "Incorrect Hadamard product with QuasiSVD"
    assert np.allclose(X.hadamard(Y).full(), X.full() * Y_full), "Incorrect Hadamard product with QuasiSVD"
    print('QuasiSVD-SVD Hadamard product passed')

test_SVD_hadamard()

#%% Test SVD with complex values
def test_SVD_complex():
    np.random.seed(0)
    rank = 3
    A = np.random.randn(20, 4) + 1j * np.random.randn(20, 4)
    B = np.random.randn(4, 18) + 1j * np.random.randn(4, 18)
    YA = SVD.from_dense(A)
    YB = SVD.from_dense(B)
    # Check that rank is correct
    assert YA.rank == 4, "Incorrect rank of complex SVD"
    assert YB.rank == 4, "Incorrect rank of complex SVD"
    # Check that full matrix is correct
    assert np.allclose(YA.full(), A), "Incorrect full matrix of complex SVD"
    assert np.allclose(YB.full(), B), "Incorrect full matrix of complex SVD"
    # Check that singular values are correct
    assert np.allclose(YA.sing_vals, la.svd(A, compute_uv=False)), "Incorrect singular values of complex SVD"
    assert np.allclose(YB.sing_vals, la.svd(B, compute_uv=False)), "Incorrect singular values of complex SVD"
    # Check that dot product is correct
    assert np.allclose(YA.dot(YB).full(), A @ B), "Incorrect dot product of complex SVD"
    # Check that addition is correct
    assert np.allclose((YA + YA).full(), A + A), "Incorrect addition of complex SVD"
    assert np.allclose((YB + YB).full(), B + B), "Incorrect addition of complex SVD"
    # Check that subtraction is correct
    assert np.allclose((YA - YA).full(), A - A), "Incorrect subtraction of complex SVD"
    # Check that scalar multiplication is correct
    assert np.allclose((YA * 2).full(), 2 * A), "Incorrect scalar multiplication of complex SVD"
    # Check that complex multiplication is correct
    assert np.allclose((YA * 1j).full(), 1j * A), "Incorrect complex multiplication of complex SVD"
    # Check that Hadamard product is correct
    assert np.allclose(YA.hadamard(YA).full(), A * A), "Incorrect Hadamard product of complex SVD"
    assert np.allclose(YB.hadamard(YB).full(), B * B), "Incorrect Hadamard product of complex SVD"
    print('Complex SVD passed')

test_SVD_complex()


# %% Test randomized SVD
def test_randomized_SVD():
    np.random.seed(0)

    # Test with a small random matrix
    rank = 4
    oversample = 5
    A = np.random.randn(20, rank)
    B = np.random.randn(rank, 18)
    X_full = A @ B
    X = SVD.truncated_svd(X_full)
    # Randomized SVD
    assert np.allclose(SVD.randomized_svd(X_full, rank, oversample).full(), X.truncate(rank).full()), "Incorrect randomized SVD"
    # error = np.linalg.norm(X_full - SVD.randomized_svd(X_full, rank, oversample).full(), 'fro') / np.linalg.norm(X_full, 'fro')
    # print('Relative error of randomized SVD:', error)
    print('Small randomized SVD passed')

    # Test with a larger random matrix
    effective_rank = 10
    Y = SVD.generate_random((500, 400), np.logspace(0,-14, effective_rank))
    # Randomized SVD
    approx_rank = 10
    oversample = 10
    assert np.allclose(SVD.randomized_svd(Y, approx_rank, oversample).full(), Y.truncate(effective_rank).full()), "Incorrect randomized SVD"
    error = np.linalg.norm(Y.truncate(effective_rank) - SVD.randomized_svd(Y, approx_rank, oversample).full(), 'fro') / np.linalg.norm(Y.full(), 'fro')
    print('Relative error of randomized SVD:', error)
    print('Large randomized SVD passed')

    # Test with a larger random matrix and subspace iteration
    approx_rank = 10
    oversample = 10
    nb_iter = 2
    assert np.allclose(SVD.randomized_svd(Y, approx_rank, oversample, nb_iter).full(), Y.truncate(approx_rank).full()), "Incorrect randomized SVD with subspace iteration"
    error = np.linalg.norm(Y.truncate(approx_rank) - SVD.randomized_svd(Y, approx_rank, oversample, nb_iter).full(), 'fro') / np.linalg.norm(X_full, 'fro')
    print('Relative error of randomized SVD with subspace iteration:', error)
    print('Randomized SVD with subspace iteration passed')

test_randomized_SVD()

#%% Test generalized Nystroem
def test_generalized_nystroem():
    # Test with a small random matrix
    rank = 4
    A = np.random.randn(20, rank)
    B = np.random.randn(rank, 18)
    X_full = A @ B
    X = SVD.truncated_svd(X_full)

    # Small plain GN
    approx_rank = 4
    oversamplings = (2,4)
    assert np.allclose(SVD.generalized_nystroem(X, approx_rank, oversamplings).full(), X.truncate(rank).full()), "Incorrect generalized Nystroem"
    error = np.linalg.norm(X_full - SVD.generalized_nystroem(X_full, approx_rank, oversamplings).full(), 'fro') / np.linalg.norm(X_full, 'fro')
    print('Relative error of plain GN:', error)
    print('Small GN passed')

    # Small stable GN
    approx_rank = 4
    oversamplings = (2,4)
    epsilon = 1e-12
    assert np.allclose(SVD.generalized_nystroem(X, approx_rank, oversamplings, epsilon).full(), X.truncate(rank).full()), "Incorrect generalized Nystroem"
    error = np.linalg.norm(X_full - SVD.generalized_nystroem(X_full, approx_rank, oversamplings, epsilon).full(), 'fro') / np.linalg.norm(X_full, 'fro')
    print('Relative error of stable GN:', error)
    print('Small GN passed')

    # Test with large random matrix
    effective_rank = 10
    Y = SVD.generate_random((500, 400), np.logspace(0,-14, effective_rank))
    # Plain GN
    approx_rank = 10
    oversamplings = (2, 7)
    assert np.allclose(SVD.generalized_nystroem(Y, approx_rank, oversamplings).full(), Y.truncate(effective_rank).full()), "Incorrect plain GN for large matrices"
    error = np.linalg.norm(Y.truncate(effective_rank) - SVD.generalized_nystroem(Y, approx_rank, oversamplings).full(), 'fro') / np.linalg.norm(Y.full(), 'fro')
    print('Relative error of plain GN:', error)
    print('Large plain GN passed')

    # Test with a larger random matrix and subspace iteration
    approx_rank = 10
    oversamplings = (2, 7)
    epsilon = 1e-12
    assert np.allclose(SVD.generalized_nystroem(Y, approx_rank, oversamplings, epsilon).full(), Y.truncate(approx_rank).full()), "Incorrect stable GN"
    error = np.linalg.norm(Y.truncate(approx_rank) - SVD.generalized_nystroem(Y, approx_rank, oversamplings, epsilon).full(), 'fro') / np.linalg.norm(X_full, 'fro')
    print('Relative error of stable GN:', error)
    print('Large stable GN passed')



test_generalized_nystroem()

# %%
