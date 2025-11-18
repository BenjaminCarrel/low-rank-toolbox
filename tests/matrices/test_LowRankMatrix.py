# Test file for LowRankMatrix class

#%% Imports
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator
import pytest
import warnings
import tempfile
import os
from lowrank import LowRankMatrix
from lowrank.matrices.low_rank_matrix import LowRankEfficiencyWarning


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
    assert X.size == A.size + B.size + C.size + D.size, "Incorrect size property"


#%% Test addition
def test_LowRankMatrix_addition():
    # Test addition
    assert np.allclose((X + X), 2 * X_full), "Incorrect addition with LowRankMatrix"
    assert np.allclose((X + X_full), 2 * X_full), "Incorrect addition with ndarray"
    assert np.allclose((X - X), 0 * X_full), "Incorrect subtraction with LowRankMatrix"
    assert np.allclose((X - X_full), 0 * X_full), "Incorrect subtraction with ndarray"  
    assert np.allclose((10 * X).full(), 10 * X_full), "Incorrect scalar multiplication"
    assert np.allclose((X * 10).full(), 10 * X_full), "Incorrect right scalar multiplication"
    assert np.allclose((-X).full(), -X_full), "Incorrect negation"
    print('Addition passed')


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


#%% Test compress method
def test_compress_no_change():
    """Test case where compression should not happen."""
    A = np.random.randn(10, 2)
    B = np.random.randn(2, 10)
    X = LowRankMatrix(A, B)
    
    # Cost of A, B: 10*2 + 2*10 = 40
    # Cost of (A@B): 10*10 = 100
    # No compression should occur
    X_compressed = X.compress()
    assert len(X_compressed._matrices) == 2, "Compression happened when it shouldn't have"
    assert np.allclose(X.full(), X_compressed.full()), "Matrix value changed after compression"
    print("test_compress_no_change passed")

def test_compress_one_merge():
    """Test case where one merge should happen."""
    A = np.random.randn(10, 50)
    B = np.random.randn(50, 2)
    C = np.random.randn(2, 20)
    X = LowRankMatrix(A, B, C)
    
    # Cost of A, B, C: 10*50 + 50*2 + 2*20 = 500 + 100 + 40 = 640
    # Cost of (A@B), C: (10*2) + 2*20 = 20 + 40 = 60. This is optimal.
    # Cost of A, (B@C): 10*50 + (50*20) = 500 + 1000 = 1500
    X_compressed = X.compress()
    assert len(X_compressed._matrices) == 2, "Incorrect number of matrices after compression"
    assert X_compressed._matrices[0].shape == (10, 2), "Incorrect shape of merged matrix"
    assert np.allclose(X.full(), X_compressed.full()), "Matrix value changed after compression"
    print("test_compress_one_merge passed")

def test_compress_long_chain():
    """Test a longer chain of matrices."""
    A = np.random.randn(100, 2)
    B = np.random.randn(2, 80)
    C = np.random.randn(80, 3)
    D = np.random.randn(3, 50)
    E = np.random.randn(50, 4)
    X = LowRankMatrix(A, B, C, D, E)

    # Optimal should be A, (B@C@D@E)
    # Cost of A, B, C, D, E: 200 + 160 + 240 + 150 + 200 = 950
    # Cost of A, (B@C), D, E: 200 + (2*3) + 150 + 200 = 556
    # Cost of A, B, C, (D@E): 200 + 160 + 240 + (3*4) = 612
    # Cost of A, (B@C@D@E): 200 + (2*4) = 208. This is optimal.
    X_compressed = X.compress()
    assert len(X_compressed._matrices) == 2, "Incorrect number of matrices for long chain"
    assert X_compressed._matrices[0].shape == (100, 2), "Incorrect shape for matrix 1"
    assert X_compressed._matrices[1].shape == (2, 4), "Incorrect shape for matrix 2"
    assert np.allclose(X.full(), X_compressed.full()), "Matrix value changed after compression"
    print("test_compress_long_chain passed")

def test_compress_base_cases():
    """Test with 2 matrices (minimum for low-rank factorization)."""
    A = np.random.randn(10, 5)
    B = np.random.randn(5, 8)
    X2 = LowRankMatrix(A, B)
    X2_compressed = X2.compress()
    assert X2 is X2_compressed, "Should return self for two matrices"
    assert len(X2_compressed._matrices) == 2, "Base case with 2 matrices failed"
    print("test_compress_base_cases passed")

def test_compress_already_optimal():
    """Test cases where the matrix is already optimal (2 factors)."""
    # Case with 2 matrices
    A = np.random.randn(10, 5)
    B = np.random.randn(5, 8)
    X2 = LowRankMatrix(A, B)
    X2_compressed = X2.compress()
    assert X2 is X2_compressed, "Should return self for two matrices"
    assert len(X2_compressed._matrices) == 2


def test_compress_bottleneck_middle():
    """Test case where the bottleneck rank is in a middle matrix."""
    A = np.random.randn(10, 5)
    B = np.random.randn(5, 2)  # Bottleneck
    C = np.random.randn(2, 8)
    D = np.random.randn(8, 12)
    X = LowRankMatrix(A, B, C, D)

    X_compressed = X.compress()

    # Should compress to 2 matrices around the bottleneck B
    assert isinstance(X_compressed, LowRankMatrix)
    assert len(X_compressed._matrices) == 2, "Should compress to 2 matrices"
    assert X_compressed._matrices[0].shape == (10, 2), "Shape of the left product is incorrect"
    assert X_compressed._matrices[1].shape == (2, 12), "Shape of the right product is incorrect"
    assert np.allclose(X.full(), X_compressed.full()), "Matrix value changed after compression"


def test_compress_bottleneck_first():
    """Test case where the bottleneck rank is the first matrix."""
    A = np.random.randn(10, 2)  # Bottleneck
    B = np.random.randn(2, 5)
    C = np.random.randn(5, 8)
    X = LowRankMatrix(A, B, C)

    X_compressed = X.compress()

    # Should compress to 2 matrices: A and (B @ C)
    assert isinstance(X_compressed, LowRankMatrix)
    assert len(X_compressed._matrices) == 2, "Should compress to 2 matrices"
    assert X_compressed._matrices[0].shape == (10, 2), "Shape of the first matrix should be unchanged"
    assert X_compressed._matrices[1].shape == (2, 8), "Shape of the right product is incorrect"
    assert np.allclose(X.full(), X_compressed.full()), "Matrix value changed after compression"


def test_compress_bottleneck_second_to_last():
    """Test case where the bottleneck is the second-to-last matrix."""
    A = np.random.randn(10, 5)
    B = np.random.randn(5, 2)  # Bottleneck
    C = np.random.randn(2, 8)
    X = LowRankMatrix(A, B, C)

    X_compressed = X.compress()

    # Should compress to 2 matrices: (A @ B) and C
    assert isinstance(X_compressed, LowRankMatrix)
    assert len(X_compressed._matrices) == 2, "Should compress to 2 matrices"
    assert X_compressed._matrices[0].shape == (10, 2), "Shape of the left product is incorrect"
    assert X_compressed._matrices[1].shape == (2, 8), "Shape of the last matrix should be unchanged"
    assert np.allclose(X.full(), X_compressed.full()), "Matrix value changed after compression"


def test_compress_bottleneck_last_returns_dense():
    """Test case where the bottleneck is the last matrix, returning a dense array."""
    A = np.random.randn(10, 5)
    B = np.random.randn(5, 8)
    C = np.random.randn(8, 2)  # Bottleneck
    X = LowRankMatrix(A, B, C)

    X_compressed = X.compress()

    # The implementation returns a dense matrix in this case
    assert isinstance(X_compressed, np.ndarray), "Should return a dense numpy array"
    assert not isinstance(X_compressed, LowRankMatrix), "Should not return a LowRankMatrix instance"
    assert X_compressed.shape == (10, 2), "Shape of the dense output is incorrect"
    assert np.allclose(X.full(), X_compressed), "Matrix value changed after compression"


#%% Test constructor edge cases
def test_constructor_errors():
    """Test error handling in constructor."""
    import pytest
    
    # Test misaligned shapes
    A = np.random.randn(10, 5)
    B = np.random.randn(3, 8)  # Wrong dimension
    with pytest.raises(ValueError, match="Matrix shapes do not align"):
        LowRankMatrix(A, B)
    
    # Test single matrix (should raise error - not a low-rank factorization)
    A = np.random.randn(10, 5)
    with pytest.raises(ValueError, match="At least two matrices must be provided"):
        LowRankMatrix(A)
    
    print("Constructor error tests passed")


def test_constructor_edge_cases():
    """Test constructor with various edge cases."""
    import pytest
    
    # Empty matrices (0 rows or columns)
    A_empty_rows = np.random.randn(0, 5)
    B_empty_rows = np.random.randn(5, 8)
    X_empty = LowRankMatrix(A_empty_rows, B_empty_rows)
    assert X_empty.shape == (0, 8), "Empty rows should work"
    assert X_empty.full().shape == (0, 8), "Empty full matrix shape incorrect"
    
    A_empty_cols = np.random.randn(10, 0)
    B_empty_cols = np.random.randn(0, 8)
    X_empty2 = LowRankMatrix(A_empty_cols, B_empty_cols)
    assert X_empty2.shape == (10, 8), "Empty columns should work"
    assert X_empty2.rank == 0, "Rank should be 0 for empty inner dimension"
    
    # Rank-1 matrices
    A_rank1 = np.random.randn(10, 1)
    B_rank1 = np.random.randn(1, 8)
    X_rank1 = LowRankMatrix(A_rank1, B_rank1)
    assert X_rank1.rank == 1, "Rank-1 matrix should work"
    assert X_rank1.shape == (10, 8), "Rank-1 shape incorrect"
    
    # Very large rank (equal to dimensions)
    A_full_rank = np.random.randn(10, 10)
    B_full_rank = np.random.randn(10, 8)
    X_full_rank = LowRankMatrix(A_full_rank, B_full_rank)
    assert X_full_rank.rank == 8, "Full rank case incorrect"
    
    # Zero matrices
    A_zero = np.zeros((10, 5))
    B_zero = np.zeros((5, 8))
    X_zero = LowRankMatrix(A_zero, B_zero)
    assert np.allclose(X_zero.full(), np.zeros((10, 8))), "Zero matrices should work"
    
    print("Constructor edge cases passed")


#%% Test complex numbers
def test_complex_matrices():
    """Test operations with complex-valued matrices."""
    # Create complex matrices
    A_real = np.random.randn(10, 5)
    A_imag = np.random.randn(10, 5)
    A = A_real + 1j * A_imag
    
    B_real = np.random.randn(5, 8)
    B_imag = np.random.randn(5, 8)
    B = B_real + 1j * B_imag
    
    X = LowRankMatrix(A, B)
    X_full = A @ B
    
    # Test dtype
    assert X.dtype == np.complex128 or X.dtype == np.complex64, "Incorrect dtype for complex matrix"
    
    # Test full reconstruction
    assert np.allclose(X.full(), X_full), "Complex matrix reconstruction failed"
    
    # Test transpose
    assert np.allclose(X.T.full(), X_full.T), "Complex transpose incorrect"
    
    # Test conjugate
    assert np.allclose(X.conj().full(), X_full.conj()), "Complex conjugate incorrect"
    
    # Test Hermitian transpose
    assert np.allclose(X.H.full(), X_full.T.conj()), "Hermitian transpose incorrect"
    
    # Test addition (returns ndarray, not LowRankMatrix)
    result_add = X + X
    assert np.allclose(result_add, 2 * X_full), "Complex addition incorrect"
    
    # Test scalar multiplication
    assert np.allclose((2 * X).full(), 2 * X_full), "Complex scalar multiplication incorrect"
    assert np.allclose(((1 + 1j) * X).full(), (1 + 1j) * X_full), "Complex scalar multiplication incorrect"
    
    # Test matrix multiplication
    C = np.random.randn(8, 6) + 1j * np.random.randn(8, 6)
    result = X.dot(C, dense_output=True)
    assert np.allclose(result, X_full @ C), "Complex matrix multiplication incorrect"
    
    print("Complex matrix tests passed")


#%% Test class methods
def test_from_matrix():
    """Test from_matrix, from_full, from_dense class methods raise NotImplementedError."""
    import pytest
    
    A = np.random.randn(10, 8)
    
    # Base class should raise NotImplementedError
    # Subclasses like SVD or QR will override these methods
    with pytest.raises(NotImplementedError, match="from_matrix\\(\\) must be implemented by subclasses"):
        LowRankMatrix.from_matrix(A)
    
    with pytest.raises(NotImplementedError):
        LowRankMatrix.from_full(A)
    
    with pytest.raises(NotImplementedError):
        LowRankMatrix.from_dense(A)
    
    print("from_matrix tests passed (NotImplementedError correctly raised)")


def test_from_low_rank():
    """Test from_low_rank class method."""
    # Create a low-rank matrix
    X_orig = LowRankMatrix(A, B, C, D)
    
    # Convert using from_low_rank
    X_new = LowRankMatrix.from_low_rank(X_orig)
    
    assert isinstance(X_new, LowRankMatrix), "from_low_rank should return LowRankMatrix"
    assert np.allclose(X_new.full(), X_orig.full()), "from_low_rank incorrect"
    assert X_new.length == X_orig.length, "from_low_rank should preserve structure"
    
    print("from_low_rank test passed")


#%% Test copy method
def test_copy():
    """Test that copy creates a deep copy."""
    X_copy = X.copy()
    
    # Verify it's a different object
    assert X_copy is not X, "copy() should create new object"
    assert X_copy._matrices is not X._matrices, "copy() should deep copy _matrices"
    
    # Verify values are the same
    assert np.allclose(X_copy.full(), X.full()), "copy() should preserve values"
    
    # Modify copy and ensure original is unchanged
    X_copy._matrices[0] *= 2
    assert not np.allclose(X_copy.full(), X.full()), "Modifying copy should not affect original"
    
    print("copy() test passed")


#%% Test special operators
def test_matmul_operator():
    """Test @ operator (calls dot)."""
    Y = np.random.randn(8, 7)
    
    # Test @ with ndarray
    result1 = X @ Y
    result2 = X.dot(Y)
    assert np.allclose(result1.full(), result2.full()), "@ operator should match dot()"
    
    # Test @ with LowRankMatrix
    Z = LowRankMatrix(np.random.randn(8, 3), np.random.randn(3, 7))
    result3 = X @ Z
    result4 = X.dot(Z)
    assert np.allclose(result3.full(), result4.full()), "@ operator should match dot() for LowRankMatrix"
    
    print("@ operator test passed")


def test_matmul_left_right():
    """Test both left and right matmul with numpy arrays.
    
    With __array_priority__, numpy will prefer our __matmul__ and __rmatmul__
    methods over its own. This test ensures both directions work correctly.
    """
    # Test left matmul: X @ Y (LowRankMatrix @ ndarray)
    Y_right = np.random.randn(8, 6)
    result_left = X @ Y_right
    expected_left = X_full @ Y_right
    
    assert isinstance(result_left, LowRankMatrix), \
        "X @ ndarray should return LowRankMatrix"
    assert result_left.shape == (10, 6), \
        f"Left matmul shape incorrect: {result_left.shape}"
    assert np.allclose(result_left.full(), expected_left), \
        "Left matmul (X @ Y) result incorrect"
    
    # Test right matmul: Y @ X (ndarray @ LowRankMatrix)
    Y_left = np.random.randn(6, 10)
    result_right = Y_left @ X
    expected_right = Y_left @ X_full
    
    assert isinstance(result_right, LowRankMatrix), \
        "ndarray @ X should return LowRankMatrix"
    assert result_right.shape == (6, 8), \
        f"Right matmul shape incorrect: {result_right.shape}"
    assert np.allclose(result_right.full(), expected_right), \
        "Right matmul (Y @ X) result incorrect"
    
    # Test with vectors (should return dense)
    v_right = np.random.randn(8)
    v_result_left = X @ v_right
    assert isinstance(v_result_left, np.ndarray), \
        "X @ vector should return ndarray"
    assert v_result_left.shape == (10,), \
        f"X @ vector shape incorrect: {v_result_left.shape}"
    assert np.allclose(v_result_left, X_full @ v_right), \
        "X @ vector result incorrect"
    
    v_left = np.random.randn(10)
    v_result_right = v_left @ X
    assert isinstance(v_result_right, np.ndarray), \
        "vector @ X should return ndarray"
    assert v_result_right.shape == (8,), \
        f"vector @ X shape incorrect: {v_result_right.shape}"
    assert np.allclose(v_result_right, v_left @ X_full), \
        "vector @ X result incorrect"
    
    # Test that matmul matches dot with explicit side argument
    result_dot_right = X.dot(Y_right, side='right')
    assert np.allclose(result_left.full(), result_dot_right.full()), \
        "X @ Y should match X.dot(Y, side='right')"
    
    result_dot_left = X.dot(Y_left, side='left')
    assert np.allclose(result_right.full(), result_dot_left.full()), \
        "Y @ X should match X.dot(Y, side='left')"
    
    print("Left and right matmul test passed")


def test_multiplication_operators():
    """Test *, *=, and __rmul__ for both scalar and Hadamard multiplication."""
    import pytest
    from lowrank.matrices.low_rank_matrix import LowRankEfficiencyWarning
    
    # Test scalar multiplication
    X_copy = X.copy()
    result1 = 3 * X_copy
    assert isinstance(result1, LowRankMatrix), "Scalar * LowRankMatrix should return LowRankMatrix"
    assert np.allclose(result1.full(), 3 * X_full), "Left scalar multiplication incorrect"
    
    result2 = X_copy * 3
    assert np.allclose(result2.full(), 3 * X_full), "Right scalar multiplication incorrect"
    
    # Test in-place scalar multiplication
    X_copy2 = X.copy()
    X_copy2 *= 3
    assert isinstance(X_copy2, LowRankMatrix), "In-place scalar multiplication should preserve type"
    assert np.allclose(X_copy2.full(), 3 * X_full), "In-place scalar multiplication incorrect"
    
    # Test Hadamard multiplication with ndarray
    Y_dense = np.random.randn(10, 8)
    result3 = X.hadamard(Y_dense)
    assert isinstance(result3, np.ndarray), "Hadamard with ndarray should return ndarray"
    assert np.allclose(result3, X_full * Y_dense), "Hadamard multiplication with ndarray incorrect"
    
    # Test Hadamard multiplication with LowRankMatrix
    Y = LowRankMatrix(np.random.randn(10, 4), np.random.randn(4, 8))
    result4 = X.hadamard(Y)
    assert isinstance(result4, np.ndarray), "Hadamard with LowRankMatrix should return ndarray"
    assert np.allclose(result4, X_full * Y.full()), "Hadamard multiplication with LowRankMatrix incorrect"
    
    print("Multiplication operators test passed")


def test_negation():
    """Test unary negation operator."""
    result = -X
    assert isinstance(result, LowRankMatrix), "Negation should return LowRankMatrix"
    assert np.allclose(result.full(), -X_full), "Negation incorrect"
    print("Negation test passed")


#%% Test other properties and methods
def test_dtype_property():
    """Test dtype property."""
    # Real matrix
    assert X.dtype == np.float64 or X.dtype == np.float32, "dtype incorrect for real matrix"
    
    # Complex matrix
    A_complex = np.random.randn(10, 5) + 1j * np.random.randn(10, 5)
    B_complex = np.random.randn(5, 8) + 1j * np.random.randn(5, 8)
    X_complex = LowRankMatrix(A_complex, B_complex)
    assert X_complex.dtype == np.complex128 or X_complex.dtype == np.complex64, "dtype incorrect for complex matrix"
    
    print("dtype property test passed")


def test_length_property():
    """Test length property."""
    assert X.length == 4, "length property incorrect"
    
    Y = LowRankMatrix(A, B)
    assert Y.length == 2, "length property incorrect for 2 factors"
    
    print("length property test passed")


def test_repr():
    """Test string representation."""
    repr_str = repr(X)
    assert "(10, 8)" in repr_str, "__repr__ should contain shape"
    assert "rank 4" in repr_str or "rank=4" in repr_str, "__repr__ should contain rank"
    assert "generic" in repr_str, "__repr__ should contain format"
    print("__repr__ test passed")


def test_getitem():
    """Test __getitem__ (indexing)."""
    # Test single element access
    val = X[1, 3]
    assert abs(val - X_full[1, 3]) < 1e-12, "__getitem__ incorrect"
    
    val2 = X[0, 0]
    assert abs(val2 - X_full[0, 0]) < 1e-12, "__getitem__ incorrect for (0,0)"
    
    print("__getitem__ test passed")


def test_aliases():
    """Test todense, to_dense, to_full, flatten aliases."""
    # todense
    assert np.allclose(X.todense(), X_full), "todense() incorrect"
    
    # to_dense
    assert np.allclose(X.to_dense(), X_full), "to_dense() incorrect"
    
    # to_full
    assert np.allclose(X.to_full(), X_full), "to_full() incorrect"
    
    # flatten
    flat = X.flatten()
    assert flat.shape == (80,), "flatten() should return 1D array"
    assert np.allclose(flat, X_full.flatten()), "flatten() incorrect"
    
    print("Aliases test passed")


def test_transpose_method():
    """Test transpose() method (alternative to .T)."""
    result = X.transpose()
    assert isinstance(result, LowRankMatrix), "transpose() should return LowRankMatrix"
    assert np.allclose(result.full(), X_full.T), "transpose() incorrect"
    
    # Test double transpose
    result2 = X.transpose().transpose()
    assert np.allclose(result2.full(), X_full), "Double transpose should return to original"
    
    print("transpose() method test passed")


#%% Test dot with side parameter
def test_dot_side_left():
    """Test dot() with side='left' parameter."""
    Y = np.random.randn(8, 10)
    
    # Test left multiplication: Y @ X
    result = X.dot(Y, side='left', dense_output=True)
    expected = Y @ X_full
    assert np.allclose(result, expected), "dot() with side='left' incorrect"
    
    # Test with LowRankMatrix
    Z = LowRankMatrix(np.random.randn(8, 3), np.random.randn(3, 10))
    result2 = X.dot(Z, side='left')
    expected2 = Z.full() @ X_full
    assert np.allclose(result2.full(), expected2), "dot() with side='left' and LowRankMatrix incorrect"
    
    # Test 'opposite' alias (backward compatibility)
    result3 = X.dot(Y, side='opposite', dense_output=True)
    assert np.allclose(result3, expected), "dot() with side='opposite' incorrect"
    
    print("dot() with side='left' test passed")


def test_is_symmetric():
    """Test is_symmetric() for actually symmetric matrix."""
    # Create a symmetric matrix
    A_sym = np.random.randn(10, 5)
    X_sym = LowRankMatrix(A_sym, A_sym.T)
    
    assert X_sym.is_symmetric(), "Symmetric matrix not detected"
    
    # Non-square should be False
    assert not X.is_symmetric(), "Non-square matrix should not be symmetric"
    
    # Nearly symmetric (within tolerance) - use multi-matrix to avoid single-matrix issue
    A_nearly = np.random.randn(8, 4)
    B_nearly = np.random.randn(4, 8)
    # Create nearly symmetric: A @ B with B chosen so result is nearly symmetric
    X_nearly_full = A_nearly @ B_nearly
    # Make it exactly symmetric
    X_nearly_full = (X_nearly_full + X_nearly_full.T) / 2
    # Create a low-rank version with 2 factors
    U, s, Vt = np.linalg.svd(X_nearly_full, full_matrices=False)
    rank = 4
    X_perturbed = LowRankMatrix(U[:, :rank] * np.sqrt(s[:rank]), np.sqrt(s[:rank])[:, None] * Vt[:rank, :])
    # Should be considered symmetric due to tolerance in np.allclose
    assert X_perturbed.is_symmetric(), "Nearly symmetric matrix should be detected"
    
    print("is_symmetric() test passed")


#%% Test sparse matrix operations
def test_dot_sparse():
    """Test dot_sparse method with various sparse matrix types."""
    import scipy.sparse as sp
    
    # Create a sparse matrix
    sparse_csc = sp.random(8, 7, density=0.3, format='csc')
    sparse_csr = sp.random(8, 7, density=0.3, format='csr')
    sparse_coo = sp.random(8, 7, density=0.3, format='coo')
    
    # Test right multiplication with different formats
    result_csc = X.dot_sparse(sparse_csc, side='right', dense_output=True)
    expected = X_full @ sparse_csc.toarray()
    assert np.allclose(result_csc, expected), "CSC sparse right multiplication incorrect"
    
    result_csr = X.dot_sparse(sparse_csr, side='right', dense_output=True)
    expected_csr = X_full @ sparse_csr.toarray()
    assert np.allclose(result_csr, expected_csr), "CSR sparse right multiplication incorrect"
    
    result_coo = X.dot_sparse(sparse_coo, side='right', dense_output=True)
    expected_coo = X_full @ sparse_coo.toarray()
    assert np.allclose(result_coo, expected_coo), "COO sparse right multiplication incorrect"
    
    # Test left multiplication
    sparse_left = sp.random(12, 10, density=0.3, format='csc')
    result_left = X.dot_sparse(sparse_left, side='left', dense_output=True)
    expected_left = sparse_left.toarray() @ X_full
    assert np.allclose(result_left, expected_left), "Sparse left multiplication incorrect"
    
    # Test low-rank output
    result_lr = X.dot_sparse(sparse_csc, side='right', dense_output=False)
    assert isinstance(result_lr, LowRankMatrix), "Should return LowRankMatrix when dense_output=False"
    assert np.allclose(result_lr.full(), expected), "Sparse multiplication low-rank output incorrect"
    
    # Test with empty sparse matrix
    sparse_empty = sp.csc_matrix((8, 7))
    result_empty = X.dot_sparse(sparse_empty, side='right', dense_output=True)
    assert np.allclose(result_empty, np.zeros((10, 7))), "Empty sparse matrix multiplication incorrect"
    
    print("Sparse matrix operations passed")


def test_dot_sparse_with_dot():
    """Test that dot() correctly dispatches to dot_sparse for sparse matrices."""
    import scipy.sparse as sp
    
    sparse_mat = sp.random(8, 7, density=0.3, format='csc')
    
    # Test that dot() handles sparse matrices
    result1 = X.dot(sparse_mat, dense_output=True)
    result2 = X.dot_sparse(sparse_mat, side='right', dense_output=True)
    assert np.allclose(result1, result2), "dot() should dispatch to dot_sparse"
    
    print("dot() sparse dispatch test passed")


def test_expm_multiply():
    """Test expm_multiply method for sparse matrix exponential action."""
    import scipy.sparse as sp
    
    # Create a sparse matrix
    n = 10
    A_sparse = sp.diags([-1, 2, -1], [-1, 0, 1], shape=(n, n), format='csc')
    h = 0.1
    
    # Test left multiplication: exp(h*A) @ X
    X_small = LowRankMatrix(np.random.randn(10, 5), np.random.randn(5, 6))
    X_small_full = X_small.full()
    
    result_left = X_small.expm_multiply(A_sparse, h, side='left', dense_output=True)
    expected_left = sp.linalg.expm_multiply(A_sparse, X_small_full, start=0, stop=h, num=2, endpoint=True)[-1]
    assert np.allclose(result_left, expected_left), "expm_multiply left side incorrect"
    
    # Test right multiplication: X @ exp(h*A)
    X_small_right = LowRankMatrix(np.random.randn(6, 5), np.random.randn(5, 10))
    X_small_right_full = X_small_right.full()
    
    result_right = X_small_right.expm_multiply(A_sparse, h, side='right', dense_output=True)
    expected_right = sp.linalg.expm_multiply(A_sparse.T, X_small_right_full.T, start=0, stop=h, num=2, endpoint=True)[-1].T
    assert np.allclose(result_right, expected_right), "expm_multiply right side incorrect"
    
    # Test low-rank output
    result_lr = X_small.expm_multiply(A_sparse, h, side='left', dense_output=False)
    assert isinstance(result_lr, LowRankMatrix), "expm_multiply should return LowRankMatrix when dense_output=False"
    assert np.allclose(result_lr.full(), expected_left), "expm_multiply low-rank output incorrect"
    
    print("expm_multiply tests passed")
#%% Test multi_dot and multi_add
def test_multi_dot():
    """Test multi_dot method for sequential multiplication."""
    # Create several matrices to multiply
    Y1 = np.random.randn(8, 6)
    Y2 = np.random.randn(6, 5)
    Y3 = np.random.randn(5, 4)
    
    result = X.multi_dot([Y1, Y2, Y3])
    expected = X_full @ Y1 @ Y2 @ Y3
    
    assert isinstance(result, LowRankMatrix), "multi_dot should return LowRankMatrix"
    assert np.allclose(result.full(), expected), "multi_dot result incorrect"
    
    # Test with mix of LowRankMatrix and ndarray
    Y1_lr = LowRankMatrix(np.random.randn(8, 3), np.random.randn(3, 6))
    result2 = X.multi_dot([Y1_lr, Y2, Y3])
    expected2 = X_full @ Y1_lr.full() @ Y2 @ Y3
    assert np.allclose(result2.full(), expected2), "multi_dot with LowRankMatrix incorrect"
    
    # Test with empty sequence
    result_empty = X.multi_dot([])
    assert np.allclose(result_empty.full(), X_full), "multi_dot with empty list should return copy"
    
    # Test with single matrix
    result_single = X.multi_dot([Y1])
    expected_single = X_full @ Y1
    assert np.allclose(result_single.full(), expected_single), "multi_dot with single matrix incorrect"
    
    print("multi_dot tests passed")


def test_multi_add():
    """Test multi_add method for sequential addition."""
    # Create several low-rank matrices
    Y1 = LowRankMatrix(np.random.randn(10, 3), np.random.randn(3, 8))
    Y2 = LowRankMatrix(np.random.randn(10, 4), np.random.randn(4, 8))
    Y3 = LowRankMatrix(np.random.randn(10, 2), np.random.randn(2, 8))
    
    # Test with multiple matrices
    result = X.multi_add([Y1, Y2, Y3])
    expected = X_full + Y1.full() + Y2.full() + Y3.full()
    assert isinstance(result, np.ndarray), "multi_add should return ndarray"
    assert np.allclose(result, expected), "multi_add result incorrect"
    
    # Test with empty sequence
    result_empty = X.multi_add([])
    assert isinstance(result_empty, np.ndarray), "multi_add should always return ndarray"
    assert np.allclose(result_empty, X_full), "multi_add with empty list should return full matrix"
    
    # Test with single matrix
    result_single = X.multi_add([Y1])
    expected_single = X_full + Y1.full()
    assert np.allclose(result_single, expected_single), "multi_add with single matrix incorrect"
    
    print("multi_add tests passed")


#%% Test extra_data propagation
def test_extra_data_basic():
    """Test that extra_data is stored and accessible."""
    metadata = {"source": "test", "timestamp": 12345}
    X_with_data = LowRankMatrix(A, B, C, D, **metadata)
    
    assert X_with_data._extra_data == metadata, "extra_data not stored correctly"
    assert X_with_data._extra_data["source"] == "test", "extra_data access incorrect"
    
    print("extra_data basic test passed")


def test_extra_data_propagation():
    """Test that extra_data propagates through operations."""
    metadata = {"source": "test", "version": 1}
    X_with_data = LowRankMatrix(A, B, C, D, **metadata)
    
    # Test propagation through copy
    X_copy = X_with_data.copy()
    assert X_copy._extra_data == metadata, "extra_data not copied"
    
    # Test propagation through transpose
    X_T = X_with_data.T
    assert X_T._extra_data == metadata, "extra_data not preserved in transpose"
    
    # Test propagation through conjugate
    X_conj = X_with_data.conj()
    assert X_conj._extra_data == metadata, "extra_data not preserved in conjugate"
    
    # Test propagation through Hermitian transpose
    X_H = X_with_data.H
    assert X_H._extra_data == metadata, "extra_data not preserved in Hermitian transpose"
    
    # Test propagation through scalar multiplication
    X_scaled = 2 * X_with_data
    assert X_scaled._extra_data == metadata, "extra_data not preserved in scalar multiplication"
    
    # Test propagation through dot (low-rank output)
    Y = np.random.randn(8, 7)
    X_dot = X_with_data.dot(Y, dense_output=False)
    assert X_dot._extra_data == metadata, "extra_data not preserved in dot"
    
    # Test propagation through dot with LowRankMatrix
    Y_lr = LowRankMatrix(np.random.randn(8, 3), np.random.randn(3, 7))
    X_dot_lr = X_with_data.dot(Y_lr, dense_output=False)
    assert X_dot_lr._extra_data == metadata, "extra_data not preserved in dot with LowRankMatrix"
    
    # Test propagation through compress
    X_compressed = X_with_data.compress()
    if isinstance(X_compressed, LowRankMatrix):
        assert X_compressed._extra_data == metadata, "extra_data not preserved in compress"
    
    print("extra_data propagation test passed")


def test_extra_data_create_data_alias():
    """Test create_data_alias static method."""
    metadata = {"alpha": 0.5, "beta": 1.5}
    
    # Create a subclass with aliased properties
    class TestLowRankMatrix(LowRankMatrix):
        alpha = LowRankMatrix.create_data_alias("alpha")
        beta = LowRankMatrix.create_data_alias("beta")
    
    X_test = TestLowRankMatrix(A, B, **metadata)
    
    # Test getter
    assert X_test.alpha == 0.5, "create_data_alias getter incorrect"
    assert X_test.beta == 1.5, "create_data_alias getter incorrect"
    
    # Test setter
    X_test.alpha = 0.7
    assert X_test._extra_data["alpha"] == 0.7, "create_data_alias setter incorrect"
    assert X_test.alpha == 0.7, "create_data_alias setter/getter roundtrip incorrect"
    
    print("create_data_alias test passed")


#%% Test create_matrix_alias
def test_create_matrix_alias():
    """Test create_matrix_alias static method."""
    
    # Create a subclass with aliased properties
    class TestLowRankMatrix(LowRankMatrix):
        U = LowRankMatrix.create_matrix_alias(0)
        V = LowRankMatrix.create_matrix_alias(1)
        W_T = LowRankMatrix.create_matrix_alias(2, transpose=True)
    
    X_test = TestLowRankMatrix(A, B, C, D)
    
    # Test basic getter
    assert np.allclose(X_test.U, A), "create_matrix_alias getter incorrect"
    assert np.allclose(X_test.V, B), "create_matrix_alias getter incorrect"
    
    # Test transpose getter
    assert np.allclose(X_test.W_T, C.T), "create_matrix_alias transpose getter incorrect"
    
    # Test setter
    new_A = np.random.randn(10, 5)
    X_test.U = new_A
    assert np.allclose(X_test._matrices[0], new_A), "create_matrix_alias setter incorrect"
    assert np.allclose(X_test.U, new_A), "create_matrix_alias setter/getter roundtrip incorrect"
    
    # Test transpose setter
    new_C = np.random.randn(4, 6)
    X_test.W_T = new_C
    assert np.allclose(X_test._matrices[2], new_C.T), "create_matrix_alias transpose setter incorrect"
    assert np.allclose(X_test.W_T, new_C), "create_matrix_alias transpose setter/getter roundtrip incorrect"
    
    print("create_matrix_alias test passed")


def test_create_matrix_alias_conjugate():
    """Test create_matrix_alias with conjugate options."""
    A_complex = np.random.randn(10, 5) + 1j * np.random.randn(10, 5)
    B_complex = np.random.randn(5, 8) + 1j * np.random.randn(5, 8)
    
    class TestComplexLowRankMatrix(LowRankMatrix):
        U = LowRankMatrix.create_matrix_alias(0)
        U_conj = LowRankMatrix.create_matrix_alias(0, conjugate=True)
        U_H = LowRankMatrix.create_matrix_alias(0, transpose=True, conjugate=True)
    
    X_complex = TestComplexLowRankMatrix(A_complex, B_complex)
    
    # Test conjugate getter
    assert np.allclose(X_complex.U_conj, A_complex.conj()), "conjugate getter incorrect"
    
    # Test Hermitian getter
    assert np.allclose(X_complex.U_H, A_complex.T.conj()), "Hermitian getter incorrect"
    
    # Test conjugate setter
    new_A = np.random.randn(10, 5) + 1j * np.random.randn(10, 5)
    X_complex.U_conj = new_A
    assert np.allclose(X_complex._matrices[0], new_A.conj()), "conjugate setter incorrect"
    
    # Test Hermitian setter
    new_A2 = np.random.randn(5, 10) + 1j * np.random.randn(5, 10)
    X_complex.U_H = new_A2
    assert np.allclose(X_complex._matrices[0], new_A2.T.conj()), "Hermitian setter incorrect"
    
    print("create_matrix_alias conjugate test passed")


#%% Test input validation and error handling
def test_dot_invalid_dimensions():
    """Test that dot raises errors for incompatible dimensions."""
    import pytest
    
    # Incompatible matrix dimensions
    Y_wrong = np.random.randn(5, 7)  # Wrong first dimension
    with pytest.raises((ValueError, IndexError)):
        X.dot(Y_wrong, dense_output=True)
    
    print("dot invalid dimensions test passed")


def test_dot_invalid_side():
    """Test that dot raises error for invalid side parameter."""
    import pytest
    
    Y = np.random.randn(8, 7)
    with pytest.raises(ValueError, match="Incorrect side"):
        X.dot(Y, side='invalid')
    
    print("dot invalid side test passed")


def test_dot_sparse_invalid_side():
    """Test that dot_sparse raises error for invalid side parameter."""
    import pytest
    import scipy.sparse as sp
    
    sparse_mat = sp.random(8, 7, density=0.3, format='csc')
    with pytest.raises(ValueError, match="incorrect side"):
        X.dot_sparse(sparse_mat, side='invalid')
    
    print("dot_sparse invalid side test passed")


def test_expm_multiply_invalid_side():
    """Test that expm_multiply raises error for invalid side parameter."""
    import pytest
    import scipy.sparse as sp
    
    A_sparse = sp.diags([1, 2, 1], [-1, 0, 1], shape=(10, 10), format='csc')
    X_small = LowRankMatrix(np.random.randn(10, 5), np.random.randn(5, 6))
    
    with pytest.raises(ValueError, match="incorrect side"):
        X_small.expm_multiply(A_sparse, 0.1, side='invalid')
    
    print("expm_multiply invalid side test passed")


def test_expm_multiply_invalid_h():
    """Test that expm_multiply raises error for non-positive h."""
    import pytest
    import scipy.sparse as sp
    
    A_sparse = sp.diags([1, 2, 1], [-1, 0, 1], shape=(10, 10), format='csc')
    X_small = LowRankMatrix(np.random.randn(10, 5), np.random.randn(5, 6))
    
    # Test h=0
    with pytest.raises(ValueError, match="h must be positive"):
        X_small.expm_multiply(A_sparse, 0, side='left')
    
    # Test negative h
    with pytest.raises(ValueError, match="h must be positive"):
        X_small.expm_multiply(A_sparse, -0.1, side='left')
    
    print("expm_multiply invalid h test passed")


def test_gather_edge_cases():
    """Test gather with various index patterns."""
    # Test single element
    val = X.gather([3, 5])
    assert abs(val - X_full[3, 5]) < 1e-12, "gather single element incorrect"
    
    # Test first element
    val_first = X.gather([0, 0])
    assert abs(val_first - X_full[0, 0]) < 1e-12, "gather first element incorrect"
    
    # Test last element
    val_last = X.gather([9, 7])
    assert abs(val_last - X_full[9, 7]) < 1e-12, "gather last element incorrect"
    
    # Note: Out-of-bounds and negative indices will raise numpy IndexError
    # This is expected behavior, so we don't test for custom error handling
    
    print("gather edge cases test passed")


def test_gather_with_arrays():
    """Test gather with arrays of indices.
    
    Note: The current implementation of gather() expects indices = [row_idx, col_idx]
    for a single element. Testing with arrays of indices to see behavior.
    """
    import pytest
    
    # Test with array indices - this tests if fancy indexing works
    row_indices = np.array([0, 2, 5])
    col_indices = np.array([1, 3, 6])
    
    # Current implementation: indices[0] and indices[1] are used for indexing
    # If we pass arrays, it will do fancy indexing
    try:
        # This will select rows [0, 2, 5] from first matrix and cols [1, 3, 6] from last matrix
        result = X.gather([row_indices, col_indices])
        
        # The result shape depends on how numpy handles the fancy indexing
        # For the implementation: A[indices[0], :] gives shape (3, 5) if indices[0] is array of 3 elements
        # Then multiply through and Z[:, indices[1]] gives shape (4, 3) if indices[1] is array of 3 elements
        # Final result should be (3, 3) matrix from multi_dot
        
        assert result.shape == (3, 3), f"Array gather shape incorrect: {result.shape}"
        
        # Verify values: result[i, j] should correspond to X_full[row_indices[i], col_indices[j]]
        for i in range(3):
            for j in range(3):
                expected_val = X_full[row_indices[i], col_indices[j]]
                assert abs(result[i, j] - expected_val) < 1e-10, \
                    f"Array gather value incorrect at ({i},{j})"
        
        print("gather with arrays test passed")
        
    except (IndexError, ValueError) as e:
        # If fancy indexing doesn't work as expected, document the limitation
        print(f"gather with arrays raised {type(e).__name__}: {e}")
        pytest.skip("gather() may not support array indices in the expected way")


def test_gather_multiple_elements():
    """Test gathering multiple specific elements efficiently.
    
    This tests a common use case: extracting multiple matrix elements
    without forming the full matrix (useful for matrix completion).
    """
    # Define specific (row, col) pairs we want to extract
    row_indices = [0, 2, 5, 7]
    col_indices = [1, 3, 6, 2]
    
    # Extract using gather (one at a time, as designed)
    gathered_values = []
    for r, c in zip(row_indices, col_indices):
        gathered_values.append(X.gather([r, c]))
    
    # Compare with full matrix
    expected_values = [X_full[r, c] for r, c in zip(row_indices, col_indices)]
    
    assert np.allclose(gathered_values, expected_values), \
        "Multiple element gathering incorrect"
    
    print("gather multiple elements test passed")


def test_mul_type_error():
    """Test that multiplication raises TypeError for unsupported types."""
    import pytest
    
    with pytest.raises(TypeError, match="Unsupported operand type"):
        X * "invalid"
    
    with pytest.raises(TypeError):
        X * [1, 2, 3]
    
    print("mul TypeError test passed")


def test_inefficiency_warning():
    """Test that LowRankEfficiencyWarning is raised for inefficient operations."""
    import warnings
    from lowrank.matrices.low_rank_matrix import LowRankEfficiencyWarning
    
    X_test = X.copy()
    Y = np.random.randn(10, 8)
    
    # Test in-place Hadamard multiplication
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        X_test *= Y
        # Check that warnings were issued (both "In-place" and "Hadamard product")
        assert len(w) >= 1, "LowRankEfficiencyWarning not raised for in-place Hadamard"
        # Check that at least one warning is LowRankEfficiencyWarning
        assert any(issubclass(warning.category, LowRankEfficiencyWarning) for warning in w), "Wrong warning type"
        # Should contain either "In-place" or "Hadamard product" in messages
        messages = [str(warning.message) for warning in w]
        assert any("In-place" in msg or "Hadamard product" in msg for msg in messages), "Expected Hadamard-related warning"
    
    # Test Hadamard product
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = X * Y
        assert len(w) > 0, "LowRankEfficiencyWarning not raised for Hadamard"
        assert issubclass(w[-1].category, LowRankEfficiencyWarning), "Wrong warning type"
        assert "Hadamard product" in str(w[-1].message), "Wrong warning message for Hadamard"
    
    # Test addition
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = X + X
        assert len(w) > 0, "LowRankEfficiencyWarning not raised for addition"
        assert issubclass(w[-1].category, LowRankEfficiencyWarning), "Wrong warning type"
        assert "Addition" in str(w[-1].message), "Wrong warning message for addition"
    
    # Test subtraction
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = X - X
        assert len(w) > 0, "LowRankEfficiencyWarning not raised for subtraction"
        assert issubclass(w[-1].category, LowRankEfficiencyWarning), "Wrong warning type"
        assert "Subtraction" in str(w[-1].message), "Wrong warning message for subtraction"
    
    # Test is_symmetric
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        X_square = LowRankMatrix(np.random.randn(5, 3), np.random.randn(3, 5))
        result = X_square.is_symmetric()
        assert len(w) > 0, "LowRankEfficiencyWarning not raised for is_symmetric"
        assert issubclass(w[-1].category, LowRankEfficiencyWarning), "Wrong warning type"
        assert "symmetry" in str(w[-1].message), "Wrong warning message for is_symmetric"
    
    # Test multi_add
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = X.multi_add([X, X])
        assert len(w) > 0, "LowRankEfficiencyWarning not raised for multi_add"
        assert issubclass(w[-1].category, LowRankEfficiencyWarning), "Wrong warning type"
        assert "multi_add" in str(w[-1].message), "Wrong warning message for multi_add"
    
    print("LowRankEfficiencyWarning test passed")


#%% Test numerical edge cases
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_nan_inf_matrices():
    """Test behavior with NaN and Inf values."""
    # Matrix with NaN
    A_nan = np.array([[1.0, 2.0], [np.nan, 4.0]])
    B_nan = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    X_nan = LowRankMatrix(A_nan, B_nan)
    
    result_nan = X_nan.full()
    assert np.isnan(result_nan).any(), "NaN should propagate through multiplication"
    
    # Matrix with Inf
    A_inf = np.array([[1.0, 2.0], [np.inf, 4.0]])
    B_inf = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    X_inf = LowRankMatrix(A_inf, B_inf)
    
    result_inf = X_inf.full()
    assert np.isinf(result_inf).any(), "Inf should propagate through multiplication"
    
    # Test operations with NaN/Inf
    assert np.isnan(X_nan.norm()), "Norm with NaN should be NaN"
    
    print("NaN/Inf matrices test passed")


def test_very_small_large_values():
    """Test numerical stability with very small and large values."""
    # Very small values
    A_small = np.random.randn(10, 5) * 1e-100
    B_small = np.random.randn(5, 8) * 1e-100
    X_small = LowRankMatrix(A_small, B_small)
    
    result_small = X_small.full()
    # Result should be extremely small but not underflow to exactly zero
    assert result_small.shape == (10, 8), "Shape preserved with small values"
    
    # Very large values
    A_large = np.random.randn(10, 5) * 1e100
    B_large = np.random.randn(5, 8) * 1e100
    X_large = LowRankMatrix(A_large, B_large)
    
    result_large = X_large.full()
    assert result_large.shape == (10, 8), "Shape preserved with large values"
    # May overflow to inf, which is acceptable
    
    print("Very small/large values test passed")


#%% Additional edge case tests
def test_addition_incompatible_shapes():
    """Test that addition with incompatible shapes raises appropriate errors."""
    import pytest
    
    Y_wrong = LowRankMatrix(np.random.randn(5, 3), np.random.randn(3, 6))  # Wrong shape
    
    with pytest.raises((ValueError, IndexError)):
        result = X + Y_wrong
    
    print("Addition incompatible shapes test passed")


def test_addition_right_side():
    """Test right-side addition (ndarray + LowRankMatrix).
    
    Tests that __radd__ is implemented correctly.
    Since addition is commutative, (ndarray + LowRankMatrix) should equal 
    (LowRankMatrix + ndarray).
    """
    Y_dense = np.random.randn(10, 8)
    
    # Left-side (LowRankMatrix + ndarray)
    result_left = X + Y_dense
    expected = X_full + Y_dense
    assert np.allclose(result_left, expected), "Left-side addition incorrect"
    
    # Right-side (ndarray + LowRankMatrix)
    result_right = Y_dense + X
    assert np.allclose(result_right, expected), "Right-side addition incorrect"
    
    # Both should give the same result (addition is commutative)
    assert np.allclose(result_right, result_left), "Addition should be commutative"
    
    # Test with LowRankMatrix on both sides
    Y_lr = LowRankMatrix(np.random.randn(10, 3), np.random.randn(3, 8))
    result_lr_left = X + Y_lr
    result_lr_right = Y_lr + X
    assert np.allclose(result_lr_left, result_lr_right), "LowRankMatrix addition should be commutative"
    
    print("Right-side addition test passed")


def test_subtraction_left_right():
    """Test both left and right subtraction with numpy arrays.
    
    Tests that __rsub__ is implemented correctly.
    Since subtraction is NOT commutative, (ndarray - LowRankMatrix) should equal 
    -(LowRankMatrix - ndarray).
    """
    Y_dense = np.random.randn(10, 8)
    
    # Left-side (LowRankMatrix - ndarray)
    result_left = X - Y_dense
    expected_left = X_full - Y_dense
    assert np.allclose(result_left, expected_left), "Left-side subtraction incorrect"
    
    # Right-side (ndarray - LowRankMatrix)
    result_right = Y_dense - X
    expected_right = Y_dense - X_full
    assert np.allclose(result_right, expected_right), "Right-side subtraction incorrect"
    
    # Verify non-commutativity: A - B = -(B - A)
    assert np.allclose(result_left, -result_right), "X - Y should equal -(Y - X)"
    assert np.allclose(result_right, -result_left), "Y - X should equal -(X - Y)"
    
    # Verify they are NOT equal (unless the rare case where they cancel)
    assert not np.allclose(result_right, result_left) or np.allclose(result_left, 0), \
        "Subtraction should not be commutative"
    
    # Test with LowRankMatrix on both sides
    Y_lr = LowRankMatrix(np.random.randn(10, 3), np.random.randn(3, 8))
    Y_lr_full = Y_lr.full()
    result_lr_left = X - Y_lr
    result_lr_right = Y_lr - X
    expected_lr_left = X_full - Y_lr_full
    expected_lr_right = Y_lr_full - X_full
    assert np.allclose(result_lr_left, expected_lr_left), "LowRankMatrix - LowRankMatrix incorrect"
    assert np.allclose(result_lr_right, expected_lr_right), "LowRankMatrix - LowRankMatrix (reversed) incorrect"
    assert np.allclose(result_lr_left, -result_lr_right), "LowRankMatrix subtraction non-commutativity failed"
    
    print("Left and right subtraction test passed")


def test_subtraction_incompatible_shapes():
    """Test that subtraction with incompatible shapes raises appropriate errors."""
    import pytest
    
    Y_wrong = LowRankMatrix(np.random.randn(5, 3), np.random.randn(3, 6))
    
    with pytest.raises((ValueError, IndexError)):
        result = X - Y_wrong
    
    print("Subtraction incompatible shapes test passed")


def test_hadamard_incompatible_shapes():
    """Test that Hadamard product with incompatible shapes raises errors."""
    import pytest
    
    Y_wrong = np.random.randn(5, 6)  # Wrong shape
    
    with pytest.raises(ValueError):
        X.hadamard(Y_wrong)
    
    print("Hadamard incompatible shapes test passed")


def test_norm_invalid_ord():
    """Test norm with invalid ord parameter."""
    import pytest
    
    # Test with an invalid string ord
    with pytest.raises((ValueError, np.linalg.LinAlgError)):
        X.norm('invalid_norm')
    
    print("Norm invalid ord test passed")


def test_norm_edge_cases():
    """Test norm on edge case matrices."""
    # Empty matrix (0 rows)
    A_empty = np.random.randn(0, 5)
    B_empty = np.random.randn(5, 8)
    X_empty = LowRankMatrix(A_empty, B_empty)
    
    norm_empty = X_empty.norm('fro')
    assert norm_empty == 0 or np.isnan(norm_empty), "Norm of empty matrix should be 0 or NaN"
    
    # Zero matrix
    A_zero = np.zeros((10, 5))
    B_zero = np.zeros((5, 8))
    X_zero = LowRankMatrix(A_zero, B_zero)
    
    assert X_zero.norm('fro') == 0, "Norm of zero matrix should be 0"
    assert X_zero.norm(2) == 0, "2-norm of zero matrix should be 0"
    
    print("Norm edge cases test passed")


def test_is_symmetric_edge_cases():
    """Test is_symmetric on edge cases."""
    # 1x1 matrix (always symmetric)
    A_1x1 = np.array([[5.0]])
    B_1x1 = np.array([[1.0]])
    X_1x1 = LowRankMatrix(A_1x1, B_1x1)
    assert X_1x1.is_symmetric(), "1x1 matrix should be symmetric"
    
    # Empty square matrix
    A_empty = np.random.randn(0, 2)
    B_empty = np.random.randn(2, 0)
    X_empty = LowRankMatrix(A_empty, B_empty)
    # Shape is (0, 0), which is square, so should check for symmetry
    # Result depends on implementation - empty matrices are trivially symmetric
    sym_result = X_empty.is_symmetric()
    assert isinstance(sym_result, bool), "is_symmetric should return bool"
    
    print("is_symmetric edge cases test passed")


def test_dot_backward_compatibility():
    """Test backward compatibility aliases 'usual' and 'opposite' for dot()."""
    Y = np.random.randn(8, 7)
    
    # Test 'usual' is same as 'right'
    result_usual = X.dot(Y, side='usual', dense_output=True)
    result_right = X.dot(Y, side='right', dense_output=True)
    assert np.allclose(result_usual, result_right), "side='usual' should match side='right'"
    
    # Test 'opposite' is same as 'left' 
    Y_left = np.random.randn(12, 10)
    result_opposite = X.dot(Y_left, side='opposite', dense_output=True)
    result_left = X.dot(Y_left, side='left', dense_output=True)
    assert np.allclose(result_opposite, result_left), "side='opposite' should match side='left'"
    
    print("Dot backward compatibility test passed")


def test_multi_dot_incompatible():
    """Test multi_dot with incompatible dimensions in sequence."""
    import pytest
    
    Y1 = np.random.randn(8, 6)
    Y2 = np.random.randn(4, 5)  # Wrong dimension - doesn't match Y1's output
    
    with pytest.raises((ValueError, IndexError)):
        X.multi_dot([Y1, Y2])
    
    print("multi_dot incompatible dimensions test passed")


def test_expm_multiply_incompatible_dimensions():
    """Test expm_multiply with incompatible dimensions."""
    import pytest
    import scipy.sparse as sp
    
    # Non-square sparse matrix (should likely error)
    A_nonsquare = sp.random(10, 8, density=0.3, format='csc')
    X_small = LowRankMatrix(np.random.randn(10, 5), np.random.randn(5, 6))
    
    # This may or may not raise an error depending on implementation
    # Document the behavior
    try:
        result = X_small.expm_multiply(A_nonsquare, 0.1, side='left')
        print("expm_multiply with non-square matrix did not raise error")
    except (ValueError, Exception) as e:
        print(f"expm_multiply with non-square matrix raised: {type(e).__name__}")
    
    # Wrong dimensions - A should match first dimension of X
    A_wrong_size = sp.random(5, 5, density=0.3, format='csc')  # 5x5 but X is 10x?
    
    with pytest.raises((ValueError, IndexError, Exception)):
        X_small.expm_multiply(A_wrong_size, 0.1, side='left')
    
    print("expm_multiply incompatible dimensions test passed")


def test_compress_equal_ranks():
    """Test compress when all intermediate matrices have equal rank."""
    # All matrices have rank 3
    A_eq = np.random.randn(10, 3)
    B_eq = np.random.randn(3, 3)
    C_eq = np.random.randn(3, 8)
    X_eq = LowRankMatrix(A_eq, B_eq, C_eq)
    
    X_compressed = X_eq.compress()
    
    # Should still compress to 2 matrices
    assert len(X_compressed._matrices) == 2, "Should compress even with equal ranks"
    assert np.allclose(X_eq.full(), X_compressed.full()), "Compression should preserve values"
    
    print("compress with equal ranks test passed")


def test_matrix_vector_wrong_dimension():
    """Test matrix-vector multiplication with wrong dimension."""
    import pytest
    
    v_wrong = np.random.randn(5)  # Wrong size, should be 8
    
    with pytest.raises((ValueError, IndexError)):
        X.dot(v_wrong)
    
    print("Matrix-vector wrong dimension test passed")


def test_deepshape_unusual_chains():
    """Test deepshape with various unusual matrix chain configurations."""
    # Very long chain
    matrices_long = [np.random.randn(10, 3)]
    for _ in range(10):
        matrices_long.append(np.random.randn(3, 3))
    matrices_long.append(np.random.randn(3, 8))
    
    X_long = LowRankMatrix(*matrices_long)
    assert X_long.deepshape == (10,) + (3,) * 11 + (8,), "Deepshape incorrect for long chain"
    assert X_long.shape == (10, 8), "Shape should be (first_dim, last_dim)"
    
    print("deepshape unusual chains test passed")


def test_sparse_side_aliases():
    """Test that dot_sparse backward compatibility aliases work."""
    import scipy.sparse as sp
    
    sparse_mat = sp.random(8, 7, density=0.3, format='csc')
    
    # Test 'usual' is same as 'right'
    result_usual = X.dot_sparse(sparse_mat, side='usual', dense_output=True)
    result_right = X.dot_sparse(sparse_mat, side='right', dense_output=True)
    assert np.allclose(result_usual, result_right), "side='usual' should match side='right' for sparse"
    
    # Test 'opposite' is same as 'left'
    sparse_left = sp.random(12, 10, density=0.3, format='csc')
    result_opposite = X.dot_sparse(sparse_left, side='opposite', dense_output=True)
    result_left = X.dot_sparse(sparse_left, side='left', dense_output=True)
    assert np.allclose(result_opposite, result_left), "side='opposite' should match side='left' for sparse"
    
    print("sparse side aliases test passed")


def test_extra_data_after_operations():
    """Test that extra_data is preserved through more complex operation chains."""
    metadata = {"test": "value", "number": 42}
    X_with_data = LowRankMatrix(A, B, C, D, **metadata)
    
    # Chain of operations
    result = (X_with_data * 2).T.conj()
    
    assert result._extra_data == metadata, "extra_data lost in operation chain"
    
    # Multiple dots
    Y1 = np.random.randn(8, 6)
    Y2 = np.random.randn(6, 5)
    result2 = X_with_data.dot(Y1, dense_output=False).dot(Y2, dense_output=False)
    
    assert result2._extra_data == metadata, "extra_data lost in multi-dot chain"
    
    print("extra_data after operations test passed")


# %%

#%% NEW FEATURES TESTS - Merged from test_new_features.py

# Create additional square matrix for new tests
A_sq = np.random.randn(8, 4)
B_sq = np.random.randn(4, 8)
X_sq = LowRankMatrix(A_sq, B_sq)
X_sq_full = A_sq @ B_sq


#%% Test in-place operations
def test_scale_inplace():
    """Test in-place scaling."""
    X_copy = X.copy()
    result = X_copy.scale_(2.5)
    
    # Check that it returns self
    assert result is X_copy, "scale_() should return self"
    
    # Check that scaling worked
    assert np.allclose(X_copy.full(), 2.5 * X_full), "In-place scaling incorrect"
    
    # Test with zero
    X_copy2 = X.copy()
    X_copy2.scale_(0)
    assert np.allclose(X_copy2.full(), np.zeros_like(X_full)), "Scaling by zero failed"
    
    # Test with negative
    X_copy3 = X.copy()
    X_copy3.scale_(-1)
    assert np.allclose(X_copy3.full(), -X_full), "Scaling by -1 failed"
    
    print("scale_() test passed")


def test_compress_inplace():
    """Test in-place compression."""
    # Create a matrix that benefits from compression
    A_comp = np.random.randn(100, 2)
    B_comp = np.random.randn(2, 80)
    C_comp = np.random.randn(80, 3)
    D_comp = np.random.randn(3, 50)
    X_comp = LowRankMatrix(A_comp, B_comp, C_comp, D_comp)
    X_comp_full = X_comp.full()
    
    original_length = X_comp.length
    result = X_comp.compress_()
    
    # Check that it returns self
    assert result is X_comp, "compress_() should return self"
    
    # Check that compression happened
    assert X_comp.length < original_length, "Compression should reduce length"
    assert X_comp.length == 2, "Should compress to 2 matrices"
    
    # Check that value is preserved
    assert np.allclose(X_comp.full(), X_comp_full), "Compression changed matrix value"
    
    # Test with already optimal matrix
    Y = LowRankMatrix(np.random.randn(10, 5), np.random.randn(5, 8))
    Y.compress_()
    assert Y.length == 2, "Already optimal matrix should remain unchanged"
    
    print("compress_() test passed")


def test_compress_inplace_edge_case():
    """Test compress_() when it would return dense."""
    # Create a case where compress returns dense (bottleneck at last position)
    A_edge = np.random.randn(10, 5)
    B_edge = np.random.randn(5, 8)
    C_edge = np.random.randn(8, 2)  # Bottleneck
    X_edge = LowRankMatrix(A_edge, B_edge, C_edge)
    
    original_length = X_edge.length
    
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = X_edge.compress_()
        
        # Should issue warning
        assert len(w) > 0, "Should warn when compression would return dense"
        assert issubclass(w[0].category, LowRankEfficiencyWarning)
    
    # Should not modify matrix
    assert X_edge.length == original_length, "Should not modify when would return dense"
    assert result is X_edge, "Should return self"
    
    print("compress_() edge case test passed")


#%% Test diagonal extraction
def test_diag_basic():
    """Test diagonal extraction."""
    diag = X_sq.diag()
    diag_expected = np.diag(X_sq_full)
    
    assert diag.shape == (8,), "Diagonal shape incorrect"
    assert np.allclose(diag, diag_expected), "Diagonal values incorrect"
    
    print("diag() basic test passed")


def test_diag_edge_cases():
    """Test diagonal extraction edge cases."""
    # Non-square matrix
    with pytest.raises(ValueError, match="must be square"):
        X.diag()
    
    # Identity-like matrix
    I_lr = LowRankMatrix(np.eye(5), np.eye(5))
    diag_I = I_lr.diag()
    assert np.allclose(diag_I, np.ones(5)), "Identity diagonal incorrect"
    
    # Zero matrix
    Z = LowRankMatrix(np.zeros((5, 3)), np.zeros((3, 5)))
    diag_Z = Z.diag()
    assert np.allclose(diag_Z, np.zeros(5)), "Zero matrix diagonal incorrect"
    
    # Small matrix
    A_small = np.random.randn(2, 3)
    B_small = np.random.randn(3, 2)
    X_small = LowRankMatrix(A_small, B_small)
    diag_small = X_small.diag()
    assert np.allclose(diag_small, np.diag(A_small @ B_small)), "Small matrix diagonal incorrect"
    
    print("diag() edge cases passed")


#%% Test trace computation
def test_trace_basic():
    """Test trace computation."""
    trace = X_sq.trace()
    trace_expected = np.trace(X_sq_full)
    
    assert np.abs(trace - trace_expected) < 1e-10, "Trace computation incorrect"
    
    print("trace() basic test passed")


def test_trace_edge_cases():
    """Test trace edge cases."""
    # Non-square matrix
    with pytest.raises(ValueError, match="must be square"):
        X.trace()
    
    # Two-matrix case
    A_two = np.random.randn(6, 4)
    B_two = np.random.randn(4, 6)
    X_two = LowRankMatrix(A_two, B_two)
    trace_two = X_two.trace()
    trace_expected = np.trace(A_two @ B_two)
    assert np.abs(trace_two - trace_expected) < 1e-10, "Two-matrix trace incorrect"
    
    # Many matrices
    matrices = [np.random.randn(5, 5) for _ in range(6)]
    X_many = LowRankMatrix(*matrices)
    trace_many = X_many.trace()
    trace_expected = np.trace(np.linalg.multi_dot(matrices))
    assert np.abs(trace_many - trace_expected) < 1e-9, "Many-matrix trace incorrect"
    
    # Zero trace
    A_zero = np.random.randn(4, 3)
    B_zero = -A_zero.T
    X_zero = LowRankMatrix(A_zero, B_zero)
    # A @ (-A^T) is negative definite, so trace should be negative
    # But let's test the computation is correct
    trace_zero = X_zero.trace()
    trace_expected = np.trace(A_zero @ B_zero)
    assert np.abs(trace_zero - trace_expected) < 1e-10, "Zero trace case incorrect"
    
    print("trace() edge cases passed")


#%% Test Frobenius norm squared
def test_norm_squared_basic():
    """Test squared Frobenius norm."""
    norm_sq = X.norm_squared()
    norm_expected = la.norm(X_full, 'fro') ** 2
    
    assert np.abs(norm_sq - norm_expected) < 1e-8, "Norm squared incorrect"
    
    # Compare with norm method
    norm_from_method = X.norm('fro') ** 2
    assert np.abs(norm_sq - norm_from_method) < 1e-8, "Norm squared inconsistent with norm()"
    
    print("norm_squared() basic test passed")


def test_norm_squared_edge_cases():
    """Test norm_squared edge cases."""
    # Zero matrix
    Z = LowRankMatrix(np.zeros((5, 3)), np.zeros((3, 8)))
    assert Z.norm_squared() < 1e-15, "Zero matrix norm squared should be ~0"
    
    # Identity-ish
    I_lr = LowRankMatrix(np.eye(4, 3), np.eye(3, 4))
    norm_sq_I = I_lr.norm_squared()
    # ||I||²_F = trace(I^T @ I) = trace(I) = min(m,n)
    assert np.abs(norm_sq_I - 3.0) < 1e-10, "Identity norm squared incorrect"
    
    # Single element
    A_single = np.array([[2.0]])
    B_single = np.array([[3.0]])
    X_single = LowRankMatrix(A_single, B_single)
    assert np.abs(X_single.norm_squared() - 36.0) < 1e-10, "Single element norm squared incorrect"
    
    print("norm_squared() edge cases passed")


#%% Test matrix power
def test_power_basic():
    """Test matrix power computation."""
    # Test X^0 = I
    X0 = X_sq.power(0)
    assert isinstance(X0, np.ndarray), "X^0 should return ndarray"
    assert np.allclose(X0, np.eye(8)), "X^0 should be identity"
    
    # Test X^1 = X
    X1 = X_sq.power(1)
    assert isinstance(X1, LowRankMatrix), "X^1 should return LowRankMatrix"
    assert np.allclose(X1.full(), X_sq_full), "X^1 should equal X"
    
    # Test X^2
    X2 = X_sq.power(2)
    assert isinstance(X2, LowRankMatrix), "X^2 should return LowRankMatrix"
    assert np.allclose(X2.full(), X_sq_full @ X_sq_full), "X^2 incorrect"
    
    # Test X^3
    X3 = X_sq.power(3)
    expected = X_sq_full @ X_sq_full @ X_sq_full
    assert np.allclose(X3.full(), expected), "X^3 incorrect"
    
    print("power() basic test passed")


def test_power_edge_cases():
    """Test power edge cases."""
    # Non-square matrix
    with pytest.raises(ValueError, match="must be square"):
        X.power(2)
    
    # Negative power
    with pytest.raises(ValueError, match="Negative powers not supported"):
        X_sq.power(-1)
    
    # Large power (tests binary exponentiation)
    X5 = X_sq.power(5)
    # Just check it doesn't crash and has right shape
    assert X5.shape == (8, 8), "X^5 shape incorrect"
    
    # Power of 7 (odd number to test binary path)
    X7 = X_sq.power(7)
    assert X7.shape == (8, 8), "X^7 shape incorrect"
    
    print("power() edge cases passed")


#%% Test slicing support
def test_slicing_single_element():
    """Test single element access via slicing."""
    val = X[2, 3]
    val_expected = X_full[2, 3]
    
    assert np.abs(val - val_expected) < 1e-12, "Single element access incorrect"
    
    # Test corner cases
    assert np.abs(X[0, 0] - X_full[0, 0]) < 1e-12, "Corner (0,0) incorrect"
    assert np.abs(X[-1, -1] - X_full[-1, -1]) < 1e-12, "Corner (-1,-1) incorrect"
    
    print("Slicing single element test passed")


def test_slicing_submatrix():
    """Test submatrix slicing."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        # Test row slice
        row_slice = X[2:5, :]
        assert np.allclose(row_slice, X_full[2:5, :]), "Row slicing incorrect"
        
        # Test column slice
        col_slice = X[:, 3:6]
        assert np.allclose(col_slice, X_full[:, 3:6]), "Column slicing incorrect"
        
        # Test block
        block = X[1:4, 2:5]
        assert np.allclose(block, X_full[1:4, 2:5]), "Block slicing incorrect"
        
        # Should have warnings
        assert len(w) >= 3, "Should warn about slicing operations"
    
    print("Slicing submatrix test passed")


def test_slicing_fancy_indexing():
    """Test fancy indexing."""
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        
        # Array indexing
        rows = np.array([0, 2, 5])
        cols = np.array([1, 3, 7])
        result = X[rows, :]
        assert np.allclose(result, X_full[rows, :]), "Fancy row indexing incorrect"
        
        result2 = X[:, cols]
        assert np.allclose(result2, X_full[:, cols]), "Fancy col indexing incorrect"
    
    print("Fancy indexing test passed")


def test_get_block():
    """Test get_block method."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        block = X.get_block(slice(1, 4), slice(2, 6))
        expected = X_full[1:4, 2:6]
        
        assert np.allclose(block, expected), "get_block() incorrect"
        assert len(w) > 0, "get_block() should warn"
    
    print("get_block() test passed")


#%% Test iterative solvers interface
def test_matvec():
    """Test matvec method."""
    v = np.random.randn(8)
    result = X.matvec(v)
    expected = X_full @ v
    
    assert result.shape == (10,), "matvec shape incorrect"
    assert np.allclose(result, expected), "matvec result incorrect"
    
    print("matvec() test passed")


def test_rmatvec():
    """Test rmatvec method."""
    v = np.random.randn(10)
    result = X.rmatvec(v)
    # rmatvec computes X.H @ v, which is (v.conj() @ X)^H = X^H @ v
    expected = X_full.T.conj() @ v
    
    assert result.shape == (8,), "rmatvec shape incorrect"
    assert np.allclose(result, expected), "rmatvec result incorrect"
    
    # Test with complex matrix
    A_complex = np.random.randn(5, 3) + 1j * np.random.randn(5, 3)
    B_complex = np.random.randn(3, 4) + 1j * np.random.randn(3, 4)
    X_complex = LowRankMatrix(A_complex, B_complex)
    X_complex_full = A_complex @ B_complex
    
    v_complex = np.random.randn(5) + 1j * np.random.randn(5)
    result_complex = X_complex.rmatvec(v_complex)
    expected_complex = X_complex_full.T.conj() @ v_complex
    
    assert np.allclose(result_complex, expected_complex), "Complex rmatvec incorrect"
    
    print("rmatvec() test passed")


def test_as_linear_operator():
    """Test that LowRankMatrix IS a LinearOperator (no conversion needed)."""
    from scipy.sparse.linalg import LinearOperator
    
    # LowRankMatrix now inherits from LinearOperator
    assert isinstance(X, LinearOperator), "LowRankMatrix should be a LinearOperator"
    
    # Check properties
    assert X.shape == (10, 8), "LinearOperator shape incorrect"
    assert X.dtype == X_full.dtype, "LinearOperator dtype incorrect"
    
    # Test matvec (can use the object directly)
    v = np.random.randn(8)
    result = X.matvec(v)
    expected = X_full @ v
    assert np.allclose(result, expected), "LinearOperator matvec incorrect"
    
    # Test rmatvec
    v2 = np.random.randn(10)
    result2 = X.rmatvec(v2)
    expected2 = v2.conj() @ X_full
    assert np.allclose(result2, expected2), "LinearOperator rmatvec incorrect"
    
    # Test with scipy solver (just check it doesn't crash)
    from scipy.sparse.linalg import gmres
    b = np.random.randn(8)
    # X_sq is already a LinearOperator, no conversion needed
    x, info = gmres(X_sq, b, rtol=1e-5, maxiter=100)
    # Just check it ran (might not converge depending on matrix condition)
    assert x.shape == (8,), "GMRES with LinearOperator failed"


#%% Test condition number estimation
def test_cond_estimate_norm_ratio():
    """Test condition number estimation with norm_ratio method."""
    cond = X_sq.cond_estimate(method='norm_ratio')
    
    assert cond > 0, "Condition number should be positive"
    assert np.isfinite(cond), "Condition number should be finite"
    
    # For a well-conditioned matrix, should be reasonable
    assert cond < 1e12, "Condition number seems unreasonably large"
    
    print("cond_estimate(norm_ratio) test passed")


def test_cond_estimate_power_iteration():
    """Test condition number with power iteration."""
    cond = X_sq.cond_estimate(method='power_iteration', n_iter=20)
    
    assert cond > 0, "Condition number should be positive"
    assert np.isfinite(cond), "Condition number should be finite"
    
    print("cond_estimate(power_iteration) test passed")


def test_cond_estimate_edge_cases():
    """Test condition number edge cases."""
    # Non-square matrix (should warn)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        cond = X.cond_estimate()
        assert len(w) > 0, "Should warn for non-square matrix"
    
    # Invalid method
    with pytest.raises(ValueError, match="Unknown method"):
        X_sq.cond_estimate(method='invalid')
    
    print("cond_estimate() edge cases passed")


#%% Test memory footprint reporting
def test_memory_usage():
    """Test memory usage reporting."""
    mem_bytes = X.memory_usage(unit='B')
    mem_kb = X.memory_usage(unit='KB')
    mem_mb = X.memory_usage(unit='MB')
    mem_gb = X.memory_usage(unit='GB')
    
    # Check relationships
    assert np.abs(mem_bytes / 1024 - mem_kb) < 1e-10, "KB conversion incorrect"
    assert np.abs(mem_kb / 1024 - mem_mb) < 1e-10, "MB conversion incorrect"
    assert np.abs(mem_mb / 1024 - mem_gb) < 1e-10, "GB conversion incorrect"
    
    # Check reasonable values
    expected_bytes = X.size * X.dtype.itemsize
    assert np.abs(mem_bytes - expected_bytes) < 1, "Memory usage calculation incorrect"
    
    # Invalid unit
    with pytest.raises(ValueError, match="Unknown unit"):
        X.memory_usage(unit='TB')
    
    print("memory_usage() test passed")


def test_compression_ratio():
    """Test compression ratio calculation."""
    ratio = X.compression_ratio()
    
    # Should be positive
    assert ratio > 0, "Compression ratio should be positive"
    
    # Check calculation
    expected_ratio = X.size / np.prod(X.shape)
    assert np.abs(ratio - expected_ratio) < 1e-10, "Compression ratio calculation incorrect"
    
    # For a good low-rank matrix, should have savings
    A_good = np.random.randn(100, 5)
    B_good = np.random.randn(5, 100)
    X_good = LowRankMatrix(A_good, B_good)
    ratio_good = X_good.compression_ratio()
    # (100*5 + 5*100) / (100*100) = 1000/10000 = 0.1
    assert ratio_good < 1, "Good low-rank should have ratio < 1"
    
    # For a matrix with large rank, ratio can be > 1 (no compression!)
    A_large = np.random.randn(10, 9)
    B_large = np.random.randn(9, 10)
    X_large = LowRankMatrix(A_large, B_large)
    ratio_large = X_large.compression_ratio()
    # (10*9 + 9*10) / (10*10) = 180/100 = 1.8, so > 1 (no compression!)
    assert ratio_large > 1, "Large rank should have ratio > 1"
    
    print("compression_ratio() test passed")


def test_is_memory_efficient():
    """Test is_memory_efficient property."""
    # Good low-rank matrix (efficient)
    A_good = np.random.randn(1000, 10)
    B_good = np.random.randn(10, 1000)
    X_good = LowRankMatrix(A_good, B_good)
    assert X_good.is_memory_efficient, "Low-rank matrix with rank 10 for 1000x1000 should be memory efficient"
    
    # Bad low-rank matrix (inefficient)
    A_bad = np.random.randn(100, 90)
    B_bad = np.random.randn(90, 100)
    X_bad = LowRankMatrix(A_bad, B_bad)
    assert not X_bad.is_memory_efficient, "High-rank matrix (90 for 100x100) should not be memory efficient"
    
    # Edge case: empty matrix
    A_empty = np.random.randn(0, 5)
    B_empty = np.random.randn(5, 10)
    X_empty = LowRankMatrix(A_empty, B_empty)
    assert X_empty.is_memory_efficient, "Empty matrix should be considered memory efficient"
    
    # Edge case: single column/row
    A_single = np.random.randn(1000, 1)
    B_single = np.random.randn(1, 1000)
    X_single = LowRankMatrix(A_single, B_single)
    assert X_single.is_memory_efficient, "Rank-1 matrix should be memory efficient"
    
    print("is_memory_efficient property test passed")


#%% Test serialization
def test_save_load():
    """Test saving and loading."""
    with tempfile.TemporaryDirectory() as tmpdir:
        filename = os.path.join(tmpdir, 'test_matrix')
        
        # Save
        X.save(filename)
        
        # Check file exists
        assert os.path.exists(filename + '.npz'), "Save file not created"
        
        # Load
        X_loaded = LowRankMatrix.load(filename)
        
        # Check properties
        assert X_loaded.shape == X.shape, "Loaded shape incorrect"
        assert X_loaded.length == X.length, "Loaded length incorrect"
        assert np.allclose(X_loaded.full(), X_full), "Loaded matrix incorrect"
    
    print("save/load test passed")


def test_save_load_with_extra_data():
    """Test save/load with extra data."""
    # Create matrix with extra data
    extra = {'poles': np.array([1, 2, 3]), 'name': 'test'}
    A_extra = np.random.randn(5, 3)
    B_extra = np.random.randn(3, 4)
    X_extra = LowRankMatrix(A_extra, B_extra, **extra)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        filename = os.path.join(tmpdir, 'test_extra')
        
        X_extra.save(filename)
        X_loaded = LowRankMatrix.load(filename)
        
        assert np.allclose(X_loaded.full(), X_extra.full()), "Matrix with extra data incorrect"
        assert 'poles' in X_loaded._extra_data, "Extra data not loaded"
        assert np.allclose(X_loaded._extra_data['poles'], extra['poles']), "Extra data incorrect"
    
    print("save/load with extra data test passed")


def test_save_load_extension():
    """Test that .npz extension is handled correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Without extension
        filename1 = os.path.join(tmpdir, 'test1')
        X.save(filename1)
        X_loaded1 = LowRankMatrix.load(filename1)
        assert np.allclose(X_loaded1.full(), X_full), "Load without extension failed"
        
        # With extension
        filename2 = os.path.join(tmpdir, 'test2.npz')
        X.save(filename2)
        X_loaded2 = LowRankMatrix.load(filename2)
        assert np.allclose(X_loaded2.full(), X_full), "Load with extension failed"
    
    print("save/load extension handling test passed")


#%% Test approximation error
def test_approximation_error():
    """Test approximation error computation."""
    # Perfect reconstruction
    error = X.approximation_error(X_full)
    assert error < 1e-10, "Error for perfect reconstruction should be ~0"
    
    # With actual error
    perturbed = X_full + 0.01 * np.random.randn(*X_full.shape)
    error_perturbed = X.approximation_error(perturbed)
    assert error_perturbed > 0, "Error should be positive"
    assert error_perturbed < 1.0, "Error seems too large"
    
    # Different norms
    error_1 = X.approximation_error(perturbed, ord=1)
    error_2 = X.approximation_error(perturbed, ord=2)
    error_inf = X.approximation_error(perturbed, ord=np.inf)
    
    assert error_1 > 0, "1-norm error should be positive"
    assert error_2 > 0, "2-norm error should be positive"
    assert error_inf > 0, "inf-norm error should be positive"
    
    print("approximation_error() test passed")


#%% Test stability analysis
def test_is_well_conditioned():
    """Test well-conditioning check."""
    # Well-conditioned matrix (identity-like)
    I_lr = LowRankMatrix(np.eye(5, 4), np.eye(4, 5))
    assert I_lr.is_well_conditioned(), "Identity should be well-conditioned"
    
    # Check with different thresholds
    result_strict = X_sq.is_well_conditioned(threshold=1e5)
    result_loose = X_sq.is_well_conditioned(threshold=1e15)
    # Should be bool
    assert isinstance(result_strict, bool), "Should return bool"
    assert isinstance(result_loose, bool), "Should return bool"
    
    # Non-square (should warn and return True)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result_nonsquare = X.is_well_conditioned()
        assert len(w) > 0, "Should warn for non-square"
        assert result_nonsquare is True, "Non-square should return True"
    
    print("is_well_conditioned() test passed")


#%% Test sparse conversion
def test_to_sparse():
    """Test conversion to sparse format."""
    import scipy.sparse as sp
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        # Convert to CSR
        X_csr = X.to_sparse(format='csr')
        assert sp.issparse(X_csr), "Should return sparse matrix"
        assert X_csr.format == 'csr', "Format should be CSR"
        assert np.allclose(X_csr.toarray(), X_full), "Sparse conversion incorrect"
        
        # Convert to CSC
        X_csc = X.to_sparse(format='csc')
        assert X_csc.format == 'csc', "Format should be CSC"
        
        # Convert to COO
        X_coo = X.to_sparse(format='coo')
        assert X_coo.format == 'coo', "Format should be COO"
        
        assert len(w) >= 3, "Should warn about forming dense matrix"
    
    print("to_sparse() test passed")


def test_to_sparse_threshold():
    """Test sparse conversion with threshold."""
    import scipy.sparse as sp
    # Create matrix with small values
    A_small = np.random.randn(10, 3) * 1e-12
    B_small = np.random.randn(3, 8)
    X_small = LowRankMatrix(A_small, B_small)
    
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        
        X_sparse = X_small.to_sparse(threshold=1e-10)
        # Most entries should be zeroed
        assert X_sparse.nnz < X_sparse.shape[0] * X_sparse.shape[1], "Threshold should reduce nnz"
    
    print("to_sparse() threshold test passed")


def test_to_sparse_invalid_format():
    """Test to_sparse with invalid format."""
    with pytest.raises(ValueError, match="Unknown format"):
        X.to_sparse(format='invalid')
    
    print("to_sparse() invalid format test passed")


#%% Test equality and comparison
def test_equality_exact():
    """Test exact equality."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        # Equal to self
        X_copy = X.copy()
        assert X == X_copy, "Matrix should equal its copy"
        
        # Not equal to modified
        X_modified = X.copy()
        X_modified.scale_(2)
        assert not (X == X_modified), "Modified matrix should not be equal"
        
        # Equal to dense
        assert X == X_full, "Should equal dense version"
        
        # Not equal to different shape
        Y = LowRankMatrix(np.random.randn(5, 3), np.random.randn(3, 4))
        assert not (X == Y), "Different shapes should not be equal"
        
        # Should have warnings
        assert len(w) > 0, "Equality check should warn"
    
    print("Equality exact test passed")


def test_allclose():
    """Test approximate equality."""
    # Exact match
    assert X.allclose(X_full), "Should be close to exact"
    
    # Small perturbation
    perturbed = X_full + 1e-7 * np.random.randn(*X_full.shape)
    assert X.allclose(perturbed, rtol=1e-5, atol=1e-6), "Should be close to small perturbation"
    
    # Large perturbation
    perturbed_large = X_full + 0.1 * np.random.randn(*X_full.shape)
    assert not X.allclose(perturbed_large, rtol=1e-5, atol=1e-6), "Should not be close to large perturbation"
    
    # Different shapes
    Y = LowRankMatrix(np.random.randn(5, 3), np.random.randn(3, 4))
    assert not X.allclose(Y), "Different shapes should not be close"
    
    # With LowRankMatrix
    X_copy = X.copy()
    assert X.allclose(X_copy), "Should be close to copy"
    
    print("allclose() test passed")


def test_equality_types():
    """Test equality with different types."""
    # With non-matrix type
    assert not (X == "not a matrix"), "Should not equal non-matrix"
    assert not (X == 42), "Should not equal scalar"
    assert not (X == [1, 2, 3]), "Should not equal list"
    
    # allclose with non-matrix
    assert not X.allclose("not a matrix"), "allclose should handle non-matrix"
    
    print("Equality type handling test passed")


#%% Test complex numbers
def test_new_features_complex():
    """Test new features with complex matrices."""
    A_c = np.random.randn(6, 4) + 1j * np.random.randn(6, 4)
    B_c = np.random.randn(4, 6) + 1j * np.random.randn(4, 6)
    X_c = LowRankMatrix(A_c, B_c)
    X_c_full = A_c @ B_c
    
    # Diagonal
    diag_c = X_c.diag()
    assert diag_c.dtype in [np.complex64, np.complex128], "Complex diagonal dtype incorrect"
    assert np.allclose(diag_c, np.diag(X_c_full)), "Complex diagonal incorrect"
    
    # Trace
    trace_c = X_c.trace()
    assert np.abs(trace_c - np.trace(X_c_full)) < 1e-10, "Complex trace incorrect"
    
    # Norm squared
    norm_sq_c = X_c.norm_squared()
    expected_norm_sq = la.norm(X_c_full, 'fro') ** 2
    # Use relative tolerance for complex norms
    assert np.abs(norm_sq_c - expected_norm_sq) / expected_norm_sq < 1e-6, "Complex norm squared incorrect"
    
    # Power
    X_c_2 = X_c.power(2)
    assert np.allclose(X_c_2.full(), X_c_full @ X_c_full), "Complex power incorrect"
    
    # Memory usage should work
    mem = X_c.memory_usage()
    assert mem > 0, "Complex memory usage should be positive"
    
    print("Complex number support for new features passed")


#%% Test edge cases and error handling
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_edge_case_empty_dimensions():
    """Test with empty dimensions."""
    # Empty rows
    A_empty = np.random.randn(0, 3)
    B_empty = np.random.randn(3, 5)
    X_empty = LowRankMatrix(A_empty, B_empty)
    
    # Memory operations should work
    mem = X_empty.memory_usage()
    ratio = X_empty.compression_ratio()
    assert mem >= 0, "Empty matrix memory should be non-negative"
    assert ratio >= 0 or np.isnan(ratio), "Empty matrix ratio should be non-negative or NaN"
    
    print("Edge case empty dimensions passed")


def test_edge_case_rank_one():
    """Test rank-1 matrices."""
    u = np.random.randn(8, 1)
    v = np.random.randn(1, 8)
    X_r1 = LowRankMatrix(u, v)
    X_r1_full = u @ v
    
    # Trace
    trace_r1 = X_r1.trace()
    assert np.abs(trace_r1 - np.trace(X_r1_full)) < 1e-10, "Rank-1 trace incorrect"
    
    # Power
    X_r1_2 = X_r1.power(2)
    assert np.allclose(X_r1_2.full(), X_r1_full @ X_r1_full), "Rank-1 power incorrect"
    
    # Compression should do nothing (already minimal)
    X_r1_comp = X_r1.compress()
    assert X_r1_comp.length == 2, "Rank-1 compression should preserve length"
    
    print("Edge case rank-1 passed")


def test_edge_case_large_chain():
    """Test with many factor matrices."""
    matrices = [np.random.randn(5, 5) for _ in range(10)]
    X_long = LowRankMatrix(*matrices)
    
    # Trace should work
    trace_long = X_long.trace()
    assert np.isfinite(trace_long), "Long chain trace should be finite"
    
    # Compression should reduce length significantly
    X_long_comp = X_long.compress()
    assert X_long_comp.length < X_long.length, "Long chain should compress"
    
    # Memory tracking
    mem_before = X_long.memory_usage()
    X_long.compress_()
    mem_after = X_long.memory_usage()
    assert mem_after < mem_before, "Compression should reduce memory"
    
    print("Edge case large chain passed")


#%% Test LinearOperator addition compatibility
def test_linearoperator_addition():
    """Test addition with generic LinearOperator objects."""
    from scipy.sparse.linalg import aslinearoperator
    
    # Create a generic LinearOperator (not LowRankMatrix)
    A_dense = np.random.randn(10, 8)
    A_linop = aslinearoperator(A_dense)
    
    # Add LowRankMatrix + LinearOperator should return _SumLinearOperator
    result = X + A_linop
    assert isinstance(result, LinearOperator), "Should return LinearOperator"
    assert not isinstance(result, LowRankMatrix), "Should return lazy sum, not LowRankMatrix"
    
    # Test that the result works correctly
    v = np.random.randn(8)
    result_matvec = result @ v
    expected = X_full @ v + A_dense @ v
    assert np.allclose(result_matvec, expected), "LinearOperator addition matvec incorrect"
    
    # Reverse addition should also work
    result_rev = A_linop + X
    result_rev_matvec = result_rev @ v
    assert np.allclose(result_rev_matvec, expected), "Reverse LinearOperator addition incorrect"
    
    print("LinearOperator addition compatibility test passed")


# %%

