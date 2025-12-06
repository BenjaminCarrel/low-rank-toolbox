"""
Test file for QuasiSVD class defined in low_rank_toolbox/matrices/quasi_svd.py

Author: Benjamin Carrel, University of Geneva, 2023
"""

#%% Imports
import numpy as np
import scipy.linalg as la
from lowrank import LowRankMatrix, QuasiSVD
from numpy import ndarray
import pytest
import warnings

# Configure pytest to ignore all warnings in this test file
pytestmark = pytest.mark.filterwarnings("ignore")


#%% Fixtures for creating test matrices
@pytest.fixture
def simple_quasisvd():
    """Create a simple QuasiSVD for basic tests."""
    np.random.seed(1234)
    A = np.random.randn(20, 4)
    B = np.random.randn(4, 18)
    Q1, _ = la.qr(A, mode='economic')
    Q2, _ = la.qr(B.T, mode='economic')
    # Use a non-diagonal S matrix (important for QuasiSVD)
    S = np.random.randn(4, 4)
    X = QuasiSVD(Q1, S, Q2)
    X_full = Q1 @ S @ Q2.T
    return X, X_full

@pytest.fixture
def rectangular_quasisvd():
    """Create a QuasiSVD with rectangular S matrix."""
    np.random.seed(5678)
    Q1, _ = la.qr(np.random.randn(15, 5), mode='economic')
    Q2, _ = la.qr(np.random.randn(10, 3), mode='economic')
    S = np.random.randn(5, 3)
    X = QuasiSVD(Q1, S, Q2)
    X_full = Q1 @ S @ Q2.T
    return X, X_full

@pytest.fixture
def complex_quasisvd():
    """Create a complex QuasiSVD."""
    np.random.seed(9999)
    Q1, _ = la.qr(np.random.randn(10, 3) + 1j * np.random.randn(10, 3), mode='economic')
    Q2, _ = la.qr(np.random.randn(8, 3) + 1j * np.random.randn(8, 3), mode='economic')
    S = np.random.randn(3, 3) + 1j * np.random.randn(3, 3)
    X = QuasiSVD(Q1, S, Q2)
    X_full = Q1 @ S @ Q2.T.conj()
    return X, X_full

#%% ===========================
#%% INITIALIZATION AND PROPERTIES
#%% ===========================

def test_initialization_basic(simple_quasisvd):
    """Test basic initialization and properties."""
    X, X_full = simple_quasisvd
    
    # Test dimensions
    assert X.shape == (20, 18), "Incorrect shape"
    assert X.ndim == 2, "Incorrect ndim"
    assert X.deepshape == (20, 4, 4, 18), "Incorrect deepshape"
    assert X.rank == 4, "Incorrect rank"
    
    # Test matrix aliases
    assert X.U.shape == (20, 4), "Incorrect U shape"
    assert X.S.shape == (4, 4), "Incorrect S shape"
    assert X.V.shape == (18, 4), "Incorrect V shape"
    assert X.Vh.shape == (4, 18), "Incorrect Vh shape"
    assert X.Vt.shape == (4, 18), "Incorrect Vt shape"
    assert X.Ut.shape == (4, 20), "Incorrect Ut shape"
    assert X.Uh.shape == (4, 20), "Incorrect Uh shape"
    
    # Test full reconstruction
    assert np.allclose(X.full(), X_full), "Incorrect full() method"


def test_initialization_rectangular_s(rectangular_quasisvd):
    """Test initialization with rectangular S matrix."""
    X, X_full = rectangular_quasisvd
    
    assert X.shape == (15, 10), "Incorrect shape for rectangular S"
    # Deepshape is (m, r_U, r_S, q_S, q_V, n) = (15, 5, 5, 3, 3, 10)
    # But internally stored as (15, 5, 3, 10) since S is (5, 3)
    assert X.deepshape == (15, 5, 3, 10), "Incorrect deepshape for rectangular S"
    assert X.rank == 3, "Incorrect rank for rectangular S"
    assert np.allclose(X.full(), X_full), "Incorrect full reconstruction for rectangular S"


def test_initialization_complex(complex_quasisvd):
    """Test initialization with complex matrices."""
    X, X_full = complex_quasisvd
    
    assert X.shape == (10, 8), "Incorrect shape for complex QuasiSVD"
    assert X.U.dtype == np.complex128, "U should be complex"
    assert X.V.dtype == np.complex128, "V should be complex"
    assert np.allclose(X.full(), X_full), "Incorrect full reconstruction for complex"
    
    # Test conjugate transposes
    assert np.allclose(X.Vh, X.V.T.conj()), "Vh should be V.T.conj()"
    assert np.allclose(X.Uh, X.U.T.conj()), "Uh should be U.T.conj()"


def test_initialization_errors():
    """Test that initialization raises appropriate errors."""
    np.random.seed(42)
    Q1, _ = la.qr(np.random.randn(10, 3), mode='economic')
    Q2, _ = la.qr(np.random.randn(8, 3), mode='economic')
    
    # Test 1D array warning (error because we want 2D S)
    s_1d = np.array([1.0, 2.0, 3.0])
    with pytest.raises(TypeError, match="2D array"):
        QuasiSVD(Q1, s_1d, Q2)
    
    # Test dimension mismatch
    S_wrong = np.random.randn(4, 3)  # Wrong first dimension
    with pytest.raises(ValueError, match="Dimension mismatch"):
        QuasiSVD(Q1, S_wrong, Q2)
    
    S_wrong2 = np.random.randn(3, 4)  # Wrong second dimension
    with pytest.raises(ValueError, match="Dimension mismatch"):
        QuasiSVD(Q1, S_wrong2, Q2)
    
    # Test dtype mismatch
    Q1_complex = Q1.astype(np.complex128)
    S = np.random.randn(3, 3)
    with pytest.raises(TypeError, match="same dtype"):
        QuasiSVD(Q1_complex, S, Q2)


#%% ===========================
#%% PROPERTIES AND CHECKS
#%% ===========================

def test_is_orthogonal(simple_quasisvd):
    """Test orthogonality check."""
    X, _ = simple_quasisvd
    
    # Should be orthogonal since we used QR decomposition
    assert X.is_orthogonal(), "U and V should be orthogonal"
    
    # Test caching
    assert X.is_orthogonal() is True, "Result should be cached"


def test_is_symmetric():
    """Test symmetry check."""
    np.random.seed(111)
    # Create symmetric matrix
    Q, _ = la.qr(np.random.randn(10, 4), mode='economic')
    S_sym = np.random.randn(4, 4)
    S_sym = (S_sym + S_sym.T) / 2  # Symmetrize
    X_sym = QuasiSVD(Q, S_sym, Q)  # U = V for symmetry
    
    assert X_sym.is_symmetric(), "Matrix should be symmetric"
    
    # Non-square matrix cannot be symmetric
    Q1, _ = la.qr(np.random.randn(10, 3), mode='economic')
    Q2, _ = la.qr(np.random.randn(8, 3), mode='economic')
    S = np.random.randn(3, 3)
    X_nonsym = QuasiSVD(Q1, S, Q2)
    
    assert not X_nonsym.is_symmetric(), "Non-square matrix cannot be symmetric"


def test_K_L_properties(simple_quasisvd, rectangular_quasisvd, complex_quasisvd):
    """Test K and L properties."""
    # Test with simple QuasiSVD
    X_simple, _ = simple_quasisvd
    K_simple = X_simple.K
    L_simple = X_simple.L
    assert K_simple.shape == (X_simple.shape[0], X_simple.S.shape[1]), "Incorrect shape for K (simple)"
    assert L_simple.shape == (X_simple.shape[1], X_simple.S.shape[0]), "Incorrect shape for L (simple)"
    assert np.allclose(K_simple, X_simple.U @ X_simple.S), "K property is not U @ S"
    assert np.allclose(L_simple, X_simple.V @ X_simple.S.T), "L property is not V @ S.T"

    # Test with rectangular S matrix
    X_rect, _ = rectangular_quasisvd
    K_rect = X_rect.K
    L_rect = X_rect.L
    assert K_rect.shape == (15, 3), "Incorrect shape for K (rectangular)"
    assert L_rect.shape == (10, 5), "Incorrect shape for L (rectangular)"
    assert np.allclose(K_rect, X_rect.U @ X_rect.S), "K property is not U @ S (rectangular)"
    assert np.allclose(L_rect, X_rect.V @ X_rect.S.T), "L property is not V @ S.T (rectangular)"

    # Test with complex QuasiSVD
    X_complex, _ = complex_quasisvd
    K_complex = X_complex.K
    L_complex = X_complex.L
    assert K_complex.shape == (10, 3), "Incorrect shape for K (complex)"
    assert L_complex.shape == (8, 3), "Incorrect shape for L (complex)"
    assert np.allclose(K_complex, X_complex.U @ X_complex.S), "K property is not U @ S (complex)"
    assert np.allclose(L_complex, X_complex.V @ X_complex.S.T), "L property is not V @ S.T (complex)"


def test_is_singular():
    """Test singularity check."""
    np.random.seed(222)
    Q1, _ = la.qr(np.random.randn(10, 3), mode='economic')
    Q2, _ = la.qr(np.random.randn(8, 3), mode='economic')
    
    # Non-singular S (use non-diagonal matrix)
    S_nonsingular = np.eye(3) + np.random.randn(3, 3) * 0.5
    X_nonsingular = QuasiSVD(Q1, S_nonsingular, Q2)
    assert not X_nonsingular.is_singular(), "S should not be singular"
    
    # Singular S (make it clearly singular but not diagonal)
    S_singular = np.random.randn(3, 3) * 0.1
    S_singular[2, :] = S_singular[0, :] + S_singular[1, :]  # Make rank-deficient
    X_singular = QuasiSVD(Q1, S_singular, Q2)
    assert X_singular.is_singular(), "S should be singular"


def test_sing_vals(simple_quasisvd):
    """Test singular values computation."""
    X, _ = simple_quasisvd
    
    # Compute singular values
    sing_vals = X.sing_vals()
    
    # Should match SVD of S
    s_expected = la.svdvals(X.S)
    assert np.allclose(sing_vals, s_expected), "Singular values should match svdvals(S)"
    
    # Test caching
    assert X.sing_vals() is sing_vals, "Singular values should be cached"


def test_norms(simple_quasisvd):
    """Test norm computations."""
    X, X_full = simple_quasisvd
    
    # Frobenius norm
    assert abs(X.norm('fro') - la.norm(X_full, 'fro')) < 1e-10, "Incorrect Frobenius norm"
    
    # Nuclear norm
    assert abs(X.norm('nuc') - la.norm(X_full, 'nuc')) < 1e-10, "Incorrect nuclear norm"
    
    # 2-norm
    assert abs(X.norm(2) - la.norm(X_full, 2)) < 1e-10, "Incorrect 2-norm"
    
    # Test caching
    X.norm('fro')
    X.norm(2)
    assert 'fro' in X._cache, "Norms should be cached in _cache"
    assert 2 in X._cache, "Norms should be cached in _cache"


#%% ===========================
#%% ADDITION AND SUBTRACTION
#%% ===========================

def test_addition_quasisvd(simple_quasisvd):
    """Test addition of two QuasiSVD matrices."""
    X, X_full = simple_quasisvd
    
    # Addition
    Y = X + X
    assert isinstance(Y, QuasiSVD), "Addition should return QuasiSVD"
    assert Y.rank == 2 * X.rank, "Rank should double after addition"
    assert Y.is_orthogonal(), "Addition should preserve orthogonality"
    assert np.allclose(Y.full(), 2 * X_full), "Addition result incorrect"
    
    # Right addition
    Y = X.__radd__(X)
    assert isinstance(Y, QuasiSVD), "Right addition should return QuasiSVD"
    assert np.allclose(Y.full(), 2 * X_full), "Right addition result incorrect"


def test_subtraction_quasisvd(simple_quasisvd):
    """Test subtraction of two QuasiSVD matrices."""
    X, X_full = simple_quasisvd
    
    # Subtraction
    Y = X - X
    assert isinstance(Y, QuasiSVD), "Subtraction should return QuasiSVD"
    assert Y.rank == 2 * X.rank, "Rank should double after subtraction (algebraic consistency)"
    assert Y.is_orthogonal(), "Subtraction should preserve orthogonality"
    assert np.allclose(Y.full(), np.zeros_like(X_full)), "X - X should be zero"
    
    # Right subtraction
    Y = X.__rsub__(X)
    assert isinstance(Y, QuasiSVD), "Right subtraction should return QuasiSVD"
    assert np.allclose(Y.full(), np.zeros_like(X_full)), "Right subtraction result incorrect"


def test_addition_with_dense(simple_quasisvd):
    """Test addition with dense arrays."""
    X, X_full = simple_quasisvd
    
    # Addition
    Y_dense = np.random.randn(20, 18)
    Y = X + Y_dense
    assert isinstance(Y, ndarray), "Addition with dense should return ndarray"
    assert np.allclose(Y, X_full + Y_dense), "Addition with dense incorrect"
    
    # Right addition
    Y = Y_dense + X
    assert isinstance(Y, ndarray), "Right addition with dense should return ndarray"
    assert np.allclose(Y, X_full + Y_dense), "Right addition with dense incorrect"


def test_subtraction_with_dense(simple_quasisvd):
    """Test subtraction with dense arrays."""
    X, X_full = simple_quasisvd
    
    # Subtraction
    Y_dense = np.random.randn(20, 18)
    Y = X - Y_dense
    assert isinstance(Y, ndarray), "Subtraction with dense should return ndarray"
    assert np.allclose(Y, X_full - Y_dense), "Subtraction with dense incorrect"
    
    # Right subtraction
    Y = Y_dense - X
    assert isinstance(Y, ndarray), "Right subtraction with dense should return ndarray"
    assert np.allclose(Y, Y_dense - X_full), "Right subtraction with dense incorrect"


#%% ===========================
#%% SCALAR MULTIPLICATION
#%% ===========================

def test_scalar_multiplication(simple_quasisvd):
    """Test scalar multiplication."""
    X, X_full = simple_quasisvd
    
    # Float multiplication
    Y = X * 2.5
    assert isinstance(Y, QuasiSVD), "Scalar multiplication should return QuasiSVD"
    assert np.allclose(Y.full(), 2.5 * X_full), "Scalar multiplication incorrect"
    
    # Right multiplication
    Y = 2.5 * X
    assert isinstance(Y, QuasiSVD), "Right scalar multiplication should return QuasiSVD"
    assert np.allclose(Y.full(), 2.5 * X_full), "Right scalar multiplication incorrect"
    
    # Negative scalar
    Y = X * (-1)
    assert isinstance(Y, QuasiSVD), "Negative scalar multiplication should return QuasiSVD"
    assert np.allclose(Y.full(), -X_full), "Negative scalar multiplication incorrect"


def test_scalar_multiplication_complex(complex_quasisvd):
    """Test scalar multiplication with complex numbers."""
    X, X_full = complex_quasisvd
    
    # Complex scalar
    c = 1 + 2j
    Y = X * c
    assert isinstance(Y, QuasiSVD), "Complex scalar multiplication should return QuasiSVD"
    assert Y.S.dtype == np.complex128, "S should remain complex"
    assert np.allclose(Y.full(), c * X_full), "Complex scalar multiplication incorrect"


def test_inplace_scalar_multiplication(simple_quasisvd):
    """Test in-place scalar multiplication."""
    X, X_full = simple_quasisvd
    
    # In-place multiplication
    X *= 3.0
    assert isinstance(X, QuasiSVD), "In-place multiplication should return QuasiSVD"
    assert np.allclose(X.full(), 3.0 * X_full), "In-place multiplication incorrect"


#%% ===========================
#%% DOT PRODUCT (MATRIX MULTIPLICATION)
#%% ===========================

def test_dot_with_vector(simple_quasisvd):
    """Test matrix-vector multiplication."""
    X, X_full = simple_quasisvd
    np.random.seed(333)
    v = np.random.randn(18)
    
    result = X.dot(v)
    assert result.shape == (20,), "Incorrect shape for matrix-vector product"
    assert np.allclose(result, X_full @ v), "Matrix-vector multiplication incorrect"
    
    # Test dense output
    result_dense = X.dot(v, dense_output=True)
    assert isinstance(result_dense, ndarray), "dense_output should return ndarray"
    assert np.allclose(result_dense, X_full @ v), "Dense output incorrect"


def test_dot_with_dense_matrix(simple_quasisvd):
    """Test multiplication with dense matrix."""
    X, X_full = simple_quasisvd
    np.random.seed(444)
    Y = np.random.randn(18, 17)
    
    result = X.dot(Y)
    assert result.shape == (20, 17), "Incorrect shape for matrix-matrix product"
    assert isinstance(result, LowRankMatrix), "Should return LowRankMatrix"
    assert not isinstance(result, QuasiSVD), "Should not return QuasiSVD"
    assert np.allclose(result.full(), X_full @ Y), "Matrix-matrix multiplication incorrect"
    
    # Test dense output
    result_dense = X.dot(Y, dense_output=True)
    assert isinstance(result_dense, ndarray), "dense_output should return ndarray"
    assert np.allclose(result_dense, X_full @ Y), "Dense output incorrect"


def test_dot_with_quasisvd(simple_quasisvd):
    """Test multiplication between two QuasiSVD matrices."""
    X, X_full = simple_quasisvd
    np.random.seed(555)
    
    # Create another QuasiSVD (transposed to be compatible)
    Q1, _ = la.qr(np.random.randn(18, 3), mode='economic')
    Q2, _ = la.qr(np.random.randn(15, 3), mode='economic')
    S = np.random.randn(3, 3)
    Y = QuasiSVD(Q1, S, Q2)
    Y_full = Y.full()
    
    result = X.dot(Y)
    assert isinstance(result, QuasiSVD), "QuasiSVD-QuasiSVD multiplication should return QuasiSVD"
    assert result.shape == (20, 15), "Incorrect shape"
    assert result.rank == min(X.rank, Y.rank), "Rank should be min of input ranks"
    assert np.allclose(result.full(), X_full @ Y_full), "Multiplication result incorrect"


def test_dot_with_lowrank(simple_quasisvd):
    """Test multiplication with generic LowRankMatrix."""
    X, X_full = simple_quasisvd
    np.random.seed(666)
    
    Y = LowRankMatrix(np.random.randn(18, 5), np.random.randn(5, 6), 
                      np.random.randn(6, 4), np.random.randn(4, 17))
    Y_full = Y.full()
    
    result = X.dot(Y)
    assert result.shape == (20, 17), "Incorrect shape"
    assert isinstance(result, LowRankMatrix), "Should return LowRankMatrix"
    assert not isinstance(result, QuasiSVD), "Should not return QuasiSVD"
    assert np.allclose(result.full(), X_full @ Y_full), "Multiplication result incorrect"


def test_dot_left_side(simple_quasisvd):
    """Test left-side multiplication (other @ self)."""
    X, X_full = simple_quasisvd
    np.random.seed(777)
    
    Y = np.random.randn(15, 20)
    
    result = X.dot(Y, side='left')
    assert result.shape == (15, 18), "Incorrect shape for left multiplication"
    # Result is LowRankMatrix, convert to dense for comparison
    result_dense = result.full() if hasattr(result, 'full') else result
    assert np.allclose(result_dense, Y @ X_full), "Left multiplication incorrect"
    
    # Test with alias 'opposite'
    result2 = X.dot(Y, side='opposite')
    result2_dense = result2.full() if hasattr(result2, 'full') else result2
    assert np.allclose(result2_dense, result_dense), "side='opposite' should be same as side='left'"


#%% ===========================
#%% HADAMARD PRODUCT
#%% ===========================

def test_hadamard_quasisvd(simple_quasisvd):
    """Test Hadamard product between two QuasiSVD matrices."""
    X, X_full = simple_quasisvd
    
    result = X.hadamard(X)
    assert isinstance(result, QuasiSVD), "Hadamard product should return QuasiSVD"
    assert result.rank == X.rank ** 2, "Rank should be product of input ranks"
    assert np.allclose(result.full(), X_full * X_full), "Hadamard product incorrect"


def test_hadamard_dense(simple_quasisvd):
    """Test Hadamard product with dense array."""
    X, X_full = simple_quasisvd
    np.random.seed(888)
    Y = np.random.randn(20, 18)
    
    result = X.hadamard(Y)
    assert isinstance(result, ndarray), "Hadamard with dense should return ndarray"
    assert np.allclose(result, X_full * Y), "Hadamard product with dense incorrect"


def test_hadamard_errors(simple_quasisvd):
    """Test Hadamard product error handling."""
    X, _ = simple_quasisvd
    
    # Wrong shape
    Y = np.random.randn(15, 15)
    with pytest.raises(ValueError, match="same shape"):
        X.hadamard(Y)


#%% ===========================
#%% MULTI_ADD
#%% ===========================

def test_multi_add_basic():
    """Test multi_add with multiple QuasiSVD matrices."""
    np.random.seed(1111)
    
    # Create three QuasiSVD matrices
    matrices = []
    fulls = []
    for i in range(3):
        Q1, _ = la.qr(np.random.randn(10, 2), mode='economic')
        Q2, _ = la.qr(np.random.randn(8, 2), mode='economic')
        S = np.random.randn(2, 2)
        X = QuasiSVD(Q1, S, Q2)
        matrices.append(X)
        fulls.append(X.full())
    
    # Multi-add without truncation
    result = QuasiSVD.multi_add(matrices, auto_truncate=False)
    assert isinstance(result, QuasiSVD), "multi_add should return QuasiSVD"
    assert result.rank == sum(m.rank for m in matrices), "Rank should be sum of input ranks"
    assert result.is_orthogonal(), "Result should have orthogonal U and V"
    assert np.allclose(result.full(), sum(fulls)), "Multi-add result incorrect"


def test_multi_add_subtraction():
    """Test that multi_add works for X - X (algebraic consistency)."""
    np.random.seed(2222)
    Q1, _ = la.qr(np.random.randn(10, 3), mode='economic')
    Q2, _ = la.qr(np.random.randn(8, 3), mode='economic')
    S = np.random.randn(3, 3)
    X = QuasiSVD(Q1, S, Q2)
    
    # Without auto-truncation: X - X should have rank 6 but represent zero
    result = QuasiSVD.multi_add([X, -X], auto_truncate=False)
    assert isinstance(result, QuasiSVD), "Should return QuasiSVD without truncation"
    assert result.rank == 2 * X.rank, "Rank should be 2*rank(X) without truncation"
    assert np.allclose(result.full(), np.zeros(X.shape)), "X - X should be zero"


def test_multi_add_errors():
    """Test multi_add error handling."""
    np.random.seed(3333)
    Q1, _ = la.qr(np.random.randn(10, 2), mode='economic')
    Q2, _ = la.qr(np.random.randn(8, 2), mode='economic')
    S = np.random.randn(2, 2)
    X1 = QuasiSVD(Q1, S, Q2)
    
    Q3, _ = la.qr(np.random.randn(12, 2), mode='economic')
    Q4, _ = la.qr(np.random.randn(8, 2), mode='economic')
    X2 = QuasiSVD(Q3, S, Q4)  # Different shape
    
    # Different shapes
    with pytest.raises(AssertionError, match="same shape"):
        QuasiSVD.multi_add([X1, X2])


#%% ===========================
#%% MULTI_DOT
#%% ===========================
def test_multi_dot_basic():
    """Test multi_dot with multiple QuasiSVD matrices."""
    np.random.seed(4444)
    
    # Create three compatible QuasiSVD matrices
    Q1, _ = la.qr(np.random.randn(10, 3), mode='economic')
    Q2, _ = la.qr(np.random.randn(8, 3), mode='economic')
    S1 = np.random.randn(3, 3)
    X1 = QuasiSVD(Q1, S1, Q2)
    
    Q3, _ = la.qr(np.random.randn(8, 2), mode='economic')
    Q4, _ = la.qr(np.random.randn(6, 2), mode='economic')
    S2 = np.random.randn(2, 2)
    X2 = QuasiSVD(Q3, S2, Q4)
    
    Q5, _ = la.qr(np.random.randn(6, 4), mode='economic')
    Q6, _ = la.qr(np.random.randn(5, 4), mode='economic')
    S3 = np.random.randn(4, 4)
    X3 = QuasiSVD(Q5, S3, Q6)
    
    # Multi-dot
    result = QuasiSVD.multi_dot([X1, X2, X3])
    assert isinstance(result, QuasiSVD), "multi_dot should return QuasiSVD"
    assert result.shape == (10, 5), "Incorrect shape"
    assert result.rank == min(X1.rank, X3.rank), "Rank should be min of first and last"
    
    # Check correctness
    expected = X1.full() @ X2.full() @ X3.full()
    assert np.allclose(result.full(), expected), "Multi-dot result incorrect"


def test_multi_dot_two_matrices():
    """Test multi_dot with two matrices."""
    np.random.seed(5555)
    Q1, _ = la.qr(np.random.randn(12, 4), mode='economic')
    Q2, _ = la.qr(np.random.randn(10, 4), mode='economic')
    S1 = np.random.randn(4, 4)
    X1 = QuasiSVD(Q1, S1, Q2)
    
    Q3, _ = la.qr(np.random.randn(10, 3), mode='economic')
    Q4, _ = la.qr(np.random.randn(7, 3), mode='economic')
    S2 = np.random.randn(3, 3)
    X2 = QuasiSVD(Q3, S2, Q4)
    
    result = QuasiSVD.multi_dot([X1, X2])
    assert result.shape == (12, 7), "Incorrect shape"
    assert np.allclose(result.full(), X1.full() @ X2.full()), "Two-matrix multi_dot incorrect"


def test_multi_dot_errors():
    """Test multi_dot error handling."""
    np.random.seed(6666)
    Q1, _ = la.qr(np.random.randn(10, 2), mode='economic')
    Q2, _ = la.qr(np.random.randn(8, 2), mode='economic')
    S1 = np.random.randn(2, 2)
    X1 = QuasiSVD(Q1, S1, Q2)
    
    Q3, _ = la.qr(np.random.randn(10, 2), mode='economic')  # Wrong: should be 8
    Q4, _ = la.qr(np.random.randn(7, 2), mode='economic')
    S2 = np.random.randn(2, 2)
    X2 = QuasiSVD(Q3, S2, Q4)
    
    # Incompatible shapes
    with pytest.raises(ValueError, match="not aligned"):
        QuasiSVD.multi_dot([X1, X2])


#%% ===========================
#%% EDGE CASES AND SPECIAL SCENARIOS
#%% ===========================

def test_copy(simple_quasisvd):
    """Test copy method."""
    X, X_full = simple_quasisvd
    Y = X.copy()
    
    assert isinstance(Y, QuasiSVD), "Copy should return QuasiSVD"
    assert Y is not X, "Copy should create new object"
    assert Y.U is not X.U, "U should be copied"
    assert Y.S is not X.S, "S should be copied"
    assert Y.V is not X.V, "V should be copied"
    assert np.allclose(Y.full(), X_full), "Copy should have same values"


def test_transpose(simple_quasisvd):
    """Test transpose operation."""
    X, X_full = simple_quasisvd
    
    XT = X.T
    assert isinstance(XT, LowRankMatrix), "Transpose should return LowRankMatrix"
    assert XT.shape == (18, 20), "Transpose shape incorrect"
    assert np.allclose(XT.full(), X_full.T), "Transpose values incorrect"


def test_negation(simple_quasisvd):
    """Test negation operation."""
    X, X_full = simple_quasisvd
    
    Y = -X
    assert isinstance(Y, QuasiSVD), "Negation should return QuasiSVD"
    assert np.allclose(Y.full(), -X_full), "Negation incorrect"


def test_very_small_matrix():
    """Test QuasiSVD with very small dimensions but rank > 1."""
    np.random.seed(8888)
    Q1, _ = la.qr(np.random.randn(5, 2), mode='economic')
    Q2, _ = la.qr(np.random.randn(4, 2), mode='economic')
    S = np.random.randn(2, 2)
    
    X = QuasiSVD(Q1, S, Q2)
    assert X.shape == (5, 4), "Small matrix shape incorrect"
    assert X.rank == 2, "Small matrix rank incorrect"


def test_memory_efficiency_warning():
    """Test that memory efficiency warning is raised when appropriate."""
    from lowrank.matrices.low_rank_matrix import MemoryEfficiencyWarning
    
    np.random.seed(9999)
    # Create a "low-rank" matrix where rank is too high
    m, n, r = 10, 8, 7  # r is very close to min(m, n)
    Q1, _ = la.qr(np.random.randn(m, r), mode='economic')
    Q2, _ = la.qr(np.random.randn(n, r), mode='economic')
    S = np.random.randn(r, r)
    
    with pytest.warns(MemoryEfficiencyWarning, match="Memory inefficiency"):
        X = QuasiSVD(Q1, S, Q2)


#%% ===========================
#%% CONVERSIONS AND PROJECTIONS
#%% ===========================

def test_to_svd(simple_quasisvd):
    """Test conversion to SVD format."""
    from lowrank.matrices.svd import SVD
    
    X, X_full = simple_quasisvd
    
    # Convert to SVD
    X_svd = X.to_svd()
    
    # Check type
    assert isinstance(X_svd, SVD), "Should return SVD object"
    
    # Check dimensions
    assert X_svd.shape == X.shape, "Shape should be preserved"
    assert X_svd.rank == X.rank, "Rank should be preserved"
    
    # Check correctness
    assert np.allclose(X_svd.full(), X_full), "Conversion should preserve matrix values"
    
    # Check that S is diagonal
    assert np.allclose(X_svd.S, np.diag(X_svd.s)), "S should be diagonal in SVD"


def test_truncate(simple_quasisvd):
    """Test truncation method."""
    from lowrank.matrices.svd import SVD
    
    X, X_full = simple_quasisvd
    
    # Truncate by rank
    X_trunc_r = X.truncate(r=2)
    assert isinstance(X_trunc_r, SVD), "truncate should return SVD"
    assert X_trunc_r.rank == 2, "Rank should be truncated to 2"
    
    # Truncate by relative tolerance
    X_trunc_rtol = X.truncate(rtol=0.5)
    assert isinstance(X_trunc_rtol, SVD), "truncate should return SVD"
    assert X_trunc_rtol.rank <= X.rank, "Rank should be reduced"
    
    # Truncate by absolute tolerance (should keep most/all)
    X_trunc_atol = X.truncate(atol=1e-12)
    assert isinstance(X_trunc_atol, SVD), "truncate should return SVD"
    
    # Check that truncated matrix approximates original
    # (for rank truncation, may not be exact)
    assert X_trunc_r.shape == X.shape, "Shape should be preserved"


def test_generate_random():
    """Test random matrix generation."""
    np.random.seed(12345)
    
    # Generate random QuasiSVD
    shape = (50, 40)
    rank = 5
    X = QuasiSVD.generate_random(shape, rank, seed=12345)
    
    assert isinstance(X, QuasiSVD), "Should return QuasiSVD"
    assert X.shape == shape, "Shape should match"
    assert X.rank == rank, "Rank should match"
    assert X.is_orthogonal(), "U and V should be orthogonal"
    
    # Test symmetric generation
    shape_square = (30, 30)
    X_sym = QuasiSVD.generate_random(shape_square, rank, seed=12345, is_symmetric=True)
    assert X_sym.shape == shape_square, "Shape should match"
    assert X_sym.is_symmetric(), "Should be symmetric"
    
    # Test error for non-square symmetric
    with pytest.raises(ValueError, match="symmetric.*non-square"):
        QuasiSVD.generate_random((30, 20), rank, is_symmetric=True)


def test_project_onto_column_space(simple_quasisvd):
    """Test projection onto column space."""
    X, X_full = simple_quasisvd
    np.random.seed(11111)
    
    # Test with dense matrix
    Y = np.random.randn(20, 10)
    result = X.project_onto_column_space(Y, dense_output=False)
    
    from lowrank.matrices.qr import QR
    assert isinstance(result, QR), "Should return QR object"
    
    # Test with dense output
    result_dense = X.project_onto_column_space(Y, dense_output=True)
    assert isinstance(result_dense, ndarray), "Should return ndarray with dense_output=True"
    
    # Check correctness: P_U @ Y = U @ U.H @ Y
    expected = X.U @ (X.Uh @ Y)
    assert np.allclose(result_dense, expected), "Projection incorrect"


def test_project_onto_row_space(simple_quasisvd):
    """Test projection onto row space."""
    X, X_full = simple_quasisvd
    np.random.seed(22222)
    
    # Test with dense matrix
    Y = np.random.randn(20, 18)
    result = X.project_onto_row_space(Y, dense_output=False)
    
    from lowrank.matrices.qr import QR
    assert isinstance(result, QR), "Should return QR object"
    
    # Test with dense output
    result_dense = X.project_onto_row_space(Y, dense_output=True)
    assert isinstance(result_dense, ndarray), "Should return ndarray with dense_output=True"
    
    # Check correctness: Y @ P_V = Y @ V @ V.H
    expected = Y @ X.V @ X.Vh
    assert np.allclose(result_dense, expected), "Projection incorrect"


def test_project_onto_tangent_space(simple_quasisvd):
    """Test projection onto tangent space."""
    X, X_full = simple_quasisvd
    np.random.seed(33333)
    
    # Create another matrix to project
    Y = np.random.randn(20, 18)
    
    # Without truncation
    result = X.project_onto_tangent_space(Y, auto_truncate=False)
    assert isinstance(result, QuasiSVD), "Should return QuasiSVD"
    assert result.rank == 2 * X.rank, "Rank should be 2*rank(X)"
    assert result.shape == X.shape, "Shape should match"
    
    # With truncation
    from lowrank.matrices.svd import SVD
    result_trunc = X.project_onto_tangent_space(Y, auto_truncate=True)
    assert isinstance(result_trunc, SVD), "Should return SVD with truncation"
    assert result_trunc.rank <= 2 * X.rank, "Rank should be reduced"


def test_project_onto_interpolated_tangent_space():
    """Test interpolated tangent space projection (both modes)."""
    from lowrank.cssp import DEIM, QDEIM
    
    np.random.seed(44444)
    
    # Create test matrices
    m, n = 50, 40
    rank = 5
    X = QuasiSVD.generate_random((m, n), rank, seed=42)
    Y = QuasiSVD.generate_random((m, n), rank, seed=123)
    kwargs_online = {'Y': Y, 'cssp_method_u': DEIM, 'cssp_method_v': DEIM}
    
    # TEST: Online mode with DEIM
    result_online_deim = X.project_onto_interpolated_tangent_space(
        mode='online',
        **kwargs_online
    )
    assert isinstance(result_online_deim, QuasiSVD), "Should return QuasiSVD"
    assert result_online_deim.shape == X.shape, "Shape should match"
    
    # TEST: Online mode with QDEIM
    result_online_qdeim = X.project_onto_interpolated_tangent_space(
        Y=Y,
        cssp_method_u=QDEIM,
        cssp_method_v=QDEIM
    )
    assert isinstance(result_online_qdeim, QuasiSVD), "Should return QuasiSVD"
    
    # TEST: Offline mode
    p_u, M_u = DEIM(X.U, return_projector=True)
    p_v, M_v = DEIM(X.V, return_projector=True)
    Y_full = Y.full()
    Y_u = Y_full[p_u, :]
    Y_v = Y_full[:, p_v]
    Y_uv = Y_full[np.ix_(p_u, p_v)]
    kwargs_offline = {'Y_u': Y_u, 'Y_v': Y_v, 'Y_uv': Y_uv, 'M_u': M_u, 'M_v': M_v}
    
    result_offline = X.project_onto_interpolated_tangent_space(
        mode='offline',
        **kwargs_offline
    )
    assert isinstance(result_offline, QuasiSVD), "Should return QuasiSVD"
    
    # TEST: Consistency check (online vs offline)
    result_online = X.project_onto_interpolated_tangent_space(
        mode='online',
        **kwargs_online
    )
    diff = np.linalg.norm((result_online - result_offline).full())
    assert diff < 1e-10, "Online and offline modes should give same result"
    
    # TEST: Auto-truncation
    from lowrank.matrices.svd import SVD
    result_with_trunc = X.project_onto_interpolated_tangent_space(
        mode='online',
        auto_truncate=True,
        **kwargs_online
    )
    assert isinstance(result_with_trunc, SVD), "Should return SVD with truncation"
    


#%% ===========================
#%% OPTIMIZED METHODS
#%% ===========================

def test_trace_optimized(simple_quasisvd):
    """Test optimized trace computation."""
    X, X_full = simple_quasisvd
    
    # Trace only works on square matrices, so we need to create one
    np.random.seed(12345)
    m = 20
    r = 4
    Q, _ = la.qr(np.random.randn(m, r), mode='economic')
    S = np.random.randn(r, r)
    X_square = QuasiSVD(Q, S, Q)
    X_square_full = X_square.full()
    
    # Test trace
    trace_opt = X_square.trace()
    trace_full = np.trace(X_square_full)
    assert abs(trace_opt - trace_full) < 1e-10, "Optimized trace incorrect"
    
    # Test that non-square raises error
    with pytest.raises(ValueError, match="square"):
        X.trace()


def test_diag_optimized(simple_quasisvd):
    """Test optimized diagonal extraction."""
    X, X_full = simple_quasisvd
    
    # Test diagonal
    diag_opt = X.diag()
    diag_full = np.diag(X_full)
    
    assert np.allclose(diag_opt, diag_full), "Optimized diag incorrect"


def test_norm_squared_optimized(simple_quasisvd):
    """Test optimized squared norm computation."""
    X, X_full = simple_quasisvd
    
    # Test norm squared
    norm_sq_opt = X.norm_squared()
    norm_sq_full = np.sum(X_full ** 2)
    
    assert abs(norm_sq_opt - norm_sq_full) < 1e-9, "Optimized norm_squared incorrect"


def test_transpose_returns_quasisvd(simple_quasisvd):
    """Test that transpose returns QuasiSVD."""
    X, X_full = simple_quasisvd
    
    # Test .T property
    XT = X.T
    assert isinstance(XT, QuasiSVD), "Transpose should return QuasiSVD"
    assert XT.shape == (18, 20), "Transpose shape incorrect"
    assert np.allclose(XT.full(), X_full.T), "Transpose values incorrect"
    
    # Test transpose() method
    XT2 = X.transpose()
    assert isinstance(XT2, QuasiSVD), "transpose() should return QuasiSVD"
    assert np.allclose(XT2.full(), X_full.T), "transpose() values incorrect"


def test_conj_returns_quasisvd(complex_quasisvd):
    """Test that conjugate returns QuasiSVD."""
    X, X_full = complex_quasisvd
    
    X_conj = X.conj()
    assert isinstance(X_conj, QuasiSVD), "conj() should return QuasiSVD"
    assert np.allclose(X_conj.full(), X_full.conj()), "conj() values incorrect"


def test_hermitian_returns_quasisvd(complex_quasisvd):
    """Test that Hermitian conjugate returns QuasiSVD."""
    X, X_full = complex_quasisvd
    
    XH = X.H
    assert isinstance(XH, QuasiSVD), "H should return QuasiSVD"
    assert XH.shape == (8, 10), "H shape incorrect"
    assert np.allclose(XH.full(), X_full.T.conj()), "H values incorrect"


#%% ===========================
#%% NEW FEATURES
#%% ===========================

def test_rank_one_update(simple_quasisvd):
    """Test rank-1 update."""
    X, X_full = simple_quasisvd
    np.random.seed(55555)
    
    u = np.random.randn(20)
    v = np.random.randn(18)
    alpha = 0.5
    
    # Perform rank-1 update
    X_updated = X.rank_one_update(u, v, alpha)
    
    # Check type and dimensions
    assert isinstance(X_updated, QuasiSVD), "Should return QuasiSVD"
    assert X_updated.shape == X.shape, "Shape should be preserved"
    assert X_updated.rank == X.rank + 1, "Rank should increase by 1"
    
    # Check correctness
    expected = X_full + alpha * np.outer(u, v)
    assert np.allclose(X_updated.full(), expected), "Rank-1 update incorrect"
    
    # Test error handling
    with pytest.raises(ValueError, match="must have length"):
        X.rank_one_update(np.random.randn(15), v)


def test_reorthogonalize():
    """Test re-orthogonalization."""
    np.random.seed(66666)
    
    # Create QuasiSVD and corrupt orthogonality slightly
    Q1, _ = la.qr(np.random.randn(30, 5), mode='economic')
    Q2, _ = la.qr(np.random.randn(25, 5), mode='economic')
    S = np.random.randn(5, 5)
    
    # Add small perturbation to lose orthogonality
    Q1_corrupt = Q1 + np.random.randn(30, 5) * 1e-6
    Q2_corrupt = Q2 + np.random.randn(25, 5) * 1e-6
    
    X_corrupt = QuasiSVD(Q1_corrupt, S, Q2_corrupt)
    X_full = X_corrupt.full()
    
    # Re-orthogonalize with QR
    X_reorth = X_corrupt.reorthogonalize(method='qr')
    assert isinstance(X_reorth, QuasiSVD), "Should return QuasiSVD"
    assert X_reorth.is_orthogonal(), "Should be orthogonal after reorthogonalization"
    assert np.allclose(X_reorth.full(), X_full, atol=1e-10), "Matrix values should be preserved"
    
    # Re-orthogonalize with SVD
    X_reorth_svd = X_corrupt.reorthogonalize(method='svd')
    from lowrank.matrices.svd import SVD
    assert isinstance(X_reorth_svd, SVD), "SVD method should return SVD"
    assert np.allclose(X_reorth_svd.full(), X_full, atol=1e-10), "Matrix values should be preserved"
    
    # Test error for invalid method
    with pytest.raises(ValueError, match="Unknown method"):
        X_corrupt.reorthogonalize(method='invalid')


def test_numerical_health_check(simple_quasisvd):
    """Test numerical health check."""
    X, _ = simple_quasisvd
    
    # Run health check (non-verbose)
    health = X.numerical_health_check(verbose=False)
    
    # Check that all expected keys are present
    expected_keys = [
        'orthogonal_U', 'orthogonal_V',
        'orthogonality_error_U', 'orthogonality_error_V',
        'condition_number_S', 'is_singular',
        'min_singular_value', 'max_singular_value',
        'singular_value_ratio', 'compression_ratio',
        'memory_efficient', 'recommendations'
    ]
    for key in expected_keys:
        assert key in health, f"Missing key: {key}"
    
    # Check types
    assert isinstance(health['orthogonal_U'], bool)
    assert isinstance(health['orthogonal_V'], bool)
    assert isinstance(health['recommendations'], list)
    
    # For a well-formed QuasiSVD, should be healthy
    assert health['orthogonal_U'], "U should be orthogonal"
    assert health['orthogonal_V'], "V should be orthogonal"
    assert not health['is_singular'], "S should not be singular"


def test_numerical_health_check_with_issues():
    """Test health check detects issues."""
    np.random.seed(77777)
    
    # Create ill-conditioned QuasiSVD
    Q1, _ = la.qr(np.random.randn(20, 5), mode='economic')
    Q2, _ = la.qr(np.random.randn(15, 5), mode='economic')
    
    # Create singular S
    S_singular = np.random.randn(5, 5)
    S_singular[-1, :] = S_singular[0, :] + S_singular[1, :]  # Make rank-deficient
    
    X_singular = QuasiSVD(Q1, S_singular, Q2)
    health = X_singular.numerical_health_check(verbose=False)
    
    assert health['is_singular'], "Should detect singular S"
    assert len(health['recommendations']) > 0, "Should have recommendations"


def test_to_qr_conversion(simple_quasisvd):
    """Test conversion to QR format."""
    from lowrank.matrices.qr import QR
    
    X, X_full = simple_quasisvd
    
    # Convert to QR
    X_qr = X.to_qr()
    
    assert isinstance(X_qr, QR), "Should return QR object"
    assert np.allclose(X_qr.full(), X_full), "QR conversion should preserve values"


def test_from_qr_conversion():
    """Test conversion from QR format."""
    from lowrank.matrices.qr import QR
    
    np.random.seed(88888)
    
    # Create QR matrix
    Q, _ = la.qr(np.random.randn(30, 5), mode='economic')
    R = np.random.randn(5, 25)
    X_qr = QR(Q, R)
    X_full = X_qr.full()
    
    # Convert to QuasiSVD
    X = QuasiSVD.from_qr(X_qr)
    
    assert isinstance(X, QuasiSVD), "Should return QuasiSVD"
    assert X.shape == X_qr.shape, "Shape should match"
    assert np.allclose(X.full(), X_full), "from_qr should preserve values"
    assert X.is_orthogonal(), "Should have orthogonal factors"


#%% ===========================
#%% ROUND-TRIP CONVERSIONS
#%% ===========================

def test_qr_roundtrip():
    """Test QuasiSVD -> QR -> QuasiSVD roundtrip."""
    np.random.seed(99999)
    
    X = QuasiSVD.generate_random((40, 30), 6, seed=99999)
    X_full = X.full()
    
    # Convert to QR and back
    X_qr = X.to_qr()
    X_back = QuasiSVD.from_qr(X_qr)
    
    assert X_back.shape == X.shape, "Shape should be preserved"
    assert np.allclose(X_back.full(), X_full, atol=1e-10), "Roundtrip should preserve values"


#%% ===========================
#%% SVD TYPE PROPERTY
#%% ===========================

def test_svd_type_full():
    """Test svd_type property for full SVD."""
    np.random.seed(100)
    m, n = 10, 8
    U, _ = la.qr(np.random.randn(m, m), mode='economic')
    V, _ = la.qr(np.random.randn(n, n), mode='economic')
    S = np.random.randn(m, n)
    X = QuasiSVD(U, S, V)
    
    assert X.svd_type == 'full', f"Expected 'full', got '{X.svd_type}'"


def test_svd_type_reduced():
    """Test svd_type property for reduced SVD."""
    np.random.seed(101)
    m, n = 100, 80
    r = min(m, n)
    U, _ = la.qr(np.random.randn(m, r), mode='economic')
    V, _ = la.qr(np.random.randn(n, r), mode='economic')
    S = np.random.randn(r, r)
    X = QuasiSVD(U, S, V)
    
    assert X.svd_type == 'reduced', f"Expected 'reduced', got '{X.svd_type}'"


def test_svd_type_truncated():
    """Test svd_type property for truncated SVD."""
    X = QuasiSVD.generate_random((100, 80), 10)
    
    assert X.svd_type == 'truncated', f"Expected 'truncated', got '{X.svd_type}'"


def test_svd_type_unconventional():
    """Test svd_type property for unconventional quasi-SVD."""
    np.random.seed(102)
    m, n = 50, 40
    r, k = 15, 10
    U, _ = la.qr(np.random.randn(m, r), mode='economic')
    V, _ = la.qr(np.random.randn(n, k), mode='economic')
    S = np.random.randn(r, k)
    X = QuasiSVD(U, S, V)
    
    assert X.svd_type == 'unconventional', f"Expected 'unconventional', got '{X.svd_type}'"


#%% ===========================
#%% ADVANCED LINEAR ALGEBRA
#%% ===========================

def test_pseudoinverse_basic():
    """Test pseudoinverse computation."""
    np.random.seed(200)
    X = QuasiSVD.generate_random((100, 80), 15)
    X_full = X.full()
    
    # Compute pseudoinverse
    X_pinv = X.pseudoinverse()
    X_pinv_full = X_pinv.full()
    
    # Check dimensions
    assert X_pinv.shape == (80, 100), "Pseudoinverse should have transposed shape"
    
    # Check pseudoinverse properties: X @ X+ @ X ≈ X
    reconstruction = X_full @ X_pinv_full @ X_full
    assert np.allclose(reconstruction, X_full, rtol=1e-10), "X @ X+ @ X should equal X"
    
    # Check: X+ @ X @ X+ ≈ X+
    reconstruction_pinv = X_pinv_full @ X_full @ X_pinv_full
    assert np.allclose(reconstruction_pinv, X_pinv_full, rtol=1e-10), "X+ @ X @ X+ should equal X+"


def test_pseudoinverse_with_threshold():
    """Test pseudoinverse with threshold for small singular values."""
    np.random.seed(201)
    X = QuasiSVD.generate_random((50, 40), 10)
    
    # Test with different thresholds
    X_pinv1 = X.pseudoinverse(rtol=1e-10)
    X_pinv2 = X.pseudoinverse(rtol=1e-6)
    
    # Both should be valid pseudoinverses
    assert X_pinv1.shape == (40, 50), "Shape should be correct"
    assert X_pinv2.shape == (40, 50), "Shape should be correct"


def test_solve_square_matrix():
    """Test solve for square system."""
    np.random.seed(300)
    n = 100
    X = QuasiSVD.generate_random((n, n), 20)
    
    # Create right-hand side that is in the range of X
    x_true = np.random.randn(n)
    b = X.dot(x_true)
    
    # Solve system (default method='lstsq' uses pseudoinverse)
    x = X.solve(b)
    
    # Check solution reconstructs b
    b_reconstructed = X.dot(x)
    assert np.allclose(b_reconstructed, b, rtol=1e-8), "X @ x should equal b"
    
    # For rank-deficient systems, we get a least-squares solution
    # Check that residual is small
    residual = np.linalg.norm(b_reconstructed - b)
    assert residual < 1e-10, f"Residual {residual} should be small"


def test_solve_multiple_rhs():
    """Test solve with multiple right-hand sides."""
    np.random.seed(301)
    n, k = 100, 5
    X = QuasiSVD.generate_random((n, n), 20)
    
    # Create multiple right-hand sides in the range of X
    X_true = np.random.randn(n, k)
    B = X.dot(X_true, dense_output=True)
    
    # Solve system (default method='lstsq' uses pseudoinverse)
    X_sol = X.solve(B)
    
    # Check solution reconstructs B
    B_reconstructed = X.dot(X_sol, dense_output=True)
    assert np.allclose(B_reconstructed, B, rtol=1e-8), "X @ X_sol should equal B"


def test_solve_non_square_raises():
    """Test that direct solve raises error for non-square matrix."""
    X = QuasiSVD.generate_random((100, 80), 15)
    b = np.random.randn(100)
    
    with pytest.raises(ValueError, match="Direct solve requires square matrix"):
        X.solve(b, method='direct')


def test_solve_dimension_mismatch_raises():
    """Test that solve raises error for dimension mismatch."""
    X = QuasiSVD.generate_random((100, 100), 20)
    b = np.random.randn(80)  # Wrong size
    
    with pytest.raises(ValueError, match="Dimension mismatch"):
        X.solve(b)


def test_lstsq_overdetermined():
    """Test least squares for overdetermined system."""
    np.random.seed(400)
    m, n = 100, 80
    X = QuasiSVD.generate_random((m, n), 20)
    X_full = X.full()
    
    # Create right-hand side
    b = np.random.randn(m)
    
    # Compute least squares solution
    x = X.lstsq(b)
    
    # Check that x minimizes ||X @ x - b||
    residual = np.linalg.norm(X_full @ x - b)
    
    # Compare with numpy's lstsq
    x_np, residual_np, _, _ = np.linalg.lstsq(X_full, b, rcond=None)
    
    assert np.allclose(x, x_np, rtol=1e-8), "Should match numpy's lstsq"
    assert np.allclose(residual, np.sqrt(residual_np), rtol=1e-8), "Residual should match"


def test_lstsq_underdetermined():
    """Test least squares for underdetermined system."""
    np.random.seed(401)
    m, n = 80, 100
    X = QuasiSVD.generate_random((m, n), 20)
    
    # Create right-hand side in the range of X
    x_temp = np.random.randn(n)
    b = X.dot(x_temp)
    
    # Compute least squares solution
    x = X.lstsq(b)
    
    # Check solution exists
    assert x.shape == (n,), "Solution should have correct shape"
    
    # For underdetermined system, should find minimum-norm solution
    # Check that X @ x ≈ b
    b_reconstructed = X.dot(x)
    assert np.allclose(b_reconstructed, b, rtol=1e-8), "Should satisfy the equation closely"


def test_solve_with_lstsq_method():
    """Test solve with method='lstsq'."""
    np.random.seed(402)
    X = QuasiSVD.generate_random((100, 80), 20)
    b = np.random.randn(100)
    
    # Solve using lstsq method
    x1 = X.solve(b, method='lstsq')
    x2 = X.lstsq(b)
    
    assert np.allclose(x1, x2), "solve(method='lstsq') should match lstsq()"


#%% ===========================
#%% MATRIX SQUARE ROOT (sqrtm)
#%% ===========================

def test_sqrtm_basic():
    """Test basic matrix square root computation."""
    np.random.seed(500)
    
    # Create a symmetric positive definite QuasiSVD matrix
    Q, _ = la.qr(np.random.randn(20, 5), mode='economic')
    S = np.diag([4.0, 9.0, 16.0, 25.0, 36.0])  # Perfect squares for easy verification
    X = QuasiSVD(Q, S, Q)  # Symmetric by construction
    
    # Compute square root
    X_sqrt = X.sqrtm()
    
    # Verify it's a QuasiSVD
    assert isinstance(X_sqrt, QuasiSVD), "sqrtm should return QuasiSVD"
    
    # Verify X_sqrt @ X_sqrt = X
    X_reconstructed = X_sqrt.dot(X_sqrt)
    assert np.allclose(X_reconstructed.full(), X.full()), "X_sqrt @ X_sqrt should equal X"
    
    # Verify the S matrix is the square root
    S_sqrt_expected = np.diag([2.0, 3.0, 4.0, 5.0, 6.0])
    assert np.allclose(X_sqrt.S, S_sqrt_expected), "S should be square root of original S"


def test_sqrtm_non_diagonal():
    """Test sqrtm with non-diagonal S matrix."""
    np.random.seed(501)
    
    Q1, _ = la.qr(np.random.randn(15, 4), mode='economic')
    Q2, _ = la.qr(np.random.randn(12, 4), mode='economic')
    
    # Create a non-diagonal S matrix
    A = np.random.randn(4, 4)
    S = A + A.T  # Symmetric but non-diagonal
    S += np.eye(4) * 5  # Make it strictly positive definite
    
    X = QuasiSVD(Q1, S, Q2)
    
    # Compute square root
    X_sqrt = X.sqrtm()
    
    # Verify it's a QuasiSVD
    assert isinstance(X_sqrt, QuasiSVD), "sqrtm should return QuasiSVD"
    
    # Verify U and V are unchanged (only S changes)
    assert np.allclose(X_sqrt.U, X.U), "U should be unchanged"
    assert np.allclose(X_sqrt.V, X.V), "V should be unchanged"
    
    # Verify S_sqrt @ S_sqrt ≈ S
    S_sqrt_squared = X_sqrt.S @ X_sqrt.S
    assert np.allclose(S_sqrt_squared, S, atol=1e-10), "S_sqrt @ S_sqrt should equal S"


def test_sqrtm_inplace():
    """Test in-place matrix square root computation."""
    np.random.seed(502)
    
    Q, _ = la.qr(np.random.randn(10, 3), mode='economic')
    S = np.diag([1.0, 4.0, 9.0])
    X = QuasiSVD(Q, S, Q)
    X_full_original = X.full()
    
    # Store original S
    S_original = X.S.copy()
    
    # Compute square root in-place
    result = X.sqrtm(inplace=True)
    
    # Verify it returns self
    assert result is X, "inplace=True should return self"
    
    # Verify S has been modified
    assert not np.allclose(X.S, S_original), "S should be modified in-place"
    
    # Verify X_sqrt @ X_sqrt = X_original
    X_reconstructed = X.dot(X)
    assert np.allclose(X_reconstructed.full(), X_full_original), "X_sqrt @ X_sqrt should equal original X"


def test_sqrtm_complex():
    """Test sqrtm with complex matrices."""
    np.random.seed(503)
    
    # Create complex matrices
    Q1, _ = la.qr(np.random.randn(10, 3) + 1j * np.random.randn(10, 3), mode='economic')
    Q2, _ = la.qr(np.random.randn(8, 3) + 1j * np.random.randn(8, 3), mode='economic')
    
    # Create a Hermitian S matrix
    A = np.random.randn(3, 3) + 1j * np.random.randn(3, 3)
    S = A + A.conj().T  # Hermitian
    S += np.eye(3) * 5  # Make it strictly positive definite
    
    X = QuasiSVD(Q1, S, Q2)
    
    # Compute square root
    X_sqrt = X.sqrtm()
    
    # Verify result is complex
    assert X_sqrt.S.dtype == np.complex128, "Result should be complex"
    
    # Verify U and V are unchanged
    assert np.allclose(X_sqrt.U, X.U), "U should be unchanged"
    assert np.allclose(X_sqrt.V, X.V), "V should be unchanged"
    
    # Verify S_sqrt @ S_sqrt ≈ S
    S_sqrt_squared = X_sqrt.S @ X_sqrt.S
    assert np.allclose(S_sqrt_squared, S, atol=1e-10), "S_sqrt @ S_sqrt should equal S"


def test_sqrtm_preserves_orthogonality():
    """Test that sqrtm preserves orthogonality of U and V."""
    np.random.seed(505)
    
    Q1, _ = la.qr(np.random.randn(20, 5), mode='economic')
    Q2, _ = la.qr(np.random.randn(18, 5), mode='economic')
    S = np.diag([1.0, 4.0, 9.0, 16.0, 25.0])
    
    X = QuasiSVD(Q1, S, Q2)
    assert X.is_orthogonal(), "Original should be orthogonal"
    
    # Compute square root
    X_sqrt = X.sqrtm()
    
    # Verify U and V are unchanged (only S changes)
    assert np.allclose(X_sqrt.U, X.U), "U should be unchanged"
    assert np.allclose(X_sqrt.V, X.V), "V should be unchanged"
    
    # Verify orthogonality is preserved
    assert X_sqrt.is_orthogonal(), "sqrtm result should be orthogonal"


def test_sqrtm_extra_data():
    """Test that extra_data is preserved in sqrtm."""
    np.random.seed(506)
    
    Q, _ = la.qr(np.random.randn(10, 3), mode='economic')
    S = np.diag([1.0, 4.0, 9.0])
    
    extra_data = {'poles': [1, 2, 3], 'test_key': 'test_value'}
    X = QuasiSVD(Q, S, Q, **extra_data)
    
    # Compute square root
    X_sqrt = X.sqrtm(**extra_data)
    
    # Verify extra_data is preserved
    assert 'poles' in X_sqrt._extra_data, "Extra data should be preserved"
    assert 'test_key' in X_sqrt._extra_data, "Extra data should be preserved"
    assert X_sqrt._extra_data['poles'] == [1, 2, 3], "Extra data values should match"
    assert X_sqrt._extra_data['test_key'] == 'test_value', "Extra data values should match"



#%% ===========================
#%% INHERITED METHODS TESTS
#%% ===========================

def test_from_matrix():
    """Test creating QuasiSVD from dense matrix using truncated_svd."""
    from lowrank.matrices.svd import SVD
    np.random.seed(800)
    
    # Create dense matrix
    A = np.random.randn(50, 40)
    
    # Convert to QuasiSVD via truncated SVD
    X = SVD.truncated_svd(A, r=10)
    
    assert isinstance(X, QuasiSVD), "Should create QuasiSVD instance"
    assert X.shape == (50, 40), "Shape should match"
    assert X.rank == 10, "Rank should be 10"


def test_flatten():
    """Test flattening QuasiSVD to 1D array."""
    X = QuasiSVD.generate_random((20, 15), 5)
    X_full = X.full()
    
    # Flatten
    X_flat = X.flatten()
    expected = X_full.flatten()
    
    assert X_flat.shape == (300,), "Flattened shape should be 1D"
    assert np.allclose(X_flat, expected), "Flattened values should match"


#%% ===========================
#%% COMPLEX MATRIX TESTS
#%% ===========================

def test_complex_transpose():
    """Test transpose for complex matrices."""
    np.random.seed(42)
    m, n, r = 10, 8, 4
    U, _ = la.qr(np.random.randn(m, r) + 1j * np.random.randn(m, r), mode='economic')
    V, _ = la.qr(np.random.randn(n, r) + 1j * np.random.randn(n, r), mode='economic')
    S = np.random.randn(r, r) + 1j * np.random.randn(r, r)
    
    X = QuasiSVD(U, S, V)
    A = X.full()
    
    # Test transpose (without conjugate)
    X_T = X.T
    assert isinstance(X_T, QuasiSVD), "Transpose should return QuasiSVD"
    assert X_T.shape == (n, m), "Transpose shape should be swapped"
    assert np.allclose(X_T.full(), A.T), "Transpose should match A.T"
    assert not np.allclose(X_T.full(), A.T.conj()), "Transpose should NOT conjugate"


def test_complex_hermitian():
    """Test Hermitian conjugate for complex matrices."""
    np.random.seed(43)
    m, n, r = 10, 8, 4
    U, _ = la.qr(np.random.randn(m, r) + 1j * np.random.randn(m, r), mode='economic')
    V, _ = la.qr(np.random.randn(n, r) + 1j * np.random.randn(n, r), mode='economic')
    S = np.random.randn(r, r) + 1j * np.random.randn(r, r)
    
    X = QuasiSVD(U, S, V)
    A = X.full()
    
    # Test Hermitian conjugate
    X_H = X.H
    assert isinstance(X_H, QuasiSVD), "Hermitian should return QuasiSVD"
    assert X_H.shape == (n, m), "Hermitian shape should be swapped"
    assert np.allclose(X_H.full(), A.T.conj()), "Hermitian should match A.T.conj()"
    assert np.allclose(X_H.full(), A.conj().T), "Hermitian should match A.conj().T"


def test_complex_transpose_hermitian_relationship():
    """Test relationship between transpose and Hermitian for complex matrices."""
    np.random.seed(44)
    m, n, r = 12, 10, 5
    U, _ = la.qr(np.random.randn(m, r) + 1j * np.random.randn(m, r), mode='economic')
    V, _ = la.qr(np.random.randn(n, r) + 1j * np.random.randn(n, r), mode='economic')
    S = np.random.randn(r, r) + 1j * np.random.randn(r, r)
    
    X = QuasiSVD(U, S, V)
    
    # Test relationships
    assert np.allclose(X.H.full(), X.T.conj().full()), "X.H should equal X.T.conj()"
    assert np.allclose(X.H.full(), X.conj().T.full()), "X.H should equal X.conj().T"
    assert np.allclose(X.T.T.full(), X.full()), "X.T.T should equal X"
    assert np.allclose(X.H.H.full(), X.full()), "X.H.H should equal X"


def test_complex_addition():
    """Test addition with complex matrices."""
    np.random.seed(45)
    m, n, r = 8, 6, 3
    U1, _ = la.qr(np.random.randn(m, r) + 1j * np.random.randn(m, r), mode='economic')
    V1, _ = la.qr(np.random.randn(n, r) + 1j * np.random.randn(n, r), mode='economic')
    S1 = np.random.randn(r, r) + 1j * np.random.randn(r, r)
    
    U2, _ = la.qr(np.random.randn(m, r) + 1j * np.random.randn(m, r), mode='economic')
    V2, _ = la.qr(np.random.randn(n, r) + 1j * np.random.randn(n, r), mode='economic')
    S2 = np.random.randn(r, r) + 1j * np.random.randn(r, r)
    
    X1 = QuasiSVD(U1, S1, V1)
    X2 = QuasiSVD(U2, S2, V2)
    A1 = X1.full()
    A2 = X2.full()
    
    # Test addition
    Y = X1 + X2
    assert isinstance(Y, QuasiSVD), "Addition should return QuasiSVD"
    assert np.allclose(Y.full(), A1 + A2), "Complex addition incorrect"


def test_complex_subtraction():
    """Test subtraction with complex matrices."""
    np.random.seed(46)
    m, n, r = 8, 6, 3
    U1, _ = la.qr(np.random.randn(m, r) + 1j * np.random.randn(m, r), mode='economic')
    V1, _ = la.qr(np.random.randn(n, r) + 1j * np.random.randn(n, r), mode='economic')
    S1 = np.random.randn(r, r) + 1j * np.random.randn(r, r)
    
    U2, _ = la.qr(np.random.randn(m, r) + 1j * np.random.randn(m, r), mode='economic')
    V2, _ = la.qr(np.random.randn(n, r) + 1j * np.random.randn(n, r), mode='economic')
    S2 = np.random.randn(r, r) + 1j * np.random.randn(r, r)
    
    X1 = QuasiSVD(U1, S1, V1)
    X2 = QuasiSVD(U2, S2, V2)
    A1 = X1.full()
    A2 = X2.full()
    
    # Test subtraction
    Y = X1 - X2
    assert isinstance(Y, QuasiSVD), "Subtraction should return QuasiSVD"
    assert np.allclose(Y.full(), A1 - A2), "Complex subtraction incorrect"


def test_complex_multiplication():
    """Test matrix multiplication with complex matrices."""
    np.random.seed(47)
    m, n, k, r = 10, 8, 6, 4
    U1, _ = la.qr(np.random.randn(m, r) + 1j * np.random.randn(m, r), mode='economic')
    V1, _ = la.qr(np.random.randn(n, r) + 1j * np.random.randn(n, r), mode='economic')
    S1 = np.random.randn(r, r) + 1j * np.random.randn(r, r)
    
    U2, _ = la.qr(np.random.randn(n, r) + 1j * np.random.randn(n, r), mode='economic')
    V2, _ = la.qr(np.random.randn(k, r) + 1j * np.random.randn(k, r), mode='economic')
    S2 = np.random.randn(r, r) + 1j * np.random.randn(r, r)
    
    X1 = QuasiSVD(U1, S1, V1)
    X2 = QuasiSVD(U2, S2, V2)
    A1 = X1.full()
    A2 = X2.full()
    
    # Test multiplication
    Y = X1.dot(X2)
    assert isinstance(Y, QuasiSVD), "Multiplication should return QuasiSVD"
    assert np.allclose(Y.full(), A1 @ A2), "Complex multiplication incorrect"


def test_complex_hadamard():
    """Test Hadamard product with complex matrices."""
    np.random.seed(48)
    m, n, r = 8, 6, 3
    U1, _ = la.qr(np.random.randn(m, r) + 1j * np.random.randn(m, r), mode='economic')
    V1, _ = la.qr(np.random.randn(n, r) + 1j * np.random.randn(n, r), mode='economic')
    S1 = np.random.randn(r, r) + 1j * np.random.randn(r, r)
    
    U2, _ = la.qr(np.random.randn(m, r) + 1j * np.random.randn(m, r), mode='economic')
    V2, _ = la.qr(np.random.randn(n, r) + 1j * np.random.randn(n, r), mode='economic')
    S2 = np.random.randn(r, r) + 1j * np.random.randn(r, r)
    
    X1 = QuasiSVD(U1, S1, V1)
    X2 = QuasiSVD(U2, S2, V2)
    A1 = X1.full()
    A2 = X2.full()
    
    # Test Hadamard product
    Y = X1.hadamard(X2)
    assert isinstance(Y, QuasiSVD), "Hadamard should return QuasiSVD"
    assert np.allclose(Y.full(), A1 * A2), "Complex Hadamard product incorrect"


def test_complex_scalar_multiplication():
    """Test scalar multiplication with complex scalars."""
    np.random.seed(49)
    m, n, r = 8, 6, 3
    U, _ = la.qr(np.random.randn(m, r) + 1j * np.random.randn(m, r), mode='economic')
    V, _ = la.qr(np.random.randn(n, r) + 1j * np.random.randn(n, r), mode='economic')
    S = np.random.randn(r, r) + 1j * np.random.randn(r, r)
    
    X = QuasiSVD(U, S, V)
    A = X.full()
    
    # Test complex scalar
    c = 2.0 + 3.0j
    Y = X * c
    assert isinstance(Y, QuasiSVD), "Scalar multiplication should return QuasiSVD"
    assert Y.S.dtype == np.complex128, "S should be complex"
    assert np.allclose(Y.full(), c * A), "Complex scalar multiplication incorrect"


def test_complex_pseudoinverse():
    """Test pseudoinverse with complex matrices."""
    np.random.seed(50)
    m, n, r = 10, 8, 4
    U, _ = la.qr(np.random.randn(m, r) + 1j * np.random.randn(m, r), mode='economic')
    V, _ = la.qr(np.random.randn(n, r) + 1j * np.random.randn(n, r), mode='economic')
    S = np.diag(np.logspace(0, -2, r)) + 1j * np.diag(np.logspace(-1, -3, r))
    
    X = QuasiSVD(U, S, V)
    A = X.full()
    
    # Compute pseudoinverse
    X_pinv = X.pseudoinverse()
    assert isinstance(X_pinv, QuasiSVD), "Pseudoinverse should return QuasiSVD"
    
    # Test property: X @ X⁺ @ X ≈ X
    reconstruction = X.dot(X_pinv.dot(X.full(), dense_output=True), dense_output=True)
    assert np.allclose(A, reconstruction), "Complex pseudoinverse reconstruction failed"


def test_complex_solve():
    """Test solve with complex matrices."""
    np.random.seed(51)
    n, r = 10, 8
    U, _ = la.qr(np.random.randn(n, r) + 1j * np.random.randn(n, r), mode='economic')
    V, _ = la.qr(np.random.randn(n, r) + 1j * np.random.randn(n, r), mode='economic')
    S = np.random.randn(r, r) + 1j * np.random.randn(r, r)
    
    X = QuasiSVD(U, S, V)
    A = X.full()
    
    # Create complex RHS
    b = np.random.randn(n) + 1j * np.random.randn(n)
    
    # Solve using lstsq method
    x = X.solve(b, method='lstsq')
    
    # Check that solution minimizes residual
    residual = np.linalg.norm(A @ x - b)
    expected_x = np.linalg.lstsq(A, b, rcond=None)[0]
    expected_residual = np.linalg.norm(A @ expected_x - b)
    
    assert np.abs(residual - expected_residual) < 1e-10, "Complex solve residual incorrect"


def test_complex_lstsq():
    """Test least squares with complex matrices."""
    np.random.seed(52)
    m, n, r = 12, 8, 6
    U, _ = la.qr(np.random.randn(m, r) + 1j * np.random.randn(m, r), mode='economic')
    V, _ = la.qr(np.random.randn(n, r) + 1j * np.random.randn(n, r), mode='economic')
    S = np.random.randn(r, r) + 1j * np.random.randn(r, r)
    
    X = QuasiSVD(U, S, V)
    A = X.full()
    
    # Create complex RHS
    b = np.random.randn(m) + 1j * np.random.randn(m)
    
    # Solve least squares
    x = X.lstsq(b)
    
    # Compare with numpy
    x_expected = np.linalg.lstsq(A, b, rcond=None)[0]
    assert np.allclose(x, x_expected, atol=1e-10), "Complex lstsq solution incorrect"


def test_complex_sqrtm():
    """Test matrix square root with complex matrices."""
    np.random.seed(53)
    n, r = 8, 4
    U, _ = la.qr(np.random.randn(n, r) + 1j * np.random.randn(n, r), mode='economic')
    S = np.random.randn(r, r) + 1j * np.random.randn(r, r)
    
    # Create symmetric complex matrix (U = V)
    X = QuasiSVD(U, S, U)
    A = X.full()
    
    # Compute square root
    X_sqrt = X.sqrtm()
    assert isinstance(X_sqrt, QuasiSVD), "sqrtm should return QuasiSVD"
    
    # Test property: X_sqrt @ X_sqrt ≈ X
    reconstruction = X_sqrt.dot(X_sqrt.full(), dense_output=True)
    assert np.allclose(reconstruction, A, atol=1e-10), "Complex sqrtm verification failed"


def test_complex_expm_raises():
    """Test that matrix exponential raises NotImplementedError for complex matrices."""
    np.random.seed(54)
    n, r = 6, 3
    U, _ = la.qr(np.random.randn(n, r) + 1j * np.random.randn(n, r), mode='economic')
    
    # Create Hermitian S for Hermitian matrix
    S_temp = (np.random.randn(r, r) + 1j * np.random.randn(r, r)) * 0.1
    S = (S_temp + S_temp.T.conj()) / 2  # Make S Hermitian
    
    # Create Hermitian complex matrix (X = U @ S @ U.H)
    X = QuasiSVD(U, S, U)
    
    # Should raise NotImplementedError for complex matrices
    with pytest.raises(NotImplementedError, match="not implemented for complex matrices"):
        X.expm()


def test_complex_rectangular_s():
    """Test complex matrices with rectangular S."""
    np.random.seed(55)
    m, n, r1, r2 = 10, 8, 5, 3
    U, _ = la.qr(np.random.randn(m, r1) + 1j * np.random.randn(m, r1), mode='economic')
    V, _ = la.qr(np.random.randn(n, r2) + 1j * np.random.randn(n, r2), mode='economic')
    S = np.random.randn(r1, r2) + 1j * np.random.randn(r1, r2)
    
    X = QuasiSVD(U, S, V)
    A = X.full()
    
    # Test reconstruction
    assert np.allclose(X.full(), A), "Complex rectangular S reconstruction failed"
    
    # Test transpose
    assert np.allclose(X.T.full(), A.T), "Complex rectangular S transpose failed"
    
    # Test Hermitian
    assert np.allclose(X.H.full(), A.T.conj()), "Complex rectangular S Hermitian failed"


def test_complex_norms():
    """Test norms with complex matrices."""
    np.random.seed(56)
    m, n, r = 10, 8, 5
    U, _ = la.qr(np.random.randn(m, r) + 1j * np.random.randn(m, r), mode='economic')
    V, _ = la.qr(np.random.randn(n, r) + 1j * np.random.randn(n, r), mode='economic')
    S = np.random.randn(r, r) + 1j * np.random.randn(r, r)
    
    X = QuasiSVD(U, S, V)
    A = X.full()
    
    # Test Frobenius norm
    norm_fro = X.norm('fro')
    expected_fro = np.linalg.norm(A, 'fro')
    assert np.abs(norm_fro - expected_fro) < 1e-10, "Complex Frobenius norm incorrect"
    
    # Test 2-norm
    norm_2 = X.norm(2)
    expected_2 = np.linalg.norm(A, 2)
    assert np.abs(norm_2 - expected_2) < 1e-10, "Complex 2-norm incorrect"
    
    # Test nuclear norm
    norm_nuc = X.norm('nuc')
    expected_nuc = np.linalg.norm(A, 'nuc')
    assert np.abs(norm_nuc - expected_nuc) < 1e-10, "Complex nuclear norm incorrect"


def test_complex_trace():
    """Test trace with complex square matrices."""
    np.random.seed(57)
    n, r = 10, 5
    U, _ = la.qr(np.random.randn(n, r) + 1j * np.random.randn(n, r), mode='economic')
    V, _ = la.qr(np.random.randn(n, r) + 1j * np.random.randn(n, r), mode='economic')
    S = np.random.randn(r, r) + 1j * np.random.randn(r, r)
    
    X = QuasiSVD(U, S, V)
    A = X.full()
    
    # Test trace
    tr = X.trace()
    expected_tr = np.trace(A)
    assert np.abs(tr - expected_tr) < 1e-10, "Complex trace incorrect"


def test_complex_to_svd_conversion():
    """Test conversion to SVD for complex matrices."""
    from lowrank.matrices.svd import SVD
    
    np.random.seed(58)
    m, n, r = 10, 8, 5
    U, _ = la.qr(np.random.randn(m, r) + 1j * np.random.randn(m, r), mode='economic')
    V, _ = la.qr(np.random.randn(n, r) + 1j * np.random.randn(n, r), mode='economic')
    S = np.random.randn(r, r) + 1j * np.random.randn(r, r)
    
    X = QuasiSVD(U, S, V)
    A = X.full()
    
    # Convert to SVD
    X_svd = X.to_svd()
    assert isinstance(X_svd, SVD), "Should return SVD"
    assert np.allclose(X_svd.full(), A), "Complex to_svd conversion failed"
    assert X_svd.U.dtype == np.complex128, "SVD U should be complex"
    assert X_svd.V.dtype == np.complex128, "SVD V should be complex"


def test_complex_projection_column_space():
    """Test projection onto column space with complex matrices."""
    np.random.seed(59)
    m, n, r = 12, 10, 5
    U, _ = la.qr(np.random.randn(m, r) + 1j * np.random.randn(m, r), mode='economic')
    V, _ = la.qr(np.random.randn(n, r) + 1j * np.random.randn(n, r), mode='economic')
    S = np.random.randn(r, r) + 1j * np.random.randn(r, r)
    
    X = QuasiSVD(U, S, V)
    
    # Complex matrix to project
    Y = np.random.randn(m, 8) + 1j * np.random.randn(m, 8)
    
    # Project
    result = X.project_onto_column_space(Y, dense_output=True)
    
    # Expected: U @ U.H @ Y
    expected = U @ (U.T.conj() @ Y)
    assert np.allclose(result, expected), "Complex column space projection failed"


def test_complex_projection_row_space():
    """Test projection onto row space with complex matrices."""
    np.random.seed(60)
    m, n, r = 12, 10, 5
    U, _ = la.qr(np.random.randn(m, r) + 1j * np.random.randn(m, r), mode='economic')
    V, _ = la.qr(np.random.randn(n, r) + 1j * np.random.randn(n, r), mode='economic')
    S = np.random.randn(r, r) + 1j * np.random.randn(r, r)
    
    X = QuasiSVD(U, S, V)
    
    # Complex matrix to project
    Y = np.random.randn(m, n) + 1j * np.random.randn(m, n)
    
    # Project
    result = X.project_onto_row_space(Y, dense_output=True)
    
    # Expected: Y @ V @ V.H
    expected = Y @ V @ V.T.conj()
    assert np.allclose(result, expected), "Complex row space projection failed"
