import numpy as np
import scipy.linalg as la
import pytest
from lowrank.utils import givens


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def real_vector():
    """Simple real 2D vector."""
    return 3.0, 4.0


@pytest.fixture
def complex_vector():
    """Simple complex 2D vector."""
    return 1 + 2j, 3 - 4j


@pytest.fixture
def large_real_vector():
    """Large magnitude real vector to test numerical stability."""
    return 1e10, 1e10


@pytest.fixture
def small_real_vector():
    """Small magnitude real vector to test numerical stability."""
    return 1e-10, 1e-10


# ============================================================================
# Basic Functionality Tests
# ============================================================================

def test_givens_real_basic(real_vector):
    """Test Givens rotation with simple real vector."""
    x, y = real_vector
    G = givens(x, y)
    
    # Check shape
    assert G.shape == (2, 2), "Givens matrix should be 2x2"
    
    # Check result
    v = np.array([x, y])
    result = G @ v
    
    # First component should be the norm (real, non-negative)
    expected_r = np.hypot(x, y)
    assert np.isreal(result[0]), "First component should be real"
    assert np.abs(result[0] - expected_r) < 1e-10, "First component should be the norm"
    
    # Second component should be zero
    assert np.abs(result[1]) < 1e-10, "Second component should be zero"


def test_givens_complex_basic(complex_vector):
    """Test Givens rotation with simple complex vector."""
    x, y = complex_vector
    G = givens(x, y)
    
    # Check shape
    assert G.shape == (2, 2), "Givens matrix should be 2x2"
    
    # Check result
    v = np.array([x, y])
    result = G @ v
    
    # First component should be the norm (real, non-negative)
    expected_r = np.hypot(np.abs(x), np.abs(y))
    assert np.abs(np.imag(result[0])) < 1e-10, "First component should be real"
    assert result[0].real >= 0, "First component should be non-negative"
    assert np.abs(result[0] - expected_r) < 1e-10, "First component should be the norm"
    
    # Second component should be zero
    assert np.abs(result[1]) < 1e-10, "Second component should be zero"


def test_givens_returns_complex_array():
    """Test that givens always returns a complex array."""
    x, y = 1.0, 1.0
    G = givens(x, y)
    
    assert G.dtype == np.complex128, "Should return complex128 dtype"


# ============================================================================
# Unitary Property Tests
# ============================================================================

def test_givens_is_unitary_real():
    """Test that Givens matrix is unitary for real inputs."""
    x, y = 3.0, 4.0
    G = givens(x, y)
    
    # Check G @ G.H = I
    identity = G @ G.conj().T
    expected_identity = np.eye(2, dtype=np.complex128)
    
    assert np.allclose(identity, expected_identity, atol=1e-12), "G should be unitary"


def test_givens_is_unitary_complex():
    """Test that Givens matrix is unitary for complex inputs."""
    x, y = 1 + 2j, 3 - 4j
    G = givens(x, y)
    
    # Check G @ G.H = I
    identity = G @ G.conj().T
    expected_identity = np.eye(2, dtype=np.complex128)
    
    assert np.allclose(identity, expected_identity, atol=1e-12), "G should be unitary"


def test_givens_determinant():
    """Test that Givens matrix has unit determinant."""
    x, y = 2 + 3j, 1 - 5j
    G = givens(x, y)
    
    det = la.det(G)
    
    # For SU(2), det should be 1, but for general unitary it should have |det| = 1
    assert np.abs(np.abs(det) - 1.0) < 1e-10, "Determinant should have unit magnitude"


@pytest.mark.parametrize("x,y", [
    (1.0, 0.0),
    (0.0, 1.0),
    (1.0, 1.0),
    (1 + 1j, 0.0),
    (0.0, 1 + 1j),
    (1 + 1j, 1 - 1j),
    (5.0, -3.0),
    (-2.0, 7.0),
    (1e5, 1e5),
    (1e-5, 1e-5),
])
def test_givens_unitary_parametrized(x, y):
    """Test unitarity for various inputs."""
    G = givens(x, y)
    identity = G @ G.conj().T
    expected_identity = np.eye(2, dtype=np.complex128)
    
    assert np.allclose(identity, expected_identity, atol=1e-10), f"Failed for x={x}, y={y}"


# ============================================================================
# Edge Case Tests
# ============================================================================

def test_givens_y_zero():
    """Test Givens rotation when y=0."""
    x, y = 5.0, 0.0
    G = givens(x, y)
    
    v = np.array([x, y])
    result = G @ v
    
    # Result should be [|x|, 0]
    assert np.abs(result[0] - np.abs(x)) < 1e-12, "First component should be |x|"
    assert np.abs(result[1]) < 1e-12, "Second component should be zero"
    
    # G should be unitary
    assert np.allclose(G @ G.conj().T, np.eye(2, dtype=np.complex128), atol=1e-12)


def test_givens_y_zero_complex():
    """Test Givens rotation when y=0 and x is complex."""
    x, y = 3 + 4j, 0.0
    G = givens(x, y)
    
    v = np.array([x, y])
    result = G @ v
    
    # Result should be [|x|, 0]
    assert np.abs(result[0] - np.abs(x)) < 1e-12, "First component should be |x|"
    assert np.abs(result[1]) < 1e-12, "Second component should be zero"
    assert result[0].real >= 0, "First component should be non-negative"


def test_givens_x_zero():
    """Test Givens rotation when x=0."""
    x, y = 0.0, 3.0
    G = givens(x, y)
    
    v = np.array([x, y])
    result = G @ v
    
    # Result should be [|y|, 0]
    assert np.abs(result[0] - np.abs(y)) < 1e-12, "First component should be |y|"
    assert np.abs(result[1]) < 1e-12, "Second component should be zero"


def test_givens_both_zero():
    """Test Givens rotation when both x=0 and y=0."""
    x, y = 0.0, 0.0
    G = givens(x, y)
    
    v = np.array([x, y])
    result = G @ v
    
    # Result should be [0, 0]
    assert np.abs(result[0]) < 1e-12, "First component should be zero"
    assert np.abs(result[1]) < 1e-12, "Second component should be zero"
    
    # G should still be unitary (should be identity or close to it)
    assert np.allclose(G @ G.conj().T, np.eye(2, dtype=np.complex128), atol=1e-12)


def test_givens_negative_real():
    """Test Givens rotation with negative real values."""
    x, y = -3.0, -4.0
    G = givens(x, y)
    
    v = np.array([x, y])
    result = G @ v
    
    # Result should still have non-negative real first component
    expected_r = np.hypot(x, y)
    assert result[0].real >= -1e-12, "First component should be non-negative"
    assert np.abs(result[0] - expected_r) < 1e-10, "First component should be the norm"
    assert np.abs(result[1]) < 1e-10, "Second component should be zero"


# ============================================================================
# Numerical Stability Tests
# ============================================================================

def test_givens_large_values(large_real_vector):
    """Test numerical stability with large values."""
    x, y = large_real_vector
    G = givens(x, y)
    
    v = np.array([x, y])
    result = G @ v
    
    # Check norm is preserved
    expected_r = np.hypot(x, y)
    assert np.abs(result[0] - expected_r) / expected_r < 1e-10, "Relative error should be small"
    assert np.abs(result[1]) < 1e-5, "Second component should be near zero"
    
    # Check unitarity
    identity = G @ G.conj().T
    assert np.allclose(identity, np.eye(2, dtype=np.complex128), atol=1e-10)


def test_givens_small_values(small_real_vector):
    """Test numerical stability with small values."""
    x, y = small_real_vector
    G = givens(x, y)
    
    v = np.array([x, y])
    result = G @ v
    
    # Check norm is preserved (relative error)
    expected_r = np.hypot(x, y)
    if expected_r > 0:
        assert np.abs(result[0] - expected_r) / expected_r < 1e-10, "Relative error should be small"
    
    # Check unitarity
    identity = G @ G.conj().T
    assert np.allclose(identity, np.eye(2, dtype=np.complex128), atol=1e-10)


def test_givens_vastly_different_magnitudes():
    """Test when x and y have vastly different magnitudes."""
    x, y = 1e10, 1e-10
    G = givens(x, y)
    
    v = np.array([x, y])
    result = G @ v
    
    # Check norm is approximately |x| since y is negligible
    expected_r = np.hypot(x, y)
    assert np.abs(result[0] - expected_r) / expected_r < 1e-10, "Should handle magnitude difference"
    assert np.abs(result[1]) < 1e-5, "Second component should be near zero"


def test_givens_y_much_larger_than_x():
    """Test when |y| >> |x|."""
    x, y = 1.0, 1e10
    G = givens(x, y)
    
    v = np.array([x, y])
    result = G @ v
    
    expected_r = np.hypot(np.abs(x), np.abs(y))
    assert np.abs(result[0] - expected_r) / expected_r < 1e-10, "Should handle y >> x"
    assert np.abs(result[1]) < 1e-5, "Second component should be near zero"


# ============================================================================
# Matrix Structure Tests
# ============================================================================

def test_givens_matrix_form():
    """Test that Givens matrix has the correct form [[c, s], [-conj(s), conj(c)]]."""
    x, y = 1 + 1j, 2 - 1j
    G = givens(x, y)
    
    c = G[0, 0]
    s = G[0, 1]
    
    # Check structure
    assert np.abs(G[1, 0] + np.conj(s)) < 1e-12, "G[1,0] should be -conj(s)"
    assert np.abs(G[1, 1] - np.conj(c)) < 1e-12, "G[1,1] should be conj(c)"


def test_givens_normalization():
    """Test that |c|^2 + |s|^2 = 1."""
    x, y = 3 + 2j, 1 - 4j
    G = givens(x, y)
    
    c = G[0, 0]
    s = G[0, 1]
    
    norm_squared = np.abs(c)**2 + np.abs(s)**2
    assert np.abs(norm_squared - 1.0) < 1e-12, "|c|^2 + |s|^2 should equal 1"


# ============================================================================
# Rotation Property Tests
# ============================================================================

def test_givens_preserves_norm():
    """Test that Givens rotation preserves vector norm."""
    x, y = 5 + 3j, 2 - 7j
    G = givens(x, y)
    
    v = np.array([x, y])
    result = G @ v
    
    original_norm = la.norm(v)
    result_norm = la.norm(result)
    
    assert np.abs(original_norm - result_norm) < 1e-10, "Norm should be preserved"


def test_givens_result_is_real_nonnegative():
    """Test that first component of result is real and non-negative."""
    np.random.seed(42)
    
    for _ in range(10):
        x = np.random.randn() + 1j * np.random.randn()
        y = np.random.randn() + 1j * np.random.randn()
        
        G = givens(x, y)
        v = np.array([x, y])
        result = G @ v
        
        assert np.abs(np.imag(result[0])) < 1e-10, f"First component should be real for x={x}, y={y}"
        assert result[0].real >= -1e-10, f"First component should be non-negative for x={x}, y={y}"


# ============================================================================
# Inverse/Transpose Tests
# ============================================================================

def test_givens_inverse_is_conjugate_transpose():
    """Test that G^{-1} = G^H for unitary matrix."""
    x, y = 2 + 1j, 3 - 2j
    G = givens(x, y)
    
    G_inv = la.inv(G)
    G_H = G.conj().T
    
    assert np.allclose(G_inv, G_H, atol=1e-12), "Inverse should equal conjugate transpose"


def test_givens_double_application():
    """Test that applying G twice with different vectors works correctly."""
    x1, y1 = 3.0, 4.0
    x2, y2 = 1 + 1j, 2 - 1j
    
    G1 = givens(x1, y1)
    G2 = givens(x2, y2)
    
    # Both should be unitary
    assert np.allclose(G1 @ G1.conj().T, np.eye(2, dtype=np.complex128), atol=1e-12)
    assert np.allclose(G2 @ G2.conj().T, np.eye(2, dtype=np.complex128), atol=1e-12)
    
    # G1 @ G2 should also be unitary
    G_combined = G1 @ G2
    assert np.allclose(G_combined @ G_combined.conj().T, np.eye(2, dtype=np.complex128), atol=1e-12)


# ============================================================================
# Comparison with Standard Implementation Tests
# ============================================================================

def test_givens_real_matches_standard():
    """Test that real case matches standard Givens rotation formula."""
    x, y = 3.0, 4.0
    G = givens(x, y)
    
    # Standard Givens rotation for real numbers
    r = np.hypot(x, y)
    c_expected = x / r
    s_expected = y / r
    
    # Our implementation should give similar results (may differ in sign/phase)
    G_expected = np.array([[c_expected, s_expected], [-s_expected, c_expected]], dtype=np.complex128)
    
    # Check they both zero out the second component
    v = np.array([x, y])
    result = G @ v
    result_expected = G_expected @ v
    
    assert np.abs(result[1]) < 1e-12, "Our implementation should zero second component"
    assert np.abs(result_expected[1]) < 1e-12, "Standard should zero second component"
    
    # Both should give the same norm
    assert np.abs(np.abs(result[0]) - np.abs(result_expected[0])) < 1e-12


# ============================================================================
# Random Input Tests
# ============================================================================

@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_givens_random_complex(seed):
    """Test Givens rotation with random complex inputs."""
    np.random.seed(seed)
    x = np.random.randn() + 1j * np.random.randn()
    y = np.random.randn() + 1j * np.random.randn()
    
    G = givens(x, y)
    v = np.array([x, y])
    result = G @ v
    
    # Check result
    expected_r = np.hypot(np.abs(x), np.abs(y))
    assert np.abs(result[0] - expected_r) < 1e-10, f"Seed {seed}: first component incorrect"
    assert np.abs(result[1]) < 1e-10, f"Seed {seed}: second component not zero"
    
    # Check unitarity
    assert np.allclose(G @ G.conj().T, np.eye(2, dtype=np.complex128), atol=1e-12)


# ============================================================================
# Documentation Example Tests
# ============================================================================

def test_givens_documentation_real_example():
    """Test the real example from the documentation."""
    x_real, y_real = 3.0, 4.0
    G_real = givens(x_real, y_real)
    v_real = np.array([x_real, y_real])
    result_real = G_real @ v_real
    
    expected_r = np.hypot(x_real, y_real)
    assert np.abs(result_real[0] - expected_r) < 1e-10
    assert np.abs(result_real[1]) < 1e-10


def test_givens_documentation_complex_example():
    """Test the complex example from the documentation."""
    x_complex, y_complex = 1 + 2j, 3 - 4j
    G_complex = givens(x_complex, y_complex)
    v_complex = np.array([x_complex, y_complex])
    result_complex = G_complex @ v_complex
    
    expected_r = np.hypot(np.abs(x_complex), np.abs(y_complex))
    assert np.abs(result_complex[0] - expected_r) < 1e-10
    assert np.abs(result_complex[1]) < 1e-10
