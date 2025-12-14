"""
Author: Benjamin Carrel, University of Geneva, 2022

Test file for the KrylovSpace class and methods
"""

# %% Imports
import numpy as np
import pytest
import scipy.linalg as la
import scipy.sparse as sps
from lowrank.krylov.spaces.krylov_space import KrylovSpace


# %% Vector case
np.random.seed(1234)
A = sps.random(20, 20, density=0.1, format='csc')
x = np.random.rand(20,1)


#%% Krylov space
def test_vector_KrylovSpace():
    # Non-symmetric matrix
    KS = KrylovSpace(A, x)

    # Test the attributes
    assert KS.A is A, "Wrong A"
    assert KS.n == 20, "Wrong n"
    assert KS.r == 1, "Wrong r"

    # Test for m = 2
    m = 2
    Q_ref, _ = la.qr(np.column_stack((x, A@x)), mode='economic')
    KS.augment_basis()
    assert np.allclose(la.norm(KS.Q, axis=0), np.ones(2)), "The columns of Q are not normalized -> error in augment basis"
    assert la.norm(KS.Q.T.dot(KS.Q) - np.eye(2)) < 1e-10, "The columns of Q are not orthogonal -> error in augment basis"
    print('Krylov Space Basis OK.')

    assert la.norm(KS.Q.dot(KS.Q.T) - Q_ref.dot(Q_ref.T)) < 1e-10, "Wrong projection -> error in augment basis"
    # assert np.allclose(KS.Am, Q_ref.T.dot(A.dot(Q_ref))), "Wrong projected A -> error in augment basis"
    print('Krylov Space Projection (m=2) OK.')

    # Test for m = 3
    m = 3
    Q_ref, _ = la.qr(np.column_stack((x, A@x, A@A@x)), mode='economic')
    KS.augment_basis()
    assert np.allclose(la.norm(KS.Q, axis=0), np.ones(3)), "The columns of Q are not normalized -> error in augment basis"
    assert la.norm(KS.Q.T.dot(KS.Q) - np.eye(3)) < 1e-10, "The columns of Q are not orthogonal -> error in augment basis"
    assert la.norm(KS.Q.dot(KS.Q.T) - Q_ref.dot(Q_ref.T)) < 1e-10, "Wrong projection -> error in augment basis"
    # assert np.allclose(KS.Am, Q_ref.T.dot(A.dot(Q_ref))), "Wrong projected A -> error in augment basis"
    print('Krylov Space Projection (m=3) OK.')

    # Symmetric matrix
    S = A.T.dot(A)
    KS = KrylovSpace(S, x)

    # Test the attributes
    assert KS.A is S, "Wrong A"
    assert KS.n == 20, "Wrong n"
    assert KS.r == 1, "Wrong r"

    # Test for m = 2
    m = 2
    Q_ref, _ = la.qr(np.column_stack((x, S@x)), mode='economic')
    KS.augment_basis()
    assert np.allclose(la.norm(KS.Q, axis=0), np.ones(2)), "The columns of Q are not normalized -> error in augment basis"
    assert la.norm(KS.Q.T.dot(KS.Q) - np.eye(2)) < 1e-10, "The columns of Q are not orthogonal -> error in augment basis"
    print('(Symmetric) Krylov Space Basis OK.')
    assert la.norm(KS.Q.dot(KS.Q.T) - Q_ref.dot(Q_ref.T)) < 1e-10, "Wrong projection -> error in augment basis"
    # assert np.allclose(KS.Am, Q_ref.T.dot(S.dot(Q_ref))), "Wrong projected A -> error in augment basis"
    print('(Symmetric) Krylov Space Projection (m=2) OK.')

    # Test for m = 3
    m = 3
    Q_ref, _ = la.qr(np.column_stack((x, S@x, S@S@x)), mode='economic')
    KS.augment_basis()
    assert np.allclose(la.norm(KS.Q, axis=0), np.ones(3)), "The columns of Q are not normalized -> error in augment basis"
    assert la.norm(KS.Q.T.dot(KS.Q) - np.eye(3)) < 1e-10, "The columns of Q are not orthogonal -> error in augment basis"
    print('(Symmetric) Krylov Space Basis OK.')
    assert la.norm(KS.Q.dot(KS.Q.T) - Q_ref.dot(Q_ref.T)) < 1e-10, "Wrong projection -> error in augment basis"
    # assert np.allclose(KS.Am, Q_ref.T.dot(S.dot(Q_ref))), "Wrong projected A -> error in augment basis"
    print('(Symmetric) Krylov Space Projection (m=3) OK.')




#%% Matrix case
np.random.seed(1234)
A = sps.random(20, 20, density=0.1, format='csc')
X0 = np.random.rand(20, 3)


# %% Block Krylov Space
def test_block_KrylovSpace():
    # Non-symmetric case (Arnoldi)
    m = 4
    r = X0.shape[1]

    # Reference computation
    space = [(A**i).dot(X0) for i in range(m)]
    Q_ref, R = la.qr(np.column_stack(space), mode='economic')

    # Check the reference
    assert np.allclose(la.norm(Q_ref, axis=0), np.ones(m*r)), "The columns of Q are not normalized -> error in the QR decomposition"
    assert la.norm(Q_ref.T.dot(Q_ref) - np.eye(m*r)) < 1e-10, "The columns of Q are not orthogonal -> error in the QR decomposition"
    print('Reference OK.')

    # Block Krylov Space
    KS = KrylovSpace(A, X0)
    for i in range(m-1):
        KS.augment_basis()

    # Check the basis
    assert (KS.Q.shape == (20, m*r)), "Wrong shape of Q -> error in block Krylov Space"
    assert (KS.Am.shape == (m*r, m*r)), "Wrong shape of Am -> error in block Krylov Space"
    assert np.allclose(la.norm(KS.Q, axis=0), np.ones(m*r)), "The columns of Q are not normalized -> error in block Krylov Space"
    assert la.norm(KS.Q.T.dot(KS.Q) - np.eye(m*r)) < 1e-10, "The columns of Q are not orthogonal -> error in block Krylov Space"
    print('Block Krylov Space Basis OK.')

    # Compare to the reference
    assert la.norm(KS.Q.dot(KS.Q.T) - Q_ref.dot(Q_ref.T)) < 1e-10, "Wrong projection -> error in block Krylov Space"
    print('Block Krylov Space Projection OK.')
    # assert np.allclose(KS.Am, Q_ref.T.dot(A.dot(Q_ref))), "Wrong projected A -> error in block Krylov Space"

    # Symmetric case (Lanczos)
    m = 4
    r = X0.shape[1]
    S = A.T.dot(A)

    # Reference computation
    space = [(S**i).dot(X0) for i in range(m)]
    Q_ref, _ = la.qr(np.column_stack(space), mode='economic')

    # Check the reference
    assert np.allclose(la.norm(Q_ref, axis=0), np.ones(m*r)), "The columns of Q are not normalized -> error in the QR decomposition"
    assert la.norm(Q_ref.T.dot(Q_ref) - np.eye(m*r)) < 1e-10, "The columns of Q are not orthogonal -> error in the QR decomposition"
    print('Reference OK.')

    # Block Krylov Space
    KS = KrylovSpace(S, X0, symmetric=True)
    for i in range(m-1):
        KS.augment_basis()

    # Check the basis
    assert (KS.Q.shape == (20, m*r)), "Wrong shape of Q -> error in block Krylov Space"
    assert (KS.Am.shape == (m*r, m*r)), "Wrong shape of Am -> error in block Krylov Space"
    assert np.allclose(la.norm(KS.Q, axis=0), np.ones(m*r)), "The columns of Q are not normalized -> error in block Krylov Space"
    assert la.norm(KS.Q.T.dot(KS.Q) - np.eye(m*r)) < 1e-10, "The columns of Q are not orthogonal -> error in block Krylov Space"
    print('Block Krylov Space Basis OK.')

    # Compare to the reference
    assert la.norm(KS.Q.dot(KS.Q.T) - Q_ref.dot(Q_ref.T)) < 1e-10, "Wrong projection -> error in block Krylov Space"
    print('Block Krylov Space Projection OK.')


# %% Test fixtures for error handling
@pytest.fixture
def valid_krylov_matrix():
    """Create a valid sparse matrix"""
    np.random.seed(42)
    return sps.random(10, 10, density=0.3, format='csc')


@pytest.fixture
def valid_krylov_vector():
    """Create a valid vector"""
    np.random.seed(42)
    return np.random.rand(10, 1)


# %% Input validation tests
class TestKrylovInputValidation:
    """Test input validation for KrylovSpace"""
    
    def test_non_sparse_matrix(self, valid_krylov_vector):
        """Test that non-sparse matrix raises TypeError"""
        A = np.random.rand(10, 10)  # Dense matrix
        with pytest.raises(TypeError, match="A must be a sparse matrix"):
            KrylovSpace(A, valid_krylov_vector)
    
    def test_non_array_vector(self, valid_krylov_matrix):
        """Test that non-numpy array raises TypeError"""
        x = [[1], [2], [3]]  # List instead of array
        with pytest.raises(TypeError, match="X must be a numpy array"):
            KrylovSpace(valid_krylov_matrix, x)
    
    def test_non_square_matrix(self, valid_krylov_vector):
        """Test that non-square matrix raises ValueError"""
        A = sps.random(10, 15, density=0.3, format='csc')
        with pytest.raises(ValueError, match="A must be a square matrix"):
            KrylovSpace(A, valid_krylov_vector)
    
    def test_dimension_mismatch(self, valid_krylov_matrix):
        """Test that dimension mismatch raises ValueError"""
        x = np.random.rand(5, 1)  # Wrong dimension
        with pytest.raises(ValueError, match="A and X must have the same number of rows"):
            KrylovSpace(valid_krylov_matrix, x)
    
    def test_nan_in_vector(self, valid_krylov_matrix):
        """Test that NaN in vector raises ValueError"""
        x = np.random.rand(10, 1)
        x[0] = np.nan
        with pytest.raises(ValueError, match="X contains NaN or Inf values"):
            KrylovSpace(valid_krylov_matrix, x)
    
    def test_inf_in_vector(self, valid_krylov_matrix):
        """Test that Inf in vector raises ValueError"""
        x = np.random.rand(10, 1)
        x[0] = np.inf
        with pytest.raises(ValueError, match="X contains NaN or Inf values"):
            KrylovSpace(valid_krylov_matrix, x)
    
    def test_nan_in_matrix(self, valid_krylov_vector):
        """Test that NaN in matrix raises ValueError"""
        A = sps.random(10, 10, density=0.3, format='csc')
        A.data[0] = np.nan
        with pytest.raises(ValueError, match="A contains NaN or Inf values"):
            KrylovSpace(A, valid_krylov_vector)
    
    def test_inf_in_matrix(self, valid_krylov_vector):
        """Test that Inf in matrix raises ValueError"""
        A = sps.random(10, 10, density=0.3, format='csc')
        A.data[0] = np.inf
        with pytest.raises(ValueError, match="A contains NaN or Inf values"):
            KrylovSpace(A, valid_krylov_vector)


# %% Dimension overflow tests
class TestKrylovDimensionOverflow:
    """Test behavior when space size exceeds matrix dimension"""
    
    def test_krylov_dimension_warning(self, valid_krylov_matrix, valid_krylov_vector):
        """Test that augmenting beyond dimension gives warning"""
        KS = KrylovSpace(valid_krylov_matrix, valid_krylov_vector)
        # Augment to maximum
        for _ in range(9):  # n=10, start with m=1, so 9 augmentations
            KS.augment_basis()
        
        # Try to augment beyond dimension
        with pytest.warns(UserWarning, match="The next basis would exceed the dimension"):
            KS.augment_basis()


# %% Property tests for KrylovSpace
class TestKrylovProperties:
    """Test properties for KrylovSpace"""
    
    def test_krylov_size_property(self, valid_krylov_matrix, valid_krylov_vector):
        """Test size property updates correctly"""
        KS = KrylovSpace(valid_krylov_matrix, valid_krylov_vector)
        assert KS.size == 1, "Initial size should be 1"
        
        KS.augment_basis()
        assert KS.size == 2, "Size should be 2 after first augmentation"
        
        KS.augment_basis()
        assert KS.size == 3, "Size should be 3 after second augmentation"
    
    def test_krylov_basis_property(self, valid_krylov_matrix, valid_krylov_vector):
        """Test basis property returns Q"""
        KS = KrylovSpace(valid_krylov_matrix, valid_krylov_vector)
        assert np.array_equal(KS.basis, KS.Q), "basis should be same as Q"
    
    def test_reduced_A_property(self, valid_krylov_matrix, valid_krylov_vector):
        """Test reduced_A computes Q^T A Q correctly"""
        KS = KrylovSpace(valid_krylov_matrix, valid_krylov_vector)
        KS.augment_basis()
        KS.augment_basis()
        
        Q = KS.Q
        A = valid_krylov_matrix
        expected = Q.T @ A @ Q
        
        assert np.allclose(KS.reduced_A, expected), "reduced_A should be Q^T A Q"
    
    def test_Am_Ak_shortcuts(self, valid_krylov_matrix, valid_krylov_vector):
        """Test Am and Ak are shortcuts to reduced_A"""
        KS = KrylovSpace(valid_krylov_matrix, valid_krylov_vector)
        KS.augment_basis()
        
        assert np.array_equal(KS.Am, KS.reduced_A), "Am should equal reduced_A"
        assert np.array_equal(KS.Ak, KS.reduced_A), "Ak should equal reduced_A"


# %% Edge case tests for KrylovSpace
class TestKrylovEdgeCases:
    """Test edge cases and special scenarios"""
    
    def test_single_iteration(self, valid_krylov_matrix, valid_krylov_vector):
        """Test space with only initial vector"""
        KS = KrylovSpace(valid_krylov_matrix, valid_krylov_vector)
        assert KS.Q.shape == (10, 1), "Should have single column"
        assert np.allclose(la.norm(KS.Q, axis=0), 1.0), "Should be normalized"
    
    def test_dtype_consistency(self, valid_krylov_matrix, valid_krylov_vector):
        """Test dtype is consistent throughout"""
        KS = KrylovSpace(valid_krylov_matrix, valid_krylov_vector)
        original_dtype = KS.dtype
        
        KS.augment_basis()
        assert KS.Q.dtype == original_dtype, "dtype should remain consistent"
    
    def test_symmetric_detection(self):
        """Test symmetric matrix detection"""
        np.random.seed(42)
        A_nonsym = sps.random(10, 10, density=0.3, format='csc')
        A_sym = A_nonsym.T @ A_nonsym
        
        x = np.random.rand(10, 1)
        
        KS_nonsym = KrylovSpace(A_nonsym, x)
        assert not KS_nonsym.is_symmetric, "Should detect non-symmetric"
        
        KS_sym = KrylovSpace(A_sym, x)
        assert KS_sym.is_symmetric, "Should detect symmetric"
    
    def test_custom_matvec(self, valid_krylov_matrix, valid_krylov_vector):
        """Test custom matvec function"""
        call_count = [0]
        
        def custom_matvec(x):
            call_count[0] += 1
            return valid_krylov_matrix @ x
        
        KS = KrylovSpace(valid_krylov_matrix, valid_krylov_vector, matvec=custom_matvec)
        KS.augment_basis()
        
        assert call_count[0] > 0, "Custom matvec should be called"


# %% compute_all tests for KrylovSpace
class TestKrylovComputeAll:
    """Test compute_all method"""
    
    def test_krylov_compute_all_with_max_iter(self, valid_krylov_matrix, valid_krylov_vector):
        """Test compute_all with specified max_iter"""
        max_iter = 5
        KS = KrylovSpace(valid_krylov_matrix, valid_krylov_vector, max_iter=max_iter)
        KS.compute_all()
        
        # Should have m = max_iter + 1 (initial + max_iter augmentations)
        assert KS.m == max_iter + 1, f"Should have m={max_iter+1} after compute_all"


# %% Advanced functionality tests for KrylovSpace
class TestKrylovAdvancedFunctionality:
    """Test convergence and consistency"""
    
    def test_krylov_span_consistency(self, valid_krylov_matrix, valid_krylov_vector):
        """Test that Krylov space spans correct subspace"""
        KS = KrylovSpace(valid_krylov_matrix, valid_krylov_vector)
        for _ in range(3):
            KS.augment_basis()
        
        # Compute reference space
        A = valid_krylov_matrix.toarray()
        x = valid_krylov_vector
        space = np.column_stack([np.linalg.matrix_power(A, i) @ x for i in range(4)])
        Q_ref, _ = la.qr(space, mode='economic')
        
        # Check projection equality
        P_computed = KS.Q @ KS.Q.T
        P_ref = Q_ref @ Q_ref.T
        
        assert np.allclose(P_computed, P_ref), \
            "Krylov space should span same subspace as reference"
    
    def test_orthogonality_preservation(self):
        """Test orthogonality is preserved through multiple augmentations"""
        np.random.seed(42)
        test_matrix = sps.random(15, 15, density=0.4, format='csc')
        test_block = np.random.rand(15, 3)
        
        KS = KrylovSpace(test_matrix, test_block)
        
        for i in range(3):
            KS.augment_basis()
            # Check orthonormality at each step
            Q = KS.Q
            identity = np.eye(Q.shape[1])
            error = la.norm(Q.T @ Q - identity)
            assert error < 1e-10, \
                f"Orthogonality lost after {i+1} augmentations (error: {error})"
    
    def test_reduced_matrix_structure(self, valid_krylov_matrix, valid_krylov_vector):
        """Test reduced matrix has expected structure"""
        # For symmetric matrices, reduced matrix should be symmetric
        S = valid_krylov_matrix.T @ valid_krylov_matrix
        KS = KrylovSpace(S, valid_krylov_vector)
        for _ in range(3):
            KS.augment_basis()
        
        Am = KS.reduced_A
        assert np.allclose(Am, Am.T), \
            "Reduced matrix should be symmetric for symmetric A"


# %% Tests for space_structure base class properties
def test_space_structure_repr():
    """Test __repr__ method of SpaceStructure."""
    np.random.seed(400)
    A = sps.random(15, 15, density=0.3, format='csc')
    x = np.random.rand(15, 1)
    
    KS = KrylovSpace(A, x)
    KS.augment_basis()
    
    repr_str = repr(KS)
    assert "KrylovSpace" in repr_str, "repr should contain class name"
    assert "size" in repr_str, "repr should mention size"
    assert "shape" in repr_str, "repr should mention shape"


def test_space_structure_dtype():
    """Test dtype handling in SpaceStructure."""
    np.random.seed(401)
    # Float case
    A_float = sps.random(10, 10, density=0.3, format='csc')
    x_float = np.random.rand(10, 1)
    KS_float = KrylovSpace(A_float, x_float)
    assert KS_float.dtype == np.float64, "Should detect float64"
    
    # Complex case
    A_complex = sps.random(10, 10, density=0.3, format='csc', dtype=np.complex128)
    x_complex = np.random.rand(10, 1) + 1j * np.random.rand(10, 1)
    KS_complex = KrylovSpace(A_complex, x_complex)
    assert KS_complex.dtype == np.complex128, "Should detect complex128"
    
    # Mixed case (should promote)
    x_complex_mixed = np.random.rand(10, 1) + 1j * np.random.rand(10, 1)
    KS_mixed = KrylovSpace(A_float, x_complex_mixed)
    assert np.iscomplexobj(np.zeros(1, dtype=KS_mixed.dtype)), "Should promote to complex"


def test_space_structure_max_iter():
    """Test max_iter parameter in SpaceStructure."""
    np.random.seed(402)
    A = sps.random(20, 20, density=0.3, format='csc')
    x = np.random.rand(20, 1)
    
    # Default max_iter
    KS1 = KrylovSpace(A, x)
    assert KS1.max_iter == 20, f"Default max_iter should be n=20, got {KS1.max_iter}"
    
    # Custom max_iter
    KS2 = KrylovSpace(A, x, max_iter=10)
    assert KS2.max_iter == 10, f"Custom max_iter should be 10, got {KS2.max_iter}"


def test_space_structure_symmetry_detection():
    """Test automatic symmetry detection."""
    np.random.seed(403)
    
    # Non-symmetric matrix
    A_nonsym = sps.random(10, 10, density=0.3, format='csc')
    x = np.random.rand(10, 1)
    KS_nonsym = KrylovSpace(A_nonsym, x)
    # Should detect as non-symmetric
    assert not KS_nonsym.is_symmetric or (A_nonsym - A_nonsym.T).nnz == 0
    
    # Symmetric matrix
    A_dense = np.random.randn(10, 10)
    A_sym = sps.csc_matrix(A_dense + A_dense.T)
    KS_sym = KrylovSpace(A_sym, x)
    assert KS_sym.is_symmetric, "Should detect symmetry"
    
    # Manual override
    KS_override = KrylovSpace(A_nonsym, x, is_symmetric=True)
    assert KS_override.is_symmetric, "Should respect is_symmetric parameter"


def test_space_structure_extra_args():
    """Test extra_args parameter storage."""
    np.random.seed(404)
    A = sps.random(10, 10, density=0.3, format='csc')
    x = np.random.rand(10, 1)
    
    extra = {'custom_param': 42, 'another_param': 'test'}
    KS = KrylovSpace(A, x, **extra)
    
    assert 'custom_param' in KS.extra_args, "extra_args should store custom params"
    assert KS.extra_args['custom_param'] == 42, "Should preserve param values"


def test_space_structure_n_r_attributes():
    """Test n and r attributes."""
    np.random.seed(405)
    A = sps.random(15, 15, density=0.3, format='csc')
    
    # Vector case
    x_vec = np.random.rand(15, 1)
    KS_vec = KrylovSpace(A, x_vec)
    assert KS_vec.n == 15, "n should be matrix dimension"
    assert KS_vec.r == 1, "r should be 1 for vector"
    
    # Block case
    x_block = np.random.rand(15, 3)
    KS_block = KrylovSpace(A, x_block)
    assert KS_block.n == 15, "n should be matrix dimension"
    assert KS_block.r == 3, "r should be 3 for block of 3"


def test_space_structure_X_reshape():
    """Test that X must be 2D - 1D arrays should raise error."""
    np.random.seed(406)
    A = sps.random(10, 10, density=0.3, format='csc')
    
    # 1D array should raise error
    x_1d = np.random.rand(10)
    with pytest.raises((ValueError, np.linalg.LinAlgError)):
        KS = KrylovSpace(A, x_1d)


def test_space_structure_reduced_A_property():
    """Test reduced_A property computes Q^T A Q."""
    np.random.seed(407)
    A = sps.random(12, 12, density=0.4, format='csc')
    x = np.random.rand(12, 1)
    
    KS = KrylovSpace(A, x)
    KS.augment_basis()
    KS.augment_basis()
    
    Q = KS.Q
    Am = KS.reduced_A
    
    # Should equal Q^T @ A @ Q
    expected = Q.T @ A.toarray() @ Q
    
    assert np.allclose(Am, expected, atol=1e-10), "reduced_A incorrect"
    assert Am.shape == (Q.shape[1], Q.shape[1]), "reduced_A wrong shape"


def test_space_structure_Am_property():
    """Test Am property (alias for reduced_A)."""
    np.random.seed(408)
    A = sps.random(10, 10, density=0.4, format='csc')
    x = np.random.rand(10, 1)
    
    KS = KrylovSpace(A, x)
    KS.augment_basis()
    
    # Am should be alias for reduced_A
    assert np.allclose(KS.Am, KS.reduced_A), "Am should equal reduced_A"


def test_space_structure_check_inputs():
    """Test input validation in SpaceStructure."""
    np.random.seed(409)
    A = sps.random(10, 10, density=0.3, format='csc')
    x_wrong = np.random.rand(5, 1)  # Wrong dimension
    
    # Should raise error for dimension mismatch
    with pytest.raises(ValueError):
        KS = KrylovSpace(A, x_wrong)


# %%
