"""
Author: Benjamin Carrel, University of Geneva, 2022

Test file for the ExtendedKrylovSpace class and methods
"""

# %% Imports
import numpy as np
import pytest
import scipy.linalg as la
import scipy.sparse as sps
import scipy.sparse.linalg as spsla
from lowrank.krylov.spaces.extended_krylov_space import ExtendedKrylovSpace


# %% Vector case
np.random.seed(1234)
A = sps.random(20, 20, density=0.5, format='csc')
x = np.random.rand(20,1)
invA = lambda x: spsla.spsolve(A, x).reshape(x.shape)

def space(A, invA, x, m):
    def repeat_dot_and_inv_A(x, n):
        y = x
        for _ in range(n-1):
            x = A.dot(x)
        for _ in range(n):
            y = invA(y)
        return np.column_stack((x, y))
    return np.column_stack([repeat_dot_and_inv_A(x, n) for n in range(1, m+1)])


#%% Krylov space
def test_vector_KrylovSpace():
    # Non-symmetric matrix
    KS = ExtendedKrylovSpace(A, x, invA=invA)

    # Test the attributes
    assert KS.A is A, "Wrong A"
    assert KS.n == 20, "Wrong n"
    assert KS.r == 1, "Wrong r"

    # Test for m = 2
    m = 2
    Q_ref, _ = la.qr(space(A, invA, x, m), mode='economic')
    # Q_ref, _ = la.qr(np.column_stack((x, A.dot(x), invA(x), invA(invA(x)))), mode='economic')
    KS.augment_basis()
    assert np.allclose(la.norm(KS.Q, axis=0), np.ones(2*m)), "The columns of Q are not normalized -> error in augment basis"
    assert la.norm(KS.Q.T.dot(KS.Q) - np.eye(2*m)) < 1e-10, "The columns of Q are not orthogonal -> error in augment basis"
    print('Krylov Space Basis OK.')

    assert la.norm(KS.Q.dot(KS.Q.T) - Q_ref.dot(Q_ref.T)) < 1e-10, "Wrong projection -> error in augment basis"
    # assert np.allclose(KS.Am, Q_ref.T.dot(A.dot(Q_ref))), "Wrong projected A -> error in augment basis"
    print('Krylov Space Projection (m=2) OK.')

    # Test for m = 3
    m = 3
    Q_ref, _ = la.qr(space(A, invA, x, m), mode='economic')
    KS.augment_basis()
    assert np.allclose(la.norm(KS.Q, axis=0), np.ones(3*2)), "The columns of Q are not normalized -> error in augment basis"
    assert la.norm(KS.Q.T.dot(KS.Q) - np.eye(3*2)) < 1e-10, "The columns of Q are not orthogonal -> error in augment basis"
    assert la.norm(KS.Q.dot(KS.Q.T) - Q_ref.dot(Q_ref.T)) < 1e-10, "Wrong projection -> error in augment basis"
    # assert np.allclose(KS.Am, Q_ref.T.dot(A.dot(Q_ref))), "Wrong projected A -> error in augment basis"
    print('Krylov Space Projection (m=3) OK.')

    # Symmetric matrix
    S = A.T.dot(A)
    invS = lambda x: spsla.spsolve(S, x).reshape(x.shape)
    KS = ExtendedKrylovSpace(S, x, invA=invS)

    # Test the attributes
    assert KS.A is S, "Wrong A"
    assert KS.n == 20, "Wrong n"
    assert KS.r == 1, "Wrong r"

    # Test for m = 2
    m = 2
    Q_ref, _ = la.qr(space(S, invS, x, m), mode='economic')
    KS.augment_basis()
    assert np.allclose(la.norm(KS.Q, axis=0), np.ones(2*m)), "The columns of Q are not normalized -> error in augment basis"
    assert la.norm(KS.Q.T.dot(KS.Q) - np.eye(2*m)) < 1e-10, "The columns of Q are not orthogonal -> error in augment basis"
    print('(Symmetric) Krylov Space Basis OK.')
    assert la.norm(KS.Q.dot(KS.Q.T) - Q_ref.dot(Q_ref.T)) < 1e-10, "Wrong projection -> error in augment basis"
    # assert np.allclose(KS.Am, Q_ref.T.dot(S.dot(Q_ref))), "Wrong projected A -> error in augment basis"
    print('(Symmetric) Krylov Space Projection (m=2) OK.')

    # Test for m = 3
    m = 3
    Q_ref, _ = la.qr(space(S, invS, x, m), mode='economic')
    KS.augment_basis()
    assert np.allclose(la.norm(KS.Q, axis=0), np.ones(3*2)), "The columns of Q are not normalized -> error in augment basis"
    assert la.norm(KS.Q.T.dot(KS.Q) - np.eye(3*2)) < 1e-10, "The columns of Q are not orthogonal -> error in augment basis"
    print('(Symmetric) Krylov Space Basis OK.')
    assert la.norm(KS.Q.dot(KS.Q.T) - Q_ref.dot(Q_ref.T)) < 1e-10, "Wrong projection -> error in augment basis"
    # assert np.allclose(KS.Am, Q_ref.T.dot(S.dot(Q_ref))), "Wrong projected A -> error in augment basis"
    print('(Symmetric) Krylov Space Projection (m=3) OK.')




#%% Matrix case
np.random.seed(1234)
A = sps.random(20, 20, density=0.5, format='csc')
X0 = np.random.rand(20, 3)
invA = lambda x: spsla.spsolve(A, x).reshape(x.shape)

# %% Block Krylov Space
def test_block_KrylovSpace():
    # Non-symmetric case (Arnoldi)
    m = 3
    r = X0.shape[1]

    # Reference computation
    Q_ref, R = la.qr(space(A, invA, X0, m), mode='economic')

    # Check the reference
    assert np.allclose(la.norm(Q_ref, axis=0), np.ones(2*m*r)), "The columns of Q are not normalized -> error in the QR decomposition"
    assert la.norm(Q_ref.T.dot(Q_ref) - np.eye(2*m*r)) < 1e-10, "The columns of Q are not orthogonal -> error in the QR decomposition"
    print('Reference OK.')

    # Block Krylov Space
    KS = ExtendedKrylovSpace(A, X0)
    for i in range(m-1):
        KS.augment_basis()

    # Check the basis
    assert (KS.Q.shape == (20, 2*m*r)), "Wrong shape of Q -> error in block Krylov Space"
    assert (KS.Am.shape == (2*m*r, 2*m*r)), "Wrong shape of Am -> error in block Krylov Space"
    assert np.allclose(la.norm(KS.Q, axis=0), np.ones(2*m*r)), "The columns of Q are not normalized -> error in block Krylov Space"
    assert la.norm(KS.Q.T.dot(KS.Q) - np.eye(2*m*r)) < 1e-10, "The columns of Q are not orthogonal -> error in block Krylov Space"
    print('Block Krylov Space Basis OK.')

    # Compare to the reference
    assert la.norm(KS.Q.dot(KS.Q.T) - Q_ref.dot(Q_ref.T)) < 1e-10, "Wrong projection -> error in block Krylov Space"
    print('Block Krylov Space Projection OK.')
    # assert np.allclose(KS.Am, Q_ref.T.dot(A.dot(Q_ref))), "Wrong projected A -> error in block Krylov Space"

    # Symmetric case (Lanczos)
    m = 3
    r = X0.shape[1]
    S = A.T.dot(A)
    invS = lambda x: spsla.spsolve(S, x).reshape(x.shape)

    # Reference computation
    Q_ref, _ = la.qr(space(S, invS, X0, m), mode='economic')

    # Check the reference
    assert np.allclose(la.norm(Q_ref, axis=0), np.ones(2*m*r)), "The columns of Q are not normalized -> error in the QR decomposition"
    assert la.norm(Q_ref.T.dot(Q_ref) - np.eye(2*m*r)) < 1e-10, "The columns of Q are not orthogonal -> error in the QR decomposition"
    print('Reference OK.')

    # Block Krylov Space
    KS = ExtendedKrylovSpace(S, X0, invA=invS, is_symmetric=True)
    for i in range(m-1):
        KS.augment_basis()

    # Check the basis
    assert (KS.Q.shape == (20, 2*m*r)), "Wrong shape of Q -> error in block Krylov Space"
    assert (KS.Am.shape == (2*m*r, 2*m*r)), "Wrong shape of Am -> error in block Krylov Space"
    assert np.allclose(la.norm(KS.Q, axis=0), np.ones(2*m*r)), "The columns of Q are not normalized -> error in block Krylov Space"
    assert la.norm(KS.Q.T.dot(KS.Q) - np.eye(2*m*r)) < 1e-10, "The columns of Q are not orthogonal -> error in block Krylov Space"
    print('Block Krylov Space Basis OK.')

    # Compare to the reference
    print(la.norm(KS.Q.dot(KS.Q.T) - Q_ref.dot(Q_ref.T)))
    assert np.allclose(KS.Q.dot(KS.Q.T), Q_ref.dot(Q_ref.T), atol=1e-6), "Wrong projection -> error in block Krylov Space"
    print('Block Krylov Space Projection OK.')


# %% Test fixtures
@pytest.fixture
def valid_extended_matrix():
    """Create a valid sparse matrix"""
    np.random.seed(42)
    return sps.random(10, 10, density=0.3, format='csc')


@pytest.fixture
def valid_extended_vector():
    """Create a valid vector"""
    np.random.seed(42)
    return np.random.rand(10, 1)


# %% Input validation tests for ExtendedKrylovSpace
class TestExtendedKrylovInputValidation:
    """Test input validation for ExtendedKrylovSpace"""
    
    def test_non_sparse_matrix(self, valid_extended_vector):
        """Test that non-sparse matrix raises TypeError"""
        A = np.random.rand(10, 10)  # Dense matrix
        with pytest.raises(TypeError, match="A must be a sparse matrix"):
            ExtendedKrylovSpace(A, valid_extended_vector)
    
    def test_non_array_vector(self, valid_extended_matrix):
        """Test that non-numpy array raises TypeError"""
        x = [[1], [2], [3]]  # List instead of array
        with pytest.raises(TypeError, match="X must be a numpy array"):
            ExtendedKrylovSpace(valid_extended_matrix, x)


# %% New property tests for better coverage
def test_property_H1_H2():
    """Test H1 and H2 properties for Hessenberg matrices."""
    np.random.seed(300)
    A = sps.random(15, 15, density=0.4, format='csc')
    x = np.random.rand(15, 1)
    invA = lambda x: spsla.spsolve(A, x).reshape(x.shape)
    
    EKS = ExtendedKrylovSpace(A, x, invA=invA, is_symmetric=False)
    EKS.augment_basis()
    
    # Access H1 and H2 properties
    H1 = EKS.H1
    H2 = EKS.H2
    
    # For non-symmetric case, should return Hessenberg matrices
    assert H1 is not None, "H1 should not be None for non-symmetric case"
    assert H2 is not None, "H2 should not be None for non-symmetric case"
    
    # Check dimensions
    assert H1.shape[0] == H1.shape[1] + 1 or H1.shape[0] == H1.shape[1], "H1 wrong shape"
    assert H2.shape[0] == H2.shape[1] + 1 or H2.shape[0] == H2.shape[1], "H2 wrong shape"


def test_property_Q1_Q2():
    """Test Q1 and Q2 properties return correct bases."""
    np.random.seed(301)
    A = sps.random(12, 12, density=0.5, format='csc')
    x = np.random.rand(12, 1)
    invA = lambda x: spsla.spsolve(A, x).reshape(x.shape)
    
    EKS = ExtendedKrylovSpace(A, x, invA=invA)
    EKS.augment_basis()
    EKS.augment_basis()
    
    Q1 = EKS.Q1
    Q2 = EKS.Q2
    
    # Both should be orthonormal
    assert np.allclose(Q1.T @ Q1, np.eye(Q1.shape[1]), atol=1e-10), "Q1 not orthonormal"
    assert np.allclose(Q2.T @ Q2, np.eye(Q2.shape[1]), atol=1e-10), "Q2 not orthonormal"
    
    # Q1 from krylov space, Q2 from inverted krylov space
    assert Q1.shape[1] == EKS.krylov_space.size, "Q1 size mismatch"
    assert Q2.shape[1] == EKS.inverted_krylov_space.size, "Q2 size mismatch"


def test_Q_caching():
    """Test that Q property caching works correctly."""
    np.random.seed(302)
    A = sps.random(10, 10, density=0.5, format='csc')
    x = np.random.rand(10, 1)
    invA = lambda x: spsla.spsolve(A, x).reshape(x.shape)
    
    EKS = ExtendedKrylovSpace(A, x, invA=invA)
    
    # First call should compute and cache
    Q_first = EKS.Q
    initial_size = EKS.size
    
    # Second call should return cached value (same object)
    Q_second = EKS.Q
    assert Q_first is Q_second, "Q should be cached"
    
    # After augmentation, cache should be invalidated
    EKS.augment_basis()
    Q_third = EKS.Q
    assert Q_first is not Q_third, "Cache should be invalidated after augmentation"
    assert Q_third.shape[1] > Q_first.shape[1], "Q should have more columns"
    
    # Verify cache tracking
    assert EKS._cache_size == EKS.size, "Cache size tracking incorrect"


def test_basis_property_alias():
    """Test that basis property is an alias for Q."""
    np.random.seed(303)
    A = sps.random(10, 10, density=0.5, format='csc')
    x = np.random.rand(10, 1)
    invA = lambda x: spsla.spsolve(A, x).reshape(x.shape)
    
    EKS = ExtendedKrylovSpace(A, x, invA=invA)
    EKS.augment_basis()
    
    # basis should be same as Q
    assert np.allclose(EKS.basis, EKS.Q), "basis should equal Q"
    assert EKS.basis.shape == EKS.Q.shape, "basis and Q should have same shape"


def test_size_property():
    """Test size property tracks total space dimension."""
    np.random.seed(304)
    A = sps.random(15, 15, density=0.4, format='csc')
    x = np.random.rand(15, 2)  # Block case
    invA = lambda x: spsla.spsolve(A, x).reshape(x.shape)
    
    EKS = ExtendedKrylovSpace(A, x, invA=invA)
    
    initial_size = EKS.size
    assert initial_size == 4, f"Initial size should be 4 (2+2), got {initial_size}"
    
    # Augment and check size increases
    EKS.augment_basis()
    new_size = EKS.size
    assert new_size == 8, f"Size after augment should be 8 (4+4), got {new_size}"
    
    # Size should equal sum of component sizes
    assert EKS.size == EKS.krylov_space.size + EKS.inverted_krylov_space.size


def test_extended_krylov_with_custom_invA():
    """Test ExtendedKrylovSpace with custom inverse function."""
    np.random.seed(305)
    A = sps.random(12, 12, density=0.5, format='csc')
    A = A + 5 * sps.eye(12)  # Make it well-conditioned
    x = np.random.rand(12, 1)
    
    # Custom inverse using iterative solver
    def custom_invA(v):
        result, info = spsla.gmres(A, v.ravel(), rtol=1e-10)
        return result.reshape(v.shape)
    
    EKS = ExtendedKrylovSpace(A, x, invA=custom_invA)
    EKS.augment_basis()
    
    # Should still produce orthonormal basis
    Q = EKS.Q
    assert np.allclose(Q.T @ Q, np.eye(Q.shape[1]), atol=1e-8), "Q not orthonormal with custom invA"


def test_extended_krylov_symmetric_flag():
    """Test symmetric matrix handling in ExtendedKrylovSpace."""
    np.random.seed(306)
    # Create symmetric matrix
    A_dense = np.random.randn(10, 10)
    A_dense = A_dense + A_dense.T
    A = sps.csc_matrix(A_dense)
    x = np.random.rand(10, 1)
    invA = lambda x: spsla.spsolve(A, x).reshape(x.shape)
    
    # Test with is_symmetric=True
    EKS = ExtendedKrylovSpace(A, x, invA=invA, is_symmetric=True)
    EKS.augment_basis()
    
    # Both component spaces should use Lanczos
    assert EKS.krylov_space.is_symmetric, "Krylov space should detect symmetry"
    assert EKS.inverted_krylov_space.is_symmetric, "Inverted Krylov space should detect symmetry"
    
    Q = EKS.Q
    assert np.allclose(Q.T @ Q, np.eye(Q.shape[1]), atol=1e-10), "Symmetric case: Q not orthonormal"


def test_extended_krylov_multiple_augmentations():
    """Test multiple augmentation steps."""
    np.random.seed(307)
    A = sps.random(20, 20, density=0.3, format='csc')
    x = np.random.rand(20, 1)
    invA = lambda x: spsla.spsolve(A, x).reshape(x.shape)
    
    EKS = ExtendedKrylovSpace(A, x, invA=invA)
    
    sizes = [EKS.size]
    for i in range(5):
        EKS.augment_basis()
        sizes.append(EKS.size)
    
    # Size should increase with each augmentation
    for i in range(len(sizes) - 1):
        assert sizes[i+1] > sizes[i], f"Size should increase: {sizes[i]} -> {sizes[i+1]}"
    
    # Final Q should be orthonormal
    Q = EKS.Q
    assert np.allclose(Q.T @ Q, np.eye(Q.shape[1]), atol=1e-9), "Q not orthonormal after multiple augmentations"


def test_extended_krylov_block_case_properties():
    """Test properties with block (matrix) input."""
    np.random.seed(308)
    A = sps.random(15, 15, density=0.4, format='csc')
    X = np.random.rand(15, 3)  # Block of 3 vectors
    invA = lambda x: spsla.spsolve(A, x).reshape(x.shape)
    
    EKS = ExtendedKrylovSpace(A, X, invA=invA)
    EKS.augment_basis()
    
    # Properties should work with block case
    Q1 = EKS.Q1
    Q2 = EKS.Q2
    Q = EKS.Q
    
    assert Q1.shape[1] == 6, f"Q1 should have 6 columns (2 blocks × 3), got {Q1.shape[1]}"
    assert Q2.shape[1] == 6, f"Q2 should have 6 columns (2 blocks × 3), got {Q2.shape[1]}"
    
    # Combined Q should be orthonormal
    assert np.allclose(Q.T @ Q, np.eye(Q.shape[1]), atol=1e-10), "Block case: Q not orthonormal"


def test_extended_krylov_reduced_A():
    """Test reduced_A property computation."""
    np.random.seed(309)
    A = sps.random(12, 12, density=0.5, format='csc')
    x = np.random.rand(12, 1)
    invA = lambda x: spsla.spsolve(A, x).reshape(x.shape)
    
    EKS = ExtendedKrylovSpace(A, x, invA=invA)
    EKS.augment_basis()
    EKS.augment_basis()
    
    # Get reduced A
    Am = EKS.reduced_A
    Q = EKS.Q
    
    # Should equal Q^T @ A @ Q
    expected = Q.T @ A.toarray() @ Q
    
    assert np.allclose(Am, expected, atol=1e-10), "reduced_A computation incorrect"
    assert Am.shape == (Q.shape[1], Q.shape[1]), "reduced_A wrong shape"
    
    def test_non_square_matrix(self, valid_extended_vector):
        """Test that non-square matrix raises ValueError"""
        A = sps.random(10, 15, density=0.3, format='csc')
        with pytest.raises(ValueError, match="A must be a square matrix"):
            ExtendedKrylovSpace(A, valid_extended_vector)
    
    def test_dimension_mismatch(self, valid_extended_matrix):
        """Test that dimension mismatch raises ValueError"""
        x = np.random.rand(5, 1)  # Wrong dimension
        with pytest.raises(ValueError, match="A and X must have the same number of rows"):
            ExtendedKrylovSpace(valid_extended_matrix, x)
    
    def test_nan_in_vector(self, valid_extended_matrix):
        """Test that NaN in vector raises ValueError"""
        x = np.random.rand(10, 1)
        x[0] = np.nan
        with pytest.raises(ValueError, match="X contains NaN or Inf values"):
            ExtendedKrylovSpace(valid_extended_matrix, x)
    
    def test_inf_in_vector(self, valid_extended_matrix):
        """Test that Inf in vector raises ValueError"""
        x = np.random.rand(10, 1)
        x[0] = np.inf
        with pytest.raises(ValueError, match="X contains NaN or Inf values"):
            ExtendedKrylovSpace(valid_extended_matrix, x)
    
    def test_nan_in_matrix(self, valid_extended_vector):
        """Test that NaN in matrix raises ValueError"""
        A = sps.random(10, 10, density=0.3, format='csc')
        A.data[0] = np.nan
        with pytest.raises(ValueError, match="A contains NaN or Inf values"):
            ExtendedKrylovSpace(A, valid_extended_vector)
    
    def test_inf_in_matrix(self, valid_extended_vector):
        """Test that Inf in matrix raises ValueError"""
        A = sps.random(10, 10, density=0.3, format='csc')
        A.data[0] = np.inf
        with pytest.raises(ValueError, match="A contains NaN or Inf values"):
            ExtendedKrylovSpace(A, valid_extended_vector)


# %% Property tests for ExtendedKrylovSpace
class TestExtendedKrylovProperties:
    """Test properties for ExtendedKrylovSpace"""
    
    def test_extended_krylov_size(self, valid_extended_matrix, valid_extended_vector):
        """Test ExtendedKrylovSpace size is sum of components"""
        EK = ExtendedKrylovSpace(valid_extended_matrix, valid_extended_vector)
        expected_size = EK.krylov_space.size + EK.inverted_krylov_space.size
        assert EK.size == expected_size, "Size should be sum of K and IK sizes"
        
        EK.augment_basis()
        expected_size = EK.krylov_space.size + EK.inverted_krylov_space.size
        assert EK.size == expected_size, "Size should update correctly"
    
    def test_extended_krylov_caching(self, valid_extended_matrix, valid_extended_vector):
        """Test ExtendedKrylov Q property caching works"""
        EK = ExtendedKrylovSpace(valid_extended_matrix, valid_extended_vector)
        
        # Access Q multiple times
        Q1 = EK.Q
        Q2 = EK.Q
        assert Q1 is Q2, "Q should be cached and return same object"
        
        # Augment and check cache is invalidated
        EK.augment_basis()
        Q3 = EK.Q
        assert Q3 is not Q1, "Cache should be invalidated after augmentation"


# %% Advanced functionality tests for ExtendedKrylovSpace
class TestExtendedKrylovAdvancedFunctionality:
    """Test convergence and consistency"""
    
    def test_extended_krylov_combines_spaces(self, valid_extended_matrix, valid_extended_vector):
        """Test ExtendedKrylov properly combines K and IK"""
        EK = ExtendedKrylovSpace(valid_extended_matrix, valid_extended_vector)
        EK.augment_basis()
        
        # The combined basis should span both K and IK subspaces
        Q1, Q2 = EK.Q1, EK.Q2
        Q_combined = EK.Q
        
        # Q should approximately span the same space as [Q1, Q2]
        Q_concat = np.hstack([Q1, Q2])
        Q_concat_orth, _ = la.qr(Q_concat, mode='economic')
        
        P1 = Q_combined @ Q_combined.T
        P2 = Q_concat_orth @ Q_concat_orth.T
        
        assert np.allclose(P1, P2, atol=1e-8), \
            "ExtendedKrylov should span union of K and IK spaces"


# %%
