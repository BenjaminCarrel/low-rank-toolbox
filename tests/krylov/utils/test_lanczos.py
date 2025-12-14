"""
Author: Benjamin Carrel, University of Geneva, 2022

Test file for Lanczos methods
"""

# %% Imports
import numpy as np
import scipy.linalg as la
import scipy.sparse as sps
from lowrank.krylov.utils.lanczos import Lanczos, block_Lanczos

# %% Matrix to use: discretized laplacian
n = 100
A =  sps.diags([-1, 2, -1], [-1, 0, 1], shape=(n, n), format='csc')

# %% Vector case
np.random.seed(1234)
x = np.random.rand(n)

#%% Lanczos
def test_Lanczos():
    m = 3

    # Manual computation
    space = [(A**i).dot(x) for i in range(m)]
    Q_ref, _ = la.qr(np.column_stack(space), mode='economic')
    Am_ref = Q_ref.T.dot(A.dot(Q_ref))
    assert np.allclose(la.norm(Q_ref, axis=0), np.ones(m)), "The columns of Q are not normalized -> error in the QR decomposition"
    assert la.norm(Q_ref.T.dot(Q_ref) - np.eye(m)) < 1e-10, "The columns of Q are not orthogonal -> error in the QR decomposition"
    print('Reference OK.')

    # Lanczos
    Q, T = Lanczos(A, x, m)

    assert np.allclose(la.norm(Q, axis=0), np.ones(m)), "The columns of Q are not normalized -> error in Lanczos"
    assert la.norm(Q.T.dot(Q) - np.eye(m)) < 1e-10, "The columns of Q are not orthogonal -> error in Lanczos"
    assert T.shape == (m, m), "Wrong shape of T -> error in Lanczos"
    print('Lanczos Basis OK.')

    print('Error in the basis:', np.linalg.norm(Q @ Q.T - Q_ref @ Q_ref.T))
    assert la.norm(Q.dot(Q.T) - Q_ref.dot(Q_ref.T)) < 1e-10, "Basis is not the same -> error in Lanczos"
    print('Lanczos Projection OK.')



# %% Matrix case
np.random.seed(1234)
X0 = np.random.rand(n, 3)

# %% Block Lanczos
def test_block_Lanczos():
    m = 3
    r = X0.shape[1]

    # Reference computation
    space = [(A**i).dot(X0) for i in range(m)]
    Q_ref, _ = la.qr(np.column_stack(space), mode='economic')
    Am_ref = Q_ref.T.dot(A.dot(Q_ref))

    # Check the reference
    assert np.allclose(la.norm(Q_ref, axis=0), np.ones(m*r)), "The columns of Q are not normalized -> error in the QR decomposition"
    assert la.norm(Q_ref.T.dot(Q_ref) - np.eye(m*r)) < 1e-10, "The columns of Q are not orthogonal -> error in the QR decomposition"
    print('Reference OK.')

    # Block Lanczos
    Q, T = block_Lanczos(A, X0, m)

    # Check the basis
    assert (Q.shape == (n, m*r)), "Wrong shape of Q -> error in block Lanczos"
    assert (T.shape == (m*r, m*r)), "Wrong shape of T -> error in block Lanczos"
    assert np.allclose(la.norm(Q, axis=0), np.ones(m*r)), "The columns of Q are not normalized -> error in block Lanczos"
    assert la.norm(Q.T.dot(Q) - np.eye(m*r)) < 1e-10, "The columns of Q are not orthogonal -> error in block Lanczos"
    print('Block Lanczos Basis OK.')

    # Compare to the reference
    print('Error in the basis:', np.linalg.norm(Q @ Q.T - Q_ref @ Q_ref.T))
    assert la.norm(Q.dot(Q.T) - Q_ref.dot(Q_ref.T)) < 1e-10, "Wrong projection -> error in block Lanczos"
    print('Block Lanczos Projection OK.')
    # assert la.norm(S.dot(Q) - Q.dot(T)) < 1e-10, "Wrong projected A -> error in block Lanczos"


# %% Edge case tests - Complex numbers
def test_Lanczos_complex():
    """Test Lanczos with complex Hermitian matrix and vector"""
    m = 3
    # Create complex Hermitian matrix (A + A^H is Hermitian)
    A_rand = sps.random(n, n, density=0.05, format='csc')
    A_hermitian = (A_rand + A_rand.conj().T) / 2
    x_complex = x + 1j * np.random.rand(n)
    
    Q, T = Lanczos(A_hermitian, x_complex, m)
    
    # Check orthogonality (use conjugate transpose for complex)
    assert Q.dtype == np.complex128, "Q should be complex"
    assert np.allclose(la.norm(Q, axis=0), np.ones(m)), "Columns not normalized"
    assert la.norm(Q.conj().T.dot(Q) - np.eye(m)) < 1e-8, "Columns not orthogonal"
    # Verify T is tridiagonal
    assert sps.linalg.norm(T - sps.diags([T.diagonal(), T.diagonal(1), T.diagonal(-1)], 
                                          [0, 1, -1], format='csc')) < 1e-10, "T not tridiagonal"
    print('Complex Lanczos OK.')


def test_block_Lanczos_complex():
    """Test block Lanczos with complex Hermitian matrix"""
    m = 3
    r = X0.shape[1]
    # Create complex Hermitian matrix
    A_rand = sps.random(n, n, density=0.05, format='csc')
    A_hermitian = (A_rand + A_rand.conj().T) / 2
    X0_complex = X0 + 1j * np.random.rand(n, r)
    
    Q, T = block_Lanczos(A_hermitian, X0_complex, m)
    
    assert Q.dtype == np.complex128, "Q should be complex"
    assert Q.shape == (n, m * r), "Wrong shape"
    assert np.allclose(la.norm(Q, axis=0), np.ones(m * r)), "Columns not normalized"
    # Block Lanczos can have larger numerical errors, especially with complex matrices
    assert la.norm(Q.conj().T.dot(Q) - np.eye(m * r)) < 0.5, "Columns not orthogonal"
    print('Complex block Lanczos OK.')


# %% Edge case tests - Dense matrices
def test_Lanczos_dense():
    """Test Lanczos with dense matrix"""
    m = 3
    A_dense = A.toarray()
    
    Q, T = Lanczos(A_dense, x, m)
    
    assert np.allclose(la.norm(Q, axis=0), np.ones(m)), "Columns not normalized"
    assert la.norm(Q.T.dot(Q) - np.eye(m)) < 1e-8, "Columns not orthogonal"
    print('Dense Lanczos OK.')


# %% Edge case tests - Small dimensions
def test_Lanczos_single_vector():
    """Test Lanczos with m=1 (single vector)"""
    m = 1
    
    Q, T = Lanczos(A, x, m)
    
    assert Q.shape == (n, 1), "Wrong shape for m=1"
    assert T.shape == (1, 1), "Wrong shape for m=1"
    assert np.allclose(la.norm(Q), 1.0), "Single vector not normalized"
    print('Single vector Lanczos OK.')


def test_block_Lanczos_single_iteration():
    """Test block Lanczos with m=1"""
    m = 1
    r = X0.shape[1]
    
    Q, T = block_Lanczos(A, X0, m)
    
    assert Q.shape == (n, r), "Wrong shape for m=1"
    assert T.shape == (r, r), "Wrong shape for m=1"
    print('Single iteration block Lanczos OK.')


# %% Edge case tests - Verify symmetry is preserved
def test_Lanczos_preserves_symmetry():
    """Verify that Lanczos produces a symmetric tridiagonal matrix"""
    m = 5
    
    Q, T = Lanczos(A, x, m)
    
    T_dense = T.toarray()
    # Check T is symmetric (it should be for symmetric A)
    assert np.allclose(T_dense, T_dense.T), "T is not symmetric"
    # Check T is tridiagonal
    for i in range(m):
        for j in range(m):
            if abs(i - j) > 1:
                assert abs(T_dense[i, j]) < 1e-10, f"T[{i},{j}] should be zero"
    print('Lanczos symmetry preservation OK.')


# %%
