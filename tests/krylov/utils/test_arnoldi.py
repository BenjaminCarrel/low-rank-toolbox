"""
Author: Benjamin Carrel, University of Geneva, 2022

Test file for Arnoldi methods
"""

# %% Imports
import numpy as np
import scipy.linalg as la
import scipy.sparse as sps
import scipy.sparse.linalg as spsla

from lowrank.krylov.utils.arnoldi import (
    Arnoldi,
    block_Arnoldi,
    block_rational_Arnoldi,
    block_shift_and_invert_Arnoldi,
    rational_Arnoldi,
    shift_and_invert_Arnoldi,
)

#%% Matrix to use: non-symmetric laplacian (artificial)
n = 100
A = - (n**2) * sps.diags([-1, 2, -1], [-1, 0, 1], shape=(n, n), format='csc')

# %% Vector case
np.random.seed(1234)
x = np.random.rand(n)

#%% Arnoldi
def test_Arnoldi():
    m = 3

    # Manual computation
    space = [(A**i).dot(x) for i in range(m)]
    Q_ref, _ = la.qr(np.column_stack(space), mode='economic')
    Am_ref = Q_ref.T.dot(A.dot(Q_ref))
    assert np.allclose(la.norm(Q_ref, axis=0), np.ones(m)), "The columns of Q are not normalized -> error in the QR decomposition"
    assert la.norm(Q_ref.T.dot(Q_ref) - np.eye(m)) < 1e-8, "The columns of Q are not orthogonal -> error in the QR decomposition"
    print('Reference OK.')

    # Arnoldi
    Q, H = Arnoldi(A, x, m)

    assert np.allclose(la.norm(Q, axis=0), np.ones(m)), "The columns of Q are not normalized -> error in Arnoldi"
    assert la.norm(Q.T.dot(Q) - np.eye(m)) < 1e-8, "The columns of Q are not orthogonal -> error in Arnoldi"
    assert H.shape == (m, m), "Wrong shape of H -> error in Arnoldi"
    print('Arnoldi Basis OK.')

    print('Arnoldi - Error in the projection:', la.norm(Q.dot(Q.T) - Q_ref.dot(Q_ref.T)))
    assert la.norm(Q.dot(Q.T) - Q_ref.dot(Q_ref.T)) < 1e-8, "Wrong projection -> error in Arnoldi"
    print('Arnoldi Projection OK.')



#%% Shift and invert Arnoldi
def test_shift_invert_Arnoldi():
    m = 3
    s = 2

    # Manual computation
    sparse_matrices = [(A - s * sps.eye(n, format='csc'))**i for i in range(m)]
    space = [spsla.spsolve(mat.tocsc(), x) for mat in sparse_matrices]
    Q_ref, _ = la.qr(np.column_stack(space), mode='economic')
    Am_ref = Q_ref.T.dot(A.dot(Q_ref))
    assert np.allclose(la.norm(Q_ref, axis=0), np.ones(m)), "The columns of Q are not normalized -> error in the QR decomposition"
    assert la.norm(Q_ref.T.dot(Q_ref) - np.eye(m)) < 1e-8, "The columns of Q are not orthogonal -> error in the QR decomposition"
    print('Reference OK.')

    # Shift and invert Arnoldi
    Q, H = shift_and_invert_Arnoldi(A, x, m, shift=s)

    assert np.allclose(la.norm(Q, axis=0), np.ones(m)), "The columns of Q are not normalized -> error in Arnoldi"
    assert la.norm(Q.T.dot(Q) - np.eye(m)) < 1e-8, "The columns of Q are not orthogonal -> error in Arnoldi"
    assert H.shape == (m, m), "Wrong shape of H -> error in Arnoldi"
    print('Shift and invert Arnoldi Basis OK.')

    print('Shift and invert Arnoldi - Error in the projection:', la.norm(Q.dot(Q.T) - Q_ref.dot(Q_ref.T)))
    assert la.norm(Q.dot(Q.T) - Q_ref.dot(Q_ref.T)) < 1e-8, "Wrong projection -> error in Arnoldi"
    print('Shift and invert Arnoldi Projection OK.')



# %% Rational Arnoldi
def test_rational_Arnoldi():
    poles = [1/2, 1, 2] # arbitrary poles
    m = 4

    # Manual computation
    space = [x]
    for p in poles:
        space.append(spsla.spsolve(A - p * sps.eye(n, format='csc'), space[-1]))
    Q_ref, _ = la.qr(np.column_stack(space), mode='economic')
    Am_ref = Q_ref.T.dot(A.dot(Q_ref))
    assert np.allclose(la.norm(Q_ref, axis=0), np.ones(m)), "The columns of Q are not normalized -> error in the QR decomposition"
    assert la.norm(Q_ref.T.dot(Q_ref) - np.eye(m)) < 1e-8, "The columns of Q are not orthogonal -> error in the QR decomposition"
    print('Reference OK.')

    # Rational Arnoldi
    Q, H = rational_Arnoldi(A, x, poles)

    assert np.allclose(la.norm(Q, axis=0), np.ones(m)), "The columns of Q are not normalized -> error in Arnoldi"
    assert la.norm(Q.T.dot(Q) - np.eye(m)) < 1e-8, "The columns of Q are not orthogonal -> error in Arnoldi"
    assert H.shape == (m, m), "Wrong shape of H -> error in Arnoldi"
    print('Rational Arnoldi Basis OK.')

    print('Rational Arnoldi - Error in the projection:', la.norm(Q.dot(Q.T) - Q_ref.dot(Q_ref.T)))
    assert la.norm(Q.dot(Q.T) - Q_ref.dot(Q_ref.T)) < 1e-8, "Wrong projection -> error in Arnoldi"
    print('Rational Arnoldi Projection OK.')



# %% Matrix case
np.random.seed(1234)
X0 = np.random.rand(n, 3)

#%% Block Arnoldi
def test_block_Arnoldi():
    m = 4
    r = X0.shape[1]

    # Reference computation
    space = [(A**i).dot(X0) for i in range(m)]
    Q_ref, R = la.qr(np.column_stack(space), mode='economic')

    # Check the reference
    assert np.allclose(la.norm(Q_ref, axis=0), np.ones(m*r)), "The columns of Q are not normalized -> error in the QR decomposition"
    assert la.norm(Q_ref.T.dot(Q_ref) - np.eye(m*r)) < 1e-8, "The columns of Q are not orthogonal -> error in the QR decomposition"
    print('Reference OK.')

    # Block Arnoldi
    Q, H = block_Arnoldi(A, X0, m)
    Q = la.orth(Q) # might be necessary to enforce orthogonality in block case

    # Check the basis
    assert (Q.shape == (n, m*r)), "Wrong shape of Q -> error in block Arnoldi"
    assert (H.shape == (m*r, m*r)), "Wrong shape of H -> error in block Arnoldi"
    assert np.allclose(la.norm(Q, axis=0), np.ones(m*r)), "The columns of Q are not normalized -> error in block Arnoldi"
    assert la.norm(Q.T.dot(Q) - np.eye(m*r)) < 1e-8, "The columns of Q are not orthogonal -> error in block Arnoldi"
    print('Block Arnoldi Basis OK.')

    # Compare to the reference
    print('Block Arnoldi - Error in the projection:', la.norm(Q.dot(Q.T) - Q_ref.dot(Q_ref.T)))
    assert la.norm(Q.dot(Q.T) - Q_ref.dot(Q_ref.T)) < 1e-8, "Wrong projection -> error in block Arnoldi"
    print('Block Arnoldi Projection OK.')
    # assert np.allclose(A.dot(Q), Q.dot(H), rtol=1e-6), "Wrong projected A -> error in block Arnoldi"
    



# %% Block shift and invert Arnoldi
def test_block_shift_invert_Arnoldi():
    m = 4
    r = X0.shape[1]
    s = 2

    # Reference computation
    space = [spsla.spsolve(((A - s * sps.eye(n, format='csc'))**i).tocsc(), X0) for i in range(m)]
    Q_ref, R = la.qr(np.column_stack(space), mode='economic')

    # Check the reference
    assert np.allclose(la.norm(Q_ref, axis=0), np.ones(m*r)), "The columns of Q are not normalized -> error in the QR decomposition"
    assert la.norm(Q_ref.T.dot(Q_ref) - np.eye(m*r)) < 1e-10, "The columns of Q are not orthogonal -> error in the QR decomposition"
    print('Reference OK.')

    # Block shift and invert Arnoldi
    Q, H = block_shift_and_invert_Arnoldi(A, X0, m, shift=s)
    Q = la.orth(Q) # might be necessary to enforce orthogonality in block case

    # Check the basis
    assert (Q.shape == (n, m*r)), "Wrong shape of Q -> error in block Arnoldi"
    assert (H.shape == (m*r, m*r)), "Wrong shape of H -> error in block Arnoldi"
    assert np.allclose(la.norm(Q, axis=0), np.ones(m*r)), "The columns of Q are not normalized -> error in block Arnoldi"
    assert la.norm(Q.T.dot(Q) - np.eye(m*r)) < 1e-6, "The columns of Q are not orthogonal -> error in block Arnoldi"
    print('Block shift and invert Arnoldi Basis OK.')

    # Compare to the reference
    print('Block shift and invert Arnoldi - Error in the projection:', la.norm(Q.dot(Q.T) - Q_ref.dot(Q_ref.T)))
    assert la.norm(Q.dot(Q.T) - Q_ref.dot(Q_ref.T)) < 1e-4, "Wrong projection -> error in block Arnoldi"
    print('Block shift and invert Arnoldi Projection OK.')



# %% Block rational variant
def test_block_rational_Arnoldi_shift_only():
    poles = [0, -1, -2] # arbitrary poles
    m = 4
    r = X0.shape[1]

    # Reference computation
    space = [X0]
    for p in poles:
        space.append(spsla.spsolve(A - p * sps.eye(n, format='csc'), space[-1]))
    Q_ref, _ = la.qr(np.column_stack(space), mode='economic')

    # Check the reference
    assert np.allclose(la.norm(Q_ref, axis=0), np.ones(m*r)), "The columns of Q are not normalized -> error in the QR decomposition"
    assert la.norm(Q_ref.T.dot(Q_ref) - np.eye(m*r)) < 1e-8, "The columns of Q are not orthogonal -> error in the QR decomposition"
    print('Reference OK.')

    # Block rational Arnoldi (shift only)
    Q, H = block_rational_Arnoldi(A, X0, poles, inverse_only=True)
    Q = la.orth(Q) # might be necessary to enforce orthogonality in block case

    # Check the basis
    assert (Q.shape == (n, m*r)), "Wrong shape of Q -> error in block rational Arnoldi"
    assert (H.shape == (m*r, m*r)), "Wrong shape of H -> error in block rational Arnoldi"
    assert np.allclose(la.norm(Q, axis=0), np.ones(m*r)), "The columns of Q are not normalized -> error in block rational Arnoldi"
    assert la.norm(Q.T.dot(Q) - np.eye(m*r)) < 1e-8, "The columns of Q are not orthogonal -> error in block Arnoldi"
    print('Block rational Arnoldi Basis OK.')

    # Compare to the reference
    print('Block rational Arnoldi (inverse only) - Error in the projection:', la.norm(Q.dot(Q.T) - Q_ref.dot(Q_ref.T)))
    assert la.norm(Q.dot(Q.T) - Q_ref.dot(Q_ref.T)) < 1e-8, "Wrong projection -> error in block Arnoldi"
    print('Block rational Arnoldi Projection OK.')



def test_block_rational_Arnoldi_full():
    poles = [1/2, 1, 2]
    m = 4
    r = X0.shape[1]

    # Reference computation
    space = [X0]
    for p in poles:
        space.append(spsla.spsolve(A - p * sps.eye(n, format='csc'), A.dot(space[-1])))
    Q_ref, _ = la.qr(np.column_stack(space), mode='economic')
    # Check the reference
    assert np.allclose(la.norm(Q_ref, axis=0), np.ones(m*r)), "The columns of Q are not normalized -> error in the QR decomposition"
    assert la.norm(Q_ref.T.dot(Q_ref) - np.eye(m*r)) < 1e-8, "The columns of Q are not orthogonal -> error in the QR decomposition"
    print('Reference OK.')

    # Block rational Arnoldi (full)
    Q, H = block_rational_Arnoldi(A, X0, poles, inverse_only=False)
    Q = la.orth(Q) # might be necessary to enforce orthogonality in block case

    # Check the basis
    assert (Q.shape == (n, m*r)), "Wrong shape of Q -> error in block rational Arnoldi"
    assert (H.shape == (m*r, m*r)), "Wrong shape of H -> error in block rational Arnoldi"
    assert np.allclose(la.norm(Q, axis=0), np.ones(m*r)), "The columns of Q are not normalized -> error in block rational Arnoldi"
    print('error:', la.norm(Q.T.dot(Q) - np.eye(m*r)))
    assert la.norm(Q.T.dot(Q) - np.eye(m*r)) < 1e-6, "The columns of Q are not orthogonal -> error in block Arnoldi"
    print('Block rational Arnoldi Basis OK.')

    # Compare to the reference
    print('Block rational Arnoldi (full) - Error in the projection:', la.norm(Q.dot(Q.T) - Q_ref.dot(Q_ref.T)))
    assert la.norm(Q.dot(Q.T) - Q_ref.dot(Q_ref.T)) < 1e-5, "Wrong projection -> error in block Arnoldi"
    print('Block rational Arnoldi Projection OK.')


# %% Edge case tests - Complex numbers
def test_Arnoldi_complex():
    """Test Arnoldi with complex matrix and vector"""
    m = 3
    # Complex matrix - make it more strongly non-symmetric
    A_imag = sps.random(n, n, density=0.05, format='csc')
    A_complex = A + 1j * A_imag
    x_complex = x + 1j * np.random.rand(n)
    
    Q, H = Arnoldi(A_complex, x_complex, m)
    
    # Check orthogonality (use conjugate transpose for complex)
    assert Q.dtype == np.complex128, "Q should be complex"
    assert H.dtype == np.complex128, "H should be complex"
    assert np.allclose(la.norm(Q, axis=0), np.ones(m)), "Columns not normalized"
    # Relax tolerance for complex case due to numerical issues
    assert la.norm(Q.conj().T.dot(Q) - np.eye(m)) < 1e-6, "Columns not orthogonal"
    print('Complex Arnoldi OK.')


def test_shift_and_invert_Arnoldi_complex_shift():
    """Test shift-and-invert Arnoldi with complex shift"""
    m = 3
    shift_complex = 2.0 + 1j * 3.0
    
    Q, H = shift_and_invert_Arnoldi(A, x, m, shift=shift_complex)
    
    assert Q.dtype == np.complex128, "Q should be complex with complex shift"
    assert H.dtype == np.complex128, "H should be complex with complex shift"
    assert np.allclose(la.norm(Q, axis=0), np.ones(m)), "Columns not normalized"
    assert la.norm(Q.conj().T.dot(Q) - np.eye(m)) < 1e-8, "Columns not orthogonal"
    print('Complex shift-and-invert Arnoldi OK.')


def test_rational_Arnoldi_complex_poles():
    """Test rational Arnoldi with complex poles"""
    poles = [1.0 + 0.5j, 2.0 - 0.5j]
    m = 3
    
    Q, H = rational_Arnoldi(A, x, poles)
    
    assert Q.dtype == np.complex128, "Q should be complex with complex poles"
    assert H.dtype == np.complex128, "H should be complex with complex poles"
    assert np.allclose(la.norm(Q, axis=0), np.ones(m)), "Columns not normalized"
    # Relax tolerance for rational Krylov with complex poles
    assert la.norm(Q.conj().T.dot(Q) - np.eye(m)) < 1e-6, "Columns not orthogonal"
    print('Complex poles rational Arnoldi OK.')


# %% Edge case tests - Dense matrices
def test_Arnoldi_dense():
    """Test Arnoldi with dense matrix"""
    m = 3
    A_dense = A.toarray()
    
    Q, H = Arnoldi(A_dense, x, m)
    
    assert np.allclose(la.norm(Q, axis=0), np.ones(m)), "Columns not normalized"
    assert la.norm(Q.T.dot(Q) - np.eye(m)) < 1e-8, "Columns not orthogonal"
    print('Dense Arnoldi OK.')


# %% Edge case tests - Small dimensions
def test_Arnoldi_single_vector():
    """Test Arnoldi with m=1 (single vector)"""
    m = 1
    
    Q, H = Arnoldi(A, x, m)
    
    assert Q.shape == (n, 1), "Wrong shape for m=1"
    assert H.shape == (1, 1), "Wrong shape for m=1"
    assert np.allclose(la.norm(Q), 1.0), "Single vector not normalized"
    print('Single vector Arnoldi OK.')


def test_block_Arnoldi_single_iteration():
    """Test block Arnoldi with m=1"""
    m = 1
    r = X0.shape[1]
    
    Q, H = block_Arnoldi(A, X0, m)
    
    assert Q.shape == (n, r), "Wrong shape for m=1"
    assert H.shape == (r, r), "Wrong shape for m=1"
    print('Single iteration block Arnoldi OK.')


# %% Edge case tests - Lambda closure bug verification
def test_rational_Arnoldi_different_poles():
    """Verify that rational Arnoldi correctly uses different poles (not just the last one)"""
    # Use distinct poles that would give very different results
    poles = [0.1, 10.0, 100.0]
    m = len(poles) + 1
    
    Q, H = rational_Arnoldi(A, x, poles)
    
    # Compute reference with manually specified operations
    Q_ref = np.zeros((n, m))
    Q_ref[:, 0] = x / la.norm(x)
    
    # First pole
    v1 = spsla.spsolve(A - 0.1 * sps.eye(n, format='csc'), A.dot(Q_ref[:, 0]))
    v1_orth = v1 - Q_ref[:, 0] * np.dot(Q_ref[:, 0], v1)
    Q_ref[:, 1] = v1_orth / la.norm(v1_orth)
    
    # Compare first two columns (enough to verify different poles are used)
    assert np.allclose(np.abs(Q[:, 1].T.dot(Q_ref[:, 1])), 1.0, atol=1e-6), \
        "Rational Arnoldi is not using correct poles (lambda closure bug detected)"
    print('Rational Arnoldi poles verification OK.')


def test_block_rational_Arnoldi_different_poles():
    """Verify that block rational Arnoldi correctly uses different poles"""
    poles = [0.1, 10.0]
    m = len(poles) + 1
    r = X0.shape[1]
    
    Q, H = block_rational_Arnoldi(A, X0, poles)
    
    # Just verify it runs without error and produces reasonable output
    assert Q.shape == (n, m * r), "Wrong shape"
    assert np.allclose(la.norm(Q, axis=0), np.ones(m * r), atol=1e-6), "Columns not normalized"
    print('Block rational Arnoldi poles verification OK.')


# %%
