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
    space = [spsla.spsolve((A - s * sps.eye(n, format='csc'))**i, x) for i in range(m)]
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
    space = [spsla.spsolve((A - s * sps.eye(n, format='csc'))**i, X0) for i in range(m)]
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
    assert la.norm(Q.dot(Q.T) - Q_ref.dot(Q_ref.T)) < 1e-6, "Wrong projection -> error in block Arnoldi"
    print('Block shift and invert Arnoldi Projection OK.')



# %% Block rational variant
def test_block_rational_Arnoldi_shift_only():
    poles = [0, 1, 2] # arbitrary poles
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

# %%
