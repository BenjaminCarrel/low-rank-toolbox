"""
Author: Benjamin Carrel, University of Geneva, 2022

Test file for the RationalKrylovSpace class and methods
"""

# %% Imports
import numpy as np
import scipy.linalg as la
import scipy.sparse as sps
import scipy.sparse.linalg as spsla
from krylov_toolbox import RationalKrylovSpace


# %% Vector case
np.random.seed(1234)
A = sps.random(20, 20, density=0.5, format='csc')
x = np.random.rand(20,1)
poles = [1, 1, 1] # arbitrary poles

def compute_space(A, x, poles):
    space = [x]
    y = x
    for pi in poles:
        y = spsla.spsolve(A - pi*sps.eye(20), A.dot(y))
        space.append(y)
    return space


#%% Krylov space
def test_vector_RationalKrylovSpace():
    # Non-symmetric matrix
    KS = RationalKrylovSpace(A, x, poles)
    m = len(poles)+1

    # Test the attributes
    assert KS.A is A, "Wrong A"
    assert KS.n == 20, "Wrong n"
    assert KS.r == 1, "Wrong r"

    space = compute_space(A, x, poles)
    Q_ref, _ = la.qr(np.column_stack(space), mode='economic')
    for _ in range(len(poles)):
        KS.augment_basis()
    assert np.allclose(la.norm(KS.Q, axis=0), np.ones(m)), "The columns of Q are not normalized -> error in augment basis"
    assert np.allclose(KS.Q.T.dot(KS.Q), np.eye(m)), "The columns of Q are not orthogonal -> error in augment basis"
    print('Krylov Space Basis OK.')
    print(la.norm(KS.Q.dot(KS.Q.T) - Q_ref.dot(Q_ref.T)))
    assert np.allclose(KS.Q.dot(KS.Q.T), Q_ref.dot(Q_ref.T)), "Wrong projection -> error in augment basis"
    # assert np.allclose(KS.Am, Q_ref.T.dot(A.dot(Q_ref))), "Wrong projected A -> error in augment basis"
    print(KS.H)
    print('Rational Krylov Space Projection OK.')

    # Symmetric matrix
    S = A.T.dot(A)
    KS = RationalKrylovSpace(S, x, poles)

    # Test the attributes
    assert KS.A is S, "Wrong A"
    assert KS.n == 20, "Wrong n"
    assert KS.r == 1, "Wrong r"

    space = compute_space(S, x, poles)
    Q_ref, _ = la.qr(np.column_stack(space), mode='economic')
    for _ in range(len(poles)):
        KS.augment_basis()
    assert np.allclose(la.norm(KS.Q, axis=0), np.ones(m)), "The columns of Q are not normalized -> error in augment basis"
    assert la.norm(KS.Q.T.dot(KS.Q) - np.eye(m)) < 1e-10, "The columns of Q are not orthogonal -> error in augment basis"
    print('(Symmetric) Krylov Space Basis OK.')
    assert la.norm(KS.Q.dot(KS.Q.T) - Q_ref.dot(Q_ref.T)) < 1e-10, "Wrong projection -> error in augment basis"
    # assert np.allclose(KS.Am, Q_ref.T.dot(S.dot(Q_ref))), "Wrong projected A -> error in augment basis"
    print('(Symmetric) Rational Krylov Space Projection OK.')


test_vector_RationalKrylovSpace()


#%% Matrix case
np.random.seed(1234)
A = sps.random(20, 20, density=0.5, format='csc')
X0 = np.random.rand(20, 3)
poles = [1, 1, 1] # arbitrary poles


# %% Block Krylov Space
def test_block_KrylovSpace():
    # Non-symmetric case (Arnoldi)
    m = len(poles)+1
    r = X0.shape[1]

    # Reference computation
    space = compute_space(A, X0, poles)
    Q_ref, R = la.qr(np.column_stack(space), mode='economic')

    # Check the reference
    assert np.allclose(la.norm(Q_ref, axis=0), np.ones(m*r)), "The columns of Q are not normalized -> error in the QR decomposition"
    assert la.norm(Q_ref.T.dot(Q_ref) - np.eye(m*r)) < 1e-10, "The columns of Q are not orthogonal -> error in the QR decomposition"
    print('Reference OK.')

    # Block Krylov Space
    KS = RationalKrylovSpace(A, X0, poles)
    for i in range(m-1):
        KS.augment_basis()

    # Check the basis
    assert (KS.Q.shape == (20, m*r)), "Wrong shape of Q -> error in block Krylov Space"
    assert (KS.Am.shape == (m*r, m*r)), "Wrong shape of Am -> error in block Krylov Space"
    assert np.allclose(la.norm(KS.Q, axis=0), np.ones(m*r)), "The columns of Q are not normalized -> error in block Krylov Space"
    assert np.allclose(KS.Q.T.dot(KS.Q), np.eye(m*r)), "The columns of Q are not orthogonal -> error in block Krylov Space"
    print('Block Rational Krylov Space Basis OK.')

    # Compare to the reference
    assert np.allclose(KS.Q.dot(KS.Q.T), Q_ref.dot(Q_ref.T)), "Wrong projection -> error in block Krylov Space"
    print('Block Rational Krylov Space Projection OK.')
    # assert np.allclose(KS.Am, Q_ref.T.dot(A.dot(Q_ref))), "Wrong projected A -> error in block Krylov Space"

    # Symmetric case (Lanczos)
    r = X0.shape[1]
    S = A.T.dot(A)

    # Reference computation
    space = compute_space(S, X0, poles)
    Q_ref, _ = la.qr(np.column_stack(space), mode='economic')

    # Check the reference
    assert np.allclose(la.norm(Q_ref, axis=0), np.ones(m*r)), "The columns of Q are not normalized -> error in the QR decomposition"
    assert np.allclose(Q_ref.T.dot(Q_ref), np.eye(m*r)), "The columns of Q are not orthogonal -> error in the QR decomposition"
    print('Reference OK.')

    # Block Krylov Space
    KS = RationalKrylovSpace(S, X0, poles)
    for i in range(m-1):
        KS.augment_basis()

    # Check the basis
    assert (KS.Q.shape == (20, m*r)), "Wrong shape of Q -> error in block Krylov Space"
    assert (KS.Am.shape == (m*r, m*r)), "Wrong shape of Am -> error in block Krylov Space"
    assert np.allclose(la.norm(KS.Q, axis=0), np.ones(m*r)), "The columns of Q are not normalized -> error in block Krylov Space"
    assert np.allclose(KS.Q.T.dot(KS.Q), np.eye(m*r)), "The columns of Q are not orthogonal -> error in block Krylov Space"
    print('Block Krylov Space Basis OK.')

    # Compare to the reference
    assert np.allclose(KS.Q.dot(KS.Q.T), Q_ref.dot(Q_ref.T)), "Wrong projection -> error in block Krylov Space"
    print('(Symmetric) Block Krylov Space Projection OK.')


test_block_KrylovSpace()

# %%
