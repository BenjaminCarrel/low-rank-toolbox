"""
Author: Benjamin Carrel, University of Geneva, 2022

Test file for the InvertedKrylovSpace class and methods
"""

# %% Imports
import numpy as np
import scipy.linalg as la
import scipy.sparse as sps
import scipy.sparse.linalg as spsla
from lowrank.krylov.spaces.inverted_krylov_space import InvertedKrylovSpace


# %% Vector case
np.random.seed(1234)
A = sps.random(20, 20, density=0.5, format='csc')
x = np.random.rand(20, 1)
invA = lambda x: spsla.spsolve(A, x).reshape(x.shape)
def space(inv, x, m):
    def repeat(x, n):
        for i in range(n+1):
            x = inv(x)
        return x
    return np.column_stack([repeat(x, i) for i in range(m)])



#%% Inverted Krylov space
def test_vector_InvertedKrylovSpace():
    # Non-symmetric matrix
    KS = InvertedKrylovSpace(A, x, invA)

    # Test the attributes
    assert KS.A is A, "Wrong A"
    assert KS.n == 20, "Wrong n"
    assert KS.r == 1, "Wrong r"

    # Test for m = 2
    m = 2
    Q_ref, _ = la.qr(space(invA, x, m), mode='economic')
    KS.augment_basis()
    assert np.allclose(la.norm(KS.Q, axis=0), np.ones(2)), "The columns of Q are not normalized -> error in augment basis"
    assert la.norm(KS.Q.T.dot(KS.Q) - np.eye(2)) < 1e-10, "The columns of Q are not orthogonal -> error in augment basis"
    print('Inverted Krylov Space Basis OK.')

    assert la.norm(KS.Q.dot(KS.Q.T) - Q_ref.dot(Q_ref.T)) < 1e-10, "Wrong projection -> error in augment basis"
    # assert np.allclose(KS.Am, Q_ref.T.dot(A.dot(Q_ref))), "Wrong projected A -> error in augment basis"
    print('Inverted Krylov Space Projection (m=2) OK.')

    # Test for m = 3
    m = 3
    Q_ref, _ = la.qr(space(invA, x, m), mode='economic')
    KS.augment_basis()
    assert np.allclose(la.norm(KS.Q, axis=0), np.ones(3)), "The columns of Q are not normalized -> error in augment basis"
    assert la.norm(KS.Q.T.dot(KS.Q) - np.eye(3)) < 1e-10, "The columns of Q are not orthogonal -> error in augment basis"
    assert la.norm(KS.Q.dot(KS.Q.T) - Q_ref.dot(Q_ref.T)) < 1e-10, "Wrong projection -> error in augment basis"
    # assert np.allclose(KS.Am, Q_ref.T.dot(A.dot(Q_ref))), "Wrong projected A -> error in augment basis"
    print('Inverted Krylov Space Projection (m=3) OK.')

    # Symmetric matrix
    S = A.T.dot(A)
    invS = lambda x: spsla.spsolve(S, x).reshape(x.shape)
    KS = InvertedKrylovSpace(S, x, invS)

    # Test the attributes
    assert KS.A is S, "Wrong A"
    assert KS.n == 20, "Wrong n"
    assert KS.r == 1, "Wrong r"

    # Test for m = 2
    m = 2
    Q_ref, _ = la.qr(space(invS, x, m), mode='economic')
    KS.augment_basis()
    assert np.allclose(la.norm(KS.Q, axis=0), np.ones(2)), "The columns of Q are not normalized -> error in augment basis"
    assert la.norm(KS.Q.T.dot(KS.Q) - np.eye(2)) < 1e-10, "The columns of Q are not orthogonal -> error in augment basis"
    print('(Symmetric) Inverted Krylov Space Basis OK.')
    assert la.norm(KS.Q.dot(KS.Q.T) - Q_ref.dot(Q_ref.T)) < 1e-10, "Wrong projection -> error in augment basis"
    # assert np.allclose(KS.Am, Q_ref.T.dot(S.dot(Q_ref))), "Wrong projected A -> error in augment basis"
    print('(Symmetric) Inverted Krylov Space Projection (m=2) OK.')

    # Test for m = 3
    m = 3
    Q_ref, _ = la.qr(space(invS, x, m), mode='economic')
    KS.augment_basis()
    assert np.allclose(la.norm(KS.Q, axis=0), np.ones(3)), "The columns of Q are not normalized -> error in augment basis"
    assert la.norm(KS.Q.T.dot(KS.Q) - np.eye(3)) < 1e-10, "The columns of Q are not orthogonal -> error in augment basis"
    print('(Symmetric) Inverted Krylov Space Basis OK.')
    print('Projection error:' + str(la.norm(KS.Q.dot(KS.Q.T) - Q_ref.dot(Q_ref.T))))
    assert la.norm(KS.Q.dot(KS.Q.T) - Q_ref.dot(Q_ref.T)) < 1e-10, "Wrong projection -> error in augment basis"
    # assert np.allclose(KS.Am, Q_ref.T.dot(S.dot(Q_ref))), "Wrong projected A -> error in augment basis"
    print('(Symmetric) Inverted Krylov Space Projection (m=3) OK.')



#%% Matrix case
np.random.seed(1234)
A = 10*sps.random(20, 20, density=0.5, format='csc')
invA = spsla.splu(A).solve
X0 = np.random.rand(20, 3)


# %% Block Krylov Space
def test_block_InvertedKrylovSpace():
    # Non-symmetric case (Arnoldi)
    m = 4
    r = X0.shape[1]

    # Reference computation
    Q_ref, R = la.qr(space(invA, X0, m), mode='economic')

    # Check the reference
    assert np.allclose(la.norm(Q_ref, axis=0), np.ones(m*r)), "The columns of Q are not normalized -> error in the QR decomposition"
    assert la.norm(Q_ref.T.dot(Q_ref) - np.eye(m*r)) < 1e-10, "The columns of Q are not orthogonal -> error in the QR decomposition"
    print('Reference OK.')

    # Block Krylov Space
    KS = InvertedKrylovSpace(A, X0, invA)
    for i in range(m-1):
        KS.augment_basis()

    # Check the basis
    assert (KS.Q.shape == (20, m*r)), "Wrong shape of Q -> error in block Krylov Space"
    assert (KS.Am.shape == (m*r, m*r)), "Wrong shape of Am -> error in block Krylov Space"
    assert np.allclose(la.norm(KS.Q, axis=0), np.ones(m*r)), "The columns of Q are not normalized -> error in block Krylov Space"
    assert la.norm(KS.Q.T.dot(KS.Q) - np.eye(m*r)) < 1e-10, "The columns of Q are not orthogonal -> error in block Krylov Space"
    print('Block Inverted Krylov Space Basis OK.')

    # Compare to the reference
    assert la.norm(KS.Q.dot(KS.Q.T) - Q_ref.dot(Q_ref.T)) < 1e-10, "Wrong projection -> error in block Krylov Space"
    print('Block Inverted Krylov Space Projection OK.')
    # assert np.allclose(KS.Am, Q_ref.T.dot(A.dot(Q_ref))), "Wrong projected A -> error in block Krylov Space"

    # Symmetric case (Lanczos)
    m = 3
    r = X0.shape[1]
    S = A.T.dot(A)
    invS = lambda x: spsla.spsolve(S, x).reshape(x.shape)

    # Reference computation
    Q_ref, _ = la.qr(space(invS, X0, m), mode='economic')

    # Check the reference
    assert np.allclose(la.norm(Q_ref, axis=0), np.ones(m*r)), "The columns of Q are not normalized -> error in the QR decomposition"
    assert la.norm(Q_ref.T.dot(Q_ref) - np.eye(m*r)) < 1e-10, "The columns of Q are not orthogonal -> error in the QR decomposition"
    print('Reference OK.')

    # Block Krylov Space
    KS = InvertedKrylovSpace(S, X0, invS)
    for i in range(m-1):
        KS.augment_basis()

    # Check the basis
    assert (KS.Q.shape == (20, m*r)), "Wrong shape of Q -> error in block Krylov Space"
    assert (KS.Am.shape == (m*r, m*r)), "Wrong shape of Am -> error in block Krylov Space"
    assert np.allclose(la.norm(KS.Q, axis=0), np.ones(m*r)), "The columns of Q are not normalized -> error in block Krylov Space"
    assert np.allclose(KS.Q.T.dot(KS.Q), np.eye(m*r)), "The columns of Q are not orthogonal -> error in block Krylov Space"
    print('Symmetric Block Inverted Krylov Space Basis OK.')

    # Compare to the reference
    print('Projection error:' + str(la.norm(KS.Q.dot(KS.Q.T) - Q_ref.dot(Q_ref.T))))
    assert np.allclose(KS.Q.dot(KS.Q.T), Q_ref.dot(Q_ref.T), atol=1e-6), "Wrong projection -> error in block Krylov Space"
    print('Symmetric Block Inverted Krylov Space Projection OK.')




# %%
