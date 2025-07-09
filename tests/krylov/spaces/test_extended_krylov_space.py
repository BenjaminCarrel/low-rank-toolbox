"""
Author: Benjamin Carrel, University of Geneva, 2022

Test file for the ExtendedKrylovSpace class and methods
"""

# %% Imports
import numpy as np
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



# %%
