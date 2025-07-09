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
    Q, T, Q2, T2 = block_Lanczos(A, X0, m)

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

