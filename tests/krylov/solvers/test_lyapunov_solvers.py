"""
Author: Benjamin Carrel, University of Geneva, 2022

Tests for the Lyapunov solvers.
"""

# %% Imports
import numpy as np
import scipy.sparse as sps
import scipy.linalg as la
from lowrank.krylov.solvers.lyapunov_solvers import solve_sparse_low_rank_lyapunov, solve_small_lyapunov
from lowrank.matrices.svd import SVD

# %% Setup data
n = 200
r = 10
a = sps.random(n, n, density=0.1)
A = a.dot(a.T)
Ad = A.todense()
C = SVD.generate_random((n, n), np.logspace(-1, -20, r), is_symmetric=True)
Cd = C.todense()

#%% Compute the reference
X_ref = la.solve_lyapunov(Ad, Cd)

# %% Test the "small" solver
def test_lyapunov_small():
    X = solve_small_lyapunov(Ad, Cd)
    assert np.allclose(X, X_ref), "The small solver is not correct"
    print('test_sylvester_small passed')


# %% Test the "large and low rank" solver
def test_lyapunov_large_low_rank():
    X = solve_sparse_low_rank_lyapunov(A, C, tol=1e-12, extended=True)
    print(X)
    Xd = X.todense()
    print(la.norm(Xd - X_ref)/la.norm(X_ref))
    assert np.allclose(Xd, X_ref), "The large and low rank solver is not correct"
    print('test_sylvester_large_low_rank passed')

# %%
