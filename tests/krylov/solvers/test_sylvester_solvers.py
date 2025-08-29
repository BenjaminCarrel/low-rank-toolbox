"""
Author: Benjamin Carrel, University of Geneva, 2022

Tests for the Sylvester solvers.
"""

# %% Imports
import numpy as np
import scipy.linalg as la
import scipy.sparse as sps

from lowrank.krylov.solvers.sylvester_solvers import (
    solve_small_sylvester,
    solve_sparse_low_rank_sylvester,
    solve_sylvester_large_A_small_B,
)
from lowrank.matrices.svd import SVD

# %% Setup data
n = 200
r = 10
a = sps.random(n, n, density=0.1)
A = a.dot(a.T)
Ad = A.todense()
b = sps.random(n, n, density=0.1)
B = b.dot(b.T)
Bd = B.todense()
C = SVD.generate_random((n, n), np.logspace(-1, -20, r))
Cd = C.todense()

#%% Compute the reference
X_ref = la.solve_sylvester(Ad, Bd, Cd)

# %% Test the "small" solver
def test_sylvester_small():
    X = solve_small_sylvester(Ad, Bd, C)
    assert np.allclose(X, X_ref), "The small solver is not correct"
    print('test_sylvester_small passed')


# %% Test the "large and small" solver
def test_sylvester_large_small():
    X = solve_sylvester_large_A_small_B(A, Bd, Cd)
    assert np.allclose(X, X_ref), "The large and small solver is not correct"
    print('test_sylvester_large_small passed')


# %% Test the "large and low rank" solver
def test_sylvester_large_low_rank():
    X = solve_sparse_low_rank_sylvester(A, B, C, tol=1e-10, extended=True)
    # print(X)
    Xd = X.todense()
    # print(la.norm(Xd - X_ref)/la.norm(X_ref))
    assert np.allclose(Xd, X_ref), "The large and low rank solver is not correct"
    print('test_sylvester_large_low_rank passed')

# %%
