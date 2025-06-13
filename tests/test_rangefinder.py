#%% Import packages
import numpy as np
from numpy.testing import assert_allclose
from lowrank.utils import adaptive_randomized_rangefinder, randomized_rangefinder

#%% Test adaptive_randomized_rangefinder
def test_adaptive_randomized_rangefinder():
    # Test case 1
    A = np.random.randn(100, 10) @ np.random.randn(10, 50)
    Q = adaptive_randomized_rangefinder(A, tol=1e-8, failure_prob=1e-8)
    # print(Q.shape)
    assert_allclose(Q @ Q.T.conj() @ A, A, atol=1e-8), "Approximation error exceeds tolerance"

    # Test case 2
    A = np.random.randn(200, 15) @ np.random.randn(15, 300)
    Q = adaptive_randomized_rangefinder(A, tol=1e-8, failure_prob=1e-8)
    # print(Q.shape)
    assert_allclose(Q @ Q.T.conj() @ A, A, atol=1e-8), "Approximation error exceeds tolerance"

    # Test case 3
    A = np.random.randn(500, 100) @ np.random.randn(100, 300)
    Q = adaptive_randomized_rangefinder(A, tol=1e-8, failure_prob=1e-8)
    # print(Q.shape)
    assert_allclose(Q @ Q.T.conj() @ A, A, atol=1e-8), "Approximation error exceeds tolerance"

    print('Tests for adaptive_randomized_rangefinder passed')

test_adaptive_randomized_rangefinder()
# %%
def test_randomized_rangefinder():
    # Test case 1
    A = np.random.randn(100, 100)
    r = 10
    p = 5
    q = 0
    Q = randomized_rangefinder(A, r, p, q)
    assert Q.shape == (100, r+p), "Incorrect shape of the sketched matrix"

    # Test case 2
    A = np.random.randn(200, 150)
    r = 20
    p = 10
    q = 1
    Q = randomized_rangefinder(A, r, p, q)
    assert Q.shape == (200, r+p), "Incorrect shape of the sketched matrix"

    # Test case 3
    A = np.random.randn(200, 300)
    r = 30
    p = 15
    q = 2
    Q = randomized_rangefinder(A, r, p, q)
    assert Q.shape == (200, r+p), "Incorrect shape of the sketched matrix"

    print('Tests for randomized_rangefinder passed')

test_randomized_rangefinder()
# %%
