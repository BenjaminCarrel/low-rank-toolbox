"""
File for rangefinder related functions

Author: Benjamin Carrel, University of Geneva
"""

#%% Importations
import numpy as np
from numpy import ndarray
from scipy import linalg as la

#%% The randomized rangefinder
def randomized_rangefinder(A: ndarray, r: int, p: int = 5, q:int = 0, seed: int = 1234, **extra_args) -> ndarray:
    """
    The randomized rangefinder method.

    Parameters
    ----------
    A : ndarray
        The matrix to sketch.
    r : int
        The target rank.
    p : int
        The number of over-sampling.
    seed : int
        The seed for the random number generator.

    Returns
    -------
    Q : ndarray
        Estimation of the range of A.
    """
    # Check the inputs
    if r + p > min(A.shape):
        raise ValueError('Target rank + oversampling exceed matrix shape.')
    if r < 1:
        raise ValueError('The target rank must be at least 1.')
    if p < 0:
        raise ValueError('The oversampling parameter must be non-negative.')
    
    # Check for sketching matrix in extra_args
    if 'Omega' in extra_args:
        Omega = extra_args['Omega']
    else:
        # Gaussian matrix
        np.random.seed(seed)
        Omega = np.random.randn(A.shape[1], r+p)
        # if sketching == 'gaussian+orth' or sketching == 'gaussian+qr':
        #     Omega = la.orth(Omega)
        # if sketching == 'gaussian+pivots':
        #     _,_,P = la.qr(Omega, mode='economic', pivoting=True)
            # Omega = np.eye(A.shape[1])[:,P]

    # Support for complex matrix A
    if np.iscomplexobj(A):
        Omega = Omega.astype(A.dtype)
    
    # The method with power iteration
    Y = A.dot(Omega)
    Q, _, _ = la.qr(Y, mode='economic', pivoting=True)
    for _ in range(q):
        Y = A.T.conj().dot(Q)
        Q, _, _ = la.qr(Y, mode='economic', pivoting=True)
        Y = A.dot(Q)
        Q, _, _ = la.qr(Y, mode='economic', pivoting=True)

    return Q

#%% The adaptive randomized rangefinder
def adaptive_randomized_rangefinder(A: ndarray, tol: float = 1e-6, failure_prob: float = 1e-6, seed: int = 1234) -> ndarray:
    """
    The adaptive randomized rangefinder method.
    The tolerance is the error made by the approximation space ||A - QQ^H A||_F <= tol
    The failure probability is the probability that the error is larger than tol

    Reference: Finding structure with randomness: Probabilistic algorithms for constructing approximate matrix decompositions (HMT)

    NOTE: For efficiency, the method performs computations in blocks of size r (defined by the failure probability). Blocking allows to use BLAS3 operations.

    Parameters
    ----------
    A : ndarray
        The matrix to sketch.
    tol : float
        The tolerance for the approximation.
    failure_prob : float
        The failure probability.
    sketching : str
        The sketching method, see sketch_random_matrix for the available methods.
    seed : int
        The seed for the random number generator.

    Returns
    -------
    Q : ndarray
        The sketched matrix.
    """
    # Compute the sketch size according to the failure probability
    n = min(A.shape)
    r = int(np.ceil(- np.log(failure_prob / n) / np.log(10)))
    tol = tol / (10 * np.sqrt(2/np.pi))
    if r < 1:
        r = 1
    if r > n:
        print('The failure probability is too low, the rank is set to the maximum.')
        r = n
    
    # Draw first r random vectors
    np.random.seed(seed)
    Omega = np.random.randn(A.shape[1], r)
    Omega = Omega.astype(A.dtype)
    Y = A.dot(Omega)
    Qi, Ri = la.qr(Y, mode='economic')
    Q, R = Qi, Ri
    j = 0
    current_max = np.max(np.linalg.norm(Y, axis=0))
    # print(f'Current max error (j={j}): ', current_max)

    # Check the convergence
    while current_max > tol:
        j += 1
        # Draw r random vectors
        Omega = np.random.randn(A.shape[1], r)
        Omega = Omega.astype(A.dtype)
        Y = A.dot(Omega)
        Qi = Y - Q.dot(Q.T.conj()).dot(Y)
        current_max = np.max(np.linalg.norm(Y - Q.dot(Q.T.conj()).dot(Y), axis=0))
        # print(f'Current max error (j={j}): ', current_max)
        Q, R = la.qr_insert(Q, R, Qi, -1, which='col')

    return Q


