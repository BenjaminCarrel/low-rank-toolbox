"""Randomized rangefinder algorithms for range approximation.

Author: Benjamin Carrel, University of Geneva, 2024
"""

# %% Importations
import numpy as np
from numpy import ndarray
from scipy import linalg as la
from scipy.sparse.linalg import LinearOperator


# %% The randomized rangefinder
def rangefinder(
    A: LinearOperator, r: int, p: int = 5, q: int = 0, seed: int = 1234, **extra_args
) -> ndarray:
    """
    The (randomized) rangefinder method, also called HMT method.

    Reference:
        "Finding structure with randomness: Probabilistic algorithms for constructing approximate matrix decompositions",
        Halko, Martinsson and Tropp 2010.

    The mathematical formulation is:
        Q, _ = qr((A A^H)^q A Omega, mode='economic')
    where Omega is a Gaussian random matrix of shape (n, r + p).

    When A is a low-rank matrix, the rangefinder provides a good approximation
    of the range of A with high probability.

    Parameters
    ----------
    A : LinearOperator
        The matrix to sketch.
    r : int
        The target rank.
    p : int, optional
        The number of over-sampling (default: 5).
    q : int, optional
        Number of power iterations (default: 0).
    seed : int, optional
        The seed for the random number generator (default: 1234).
    **extra_args : dict
        Additional arguments. If 'Omega' is provided, use it as the sketching matrix.

    Returns
    -------
    Q : ndarray
        Estimation of the range of A.
    """
    # Check the inputs
    if r < 1:
        raise ValueError(f"Target rank must be at least 1, got r={r}.")
    if p < 0:
        raise ValueError(f"Oversampling parameter must be non-negative, got p={p}.")
    if q < 0:
        raise ValueError(f"Number of power iterations must be non-negative, got q={q}.")
    if r + p > min(A.shape):
        raise ValueError(
            f"Target rank + oversampling ({r + p}) exceeds minimum matrix dimension ({min(A.shape)})."
        )

    # Check for sketching matrix in extra_args
    if "Omega" in extra_args:
        Omega = extra_args["Omega"]
    else:
        # Gaussian matrix
        np.random.seed(seed)
        Omega = np.random.randn(A.shape[1], r + p)

    # Support for complex matrix A
    if np.iscomplexobj(A):
        Omega = Omega.astype(A.dtype)

    # The method with power iteration
    Y = A.dot(Omega)
    Q, _, _ = la.qr(Y, mode="economic", pivoting=True)
    for _ in range(q):
        Y = A.T.conj().dot(Q)
        Q, _, _ = la.qr(Y, mode="economic", pivoting=True)
        Y = A.dot(Q)
        Q, _, _ = la.qr(Y, mode="economic", pivoting=True)

    return Q


# %% The adaptive randomized rangefinder
def adaptive_rangefinder(
    A: LinearOperator, tol: float = 1e-6, failure_prob: float = 1e-6, seed: int = 1234
) -> ndarray:
    """
    The adaptive (randomized) rangefinder method.
    The tolerance is the error made by the approximation space ||A - QQ^H A||_F <= tol
    The failure probability is the probability that the error is larger than tol

    Reference:
        "Finding structure with randomness: Probabilistic algorithms for constructing approximate matrix decompositions",
        Halko, Martinsson and Tropp 2010.

    NOTE: For efficiency, the method performs computations in blocks of size r (defined by the failure probability). Blocking allows to use BLAS3 operations.

    Parameters
    ----------
    A : LinearOperator
        The matrix to sketch.
    tol : float, optional
        The tolerance for the approximation (default: 1e-6).
    failure_prob : float, optional
        The failure probability, must be in (0, 1) (default: 1e-6).
    seed : int, optional
        The seed for the random number generator (default: 1234).

    Returns
    -------
    Q : ndarray
        The sketched matrix with orthonormal columns.
    """
    # Input validation
    if tol <= 0:
        raise ValueError(f"Tolerance must be positive, got tol={tol}.")
    if not (0 < failure_prob < 1):
        raise ValueError(
            f"Failure probability must be in (0, 1), got failure_prob={failure_prob}."
        )

    # Compute the sketch size according to the failure probability (HMT Theorem 4.1)
    # Block size r grows with log(1/failure_prob) to ensure failure probability guarantee
    n = min(A.shape)
    r = int(np.ceil(-np.log(failure_prob / n) / np.log(10)))
    # Adjust tolerance to account for expected norm of random vectors (HMT Algorithm 4.2)
    # Factor 10 * sqrt(2/pi) ≈ 7.98 comes from probabilistic analysis
    tol = tol / (10 * np.sqrt(2 / np.pi))
    if r < 1:
        r = 1
    if r > n:
        import warnings

        warnings.warn(
            "Failure probability is very low; rank set to maximum matrix dimension."
        )
        r = n

    # Draw first r random vectors
    np.random.seed(seed)
    Omega = np.random.randn(A.shape[1], r)
    Omega = Omega.astype(A.dtype)
    Y = A.dot(Omega)
    Qi, Ri = la.qr(Y, mode="economic")
    Q, R = Qi, Ri
    j = 0
    current_max = np.max(np.linalg.norm(Y, axis=0))

    # Check the convergence
    while current_max > tol:
        j += 1
        # Draw r random vectors
        Omega = np.random.randn(A.shape[1], r)
        Omega = Omega.astype(A.dtype)
        Y = A.dot(Omega)
        Ej = Y - Q.dot(Q.T.conj()).dot(Y)
        current_max = np.max(np.linalg.norm(Ej, axis=0))
        Q, R = la.qr_insert(Q, R, Ej, -1, which="col")

    return Q
