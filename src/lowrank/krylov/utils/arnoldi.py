"""Arnoldi iteration algorithms for Krylov subspaces.

Author: Benjamin Carrel, University of Geneva, 2022-2023
"""

# %% Imports
from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import scipy.linalg as la
import scipy.sparse as sps
import scipy.sparse.linalg as spsla
from numpy import ndarray
from scipy.sparse import spmatrix


# %% Functions
def Arnoldi(A: ndarray | spmatrix, x: ndarray, m: int) -> tuple[ndarray, ndarray]:
    """Arnoldi algorithm.
    Computes orthogonal basis of a Krylov space:
        PK_m(A,x) = span{x, A x, A^2 x, ..., A^(m-1) x}
    where A is a non-symmetric matrix, and x is a vector.
    If A is symmetric, the Lanczos algorithm will be more efficient.

    Parameters
    ----------
    A : ndarray | spmatrix
        Matrix of shape (n,n)
    x : ndarray
        Vector of shape (n,)
    m : int
        Size of the Krylov space

    Returns
    -------
    Q : ndarray
        Orthogonal Matrix of shape (n,m) containing the basis of the Krylov space
    H : ndarray
        Hessenberg Matrix of shape (m,m) containing the projection of A on the Krylov space

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.sparse import csr_matrix
    >>> from lowrank.krylov import Arnoldi
    >>> # Create a non-symmetric matrix
    >>> A = csr_matrix([[1, 2, 0], [0, 3, 1], [1, 0, 2]])
    >>> x = np.array([1.0, 0.0, 0.0])
    >>> Q, H = Arnoldi(A, x, m=3)
    >>> # Verify orthogonality
    >>> np.allclose(Q.T @ Q, np.eye(3))
    True
    >>> # Verify Arnoldi relation: A @ Q = Q @ H (up to numerical error)
    >>> np.allclose(A @ Q, Q @ H[:3, :])
    True
    """
    # Check inputs
    assert isinstance(
        A, (np.ndarray, spmatrix)
    ), "A must be a numpy array or a scipy sparse matrix"
    assert isinstance(x, np.ndarray), "x must be a numpy array"
    assert A.shape[0] == A.shape[1], "A must be a square matrix"

    # Sanity check
    x = x.reshape(-1)
    if len(x) != A.shape[0]:
        raise ValueError("x must have the same size as A")
    if m > A.shape[0]:
        raise ValueError("m must be smaller than the dimension of the matrix")

    # dtype depends on the type of A and x
    dtype = A.dtype
    if x.dtype != dtype:
        dtype = np.promote_types(dtype, x.dtype)

    # Initialize
    n = A.shape[0]
    Q = np.zeros((n, m), dtype=dtype)
    H = np.zeros((m, m), dtype=dtype)
    Q[:, 0] = x / la.norm(x)

    # Arnoldi algorithm
    for j in np.arange(m):
        u = A.dot(Q[:, j])
        for i in np.arange(j + 1):
            H[i, j] = Q[:, i].conj().T.dot(u)
            u = u - H[i, j] * Q[:, i]
        if j < m - 1:
            H[j + 1, j] = la.norm(u)
            if H[j + 1, j] < 1e-15:
                print("Lucky breakdown.")
                break
            Q[:, j + 1] = u / H[j + 1, j]
    return Q, H


def shift_and_invert_Arnoldi(
    A: ndarray | spmatrix, x: ndarray, m: int, shift: float = 0
) -> tuple[ndarray, ndarray]:
    """Arnoldi algorithm with shift and invert.
    Computes orthogonal basis of a Krylov space:
        SK_m(A,x) = span{x, (A - sI)^(-1) x, ..., (A - sI)^(-m+1) x}
    where A is a matrix, and x is a vector.
    s is the shift.

    Parameters
    ----------
    A : ndarray | spmatrix
        Matrix of shape (n,n)
    x : ndarray
        Vector of shape (n,)
    m : int
        Size of the Krylov space
    shift : float
        Shift of the matrix. Default is 0 (only invert the matrix)

    Returns
    -------
    Q : ndarray
        Orthogonal Matrix of shape (n, m) containing the basis of the Krylov space
    H : ndarray
        Hessenberg Matrix of shape (m, m) containing the projection of A on the Krylov space

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.sparse import csr_matrix
    >>> from lowrank.krylov import shift_and_invert_Arnoldi
    >>> # Create a sparse matrix
    >>> A = csr_matrix([[4, 1, 0], [1, 3, 1], [0, 1, 2]])
    >>> x = np.array([1.0, 0.0, 0.0])
    >>> # Use shift-and-invert to find eigenvectors near shift=3
    >>> Q, H = shift_and_invert_Arnoldi(A, x, m=2, shift=3.0)
    >>> # The Krylov space emphasizes eigenvalues close to 3
    >>> Q.shape
    (3, 2)
    """
    # Check inputs
    assert isinstance(
        A, (np.ndarray, spmatrix)
    ), "A must be a numpy array or a scipy sparse matrix"
    assert isinstance(x, np.ndarray), "x must be a numpy array"
    assert A.shape[0] == A.shape[1], "A must be a square matrix"
    assert isinstance(shift, (int, float, complex)), "shift must be a number"
    assert shift != np.inf, "infty shift is not supported"

    # Sanity check
    x = x.reshape(-1)
    if len(x) != A.shape[0]:
        raise ValueError("x must have the same size as A")
    if m > A.shape[0]:
        raise ValueError("m must be smaller than the dimension of the matrix")

    # dtype depends on the type of A, X and shift
    dtype = A.dtype
    if x.dtype != dtype:
        dtype = np.promote_types(dtype, x.dtype)
    if np.iscomplex(shift):
        dtype = np.promote_types(dtype, np.complex128)

    # Initialize
    n = A.shape[0]
    Q = np.zeros((n, m), dtype=dtype)
    H = np.zeros((m, m), dtype=dtype)
    Q[:, 0] = x / la.norm(x)

    # Arnoldi algorithm
    for j in np.arange(m - 1):
        u = spsla.spsolve(A - shift * sps.eye(n, format="csc"), Q[:, j])
        for i in np.arange(j + 1):
            H[i, j] = Q[:, i].conj().T.dot(u)
            u = u - H[i, j] * Q[:, i]
        H[j + 1, j] = la.norm(u)
        if H[j + 1, j] < 1e-15:
            print("Lucky breakdown.")
            break
        Q[:, j + 1] = u / H[j + 1, j]
    return Q, H


def rational_Arnoldi(
    A: spmatrix,
    x: ndarray,
    poles: list,
    invert_only: bool = False,
    inverses: Optional[list] = None,
) -> tuple[ndarray, ndarray]:
    """Arnoldi algorithm with rational Krylov space.
    Computes orthogonal basis of a Krylov space:
        RK_m(A,x) = q_m(A) PK_m(A,x)
        q_m(A) = (A - p_1 I)^(-1) ... (A - p_m I)^(-1)
    where A is a matrix, and x is a vector.
    p_i are the poles.

    Parameters
    ----------
    A : ndarray | spmatrix
        Matrix of shape (n,n)
    x : ndarray
        Vector of shape (n,)
    poles : list
        List of poles
    invert_only : bool
        If True, only invert the matrices. Default is False.
    inverses : list
        List of inverses of the matrices (optional). Default is None.

    Returns
    -------
    Q : ndarray
        Orthogonal Matrix of shape (n, m) containing the basis of the Krylov space
    H : ndarray
        Hessenberg Matrix of shape (m, m) containing the projection of A on the Krylov space

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.sparse import csr_matrix
    >>> from lowrank.krylov import rational_Arnoldi
    >>> # Create a sparse matrix
    >>> A = csr_matrix([[4, 1, 0], [1, 3, 1], [0, 1, 2]])
    >>> x = np.array([1.0, 0.0, 0.0])
    >>> # Use poles to emphasize specific spectral regions
    >>> poles = [1.0, 2.0]
    >>> Q, H = rational_Arnoldi(A, x, poles)
    >>> Q.shape
    (3, 3)
    >>> # Verify orthogonality
    >>> np.allclose(Q.T @ Q, np.eye(3))
    True
    """
    # Check inputs
    assert isinstance(A, spmatrix), "A must be a scipy sparse matrix"
    assert isinstance(x, np.ndarray), "x must be a numpy array"
    assert A.shape[0] == A.shape[1], "A must be a square matrix"
    # infty poles are not supported yet.
    assert np.inf not in poles, "infty poles are not supported yet"
    if not invert_only:  # check that 0 in not in the list of poles
        assert (
            0 not in poles
        ), "rational krylov does not work with 0 in the list of poles (the basis is not orthogonal)"

    # Sanity check
    m = len(poles) + 1
    n = A.shape[0]
    x = x.reshape(-1)
    if len(x) != A.shape[0]:
        raise ValueError("x must have the same size as A")
    if m > A.shape[0]:
        raise ValueError("m must be smaller than the dimension of the matrix")

    # dtype depends on the type of A, X and poles
    dtype = A.dtype
    if x.dtype != dtype:
        dtype = np.promote_types(dtype, x.dtype)
    for pole in poles:
        if np.iscomplex(pole):
            dtype = np.promote_types(dtype, np.complex128)

    if invert_only:
        small_matvec = lambda v: v
    else:
        small_matvec = lambda v: A.dot(v)

    if inverses is None:
        inverses = [None for _ in poles]
    for i, pole in enumerate(poles):
        if inverses[i] is None:
            # Use default argument to capture pole value (avoid lambda closure bug)
            inverses[i] = lambda v, p=pole: spsla.spsolve(
                A - p * sps.eye(n, format="csc"), small_matvec(v)
            )

    # Initialize
    Q = np.zeros((n, m), dtype=dtype)
    H = np.zeros((m, m), dtype=dtype)
    Q[:, 0] = x / la.norm(x)

    # Arnoldi algorithm
    for j in np.arange(len(poles)):
        u = inverses[j](Q[:, j])
        for i in np.arange(j + 1):
            H[i, j] = Q[:, i].conj().T.dot(u)
            u = u - H[i, j] * Q[:, i]
        if j < m - 1:
            H[j + 1, j] = la.norm(u)
            if H[j + 1, j] < 1e-15:
                print("Lucky breakdown.")
                break
            Q[:, j + 1] = u / H[j + 1, j]
    return Q, H


def block_Arnoldi(A: ndarray | spmatrix, X: ndarray, m: int) -> tuple[ndarray, ndarray]:
    """Block Arnoldi algorithm.
    Compute orthogonal basis of a Krylov space:
        PK_m(A,X) = span{X, A X, A^2 X, ..., A^(m-1) X}
    where A is a (non-symmetric) matrix, and X is a (tall) matrix.

    See Y. Saad, Iterative methods for sparse linear systems, SIAM, 2003.

    Parameters
    ----------
    A : ndarray
        Matrix of shape (n,n)
    X : ndarray
        Matrix of shape (n,r), r > 1
    m : int
        Size of the Krylov space

    Returns
    -------
    Q : ndarray
        Matrix of shape (n,m*r) containing the basis of the Krylov space
    H : ndarray
        Matrix of shape (m*r,m*r) containing the Hessenberg matrix

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.sparse import csr_matrix
    >>> from lowrank.krylov import block_Arnoldi
    >>> # Create a matrix and block starting vector
    >>> A = csr_matrix([[1, 2, 0, 0], [0, 3, 1, 0], [1, 0, 2, 1], [0, 1, 0, 3]])
    >>> X = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]])
    >>> # Build block Krylov space with m=2 blocks
    >>> Q, H = block_Arnoldi(A, X, m=2)
    >>> Q.shape
    (4, 4)
    >>> # Verify orthogonality
    >>> np.allclose(Q.T @ Q, np.eye(4))
    True
    """
    # Check inputs
    assert isinstance(
        A, (np.ndarray, spmatrix)
    ), "A must be a numpy array or a scipy sparse matrix"
    assert isinstance(X, np.ndarray), "X must be a numpy array"
    assert A.shape[0] == A.shape[1], "A must be a square matrix"
    # Sanity check
    if X.ndim != 2:
        raise ValueError("X must be a vector or a matrix")
    n, r = X.shape
    if m * r > A.shape[0]:
        raise ValueError("The size of the Krylov space is too large")

    # Initialize
    Q = np.zeros((n, m * r), dtype=A.dtype)
    H = np.zeros((m * r, m * r), dtype=A.dtype)
    Q[:, :r], _ = la.qr(X, mode="economic")

    # Arnoldi algorithm
    for j in np.arange(m):
        Wj = A.dot(Q[:, j * r : (j + 1) * r])
        for i in np.arange(j + 1):
            H[i * r : (i + 1) * r, j * r : (j + 1) * r] = (
                Q[:, i * r : (i + 1) * r].conj().T.dot(Wj)
            )
            Wj = Wj - Q[:, i * r : (i + 1) * r].dot(
                H[i * r : (i + 1) * r, j * r : (j + 1) * r]
            )
        if j < m - 1:
            (
                Q[:, (j + 1) * r : (j + 2) * r],
                H[(j + 1) * r : (j + 2) * r, j * r : (j + 1) * r],
            ) = la.qr(Wj, mode="economic")
    return Q, H


def block_shift_and_invert_Arnoldi(
    A: ndarray | spmatrix,
    X: ndarray,
    m: int,
    shift: float = 0,
    invA: Optional[Callable] = None,
) -> tuple[ndarray, ndarray]:
    """Block Arnoldi algorithm with shift and invert.
    Compute orthogonal basis of a Krylov space:
        SK_m(A,X) = span{X, (A - sI)^(-1) X, ..., (A - sI)^(-m+1) X}
    where A is a matrix, and X is a (tall) matrix.
    s is the shift.

    Parameters
    ----------
    A : ndarray
        Matrix of shape (n,n)
    X : ndarray
        Matrix of shape (n,r), r > 1
    m : int
        Size of the Krylov space
    shift : float
        Shift of the matrix. Default is 0 (only invert the matrix)
    invA : callable
        Function that computes the matrix-vector product with the inverse of A - shift*I. Optional, faster if provided.

    Returns
    -------
    Q : ndarray
        Matrix of shape (n,m*r) containing the basis of the Krylov space
    H : ndarray
        Matrix of shape (m*r,m*r) containing the Hessenberg matrix
    """
    # Check inputs
    assert isinstance(
        A, (np.ndarray, spmatrix)
    ), "A must be a numpy array or a scipy sparse matrix"
    assert isinstance(X, np.ndarray), "X must be a numpy array"
    assert A.shape[0] == A.shape[1], "A must be a square matrix"
    assert isinstance(shift, (int, float, complex)), "shift must be a number"
    assert shift != np.inf, "infty shift is not supported"

    # Sanity check
    if X.ndim != 2:
        raise ValueError("X must be a vector or a matrix")
    n, r = X.shape
    if m * r > A.shape[0]:
        raise ValueError("The size of the Krylov space is too large")

    # dtype depends on the type of A, X and shift
    dtype = A.dtype
    if X.dtype != dtype:
        dtype = np.promote_types(dtype, X.dtype)
    if np.iscomplex(shift):
        dtype = np.promote_types(dtype, np.complex128)

    if invA is None:
        spluA = spsla.splu(A - shift * sps.eye(n, format="csc"))
        invA = lambda v: spluA.solve(v)

    # Initialize
    Q = np.zeros((n, m * r), dtype=dtype)
    H = np.zeros((m * r, m * r), dtype=dtype)
    Q[:, :r], _ = la.qr(X, mode="economic")

    # Block Arnoldi algorithm
    for j in np.arange(m):
        Wj = invA(Q[:, j * r : (j + 1) * r])
        for i in np.arange(j + 1):
            H[i * r : (i + 1) * r, j * r : (j + 1) * r] = (
                Q[:, i * r : (i + 1) * r].conj().T.dot(Wj)
            )
            Wj = Wj - Q[:, i * r : (i + 1) * r].dot(
                H[i * r : (i + 1) * r, j * r : (j + 1) * r]
            )
        if j < m - 1:
            (
                Q[:, (j + 1) * r : (j + 2) * r],
                H[(j + 1) * r : (j + 2) * r, j * r : (j + 1) * r],
            ) = la.qr(Wj, mode="economic")
    return Q, H


def block_rational_Arnoldi(
    A: ndarray | spmatrix,
    X: ndarray,
    poles: list,
    inverse_only: bool = False,
    inverses: Optional[list] = None,
) -> tuple[ndarray, ndarray]:
    """Block Arnoldi algorithm with rational Krylov space.
    Compute orthogonal basis of a Krylov space:
        RK_m(A,X) = q_m(A) PK_m(A,X)
        q_m(A) = (A - p_1 I)^(-1) ... (A - p_m I)^(-1)
    where A is a matrix, and X is a (tall) matrix.
    p_i are the poles.

    Parameters
    ----------
    A : ndarray | spmatrix
        Matrix of shape (n,n)
    X : ndarray
        Matrix of shape (n,r), r > 1
    poles : list
        List of poles
    inverse_only : bool
        If True, only invert the matrices. Default is False.
    inverses : list
        List of functions that compute the matrix-vector product with the inverse of (A - pole*I). Optional, faster if provided.

    Returns
    -------
    Q : ndarray
        Matrix of shape (n,m*r) containing the basis of the Krylov space
    H : ndarray
        Matrix of shape (m*r,m*r) containing the Hessenberg matrix
    """
    # Check inputs
    assert isinstance(A, spmatrix), "A must be a scipy sparse matrix"
    assert isinstance(X, np.ndarray), "X must be a numpy array"
    assert A.shape[0] == A.shape[1], "A must be a square matrix"
    # infty poles are not supported yet.
    assert np.inf not in poles, "infty poles are not supported yet"
    if not inverse_only:  # check that 0 in not in the list of poles
        assert (
            0 not in poles
        ), "rational krylov does not work with 0 in the list of poles (the basis is not orthogonal)"

    # Sanity check
    m = len(poles) + 1
    if X.ndim != 2:
        raise ValueError("X must be a vector or a matrix")
    n, r = X.shape
    if m * r > A.shape[0]:
        raise ValueError("The size of the Krylov space is too large")

    # dtype depends on the type of A, X and poles
    dtype = A.dtype
    if X.dtype != dtype:
        dtype = np.promote_types(dtype, X.dtype)
    for pole in poles:
        if np.iscomplex(pole):
            dtype = np.promote_types(dtype, np.complex128)

    if inverses is None:
        inverses = [None for _ in range(len(poles))]
    for i in range(len(poles)):
        if inverses[i] is None:
            # Use default argument to capture pole value (avoid lambda closure bug)
            inverses[i] = lambda v, p=poles[i]: spsla.spsolve(
                A - p * sps.eye(n, format="csc"), v
            )

    if inverse_only:
        small_matvec = lambda v: v
    else:
        small_matvec = lambda v: A.dot(v)

    # Initialize
    Q = np.zeros((n, m * r), dtype=dtype)
    H = np.zeros((m * r, m * r), dtype=dtype)
    Q[:, :r], _ = la.qr(X, mode="economic")

    # Block Arnoldi algorithm
    for j in np.arange(len(poles)):
        current_matvec = lambda v: inverses[j](small_matvec(v))

        # Arnoldi procedure
        Wj = current_matvec(Q[:, j * r : (j + 1) * r])
        for i in np.arange(j + 1):
            H[i * r : (i + 1) * r, j * r : (j + 1) * r] = (
                Q[:, i * r : (i + 1) * r].conj().T.dot(Wj)
            )
            Wj = Wj - Q[:, i * r : (i + 1) * r].dot(
                H[i * r : (i + 1) * r, j * r : (j + 1) * r]
            )
        if j < m - 1:
            (
                Q[:, (j + 1) * r : (j + 2) * r],
                H[(j + 1) * r : (j + 2) * r, j * r : (j + 1) * r],
            ) = la.qr(Wj, mode="economic")
    return Q, H
