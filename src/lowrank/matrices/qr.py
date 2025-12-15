"""QR low-rank matrix class and functions.

Authors: Benjamin Carrel and Rik Vorhaar
         University of Geneva, 2022-2025
"""

from __future__ import annotations

import warnings
from typing import Optional, Union

import numpy as np
from numpy import ndarray
from scipy import linalg as la

from ._qr_config import AUTOMATIC_TRUNCATION, DEFAULT_ATOL, DEFAULT_RTOL
from .low_rank_matrix import (
    LowRankEfficiencyWarning,
    LowRankMatrix,
    MemoryEfficiencyWarning,
)

warnings.simplefilter("once", LowRankEfficiencyWarning)
warnings.simplefilter("once", MemoryEfficiencyWarning)


# %% Define class QR
class QR(LowRankMatrix):
    """
    QR decomposition for low-rank matrices.

    Efficient storage and operations for matrices represented as the product of an
    orthogonal matrix Q and an upper triangular matrix R. Particularly well-suited
    for solving linear systems and least squares problems.

    Mathematical Representation
    ---------------------------
    Standard form: X = Q @ R
    Transposed form: X = R.H @ Q.H

    where:
    - Q ∈ ℂ^(m×r) has orthonormal columns (Q.H @ Q = I_r)
    - R ∈ ℂ^(r×n) is upper triangular
    - r = rank ≤ min(m, n)

    For real matrices, conjugate transpose (.H) is equivalent to regular transpose (.T).

    Key Differences from SVD
    ------------------------
    - **Triangular structure**: R is upper triangular (not diagonal)
    - **Solve efficiency**: O(r²) back substitution vs O(r³) for SVD
    - **No singular values**: Diagonal of R approximates importance but not sorted
    - **Operations**: Addition/subtraction preserve QR structure (unlike SVD which needs recomputation)
    - **Memory**: Same O(mr + rn) storage as SVD

    Storage Efficiency
    ------------------
    Full matrix: O(mn) storage
    QR: O(mr + rn) storage
    Memory savings when r << min(m, n)

    For 1000×800 matrix with rank 10:
    - Dense: 800,000 elements
    - QR: 18,000 elements (44x compression)
    - SVD: 18,100 elements (similar)

    Important Notes
    ---------------
    - CRITICAL: Q MUST have orthonormal columns (Q.H @ Q = I_r).
      This is NOT verified at initialization for performance.
      Invalid inputs produce incorrect results without warning.
    - Use .is_orthogonal() to verify orthonormality if needed
    - Use .is_upper_triangular() to verify R structure
    - R diagonal elements |R[i,i]| serve as importance indicators (but not sorted)
    - After operations, may return LowRankMatrix (orthogonality not preserved)
    - Transposed mode enables efficient representation of A.H without recomputation

    Attributes
    ----------
    Q : ndarray, shape (m, r)
        Orthogonal matrix with orthonormal columns
    R : ndarray, shape (r, n)
        Upper triangular matrix
    shape : tuple
        Shape of the represented matrix (m, n)
    rank : int
        Number of columns in Q (rank of the decomposition)
    _transposed : bool
        Whether in transposed mode (X = R.H @ Q.H)

    Properties
    ----------
    T : QR
        Transpose of the matrix
    H : QR
        Conjugate transpose (Hermitian) of the matrix
    conj : QR
        Complex conjugate of the matrix

    Methods Overview
    ----------------
    Solving Linear Systems:
        - solve(b) : Solve Xx = b via back/forward substitution
        - lstsq(b) : Least squares solution
        - pseudoinverse() : Moore-Penrose pseudoinverse

    Matrix Operations:
        - __add__ : Addition (preserves QR for matching transposed modes)
        - __sub__ : Subtraction (preserves QR for matching transposed modes)
        - __mul__ : Scalar or Hadamard multiplication
        - dot : Matrix-vector/matrix-matrix multiplication
        - hadamard : Element-wise multiplication

    Validation:
        - is_orthogonal() : Check Q orthonormality
        - is_upper_triangular() : Check R triangular structure
        - cond() : Condition number estimation

    Truncation:
        - truncate() : Remove columns with small R diagonal elements

    Conversions:
        - to_svd() : Convert to SVD format
        - from_svd() : Create from SVD
        - from_matrix() : Compute QR decomposition
        - from_low_rank() : QR of generic low-rank matrix

    Norms:
        - norm() : Matrix norms (Frobenius optimized for QR)

    Class Methods
    -------------
    - from_matrix(A) : Compute QR decomposition of matrix A
    - from_low_rank(LR) : Compute QR of low-rank matrix
    - from_svd(svd) : Convert SVD to QR (Q=U, R=S@V.H)
    - qr(A) : Alias for from_matrix
    - generate_random(shape) : Generate random QR for testing

    Examples
    --------
    **Creating QR from a matrix:**

    >>> import numpy as np
    >>> from lowrank.matrices import QR
    >>>
    >>> A = np.random.randn(100, 80)
    >>> X = QR.from_matrix(A)
    >>> print(X.shape)  # (100, 80)
    >>> print(X.rank)   # 80
    >>> np.allclose(X.full(), A)  # True

    **Solving linear systems (faster than SVD):**

    >>> b = np.random.randn(100)
    >>> x = X.solve(b)
    >>> np.allclose(A @ x, b)  # True
    >>>
    >>> # Least squares for overdetermined systems
    >>> x_ls = X.lstsq(b)
    >>> residual = np.linalg.norm(A @ x_ls - b)

    **Efficient operations:**

    >>> # Addition preserves QR structure (same transposed mode)
    >>> Y = QR.from_matrix(np.random.randn(100, 80))
    >>> Z = X + Y  # Returns QR with rank = rank(X) + rank(Y)
    >>>
    >>> # Frobenius norm computed from R only
    >>> norm_fro = X.norm('fro')  # O(rn) vs O(mnr) for full matrix

    **Transposed mode for efficient A.H representation:**

    >>> # Instead of computing QR(A.H), use transposed flag
    >>> X_t = QR.from_matrix(A, transposed=True)
    >>> np.allclose(X_t.full(), A.T)  # True (for real A)
    >>>
    >>> # Conjugate transpose flips mode
    >>> X_h = X.H  # Returns transposed QR
    >>> assert X_h._transposed != X._transposed

    **Validation and diagnostics:**

    >>> X.is_orthogonal()  # Verify Q orthonormality
    >>> X.is_upper_triangular()  # Verify R structure
    >>> cond_approx = X.cond(exact=False)  # Fast diagonal approximation
    >>> cond_exact = X.cond(exact=True)   # Exact via SVD of R

    **Truncation for rank reduction:**

    >>> X_trunc = X.truncate(r=10)  # Keep first 10 columns
    >>> X_trunc = X.truncate(atol=1e-10)  # Remove small R[i,i]

    **Conversion between formats:**

    >>> from lowrank.matrices import SVD
    >>> X_svd = X.to_svd()  # Convert to SVD (requires SVD of R)
    >>> Y = QR.from_svd(X_svd)  # Convert back (Q=U, R=S@V.H)

    **Memory efficiency:**

    >>> X.compression_ratio()  # < 1.0 means memory savings
    >>> X.memory_usage('MB')  # Actual memory used

    See Also
    --------
    SVD : Singular Value Decomposition (diagonal middle matrix)
    QuasiSVD : Generalized SVD with non-diagonal middle matrix
    LowRankMatrix : Base class for low-rank representations

    References
    ----------
    .. [1] Trefethen, L. N., & Bau III, D. (1997). Numerical Linear Algebra. SIAM.
    .. [2] Golub, G. H., & Van Loan, C. F. (2013). Matrix Computations (4th ed.).
           Johns Hopkins University Press.
    .. [3] Demmel, J. W. (1997). Applied Numerical Linear Algebra. SIAM.

    Notes
    -----
    - QR is particularly efficient for solving linear systems and least squares
    - For repeated solves with same A, QR is faster than LU or SVD
    - Column pivoting available via scipy.linalg.qr(pivoting=True) but not
      integrated into this class (see documentation for details)
    - Operations exploit orthogonality and triangular structure for efficiency
    - Condition number can be estimated quickly from R diagonal (O(r) vs O(r³))
    """

    ## ATTRIBUTES
    _format = "QR"

    def __init__(self, Q: ndarray, R: ndarray, transposed: bool = False, **extra_data):
        """
        Initialize a QR decomposition.

        Parameters
        ----------
        Q : ndarray
            Orthogonal matrix (m x r) with orthonormal columns.
            ASSUMED to have orthonormal columns: Q.H @ Q = I_r.
            Not verified at initialization for performance.
        R : ndarray
            Upper triangular matrix (r x n)
        transposed : bool, optional
            If True, represents X = R.H @ Q.H instead of X = Q @ R, by default False
            For real matrices, this is equivalent to X = R.T @ Q.T
        **extra_data
            Additional data to store with the matrix

        Notes
        -----
        For the standard form (transposed=False): X = Q @ R
        For the transposed form (transposed=True): X = R.H @ Q.H

        The transposed form is useful when you have computed QR(A.H) and want
        to represent A efficiently.
        """
        # Store transposed flag
        self._transposed = transposed

        # Initialize cache for expensive computations
        self._cache = {}

        # Initialize based on whether we're in conjugate mode
        if self._transposed:
            # X = R.H @ Q.H
            super().__init__(R.T.conj(), Q.T.conj(), **extra_data)
        else:
            # Standard: X = Q @ R
            super().__init__(Q, R, **extra_data)

    ## MATRIX ALIASES
    @property
    def Q(self) -> ndarray:
        """Orthogonal matrix Q with orthonormal columns.

        Returns
        -------
        ndarray, shape (m, r)
            The orthogonal factor with Q.H @ Q = I_r.

        Notes
        -----
        In standard mode (transposed=False): directly returns stored Q.
        In transposed mode (transposed=True): reconstructs Q from internal storage.

        ASSUMPTION: Orthonormality is assumed, not verified. Use .is_orthogonal()
        to validate if needed.

        Examples
        --------
        >>> X = QR.from_matrix(np.random.randn(100, 80))
        >>> Q = X.Q
        >>> print(Q.shape)  # (100, 80)
        >>> print(np.allclose(Q.T @ Q, np.eye(80)))  # True
        """
        if self._transposed:
            # In conjugate mode, Q is actually stored as second matrix (conjugated)
            return self._matrices[1].T.conj()
        else:
            # Standard mode: Q is first matrix
            return self._matrices[0]

    @property
    def R(self) -> ndarray:
        """Upper triangular matrix R.

        Returns
        -------
        ndarray, shape (r, n)
            The upper triangular factor where R[i, j] ≈ 0 for i > j.

        Notes
        -----
        In standard mode (transposed=False): directly returns stored R.
        In transposed mode (transposed=True): reconstructs R from internal storage.

        The diagonal elements |R[i,i]| indicate column importance (but unlike SVD,
        they are NOT sorted by magnitude).

        ASSUMPTION: Upper triangular structure is assumed. Use .is_upper_triangular()
        to validate if needed.

        Examples
        --------
        >>> X = QR.from_matrix(np.random.randn(100, 80))
        >>> R = X.R
        >>> print(R.shape)  # (80, 80)
        >>> print(np.allclose(np.tril(R, -1), 0))  # True (lower triangle is zero)
        >>> print(np.diag(R))  # Diagonal elements (importance indicators)
        """
        if self._transposed:
            # In conjugate mode, R is actually stored as first matrix (conjugated)
            return self._matrices[0].T.conj()
        else:
            # Standard mode: R is second matrix
            return self._matrices[1]

    ## PROPERTIES
    @property
    def T(self) -> "QR":
        """Return the transpose of the QR matrix.

        For real matrices, .T is equivalent to .H
        For complex matrices, .T transposes without conjugating.

        Returns
        -------
        QR
            Transposed QR matrix

        Notes
        -----
        If X = Q @ R, then X.T = R.T @ Q.T = conj(R).H @ conj(Q).H
        If X = R.H @ Q.H, then X.T = (R.H @ Q.H).T = Q.conj() @ R.conj() = conj(Q) @ conj(R)

        To get transpose (not conjugate transpose), we conjugate Q and R and flip the mode.

        Examples
        --------
        >>> A = np.random.randn(100, 80)
        >>> X = QR.from_matrix(A)
        >>> X_t = X.T
        >>> np.allclose(X_t.full(), A.T)  # True
        >>> assert X_t._transposed != X._transposed  # Mode flipped
        """
        if self._transposed:
            # Was R.H @ Q.H, transpose gives Q.conj() @ R.conj()
            return QR(
                self.Q.conj(), self.R.conj(), transposed=False, **self._extra_data
            )
        else:
            # Was Q @ R, transpose gives R.T @ Q.T = conj(R).H @ conj(Q).H
            return QR(self.Q.conj(), self.R.conj(), transposed=True, **self._extra_data)

    @property
    def conj(self) -> "QR":
        """Return the complex conjugate of the QR matrix.

        Returns
        -------
        QR
            Complex conjugate of the QR matrix with same transposed mode

        Notes
        -----
        If X = Q @ R, then conj(X) = conj(Q) @ conj(R)
        If X = R.H @ Q.H, then conj(X) = conj(R).H @ conj(Q).H = conj(R.H) @ conj(Q.H)

        The transposed flag is preserved (unlike .T which flips it).

        Examples
        --------
        >>> A = np.random.randn(100, 80) + 1j * np.random.randn(100, 80)
        >>> X = QR.from_matrix(A)
        >>> X_conj = X.conj
        >>> np.allclose(X_conj.full(), A.conj())  # True
        >>> assert X_conj._transposed == X._transposed  # Mode preserved
        """
        return QR(
            self.Q.conj(),
            self.R.conj(),
            transposed=self._transposed,
            **self._extra_data,
        )

    @property
    def H(self) -> "QR":
        """Return the conjugate transpose (Hermitian) of the QR matrix.

        Returns
        -------
        QR
            Conjugate transpose with flipped transposed mode

        Notes
        -----
        If X = Q @ R, then X.H = R.H @ Q.H (transposed mode)
        If X = R.H @ Q.H, then X.H = (R.H @ Q.H).H = Q @ R (standard mode)

        The transposed flag is always flipped (unlike .conj which preserves it).
        For real matrices, .H is equivalent to .T

        Examples
        --------
        >>> A = np.random.randn(100, 80) + 1j * np.random.randn(100, 80)
        >>> X = QR.from_matrix(A)
        >>> X_h = X.H
        >>> np.allclose(X_h.full(), A.T.conj())  # True
        >>> assert X_h._transposed != X._transposed  # Mode flipped
        >>>
        >>> # Double conjugate transpose returns to original mode
        >>> X_hh = X.H.H
        >>> assert X_hh._transposed == X._transposed
        >>> np.allclose(X_hh.full(), X.full())  # True
        """
        if self._transposed:
            # Was R.H @ Q.H, conjugate transpose gives Q @ R
            return QR(self.Q, self.R, transposed=False, **self._extra_data)
        else:
            # Was Q @ R, conjugate transpose gives R.H @ Q.H
            return QR(self.Q, self.R, transposed=True, **self._extra_data)

    def to_svd(self):
        """Convert QR to SVD format.

        Computes SVD of R: R = U_R @ S @ V_R.H
        Then: X = Q @ R = (Q @ U_R) @ S @ V_R.H

        Returns
        -------
        SVD
            Equivalent SVD representation with same shape and rank

        Notes
        -----
        **Algorithm:**

        For standard mode (X = Q @ R):
            1. Compute SVD of R: R = U_R @ diag(s) @ V_R.H
            2. Construct: X = Q @ R = (Q @ U_R) @ diag(s) @ V_R.H
            3. Return SVD(Q @ U_R, s, V_R)

        For transposed mode (X = R.H @ Q.H):
            1. Compute SVD of R: R = U_R @ diag(s) @ V_R.H
            2. X.H = (R.H @ Q.H).H = Q @ R = Q @ U_R @ diag(s) @ V_R.H
            3. Return SVD(Q @ U_R, s, V_R)

        **Computational cost:**
        - SVD of R: O(r²n) where r = rank, n = number of columns
        - Matrix multiplication Q @ U_R: O(mr²) where m = number of rows
        - Total: O(r²(m + n))

        **Orthogonality:**
        - Input: Q has orthonormal columns, R is upper triangular
        - Output: U = Q @ U_R has orthonormal columns (product of orthogonal matrices)
        - Output: V = V_R has orthonormal columns (from SVD)
        - Result is a valid SVD with orthonormal U and V

        **Transposed mode:**
        Both standard and transposed modes produce the same SVD representation.
        This is because SVD naturally represents X, not X.H.

        Examples
        --------
        >>> A = np.random.randn(100, 80)
        >>> X_qr = QR.from_matrix(A)
        >>> X_svd = X_qr.to_svd()
        >>> print(X_svd.shape)  # (100, 80)
        >>> print(X_svd.rank)   # Same as X_qr.rank
        >>> np.allclose(X_qr.full(), X_svd.full())  # True

        >>> # Transposed mode
        >>> X_qr_t = QR.from_matrix(A, transposed=True)
        >>> X_svd_t = X_qr_t.to_svd()
        >>> np.allclose(X_qr_t.full(), X_svd_t.full())  # True

        See Also
        --------
        from_svd : Convert SVD to QR format
        SVD : Singular Value Decomposition class
        """
        # Import SVD locally to avoid circular import
        from .svd import SVD

        # Compute SVD of R matrix
        U_R, s, Vt_R = la.svd(self.R, full_matrices=False)
        V_R = Vt_R.T.conj()

        if not self._transposed:
            # Standard mode: X = Q @ R = Q @ (U_R @ diag(s) @ V_R.H) = (Q @ U_R) @ diag(s) @ V_R.H
            U_new = self.Q @ U_R
            V_new = V_R
        else:
            # Transposed mode: X = R.H @ Q.H
            # We have R = U_R @ diag(s) @ V_R.H
            # So: X = R.H @ Q.H = (U_R @ diag(s) @ V_R.H).H @ Q.H
            #       = V_R @ diag(s) @ U_R.H @ Q.H
            # This is SVD with U=V_R, s=s, V.H=U_R.H @ Q.H, i.e., V=(Q @ U_R).conj()
            U_new = V_R
            V_new = (self.Q @ U_R).conj()

        # Return SVD with same extra_data
        return SVD(U_new, s, V_new, **self._extra_data)

    def is_orthogonal(self, tol: float = 1e-12) -> bool:
        """Check if Q has orthonormal columns within a tolerance.

        Parameters
        ----------
        tol : float, optional
            Tolerance for orthogonality check, by default 1e-12

        Returns
        -------
        bool
            True if Q.H @ Q is approximately equal to identity

        Notes
        -----
        Verifies that Q.H @ Q = I_k (orthonormal columns), where k is the number of columns in Q.
        This is always true when QR is constructed via from_matrix() or qr(),
        but may not hold if Q and R are provided manually.
        Note that Q may have more columns than the reported rank after Hadamard products.

        Computational cost: O(k²m) where k is the number of columns in Q.
        Result is cached for the default tolerance.
        """
        # Check cache for default tolerance
        cache_key = ("is_orthogonal", tol)
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Use actual number of columns in Q, not self.rank
        k = self.Q.shape[1]
        identity = np.eye(k)
        Q_H_Q = self.Q.T.conj() @ self.Q
        result = np.allclose(Q_H_Q, identity, atol=tol)

        # Cache result for default tolerance
        if tol == 1e-12:
            self._cache[cache_key] = result

        return result

    def is_upper_triangular(self, tol: float = 1e-12) -> bool:
        """Check if R is upper triangular within a tolerance.

        Parameters
        ----------
        tol : float, optional
            Tolerance for triangularity check, by default 1e-12

        Returns
        -------
        bool
            True if all elements below the diagonal of R are approximately zero

        Notes
        -----
        Verifies that R[i,j] ≈ 0 for all i > j (strict lower triangular part is zero).
        This is always true when QR is constructed via from_matrix() or qr(),
        but may not hold if Q and R are provided manually.

        For methods like solve() and lstsq(), having an upper triangular R enables
        efficient back substitution. If R is not upper triangular, these methods
        may produce incorrect results.

        Computational cost: O(r²) where r = min(rank, n) is the number of rows in R.
        Result is cached for the default tolerance.

        Examples
        --------
        >>> X = QR.from_matrix(A)
        >>> X.is_upper_triangular()  # True
        True
        >>> Q = np.random.randn(10, 5)
        >>> R = np.random.randn(5, 8)  # Not upper triangular
        >>> Y = QR(Q, R)
        >>> Y.is_upper_triangular()  # False
        False
        """
        # Check cache for default tolerance
        cache_key = ("is_upper_triangular", tol)
        if cache_key in self._cache:
            return self._cache[cache_key]

        r, n = self.R.shape

        # Check if strict lower triangular part is zero
        # For efficiency, only check the lower triangle (i > j)
        for i in range(
            1, r
        ):  # Start from row 1 (skip first row which has no elements below)
            # Check elements [i, 0:i] (all elements before the diagonal in row i)
            if not np.allclose(self.R[i, :i], 0, atol=tol):
                if tol == 1e-12:
                    self._cache[cache_key] = False
                return False

        # All elements below diagonal are zero
        if tol == 1e-12:
            self._cache[cache_key] = True
        return True

    def norm(self, ord: str | int = "fro") -> float:
        """Compute matrix norm with QR-specific optimizations.

        Parameters
        ----------
        ord : str or int, optional
            Norm type. Default is 'fro' (Frobenius).
            - 'fro': Frobenius norm (optimized: ||Q @ R||_F = ||R||_F for orthogonal Q)
            - 2: Spectral norm (requires full matrix)
            - 1, -1, inf, -inf: Other matrix norms (require full matrix)

        Returns
        -------
        float
            Matrix norm

        Notes
        -----
        For orthogonal Q:
            ||Q @ R||_F = ||R||_F  (Frobenius norm preserved)
            ||Q @ R||_2 ≠ ||R||_2  (spectral norm NOT preserved)

        Results are cached for efficiency.

        Examples
        --------
        >>> X = QR.from_matrix(A)
        >>> X.norm('fro')  # Efficient: computed from R only
        >>> X.norm(2)      # Requires full matrix
        """
        # Check cache
        if ord in self._cache:
            return self._cache[ord]

        # Frobenius norm can be computed efficiently from R
        if ord == "fro":
            result = float(np.linalg.norm(self.R, "fro"))
        else:
            # Other norms require full matrix
            result = float(np.linalg.norm(self.full(), ord))  # type: ignore[arg-type]

        # Cache result
        self._cache[ord] = result
        return result

    def __repr__(self) -> str:
        """String representation of the QR matrix"""
        mode = " (conjugate)" if self._transposed else ""
        return f"{self.shape} QR decomposition with rank {self.rank}{mode}"

    ## STANDARD OPERATIONS
    def __add__(self, other: QR | LowRankMatrix | ndarray) -> Union[QR, ndarray]:
        """Addition optimized for QR matrices with same transposed mode.

        Efficiently adds two QR matrices while preserving the QR structure when both
        matrices have the same transposed flag (both standard or both transposed).
        The resulting rank is sum of the input ranks, but can be reduced with truncation.

        Parameters
        ----------
        other : QR, LowRankMatrix, or ndarray
            Matrix to add. If QR with matching transposed flag, preserves QR format.
            Otherwise, falls back to parent class addition.

        Returns
        -------
        QR, LowRankMatrix, or ndarray
            Result of addition.
            - If both are QR with same transposed flag: returns QR with rank = rank1 + rank2
            - If mismatched transposed flags: falls back to LowRankMatrix addition
            - If other is ndarray: returns ndarray

        Notes
        -----
        **Algorithm for matching QR + QR:**

        For standard mode (X1 = Q1 @ R1, X2 = Q2 @ R2):
            1. Concatenate Q matrices: Q_new = [Q1, Q2]
            2. Stack R matrices: R_stacked = [R1; R2]
            3. QR decompose R_stacked: R_stacked = Q_R @ R_new
            4. Update Q_new: Q_new = Q_new @ Q_R
            5. Result: X1 + X2 = Q_new @ R_new

        For transposed mode (X1 = R1.H @ Q1.H, X2 = R2.H @ Q2.H):
            Same algorithm applies with transposed flag preserved.

        **Computational cost:**
        - Concatenation: O(mr1 + mr2 + nr1 + nr2)
        - QR of stacked R: O((r1+r2)²min(r1+r2, n))
        - Q update: O(m(r1+r2)²)
        - Total: O(m(r1+r2)² + n(r1+r2)²) where r1, r2 are the ranks

        **Rank behavior:**
        The resulting rank is r1 + r2. This may be larger than optimal if the
        matrices have overlapping subspaces. Consider using truncation if needed.

        **Transposed mode handling:**
        Addition is only optimized when both QR matrices have the same transposed flag.
        If flags differ, the method falls back to generic LowRankMatrix addition,
        which may lose the QR structure.

        Examples
        --------
        >>> A = np.random.randn(100, 80)
        >>> B = np.random.randn(100, 80)
        >>> X = QR.from_matrix(A)
        >>> Y = QR.from_matrix(B)
        >>> Z = X + Y
        >>> print(isinstance(Z, QR))  # True
        >>> print(Z.rank)  # rank(X) + rank(Y)
        >>> np.allclose(Z.full(), A + B)  # True

        **With mismatched modes:**
        >>> X_std = QR.from_matrix(A, transposed=False)
        >>> Y_trans = QR.from_matrix(B, transposed=True)
        >>> Z = X_std + Y_trans  # Falls back to LowRankMatrix
        >>> print(isinstance(Z, QR))  # False (loses QR structure)

        See Also
        --------
        __sub__ : Subtraction (similar algorithm)
        SVD.__add__ : SVD addition with automatic truncation option
        """
        if isinstance(other, QR):
            _condition = (self._transposed and other._transposed) or (
                not self._transposed and not other._transposed
            )
            if _condition:
                new_Q = np.hstack((self.Q, other.Q))
                new_R = np.vstack((self.R, other.R))
                # compute QR of the new R
                Q, R = la.qr(new_R, mode="economic")
                # update Q
                new_Q = new_Q.dot(Q)
                if self._transposed:
                    return QR(new_Q, R, transposed=True)
                else:
                    return QR(new_Q, R)
            else:
                # Mismatched transposed flags, fall back to parent
                return super().__add__(other)
        else:
            return super().__add__(other)

    def __sub__(self, other: QR | LowRankMatrix | ndarray) -> Union[QR, ndarray]:
        """Subtraction optimized for QR matrices with same transposed mode.

        Efficiently subtracts two QR matrices while preserving the QR structure when both
        matrices have the same transposed flag (both standard or both transposed).
        The resulting rank is sum of the input ranks (not difference!).

        Parameters
        ----------
        other : QR, LowRankMatrix, or ndarray
            Matrix to subtract. If QR with matching transposed flag, preserves QR format.
            Otherwise, falls back to parent class subtraction.

        Returns
        -------
        QR, LowRankMatrix, or ndarray
            Result of subtraction.
            - If both are QR with same transposed flag: returns QR with rank = rank1 + rank2
            - If mismatched transposed flags: falls back to LowRankMatrix subtraction
            - If other is ndarray: returns ndarray

        Notes
        -----
        **Algorithm for matching QR - QR:**

        For standard mode (X1 = Q1 @ R1, X2 = Q2 @ R2):
            1. Concatenate Q matrices: Q_new = [Q1, Q2]
            2. Stack R matrices with negation: R_stacked = [R1; -R2]
            3. QR decompose R_stacked: R_stacked = Q_R @ R_new
            4. Update Q_new: Q_new = Q_new @ Q_R
            5. Result: X1 - X2 = Q_new @ R_new

        For transposed mode (X1 = R1.H @ Q1.H, X2 = R2.H @ Q2.H):
            Same algorithm applies with transposed flag preserved.

        **Computational cost:**
        Same as addition: O(m(r1+r2)² + n(r1+r2)²) where r1, r2 are the ranks

        **Rank behavior:**
        IMPORTANT: The resulting rank is r1 + r2, NOT |r1 - r2|.
        Even when subtracting identical matrices (X - X), the result has rank 2*rank(X)
        before numerical cancellation in the QR decomposition reduces it.

        For X - X, the QR algorithm produces R ≈ 0 (within numerical precision),
        so the matrix is effectively zero, but stored with rank 2*rank(X).

        **Transposed mode handling:**
        Subtraction is only optimized when both QR matrices have the same transposed flag.
        If flags differ, the method falls back to generic LowRankMatrix subtraction.

        Examples
        --------
        >>> A = np.random.randn(100, 80)
        >>> B = np.random.randn(100, 80)
        >>> X = QR.from_matrix(A)
        >>> Y = QR.from_matrix(B)
        >>> Z = X - Y
        >>> print(isinstance(Z, QR))  # True
        >>> print(Z.rank)  # rank(X) + rank(Y)
        >>> np.allclose(Z.full(), A - B)  # True

        **Subtracting a matrix from itself:**
        >>> Z_zero = X - X
        >>> print(Z_zero.rank)  # 2 * rank(X) (not 0!)
        >>> np.allclose(Z_zero.full(), 0)  # True (numerically zero)

        **With mismatched modes:**
        >>> X_std = QR.from_matrix(A, transposed=False)
        >>> Y_trans = QR.from_matrix(B, transposed=True)
        >>> Z = X_std - Y_trans  # Falls back to LowRankMatrix
        >>> print(isinstance(Z, QR))  # False

        See Also
        --------
        __add__ : Addition (similar algorithm)
        SVD.__sub__ : SVD subtraction with automatic truncation option
        """
        if isinstance(other, QR):
            _condition = (self._transposed and other._transposed) or (
                not self._transposed and not other._transposed
            )
            if _condition:
                new_Q = np.hstack((self.Q, other.Q))
                new_R = np.vstack((self.R, -other.R))
                # compute QR of the new R
                Q, R = la.qr(new_R, mode="economic")
                # update Q
                new_Q = new_Q.dot(Q)
                if self._transposed:
                    return QR(new_Q, R, transposed=True)
                else:
                    return QR(new_Q, R)
            else:
                # Mismatched transposed flags, fall back to parent
                return super().__sub__(other)
        else:
            return super().__sub__(other)

    def __mul__(self, other: float | int | QR) -> QR | ndarray:  # type: ignore[override]
        """
        Scalar multiplication or element-wise multiplication by another matrix.

        Parameters
        ----------
        other : float, int, or QR
            Scalar value to multiply, or another QR matrix for Hadamard product

        Returns
        -------
        QR or ndarray
            Resulting QR matrix after multiplication, or ndarray for Hadamard with ndarray

        Notes
        -----
        For scalar c: c * (Q @ R) = Q @ (c * R)
        This preserves the orthogonality of Q while scaling the matrix.

        Examples
        --------
        >>> X = QR.from_matrix(A)
        >>> Y = 3.0 * X  # Equivalent to Q @ (3*R)
        """
        if isinstance(other, ndarray | QR):
            return self.hadamard(other)
        elif isinstance(other, (float, int, np.number)):
            # Only scale R to preserve orthogonality of Q
            return QR(self.Q, self.R * other, transposed=self._transposed)
        # Note: This narrows parent's signature but is safe for QR-specific operations
        # Parent allows LowRankMatrix, but QR.hadamard() handles this via isinstance check
        raise TypeError(f"Unsupported operand type(s) for *: 'QR' and '{type(other)}'")

    def __rmul__(self, other: float | int | QR) -> QR | ndarray:  # type: ignore[override]
        """
        Right scalar multiplication (commutative).

        Parameters
        ----------
        other : float or int
            Scalar value to multiply

        Returns
        -------
        QR or ndarray
            Resulting QR matrix after multiplication
        """
        return self.__mul__(other)

    def __imul__(self, other: float | int | QR) -> QR:  # type: ignore[override]
        """
        In-place scalar multiplication.

        Parameters
        ----------
        other : float, int, or QR
            Scalar value to multiply, or another QR matrix

        Returns
        -------
        QR
            Self after in-place multiplication

        Notes
        -----
        Only scales R to preserve orthogonality of Q.
        Modifies the internal R matrix directly.
        """
        if isinstance(other, QR):
            warnings.warn(
                "In-place multiplication between two QR is not implemented, falling back to regular multiplication (no memory saved).",
                UserWarning,
            )
            result = self.__mul__(other)  # type: ignore[return-value]
            # Note: __mul__ can return ndarray for Hadamard, but we only reach here for QR input
            return result  # type: ignore[return-value]
        elif isinstance(other, (float, int, np.number)):
            # Only scale R in-place
            self._matrices[1] *= other
            return self
        raise TypeError(f"Unsupported operand type(s) for *=: 'QR' and '{type(other)}'")

    def dot(
        self,
        other: Union[QR, LowRankMatrix, ndarray],
        side="right",
        dense_output: bool = False,
    ) -> Union[QR, LowRankMatrix, ndarray]:
        # Multiply self @ other
        if side == "right" or side == "usual":
            # QR @ AB -> keep QR format
            if isinstance(other, LowRankMatrix) and not self._transposed:
                M = other.dot(self.R, side="left").todense()  # type: ignore[union-attr]
                output = QR(self.Q, M)
            # RQ @ AB -> generic low-rank format
            elif self._transposed:
                output = super().dot(other, side="right")  # type: ignore[assignment]
            # QR @ other -> keep QR format
            else:
                M = self.R.dot(other)
                output = QR(self.Q, M)
        # Multiply other @ self
        elif side == "opposite" or side == "left":
            # AB @ RQ -> keep QR conjugate format
            if isinstance(other, LowRankMatrix) and self._transposed:
                M = other.dot(self.R.T.conj())
                output = QR(self.Q, M.T.conj(), transposed=True)
            # AB @ QR -> generic low-rank format
            elif not self._transposed:
                output = super().dot(other, side="left")  # type: ignore[assignment]
            # other @ RQ -> keep QR conjugate format
            else:
                M = other.dot(self.R.T.conj())
                output = QR(self.Q, M.T.conj(), transposed=True)

        if dense_output:
            return output.todense()
        else:
            return output

    def hadamard(self, other: QR | ndarray) -> QR | ndarray:
        """Element-wise (Hadamard) product with another matrix.

        Parameters
        ----------
        other : QR or ndarray
            Other matrix to perform Hadamard product with

        Returns
        -------
        QR, or ndarray
            Resulting matrix after Hadamard product

        Notes
        -----
        The rank will be multiplied by the rank of the other matrix.
        This operation may be inefficient for large ranks.
        In that case a warning is raised, and one should consider
        converting to dense with .todense() first.
        """
        if isinstance(other, QR):
            condition = (self._transposed and other._transposed) or (
                not self._transposed and not other._transposed
            )
            if not condition:
                raise ValueError(
                    "Hadamard product is only implemented between two QR matrices in the same mode (both transposed or both standard)."
                )
            Q_new = np.zeros(
                (self.Q.shape[0], self.Q.shape[1] * other.Q.shape[1]),
                dtype=self.Q.dtype,
            )
            R_new = np.zeros(
                (self.R.shape[0] * other.R.shape[0], self.R.shape[1]),
                dtype=self.R.dtype,
            )

            for i in range(self.Q.shape[0]):
                Q_new[i, :] = np.kron(self.Q[i, :], other.Q[i, :])
            for j in range(self.R.shape[1]):
                R_new[:, j] = np.kron(self.R[:, j], other.R[:, j])

            # Warn if Kronecker product creates more columns than rows
            if Q_new.shape[1] > Q_new.shape[0]:
                warnings.warn(
                    f"Hadamard product creates Q with {Q_new.shape[1]} columns but only {Q_new.shape[0]} rows. "
                    f"The result is algebraically consistent but may be inefficient. "
                    f"Consider using dense arrays instead.",
                    LowRankEfficiencyWarning,
                    stacklevel=2,
                )

            # Re-orthogonalize
            Q_new, R_tilde = la.qr(Q_new, mode="economic")
            R_new = R_tilde.dot(R_new)

            return QR(Q_new, R_new, transposed=self._transposed)
        else:
            return super().hadamard(other)

    ## CLASS METHODS
    @classmethod
    def from_matrix(
        cls,
        matrix: ndarray,
        mode: str = "economic",
        transposed: bool = False,
        **extra_data,
    ):
        """Compute the QR decomposition of a matrix.

        Parameters
        ----------
        matrix : ndarray
            Matrix to decompose
        mode : str, optional
            QR mode ('economic' or 'complete'), by default 'economic'
        transposed : bool, optional
            If True, return the transposed form X = R.H @ Q.H, by default False
            Standard form (False): X = Q @ R
            Transposed form (True): X = R.H @ Q.H = matrix.H
        **extra_data
            Additional data to store

        Returns
        -------
        QR
            QR decomposition object where full() reconstructs the input matrix

        Examples
        --------
        >>> A = np.random.randn(100, 80)
        >>> X = QR.from_matrix(A)  # X = Q @ R, X.full() == A
        >>> X_T = QR.from_matrix(A, transposed=True)  # X_T = R.H @ Q.H, X_T.full() == A.H
        """
        # Compute QR decomposition of the input matrix
        Q, R = la.qr(matrix, mode=mode)
        return cls(Q, R, transposed=transposed, **extra_data)

    @classmethod
    def from_low_rank(  # type: ignore[override]
        cls, low_rank: LowRankMatrix, transposed: bool = False, **extra_data
    ):
        """Compute the QR decomposition from a low-rank representation.

        Parameters
        ----------
        low_rank : LowRankMatrix
            Low rank matrix to decompose
        transposed : bool, optional
            If True, compute transposed form, by default False
        **extra_data
            Additional data to store

        Returns
        -------
        QR
            QR decomposition object
        """
        Q, S = la.qr(low_rank._matrices[0], mode="economic")
        R = np.linalg.multi_dot([S, *low_rank._matrices[1:]])
        return cls(Q, R, transposed=transposed, **extra_data)

    @classmethod
    def from_svd(cls, svd, transposed: bool = False, **extra_data):
        """Convert SVD to QR format.

        Simply uses Q = U and R = S @ V.H from the SVD representation.
        This is efficient and preserves orthogonality.

        Parameters
        ----------
        svd : SVD
            SVD object to convert from (X = U @ diag(s) @ V.H)
        transposed : bool, optional
            If True, compute transposed form, by default False
        **extra_data
            Additional data to store (overrides svd._extra_data)

        Returns
        -------
        QR
            QR decomposition object where full() reconstructs the SVD matrix

        Notes
        -----
        **Algorithm:**

        For standard mode (transposed=False):
            - Set Q = U (already orthogonal from SVD)
            - Set R = diag(s) @ V.H = S @ V.H
            - Result: X = Q @ R = U @ S @ V.H (same as SVD)

        For transposed mode (transposed=True):
            - Same Q and R as above
            - Interpretation: X = R.H @ Q.H = V @ S @ U.H = (U @ S @ V.H).H

        **Computational cost:**
        - Matrix multiplication S @ V.H: O(r²n) where r = rank
        - No QR decomposition needed (already have orthogonal Q)
        - Total: O(r²n)

        **Advantages over to_svd():**
        - Cheaper: O(r²n) vs O(r²(m+n))
        - No loss of orthogonality (U is already orthogonal)
        - Preserves singular value information in R structure
        - No SVD computation needed
        - Preserves singular value structure in R

        **When to use:**
        - When you have an SVD and need QR format
        - When you want to exploit triangular structure of R
        - For efficient solve operations (QR is faster than SVD)

        Examples
        --------
        >>> A = np.random.randn(100, 80)
        >>> X_svd = SVD.from_matrix(A)
        >>> X_qr = QR.from_svd(X_svd)
        >>> print(X_qr.shape)  # (100, 80)
        >>> print(X_qr.rank)   # Same as X_svd.rank
        >>> np.allclose(X_svd.full(), X_qr.full())  # True

        >>> # Verify Q is orthogonal and R is upper triangular
        >>> print(X_qr.is_orthogonal())  # True
        >>> print(X_qr.is_upper_triangular())  # False (R = S @ V.H not triangular!)

        >>> # Round-trip conversion
        >>> X_svd2 = X_qr.to_svd()
        >>> np.allclose(X_svd.full(), X_svd2.full())  # True

        See Also
        --------
        to_svd : Convert QR to SVD format (requires SVD computation of R)
        SVD : Singular Value Decomposition class
        """
        # Q = U (already orthogonal)
        # R = S @ V.H where S is diagonal matrix of singular values
        Q = svd.U
        R = svd.S @ svd.Vh  # S @ V.H

        # Merge extra_data: svd's extra_data as base, override with provided extra_data
        merged_extra_data = {**svd._extra_data, **extra_data}

        return cls(Q, R, transposed=transposed, **merged_extra_data)

    @classmethod
    def qr(
        cls,
        matrix: ndarray,
        mode: str = "economic",
        transposed: bool = False,
        **extra_data,
    ) -> QR:
        """Compute a QR decomposition (alias for from_matrix).

        Parameters
        ----------
        matrix : ndarray
            Matrix to decompose
        mode : str, optional
            QR mode ('economic' or 'complete'), by default 'economic'
        transposed : bool, optional
            If True, compute transposed form, by default False
        **extra_data
            Additional data to store

        Returns
        -------
        QR
            QR decomposition object
        """
        return cls.from_matrix(matrix, mode=mode, transposed=transposed, **extra_data)

    @classmethod
    def generate_random(
        cls, shape: tuple, transposed: bool = False, seed: Optional[int] = None, **extra_data
    ) -> QR:
        """Generate a random QR decomposition.

        Parameters
        ----------
        shape : tuple
            Shape of the matrix to generate (m, n)
        transposed : bool, optional
            If True, generate transposed form, by default False
        seed : int, optional
            Random seed for reproducibility, by default None
        **extra_data
            Additional data to store

        Returns
        -------
        QR
            Random QR decomposition object

        Examples
        --------
        >>> X = QR.generate_random((100, 80), seed=42)
        >>> print(X.shape)  # (100, 80)
        >>> print(X.is_orthogonal())  # True
        """
        if seed is not None:
            np.random.seed(seed)
        A = np.random.randn(*shape)
        Q, R = la.qr(A, mode="economic")
        return cls(Q, R, transposed=transposed, **extra_data)

    def truncate(
        self,
        r: Optional[int] = None,
        rtol: Optional[float] = DEFAULT_RTOL,
        atol: float = DEFAULT_ATOL,
        inplace: bool = False,
    ) -> "QR":
        """Truncate QR to lower rank by removing columns with small R diagonal elements.

        The rank is prioritized over the tolerance.
        The relative tolerance is prioritized over absolute tolerance.
        NOTE: By default, the truncation is done with respect to the machine precision (DEFAULT_ATOL).

        Parameters
        ----------
        r : int, optional
            Target rank (number of columns to keep). If None, use tolerance.
        rtol : float, optional
            Relative tolerance. Remove columns where |R[i,i]| < rtol * max(|R[i,i]|).
            Default is DEFAULT_RTOL (None).
        atol : float, optional
            Absolute tolerance. Remove columns where |R[i,i]| < atol.
            Default is DEFAULT_ATOL (~2.22e-14).
        inplace : bool, optional
            If True, modify in place. Default False.

        Returns
        -------
        QR
            Truncated QR decomposition

        Notes
        -----
        For QR, truncation keeps the first r columns of Q and first r rows of R.
        The diagonal elements of R, |R[i,i]|, serve as importance indicators.

        Unlike SVD where singular values are sorted, R diagonal elements are NOT
        necessarily in decreasing order. However, for numerically stable QR decompositions,
        large diagonal elements indicate more important directions.

        **Algorithm:**
        1. If r is specified: keep first r columns/rows
        2. If rtol is specified: keep columns where |R[i,i]| > rtol * max(|R[i,i]|)
        3. If atol is specified: keep columns where |R[i,i]| > atol

        **Priority:** r > rtol > atol

        Examples
        --------
        >>> X = QR.from_matrix(A)
        >>> X_trunc = X.truncate(r=10)  # Keep only first 10 columns
        >>> X_trunc = X.truncate(rtol=1e-10)  # Remove columns with small R[i,i]
        >>> X_trunc = X.truncate(atol=1e-12)  # Absolute threshold

        See Also
        --------
        SVD.truncate : Similar method for SVD (uses singular values)
        """
        # If all are None, do nothing
        if r is None and rtol is None and atol is None:
            return self

        # Compute the rank associated to the tolerance
        if r is None:
            # Get absolute values of R diagonal
            R_diag = np.abs(np.diag(self.R))

            if rtol is not None:
                # Relative tolerance: compare to max diagonal element
                r = np.sum(R_diag > R_diag[0] * rtol) if len(R_diag) > 0 else 0
            else:
                # Absolute tolerance
                r = np.sum(R_diag > atol)

        # Truncate
        (m, n) = self.shape
        if r == 0:  # trivial case
            Q_new = np.zeros((m, 0), dtype=self.Q.dtype)
            R_new = np.zeros((0, n), dtype=self.R.dtype)
        else:  # general case
            Q_new = self.Q[:, :r]
            R_new = self.R[:r, :]

        if inplace:
            # Update internal storage based on transposed mode
            if self._transposed:
                self._matrices[0] = R_new.T.conj()
                self._matrices[1] = Q_new.T.conj()
            else:
                self._matrices[0] = Q_new
                self._matrices[1] = R_new
            self._cache.clear()  # Invalidate cached values after modification
            return self
        else:
            return QR(Q_new, R_new, transposed=self._transposed, **self._extra_data)

    def solve(self, b: ndarray, method: str = "direct") -> ndarray:
        """Solve Xx = b using QR factorization.

        For X = Q @ R, solve via:
            Q @ R @ x = b
            R @ x = Q.T @ b
            x = R^{-1} @ Q.T @ b  (back substitution)

        Parameters
        ----------
        b : ndarray
            Right-hand side, shape (m,) or (m, k)
        method : str, optional
            'direct' (default): Use back substitution (fast, requires full rank)
            'lstsq': Use least squares (handles rank deficiency)

        Returns
        -------
        ndarray
            Solution x, shape (n,) or (n, k)

        Raises
        ------
        ValueError
            If matrix dimensions don't match RHS dimensions
        np.linalg.LinAlgError
            If matrix is singular (method='direct')

        Notes
        -----
        QR solve is faster than SVD solve: O(n²) vs O(n³)
        Best for overdetermined systems (more equations than unknowns)

        For standard mode (X = Q @ R):
            - Solve R @ x = Q.H @ b via back substitution

        For transposed mode (X = R.H @ Q.H):
            - Solve R.H @ Q.H @ x = b
            - Equivalent to Q @ R @ y = b where y = conj(x)

        Examples
        --------
        >>> A = np.random.randn(100, 80)
        >>> b = np.random.randn(100)
        >>> X = QR.from_matrix(A)
        >>> x = X.solve(b)
        >>> np.allclose(A @ x, b)
        True
        """
        # Validate input
        b = np.asarray(b)
        if b.shape[0] != self.shape[0]:
            raise ValueError(
                f"Dimension mismatch: matrix has {self.shape[0]} rows but b has {b.shape[0]} rows"
            )

        if method == "lstsq":
            return self.lstsq(b)

        # Direct method using back substitution
        if not self._transposed:
            # Standard mode: X = Q @ R, solve Q @ R @ x = b
            # Step 1: Compute Q.H @ b
            y = self.Q.T.conj() @ b
            # Step 2: Solve R @ x = y via back substitution
            x = la.solve_triangular(self.R, y)
        else:
            # Transposed mode: X = R.H @ Q.H, solve R.H @ Q.H @ x = b
            # Let y = Q.H @ x, then R.H @ y = b
            # Step 1: Solve R.H @ y = b via forward substitution (R.H is lower triangular)
            y = la.solve_triangular(self.R.T.conj(), b, lower=True)
            # Step 2: Solve Q.H @ x = y, so x = Q @ y
            x = self.Q @ y

        return x

    def lstsq(self, b: ndarray, rcond: Optional[float] = None) -> ndarray:
        """Least squares solution via QR decomposition.

        Computes x that minimizes ||Xx - b||_2

        Parameters
        ----------
        b : ndarray
            Right-hand side, shape (m,) or (m, k)
        rcond : float, optional
            Relative condition number threshold for rank determination.
            Singular values smaller than rcond * largest_singular_value are treated as zero.
            If None, machine precision is used.

        Returns
        -------
        ndarray
            Least squares solution, shape (n,) or (n, k)

        Notes
        -----
        For overdetermined systems (m > n), finds x minimizing ||Ax - b||_2
        For underdetermined systems (m < n), finds minimum norm solution

        Uses QR decomposition:
        - For X = Q @ R: x = R^+ @ Q.H @ b where R^+ is pseudoinverse of R
        - Handles rank deficiency by zeroing small diagonal elements of R

        Examples
        --------
        >>> A = np.random.randn(100, 80)
        >>> b = np.random.randn(100)
        >>> X = QR.from_matrix(A)
        >>> x = X.lstsq(b)
        >>> residual = np.linalg.norm(A @ x - b)
        """
        # Validate input
        b = np.asarray(b)
        if b.shape[0] != self.shape[0]:
            raise ValueError(
                f"Dimension mismatch: matrix has {self.shape[0]} rows but b has {b.shape[0]} rows"
            )

        if not self._transposed:
            # Standard mode: X = Q @ R, solve via Q.H @ b and pseudoinverse of R
            y = self.Q.T.conj() @ b
            # Use lstsq on R
            x, _, _, _ = la.lstsq(self.R, y, cond=rcond)
        else:
            # Transposed mode: X = R.H @ Q.H
            # Solve R.H @ Q.H @ x = b
            # Use lstsq on full matrix for transposed case (less efficient but correct)
            x, _, _, _ = la.lstsq(self.full(), b, cond=rcond)

        return x

    def pseudoinverse(self, rcond: Optional[float] = None) -> ndarray:
        """Compute Moore-Penrose pseudoinverse using QR factorization.

        For X = Q @ R, computes X^+ = R^+ @ Q.H

        Parameters
        ----------
        rcond : float, optional
            Cutoff for small singular values in R.
            Singular values smaller than rcond * largest_singular_value are set to zero.
            If None, machine precision is used.

        Returns
        -------
        ndarray
            Pseudoinverse matrix, shape (n, m)

        Notes
        -----
        The pseudoinverse satisfies:
        1. X @ X^+ @ X = X
        2. X^+ @ X @ X^+ = X^+
        3. (X @ X^+).H = X @ X^+
        4. (X^+ @ X).H = X^+ @ X

        For full rank matrices:
        - Overdetermined (m > n): X^+ = (X.H @ X)^{-1} @ X.H = R^{-1} @ Q.H
        - Underdetermined (m < n): X^+ = X.H @ (X @ X.H)^{-1}

        Computational cost: O(n²m) for computing R^+ and matrix multiplication

        Examples
        --------
        >>> A = np.random.randn(100, 80)
        >>> X = QR.from_matrix(A)
        >>> X_pinv = X.pseudoinverse()
        >>> np.allclose(A @ X_pinv @ A, A)  # Property 1
        True
        >>> np.allclose(X_pinv @ A @ X_pinv, X_pinv)  # Property 2
        True
        """
        if not self._transposed:
            # Standard mode: X = Q @ R
            # X^+ = R^+ @ Q.H
            R_pinv = np.linalg.pinv(self.R, rcond=rcond)
            X_pinv = R_pinv @ self.Q.T.conj()
        else:
            # Transposed mode: X = R.H @ Q.H
            # X^+ = (R.H @ Q.H)^+ = Q @ R^{-H}
            R_pinv = np.linalg.pinv(self.R, rcond=rcond)
            X_pinv = self.Q @ R_pinv.T.conj()

        return X_pinv

    def cond(self, p: int | str = 2, exact: bool = False) -> float:
        """Compute condition number efficiently using R matrix.

        For QR = Q @ R with orthogonal Q:
            cond(Q @ R) = cond(R)

        The condition number is preserved by orthogonal transformations.

        Parameters
        ----------
        p : int or str, optional
            Norm type. Default is 2 (spectral norm).
            - 2: Spectral condition number (ratio of max/min singular values)
            - 'fro': Frobenius norm condition number
            - 1, -1, inf, -inf: Other matrix norms
        exact : bool, optional
            If True, compute exact condition number using numpy.linalg.cond.
            If False (default), use fast diagonal approximation for p=2.
            For other norms, always uses exact computation.

        Returns
        -------
        float
            Condition number of the matrix

        Raises
        ------
        ValueError
            If R has zeros on diagonal (singular matrix)

        Notes
        -----
        **Fast approximation (p=2, exact=False):**
        For upper triangular R, estimates the 2-norm condition number
        from diagonal elements: cond_2(R) ≈ max(|diag(R)|) / min(|diag(R)|)

        This is a LOWER BOUND for the true condition number. The exact value
        requires SVD of R: cond_2(R) = σ_max(R) / σ_min(R)

        The approximation is exact for diagonal matrices and reasonable for
        well-conditioned matrices, but can significantly underestimate the
        condition number for ill-conditioned non-diagonal triangular matrices.

        **Exact computation (exact=True or other norms):**
        Uses numpy.linalg.cond(R, p) which computes the exact condition number.
        This requires SVD for p=2, which is O(r³).

        **Computational cost:**
        - p=2, exact=False: O(r) - diagonal extraction only
        - p=2, exact=True: O(r³) - requires SVD of R
        - Other norms: O(r²) to O(r³) depending on norm type

        **Caching:**
        Results are cached for repeated queries with the same (norm, exact) pair.

        **Singularity:**
        If any diagonal element is zero (within machine precision),
        the approximation returns np.inf to indicate singular matrix.

        **Why use approximation?**
        For large ranks, computing exact condition number via SVD is expensive.
        The diagonal approximation provides a quick lower bound that's sufficient
        for many applications (checking if matrix is well-conditioned).

        Examples
        --------
        >>> A = np.random.randn(100, 80)
        >>> X = QR.from_matrix(A)
        >>>
        >>> # Fast approximation
        >>> cond_approx = X.cond(2)  # O(r)
        >>>
        >>> # Exact computation
        >>> cond_exact = X.cond(2, exact=True)  # O(r³)
        >>>
        >>> # For diagonal matrices, approximation is exact
        >>> A_diag = np.diag([1e10, 1e5, 1e0, 1e-5, 1e-10])
        >>> X_diag = QR.from_matrix(A_diag)
        >>> print(X_diag.cond(2))  # 1e20 (exact for diagonal)
        >>>
        >>> # Other norms
        >>> cond_fro = X.cond('fro')  # Always exact

        See Also
        --------
        numpy.linalg.cond : General condition number computation
        QuasiSVD.cond_estimate : Similar method for QuasiSVD
        is_upper_triangular : Check if R is upper triangular
        """
        # Check cache
        cache_key = ("cond", p, exact)
        if cache_key in self._cache:
            return self._cache[cache_key]

        # For p=2, choose between approximation and exact
        if p == 2 and not exact:
            # Fast diagonal approximation (lower bound)
            diag_R = np.diag(self.R)
            abs_diag = np.abs(diag_R)

            # Check for zeros (singular matrix)
            min_diag = np.min(abs_diag)
            if min_diag < np.finfo(self.R.dtype).eps:
                result = np.inf
            else:
                max_diag = np.max(abs_diag)
                result = max_diag / min_diag
        else:
            # Exact computation using numpy (requires SVD for p=2)
            result = np.linalg.cond(self.R, p)  # type: ignore[arg-type]

        # Cache result
        self._cache[cache_key] = result
        return result
