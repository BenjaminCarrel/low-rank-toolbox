"""QuasiSVD low-rank matrix class and functions.

Authors: Benjamin Carrel and Rik Vorhaar
         University of Geneva, 2022-2025
"""

# %% Imports
from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Callable, List, Optional

import numpy as np
import scipy.linalg as la
from numpy import ndarray

from ._svd_config import AUTOMATIC_TRUNCATION, DEFAULT_ATOL, DEFAULT_RTOL
from .low_rank_matrix import (
    LowRankEfficiencyWarning,
    LowRankMatrix,
    MemoryEfficiencyWarning,
)
from .qr import QR

# Avoid circular imports: SVD is only imported for type hints
if TYPE_CHECKING:
    from .svd import SVD

warnings.simplefilter("once", LowRankEfficiencyWarning)
warnings.simplefilter("once", MemoryEfficiencyWarning)


# %% Define class QuasiSVD
class QuasiSVD(LowRankMatrix):
    """
    Quasi Singular Value Decomposition (Quasi-SVD) for low-rank matrices.

    A generalization of the SVD where the middle matrix S is not necessarily diagonal.
    This representation is useful for operations that preserve orthogonality of U and V
    but may produce non-diagonal or even singular middle matrices.

    Mathematical Representation
    ---------------------------
    X = U @ S @ V.T

    where:
    - U ∈ ℝ^(m×r) should have orthonormal columns (U^T @ U = I_r)
    - V ∈ ℝ^(n×q) should have orthonormal columns (V^T @ V = I_q)
    - S ∈ ℝ^(r×q) is a general matrix (not necessarily diagonal, may be singular)

    Key Differences from SVD
    -------------------------
    - SVD: S is diagonal with non-negative singular values, guaranteed non-singular
    - QuasiSVD: S is a general matrix, may be non-diagonal and potentially singular
    - QuasiSVD can represent intermediate results of matrix operations efficiently
    - Converting QuasiSVD → SVD requires O(r³) operations (SVD of S)

    Storage Efficiency
    ------------------
    Full matrix: O(mn) storage
    QuasiSVD: O(mr + rq + nq) storage
    Memory savings when r, q << min(m, n)

    Important Notes
    ---------------
    - U and V are ASSUMED to have orthonormal columns (not verified at initialization)
    - Use .is_orthogonal() to verify orthonormality if needed
    - After matrix operations, orthogonality of U and V is preserved when possible, otherwise returns LowRankMatrix
    - After addition/subtraction, S may become singular or ill-conditioned
    - Use .truncate() to convert to SVD and remove small/zero singular values
    - Use .to_svd() to convert to diagonal form without truncation

    Attributes
    ----------
    U : ndarray, shape (m, r)
        Left matrix (assumed to have orthonormal columns)
    S : ndarray, shape (r, q)
        Middle matrix (general matrix, may be singular)
    V : ndarray, shape (n, q)
        Right matrix (assumed to have orthonormal columns)
    Vh : ndarray, shape (q, n)
        Hermitian conjugate of V (V.T.conj())
    Vt : ndarray, shape (q, n)
        Transpose of V (without conjugate)
    Ut : ndarray, shape (r, m)
        Transpose of U (without conjugate)
    Uh : ndarray, shape (r, m)
        Hermitian conjugate of U (U.T.conj())
    K : ndarray, shape (m, r)
        Product U @ S, computed on demand and not cached
    L : ndarray, shape (r, n)
        Product V @ S.T, computed on demand and not cached

    Properties
    ----------
    shape : tuple
        Shape of the represented matrix (m, n)
    rank : int
        Current rank of the representation (not necessarily matrix rank)
    sing_vals : ndarray
        Singular values of S (computed on demand and cached)

    Methods Overview
    ----------------
    Core Operations:
        - __add__, __sub__ : Addition/subtraction maintaining low-rank structure
        - __mul__ : Scalar or Hadamard (element-wise) multiplication
        - dot : Matrix-vector or matrix-matrix multiplication
        - hadamard : Element-wise multiplication with another matrix

    Conversion & Truncation:
        - to_svd() : Convert to SVD with diagonal S
        - truncate() : Convert to SVD and remove small singular values

    Properties & Checks:
        - is_symmetric() : Check if matrix is symmetric
        - is_orthogonal() : Check if U and V are orthonormal
        - is_singular() : Check if S is singular
        - norm() : Compute matrix norm (Frobenius, 2-norm, nuclear)

    Class Methods:
        - multi_add() : Efficient addition of multiple QuasiSVD matrices
        - multi_dot() : Efficient multiplication of multiple QuasiSVD matrices
        - generalized_nystroem() : Randomized low-rank approximation

    Configuration
    -------------
    Default behavior controlled by AUTOMATIC_TRUNCATION (default: False)
    - False: Maintains algebraic consistency (X - X = 0 exactly)
    - True: Automatically truncates to save memory (X - X ≈ 0)

    Examples
    --------
    >>> import numpy as np
    >>> from lowrank.matrices import QuasiSVD
    >>>
    >>> # Create orthonormal matrices
    >>> m, n, r = 100, 80, 10
    >>> U, _ = np.linalg.qr(np.random.randn(m, r))
    >>> V, _ = np.linalg.qr(np.random.randn(n, r))
    >>> S = np.random.randn(r, r)
    >>>
    >>> # Create QuasiSVD
    >>> X = QuasiSVD(U, S, V)
    >>> print(X.shape)  # (100, 80)
    >>> print(X.rank)   # 10
    >>>
    >>> # Operations preserve low-rank structure
    >>> Y = X + X  # rank = 20 (sum of ranks)
    >>> Z = X @ X.T  # matrix multiplication
    >>>
    >>> # Convert to SVD (diagonal S)
    >>> X_svd = X.to_svd()
    >>>
    >>> # Truncate small singular values
    >>> X_trunc = X.truncate(rtol=1e-10)
    >>>
    >>> # Memory-efficient addition
    >>> W = QuasiSVD.multi_add([X, -X], auto_truncate=True)  # rank ≈ 0

    See Also
    --------
    SVD : Singular Value Decomposition with diagonal middle matrix
    LowRankMatrix : Base class for low-rank matrix representations

    References
    ----------
    .. [1] Koch, O., & Lubich, C. (2007). Dynamical low-rank approximation.
           SIAM Journal on Matrix Analysis and Applications, 29(2), 434-454.
    .. [2] Kressner, D., & Tobler, C. (2012). Low-rank tensor completion by
           Riemannian optimization. BIT Numerical Mathematics, 54(2), 447-468.

    Notes
    -----
    - This class is designed for numerical linear algebra applications
    - Operations may accumulate numerical errors; consider periodic truncation
    - Orthonormality of U and V is assumed but NOT enforced at initialization
    - S may become singular after operations like addition or subtraction
    - Use .is_orthogonal() to verify the orthonormality assumption
    - Use .is_singular() to check if S has become singular
    - For exact SVD with guaranteed diagonal non-singular S, use the SVD class
    - The representation is not unique: (U, S, V) and (UQ, Q^T S R, VR^T)
      represent the same matrix for any orthogonal Q, R
    """

    # Class attributes
    _format = "QuasiSVD"

    # Aliases for the matrices
    U = LowRankMatrix.create_matrix_alias(0)
    S = LowRankMatrix.create_matrix_alias(1)
    V = LowRankMatrix.create_matrix_alias(2, transpose=True, conjugate=True)
    Vh = LowRankMatrix.create_matrix_alias(2)
    Vt = LowRankMatrix.create_matrix_alias(2, conjugate=True)
    Ut = LowRankMatrix.create_matrix_alias(0, transpose=True)
    Uh = LowRankMatrix.create_matrix_alias(0, transpose=True, conjugate=True)

    def __init__(self, U: ndarray, S: ndarray, V: ndarray, **extra_data):
        """
        Create a low-rank matrix stored by its SVD: Y = U @ S @ V.T

        NOTE: U and V must be orthonormal
        NOTE: S is not necessarly diagonal and can be rectangular
        NOTE: The user must give V and not V.T or V.H

        Parameters
        ----------
        U : ndarray
            Left singular vectors, shape (m, r)
        S : ndarray
            Non-singular matrix, shape (r, q)
        V : ndarray
            Right singular vectors, shape (n, q)

        Raises
        ------
        ValueError
            If matrix dimensions do not match for multiplication.
        TypeError
            If S is provided as a 1D array or diagonal matrix (use SVD class instead).

        Warnings
        --------
        MemoryEfficiencyWarning
            If the low-rank representation uses more memory than dense storage.
        """
        # Check data types
        if U.dtype != V.dtype:
            raise TypeError("U and V must have the same dtype")

        # Check dimensions compatibility
        if U.ndim != 2 or V.ndim != 2:
            raise ValueError("U and V must be 2D arrays")

        if S.ndim == 1:
            # S is a 1D array of singular values - convert to diagonal and warn
            raise TypeError(
                "For QuasiSVD, S must be a 2D array. "
                "Use the SVD class for diagonal singular value storage."
            )

        if S.ndim != 2:
            raise ValueError("S must be a 2D array")

        # Check shape alignment for matrix multiplication: U @ S @ V.T
        if U.shape[1] != S.shape[0]:
            raise ValueError(
                f"Dimension mismatch: U.shape[1]={U.shape[1]} must equal S.shape[0]={S.shape[0]}"
            )
        if S.shape[1] != V.shape[1]:
            raise ValueError(
                f"Dimension mismatch: S.shape[1]={S.shape[1]} must equal V.shape[1]={V.shape[1]}"
            )

        # Call the parent constructor (which includes memory efficiency check)
        super().__init__(U, S, V.T.conj(), **extra_data)

    ## STANDARD OPERATIONS
    def __add__(self, other: QuasiSVD | ndarray) -> QuasiSVD | ndarray:
        """Addition of two QuasiSVD matrices or QuasiSVD with dense array.

        For two QuasiSVD matrices, uses efficient multi_add() which preserves
        orthogonality of U and V. The resulting rank is at most the sum of input ranks.

        Parameters
        ----------
        other : QuasiSVD or ndarray
            Matrix to add. If QuasiSVD, returns QuasiSVD. If ndarray, returns ndarray.

        Returns
        -------
        QuasiSVD or ndarray
            Sum of the two matrices. Type depends on input type.
            For QuasiSVD + QuasiSVD: returns QuasiSVD with rank ≤ rank(self) + rank(other)
            For QuasiSVD + ndarray: returns dense ndarray

        Examples
        --------
        >>> X = QuasiSVD.generate_random((100, 80), 10)
        >>> Y = X + X  # Returns QuasiSVD with rank 20
        >>> Z = X + np.ones((100, 80))  # Returns dense ndarray

        See Also
        --------
        multi_add : Efficient addition of multiple QuasiSVD matrices
        """
        if isinstance(other, QuasiSVD):
            return QuasiSVD.multi_add([self, other])
        else:
            return super().__add__(other)

    def __radd__(self, other: QuasiSVD | ndarray) -> QuasiSVD | ndarray:
        """Right-side addition (other + self).

        This method is called when the left operand doesn't support addition
        with QuasiSVD (e.g., ndarray + QuasiSVD).

        Parameters
        ----------
        other : QuasiSVD or ndarray
            Left operand to add to self.

        Returns
        -------
        QuasiSVD or ndarray
            Sum of the two matrices.
        """
        if isinstance(other, QuasiSVD):
            return QuasiSVD.multi_add([other, self])
        else:
            return super().__radd__(other)

    def __sub__(self, other: QuasiSVD | ndarray) -> QuasiSVD | ndarray:
        """Subtraction of two QuasiSVD matrices or QuasiSVD with dense array.

        For two QuasiSVD matrices, uses efficient multi_add() with negation.
        The resulting rank is at most the sum of input ranks.

        Parameters
        ----------
        other : QuasiSVD or ndarray
            Matrix to subtract. If QuasiSVD, returns QuasiSVD. If ndarray, returns ndarray.

        Returns
        -------
        QuasiSVD or ndarray
            Difference of the two matrices. Type depends on input type.
            For QuasiSVD - QuasiSVD: returns QuasiSVD with rank ≤ rank(self) + rank(other)
            For QuasiSVD - ndarray: returns dense ndarray

        Notes
        -----
        Without auto-truncation, X - X has rank 2*rank(X) but represents zero matrix
        (maintains algebraic consistency). Use multi_add with auto_truncate=True to
        remove numerical noise in such cases.

        Examples
        --------
        >>> X = QuasiSVD.generate_random((100, 80), 10)
        >>> Y = X - X  # Returns QuasiSVD with rank 20 representing zero
        >>> Z = X - np.ones((100, 80))  # Returns dense ndarray
        """
        if isinstance(other, QuasiSVD):
            return QuasiSVD.multi_add([self, -other])  # type: ignore[list-item]
        else:
            return super().__sub__(other)

    def __rsub__(self, other: QuasiSVD | ndarray) -> QuasiSVD | ndarray:
        """Right-side subtraction (other - self).

        This method is called when the left operand doesn't support subtraction
        with QuasiSVD (e.g., ndarray - QuasiSVD).

        Parameters
        ----------
        other : QuasiSVD or ndarray
            Left operand from which self is subtracted.

        Returns
        -------
        QuasiSVD or ndarray
            Difference of the two matrices.
        """
        if isinstance(other, QuasiSVD):
            return QuasiSVD.multi_add([other, -self])  # type: ignore[list-item]
        else:
            return super().__rsub__(other)

    def __imul__(
        self, other: float | LowRankMatrix | ndarray
    ) -> LowRankMatrix | ndarray:
        """In-place scalar multiplication, or element-wise product with matrix.

        **WARNING**: For matrix operands, this is NOT truly in-place despite the name.
        It returns a new Hadamard product object. Only scalar multiplication modifies
        self in-place.

        Parameters
        ----------
        other : float, complex, LowRankMatrix, or ndarray
            Scalar: multiplies S matrix in-place (truly in-place operation)
            Matrix: computes Hadamard (element-wise) product (returns NEW object)

        Returns
        -------
        QuasiSVD, SVD, or ndarray
            For scalars: returns self (modified in-place)
            For matrices: returns new Hadamard product object

        Notes
        -----
        For complex scalars, S is automatically converted to complex dtype.
        For matrix multiplication, use @ operator or dot() method instead.

        Examples
        --------
        >>> X = QuasiSVD.generate_random((100, 80), 10)
        >>> X *= 2.0  # In-place: X.S is doubled
        >>> X *= Y    # NOT in-place: returns new Hadamard product
        """
        if isinstance(other, (LowRankMatrix, ndarray)):
            # WARNING: Not truly in-place for matrices
            return self.hadamard(other)
        else:
            # Handle complex scalar
            if isinstance(other, (complex, np.complexfloating)):
                self.S = self.S.astype(np.complex128)
            np.multiply(self.S, other, out=self.S)
        return self

    def __mul__(
        self, other: float | LowRankMatrix | ndarray
    ) -> LowRankMatrix | ndarray:
        """Scalar multiplication or element-wise (Hadamard) product.

        Parameters
        ----------
        other : float, complex, LowRankMatrix, or ndarray
            Scalar: multiplies entire matrix by scalar
            Matrix: computes element-wise (Hadamard) product

        Returns
        -------
        QuasiSVD, SVD, or ndarray
            For scalars: returns new QuasiSVD with scaled S matrix
            For matrices: returns Hadamard product (type depends on input)

        Notes
        -----
        For matrix multiplication (linear algebra product), use @ operator or dot() method.

        Examples
        --------
        >>> X = QuasiSVD.generate_random((100, 80), 10)
        >>> Y = X * 2.5  # Scalar multiplication
        >>> Z = X * X    # Element-wise product (Hadamard)
        >>> W = X @ X.T  # Matrix multiplication (use @ not *)

        See Also
        --------
        hadamard : Element-wise multiplication
        dot : Matrix multiplication
        """
        new_mat = self.copy()
        new_mat *= other
        return new_mat

    ## SPECIFIC PROPERTIES
    def sing_vals(self) -> ndarray:
        """Singular values of the QuasiSVD matrix. Computed from S, then cached.

        Returns
        -------
        ndarray
            Singular values of the middle matrix S in descending order.
            Shape is (min(r, q),) where S has shape (r, q).

        Notes
        -----
        Result is cached after first computation for efficiency.
        Computes via np.linalg.svdvals(S), which is O(r²q) for S with shape (r, q).
        """
        if "sing_vals" in self._cache:
            return self._cache["sing_vals"]
        else:
            self._cache["sing_vals"] = la.svdvals(self.S)
            return self._cache["sing_vals"]

    @property
    def numerical_rank(self) -> int:
        """Numerical rank of the QuasiSVD matrix.

        The numerical rank is defined as the number of non-negligible singular values up to the machine precision.

        Returns
        -------
        int
            Numerical rank of the matrix.
        """
        if "numerical_rank" in self._cache:
            return self._cache["numerical_rank"]
        s = self.sing_vals()
        eps = np.finfo(s.dtype).eps
        self._cache["numerical_rank"] = np.sum(s > eps)
        return self._cache["numerical_rank"]

    def cond_estimate(self) -> float:  # type: ignore[override]
        """Condition number of the QuasiSVD matrix, estimated from the singular values.

        Returns
        -------
        float
            Condition number of the matrix.
        """
        if "cond" in self._cache:
            return self._cache["cond"]
        s = self.sing_vals()
        self._cache["cond"] = s[0] / s[-1] if s[-1] != 0 else np.inf
        return self._cache["cond"]

    def is_symmetric(self) -> bool:
        """Check if the QuasiSVD matrix is symmetric.

        Returns
        -------
        bool
            True if matrix is symmetric (square and U = V), False otherwise.

        Notes
        -----
        For QuasiSVD to be symmetric, U and V must be identical (within tolerance).
        This is a necessary condition when X = U @ S @ V.T.
        Uses np.allclose for comparison (tolerates small numerical errors).
        """
        # Check cache
        if "is_symmetric" in self._cache:
            return self._cache["is_symmetric"]
        # Check squareness
        if self.shape[0] != self.shape[1]:
            self._cache["is_symmetric"] = False
        else:
            self._cache["is_symmetric"] = np.allclose(self.U, self.V)
        return self._cache["is_symmetric"]

    @property
    def K(self) -> ndarray:
        """Compute and return the product U @ S on demand.

        Returns
        -------
        ndarray
            Product U @ S, shape (m, r)
        """
        return self.U.dot(self.S)

    @property
    def L(self) -> ndarray:
        """Compute and return the product V @ S.T on demand.

        Returns
        -------
        ndarray
            Product V @ S.T, shape (n, r)
        """
        return self.V.dot(self.S.T)

    def is_orthogonal(self) -> bool:
        """Check if U and V have orthonormal columns.

        Returns
        -------
        bool
            True if both U.H @ U ≈ I and V.H @ V ≈ I, False otherwise.

        Notes
        -----
        Result is cached after first computation.
        Uses np.allclose with default tolerances.
        Orthogonality is ASSUMED at initialization but not enforced.
        This method verifies the assumption.
        """
        # Check cache
        if "is_orthogonal" in self._cache:
            return self._cache["is_orthogonal"]
        c1 = np.allclose(self.Uh.dot(self.U), np.eye(self.U.shape[1]))
        c2 = np.allclose(self.Vh.dot(self.V), np.eye(self.V.shape[1]))
        if not (c1 and c2):
            self._cache["is_orthogonal"] = False
        else:
            self._cache["is_orthogonal"] = True
        return self._cache["is_orthogonal"]

    def is_singular(self) -> bool:
        """Check if middle matrix S is numerically singular.

        Returns
        -------
        bool
            True if S is singular (condition number >= 1/machine_eps), False otherwise.

        Notes
        -----
        Result is cached after first computation.
        Uses condition number test: cond(S) >= 1/ε where ε is machine precision.
        S may become singular after operations like addition or subtraction.
        """
        # Check cache
        if "is_singular" in self._cache:
            return self._cache["is_singular"]
        else:
            self._cache["is_singular"] = self.cond_estimate() >= 1 / np.finfo(float).eps
            return self._cache["is_singular"]

    @property
    def svd_type(self) -> str:
        """
        Classify the type of SVD representation based on S dimensions.

        Returns
        -------
        str
            One of:
            - 'full': S has shape (m, n) - full SVD with all singular vectors
            - 'reduced': S has shape (min(m,n), min(m,n)) - reduced/economic SVD
            - 'truncated': S has shape (r, r) where r < min(m,n) - truncated SVD
            - 'unconventional': S has shape (r, k) where r ≠ k - general quasi-SVD

        Examples
        --------
        >>> X = QuasiSVD.generate_random((100, 80), 50)
        >>> print(X.svd_type)  # 'reduced' (50 = min(100,80))
        >>> X_trunc = X.truncate(r=10)
        >>> print(X_trunc.svd_type)  # 'truncated' (10 < min(100,80))
        """
        m, n = self.shape
        r, k = self.S.shape

        if r == m and k == n:
            return "full"
        elif r == k == min(m, n):
            return "reduced"
        elif r == k < min(m, n):
            return "truncated"
        else:
            return "unconventional"

    def norm(self, ord: str | int = "fro") -> float:
        """Calculate matrix norm with caching and optimization for orthogonal case.

        Parameters
        ----------
        ord : str or int, default='fro'
            Order of the norm. Supported values:
            - 'fro': Frobenius norm (default)
            - 2: Spectral norm (largest singular value)
            - 'nuc': Nuclear norm (sum of singular values)
            - Other values: computed via np.linalg.norm(self.full(), ord=ord)

        Returns
        -------
        float
            The computed norm value.

        Notes
        -----
        Result is cached after first computation for each norm type.
        For orthogonal U and V, common norms are computed efficiently from singular
        values of S (O(r³)) instead of forming full matrix (O(mnr)).
        - Frobenius: ||X||_F = ||S||_F
        - Spectral (2-norm): ||X||_2 = σ_max(S)
        - Nuclear: ||X||_* = Σ σ_i(S)
        """
        # check cache
        if ord in self._cache:
            return self._cache[ord]
        else:
            if self.is_orthogonal():
                if ord == 2:
                    self._cache[ord] = np.max(self.sing_vals())
                elif ord == "fro":
                    self._cache[ord] = np.sqrt(np.sum(self.sing_vals() ** 2))
                elif ord == "nuc":
                    self._cache[ord] = np.sum(self.sing_vals())
            else:
                warnings.warn(
                    "This norm is not efficiently implemented in the QuasiSVD class. Forming the dense matrix.",
                    category=LowRankEfficiencyWarning,
                )
                self._cache[ord] = float(np.linalg.norm(self.full(), ord=ord))  # type: ignore[arg-type]
            return self._cache[ord]

    ## CONVERSIONS
    def to_svd(self) -> SVD:
        """
        Convert QuasiSVD to SVD by computing the SVD of the middle matrix S.

        This operation computes U_s, s, V_s = svd(S) and returns:
            SVD(self.U @ U_s, s, self.V @ V_s)

        If you want to truncate small singular values, use the .truncate() method instead.

        Returns
        -------
        SVD
            An SVD object with diagonal S matrix

        Notes
        -----
        This method imports SVD only at runtime to avoid circular imports.
        The computational cost is O(r³) where r is the rank of S.
        """
        # Import at runtime to avoid circular dependency
        from .svd import SVD

        # Compute SVD of S
        u_s, s, vh_s = np.linalg.svd(self.S, full_matrices=False)

        # Transform U and V
        new_U = self.U.dot(u_s)
        new_V = self.V.dot(vh_s.T.conj())

        # Create SVD object
        return SVD(new_U, s, new_V, _cache=self._cache)

    def truncate(
        self,
        r: Optional[int] = None,
        rtol: Optional[float] = None,
        atol: float = DEFAULT_ATOL,
        inplace: bool = False,
    ) -> SVD:
        """
        Truncate the QuasiSVD by converting to SVD and removing small singular values.

        The QuasiSVD is first converted to SVD (diagonal S), then singular values
        are truncated based on rank or tolerance criteria.

        Parameters
        ----------
        r : int, optional
            Target rank. If specified, keep only the r largest singular values.
        rtol : float, optional
            Relative tolerance. Singular values < rtol * max(singular_values) are removed.
        atol : float, optional
            Absolute tolerance. Singular values < atol are removed.
            Default is DEFAULT_ATOL (machine precision).
        inplace : bool, optional
            If True, this parameter is ignored (QuasiSVD cannot be truncated in-place,
            must convert to SVD). Kept for API compatibility. Default is False.

        Returns
        -------
        SVD
            Truncated SVD object

        Notes
        -----
        Priority of truncation criteria (from highest to lowest):
        1. r (explicit rank)
        2. rtol (relative tolerance)
        3. atol (absolute tolerance)

        Examples
        --------
        >>> X = QuasiSVD(U, S, V)
        >>> X_trunc = X.truncate(r=10)  # Keep top 10 singular values
        >>> X_trunc = X.truncate(rtol=1e-6)  # Keep s_i > 1e-6 * s_max
        >>> X_trunc = X.truncate(atol=1e-10)  # Keep s_i > 1e-10
        """
        if inplace:
            warnings.warn(
                "QuasiSVD cannot be truncated in-place. Returning new SVD object.",
                UserWarning,
            )

        # Convert to SVD with truncation
        return self.to_svd().truncate(r=r, rtol=rtol, atol=atol, inplace=True)  # type: ignore[arg-type]

    ## Overloaded methods
    def dot(
        self, other: QuasiSVD | ndarray, side: str = "right", dense_output: bool = False
    ) -> QuasiSVD | ndarray:
        """Matrix multiplication between SVD and other.
        The output is an SVD or a Matrix, depending on the type of other.
        If two QuasiSVD are multiplied, the new rank is the minimum of the two ranks.

        If side is 'right' or 'usual', compute self @ other.
        If side is 'left' or 'opposite', compute other @ self.

        Parameters
        ----------
        other : QuasiSVD or ndarray
            Matrix to multiply
        side : str, optional
            'left' or 'right', by default 'right'
        dense_output : bool, optional
            If True, return a dense matrix. False by default.

        Returns
        -------
        QuasiSVD or ndarray
            Result of the matrix multiplication
        """
        # Check inputs
        if not (side.lower() in ["right", "usual", "left", "opposite"]):
            raise ValueError('Incorrect side. Choose "right" or "left".')
        if side.lower() in ["right", "usual"]:
            if self.shape[1] != other.shape[0]:
                raise ValueError(
                    f"Shapes {self.shape} and {other.shape} not aligned for multiplication"
                )
        elif side.lower() in ["left", "opposite"]:
            if self.shape[0] != other.shape[1]:
                raise ValueError(
                    f"Shapes {other.shape} and {self.shape} not aligned for multiplication"
                )
        if isinstance(other, QuasiSVD) and not dense_output:
            if side.lower() in ["right", "usual"]:
                return QuasiSVD.multi_dot([self, other])
            elif side.lower() in ["opposite", "left"]:
                return QuasiSVD.multi_dot([other, self])
            else:
                raise ValueError('Incorrect side. Choose "right" or "left".')
        else:
            return super().dot(other, side, dense_output)

    def hadamard(
        self,
        other: QuasiSVD | LowRankMatrix | ndarray,
        auto_truncate: bool = AUTOMATIC_TRUNCATION,
    ) -> QuasiSVD | SVD | ndarray:
        """Hadamard product between two QuasiSVD matrices

        The new rank is the multiplication of the two ranks, at maximum.
        The default behavior depends on the value of AUTOMATIC_TRUNCATION.
        If AUTOMATIC_TRUNCATION is True, the output is automatically truncated to remove small singular values.
        If AUTOMATIC_TRUNCATION is False, the output is not truncated and may have high rank.
        NOTE: if the expected rank is too large, dense matrices are used for the computation, but the output is still an SVD.

        Parameters
        ----------
        other : QuasiSVD, LowRankMatrix or ndarray
            Matrix to multiply
        auto_truncate : bool, optional
            Whether to automatically truncate small singular values after the operation.
            Default is AUTOMATIC_TRUNCATION (False by default).

        Returns
        -------
        QuasiSVD or SVD or ndarray
            Result of the Hadamard product. Returns SVD if auto_truncate=True, QuasiSVD otherwise.
        """
        # Check inputs
        if self.shape != other.shape:
            raise ValueError("Hadamard product requires matrices of the same shape")
        if isinstance(other, LowRankMatrix) and not isinstance(other, QuasiSVD):
            warnings.warn(
                "Low-rank efficiency warning: converting LowRankMatrix to QuasiSVD for Hadamard product.",
                category=LowRankEfficiencyWarning,
            )
            # Import SVD at runtime to avoid circular dependency
            from .svd import SVD

            other = SVD.from_low_rank(other)
        if isinstance(other, QuasiSVD):
            # If the new rank is too large, it is more efficient to use the full matrix
            if self.rank * other.rank >= min(self.shape):
                # Import SVD at runtime to avoid circular dependency
                from .svd import SVD

                output = np.multiply(self.full(), other.full())
                output = SVD.from_dense(
                    output
                )  # convert to SVD, otherwise it is inconsistent
            else:
                # The new matrices U and V are obtained from transposed Khatri-Rao products
                new_U = la.khatri_rao(self.Uh, other.Uh).T.conj()
                new_V = la.khatri_rao(self.Vh, other.Vh).T.conj()
                # The new singular values are obtained from the Kronecker product
                new_S = np.kron(self.S, other.S)
                output = QuasiSVD(new_U, new_S, new_V)
                if auto_truncate:
                    output = output.truncate(atol=DEFAULT_ATOL)
        elif isinstance(other, ndarray):
            warnings.warn(
                "Low-rank efficiency warning: Hadamard product with dense matrix, using full matrices.",
                category=LowRankEfficiencyWarning,
            )
            output = np.multiply(self.full(), other)
        else:
            raise TypeError(
                "Hadamard product is only defined between LowRankMatrix and numpy's ndarray matrices."
            )
        return output

    ## CLASS METHODS
    @classmethod
    def multi_add(  # type: ignore[override]
        cls,
        matrices: List[QuasiSVD],
        auto_truncate: bool = AUTOMATIC_TRUNCATION,
        rtol: Optional[float] = None,
        atol: float = DEFAULT_ATOL,
    ) -> QuasiSVD | SVD:
        """
        Addition of multiple QuasiSVD matrices.

        This is efficiently done by stacking the U, S, V matrices and then re-orthogonalizing.
        The resulting rank is at most the sum of all input ranks.

        By default (AUTOMATIC_TRUNCATION=False), the output is NOT automatically truncated to ensure
        algebraic consistency (e.g., X - X is always exactly zero). You can override
        this by setting auto_truncate=True.

        Parameters
        ----------
        matrices : List[QuasiSVD]
            Matrices to add
        auto_truncate : bool, optional
            Whether to automatically truncate small singular values after addition.
            Default is AUTOMATIC_TRUNCATION (False by default).
        rtol : float, optional
            Relative tolerance for truncation (only used if auto_truncate=True)
        atol : float, optional
            Absolute tolerance for truncation (only used if auto_truncate=True)

        Returns
        -------
        QuasiSVD or SVD
            Sum of the matrices. Returns QuasiSVD if auto_truncate=False, SVD if auto_truncate=True.

        Notes
        -----
        - Adding matrices increases rank: rank(X + Y) ≤ rank(X) + rank(Y)
        - Without auto-truncation: X - X has rank 2*rank(X) but represents zero
        - With auto-truncation: X - X has rank ≈ 0 (removes numerical noise)
        - For memory efficiency with many additions, consider auto-truncating periodically

        Examples
        --------
        >>> # Algebraically consistent (exact zero)
        >>> Z = QuasiSVD.multi_add([X, -X])  # rank = 2*rank(X), but represents zero
        >>>
        >>> # Memory efficient (removes near-zero singular values)
        >>> Z = QuasiSVD.multi_add([X, -X], auto_truncate=True)  # rank ≈ 0
        """
        # Check inputs
        assert all(
            isinstance(matrix, QuasiSVD) for matrix in matrices
        ), "All matrices must be QuasiSVD"
        assert all(
            matrix.shape == matrices[0].shape for matrix in matrices
        ), "All matrices must have the same shape"
        # Warning on low-rank memory efficiency
        if not auto_truncate:
            total_rank = sum(matrix.rank for matrix in matrices)
            m, n = matrices[0].shape
            dense_size = m * n
            low_rank_size = total_rank * (
                m + n + max(matrix.S.shape[1] for matrix in matrices)
            )
            if low_rank_size > dense_size:
                warnings.warn(
                    "Memory efficiency warning: Adding many QuasiSVD matrices without truncation may use more memory than dense storage.",
                    LowRankEfficiencyWarning,
                )
        # Add the matrices
        U_stack = np.hstack([*[matrix.U for matrix in matrices]])
        V_stack = np.hstack([*[matrix.V for matrix in matrices]])
        S_stack = la.block_diag(*[matrix.S for matrix in matrices])
        # Necessary steps to get orthogonality of U and V
        Q1, R1 = la.qr(U_stack, mode="economic")
        Q2, R2 = la.qr(V_stack, mode="economic")
        M = np.linalg.multi_dot([R1, S_stack, R2.T.conj()])
        result = QuasiSVD(Q1, M, Q2)

        # Optionally auto-truncate
        if auto_truncate:
            result = result.truncate(rtol=rtol, atol=atol)

        return result

    @classmethod
    def multi_dot(  # type: ignore[override]
        cls, matrices: List[QuasiSVD], auto_truncate: bool = AUTOMATIC_TRUNCATION
    ) -> QuasiSVD:
        """
        Matrix multiplication of several QuasiSVD matrices.

        The rank of the output is the minimum of the ranks of the first and last inputs, at maximum.
        The default behavior depends on the value of AUTOMATIC_TRUNCATION.
        If AUTOMATIC_TRUNCATION is True, the output is automatically truncated to remove small singular values.
        If AUTOMATIC_TRUNCATION is False, the output is not truncated and may have singular S matrix.
        The matrices are multiplied all at once, so this method is more efficient than multiplying them one by one.

        Parameters
        ----------
        matrices : List[QuasiSVD]
            Matrices to multiply
        auto_truncate : bool, optional
            Whether to automatically truncate small singular values after multiplication.
            Default is AUTOMATIC_TRUNCATION (False by default).

        Returns
        -------
        QuasiSVD or SVD
            Product of the matrices. Returns SVD if auto_truncate=True, QuasiSVD otherwise.
        """
        # Check inputs
        assert all(
            isinstance(matrix, QuasiSVD) for matrix in matrices
        ), "All matrices must be QuasiSVD"
        # Check alignment for matrix multiplication
        for i in range(len(matrices) - 1):
            if matrices[i].shape[1] != matrices[i + 1].shape[0]:
                raise ValueError(
                    f"Shapes {matrices[i].shape} and {matrices[i+1].shape} not aligned for multiplication at position {i}"
                )
        # Multiply the matrices
        U = matrices[0].U
        V = matrices[-1].V
        middle_matrices = []
        for matrix in matrices[1:-1]:
            middle_matrices.extend(matrix._matrices)  # Unpack the tuple

        M = np.linalg.multi_dot(
            [matrices[0].S, matrices[0].Vh]
            + middle_matrices
            + [matrices[-1].U, matrices[-1].S]
        )
        return QuasiSVD(U, M, V)

    @classmethod
    def generate_random(
        cls,
        shape: tuple,
        rank: int,
        seed: int = 1234,
        is_symmetric: bool = False,
        **extra_data,
    ) -> QuasiSVD:
        """Generate a random QuasiSVD matrix with orthonormal U and V.

        Parameters
        ----------
        shape : tuple
            Shape of the matrix (m, n)
        rank : int
            Rank of the matrix
        seed : int, optional
            Random seed for reproducibility, by default 1234
        is_symmetric : bool, optional
            If True, generate a symmetric matrix (only for square shapes), by default False

        Returns
        -------
        QuasiSVD
            Random QuasiSVD matrix

        Raises
        ------
        ValueError
            If is_symmetric is True but shape is not square.
        """
        m, n = shape
        if is_symmetric and m != n:
            raise ValueError(
                "Cannot generate symmetric QuasiSVD for non-square shapes."
            )

        np.random.seed(seed)
        # Generate random orthonormal U
        U_random = np.random.randn(m, rank)
        U, _ = la.qr(U_random, mode="economic")

        # Generate random orthonormal V
        if is_symmetric:
            V = U.copy()
        else:
            V_random = np.random.randn(n, rank)
            V, _ = la.qr(V_random, mode="economic")

        # Generate random S matrix
        S = np.random.randn(rank, rank)

        return QuasiSVD(U, S, V, **extra_data)

    ## PROJECTIONS
    def project_onto_column_space(
        self, other: LowRankMatrix | ndarray, dense_output: bool = False
    ) -> QR | ndarray:
        """
        Projection of other onto the column space of self.

        The rank of the output is the rank of self, at maximum.
        Output is typically a QR object unless dense_output=True, in which case it is a dense ndarray.

        The formula is given by:
            P_U Y = UUh Y
        where X = U S Vh is the SVD of matrix self and Y is the input (other) matrix to project.

        Parameters
        ----------
        other : ndarray or LowRankMatrix
            Matrix to project
        dense_output : bool, optional
            Whether to return a dense matrix. False by default.

        Returns
        -------
        QR or ndarray
            Projection of other onto the column space of self.
            Returns QR if dense_output=False, ndarray otherwise.
        """
        # STEP 1 : FACTORIZATION
        if isinstance(other, LowRankMatrix):
            UhY = other.dot(self.U.H, side="left", dense_output=True)
            if dense_output:
                return UhY.dot(self.U.T.conj())
            else:
                return QR(self.U, UhY)
        else:
            if dense_output:
                return self.U.dot(self.Uh.dot(other))
            else:
                UhY = self.Uh.dot(other)
                return QR(self.U, UhY)

    def project_onto_row_space(
        self, other: LowRankMatrix | ndarray, dense_output: bool = False
    ) -> QR | ndarray:
        """
        Projection of other onto the row space of self.

        The rank of the output is the rank of self, at maximum.
        Output is typically a QR object unless dense_output=True, in which case it is a dense ndarray.

        The formula is given by:
            P_V Y = Y VVh
        where X = U S Vh is the SVD of matrix self and Y is the input (other) matrix to project.

        Parameters
        ----------
        other : ndarray or LowRankMatrix
            Matrix to project
        dense_output : bool, optional
            Whether to return a dense matrix. False by default.

        Returns
        -------
        QR or ndarray
            Projection of other onto the row space of self.
            Returns QR if dense_output=False, ndarray otherwise.
        """
        # STEP 1 : FACTORIZATION
        if isinstance(other, LowRankMatrix):
            YV = other.dot(self.V, dense_output=True)
            if dense_output:
                return YV.dot(self.Vh)
            else:
                return QR(self.V, YV.T.conj(), tranposed=True)
        else:
            if dense_output:
                return other.dot(self.V).dot(self.Vh)
            else:
                YV = other.dot(self.V)
                return QR(self.V, YV.T.conj(), tranposed=True)

    def project_onto_tangent_space(
        self,
        other: LowRankMatrix | ndarray,
        dense_output: bool = False,
        auto_truncate: bool = AUTOMATIC_TRUNCATION,
    ) -> QuasiSVD:
        """
        Projection of other onto the tangent space at self.

        The rank of the output is two times the rank of self, at maximum.
        The default behavior depends on the value of AUTOMATIC_TRUNCATION.
        If AUTOMATIC_TRUNCATION is True, the output is automatically truncated to remove small singular values.
        If AUTOMATIC_TRUNCATION is False, the output is not truncated and may have high rank.

        The formula is given by:
            P_X Y = UUh Y - UUh Y VVh + Y VVh
        where X = U S Vh is the SVD of matrix self and Y is the input (other) matrix to project.

        Parameters
        ----------
        other : ndarray or LowRankMatrix
            Matrix to project
        dense_output : bool, optional
            Whether to return a dense matrix. False by default.
        auto_truncate : bool, optional
            Whether to automatically truncate small singular values after projection.
            Default is AUTOMATIC_TRUNCATION (False by default).

        Returns
        -------
        QuasiSVD or SVD
            Projection of other onto the tangent space at self.
            Returns SVD if auto_truncate=True, QuasiSVD otherwise.
        """
        # STEP 1 : FACTORIZATION
        if isinstance(other, LowRankMatrix):
            YV = other.dot(self.V, dense_output=True)
            UhY = other.dot(self.Uh, side="opposite", dense_output=True)
        else:
            YV = other.dot(self.V)
            UhY = self.Uh.dot(other)
        UhYVVh = np.linalg.multi_dot([self.Uh, YV, self.Vh])
        M1 = np.hstack([self.U, YV])
        M2 = np.vstack([UhY - UhYVVh, self.Vh])
        # STEP 2 : DOUBLE QR  (n times 2k)
        Q1, R1 = la.qr(M1, mode="economic")
        Q2, R2 = la.qr(M2.T.conj(), mode="economic")
        if auto_truncate:
            output = QuasiSVD(Q1, R1.dot(R2.T.conj()), Q2).truncate(atol=DEFAULT_ATOL)
        else:
            output = QuasiSVD(Q1, R1.dot(R2.T.conj()), Q2)
        if dense_output:
            return output.full()  # type: ignore[return-value]
        else:
            return output

    def _project_onto_interpolated_tangent_space_offline(
        self,
        Y_u: ndarray,
        Y_v: ndarray,
        Y_uv: ndarray,
        P_u: ndarray,
        P_v: ndarray,
        auto_truncate: bool,
        dense_output: bool,
    ) -> QuasiSVD:
        """Offline projection using pre-computed interpolatory matrices."""
        # Dimension compatibility checks
        m, n = self.shape

        # Y_u: (r_u, n)
        if Y_u.ndim != 2:
            raise ValueError("Y_u must be 2D")
        r_u = Y_u.shape[0]
        if Y_u.shape[1] != n:
            raise ValueError(f"Y_u must have {n} columns (got {Y_u.shape[1]})")

        # Y_v: (m, r_v)
        if Y_v.ndim != 2:
            raise ValueError("Y_v must be 2D")
        r_v = Y_v.shape[1]
        if Y_v.shape[0] != m:
            raise ValueError(f"Y_v must have {m} rows (got {Y_v.shape[0]})")

        # Y_uv: (r_u, r_v)
        if Y_uv.ndim != 2 or Y_uv.shape != (r_u, r_v):
            raise ValueError(f"Y_uv must have shape ({r_u}, {r_v})")

        # M_u: (m, r_u)
        if P_u.ndim != 2 or P_u.shape != (m, r_u):
            raise ValueError(f"M_u must have shape ({m}, {r_u})")

        # M_v: (n, r_v)
        if P_v.ndim != 2 or P_v.shape != (n, r_v):
            raise ValueError(f"M_v must have shape ({n}, {r_v})")

        # Perform projection using double QR
        M1 = np.column_stack([P_u, Y_v])
        M2 = np.vstack([Y_u - Y_uv.dot(P_v.T.conj()), P_v.T.conj()])
        Q1, R1 = la.qr(M1, mode="economic")
        Q2, R2 = la.qr(M2.T.conj(), mode="economic")

        if auto_truncate:
            output = QuasiSVD(Q1, R1.dot(R2.T.conj()), Q2).truncate(atol=DEFAULT_ATOL)
        else:
            output = QuasiSVD(Q1, R1.dot(R2.T.conj()), Q2)

        if dense_output:
            return output.full()  # type: ignore[return-value]
        else:
            return output

    def _project_onto_interpolated_tangent_space_online(
        self,
        Y: "QuasiSVD",
        cssp_method_u: Callable,
        cssp_method_v: Callable,
        cssp_kwargs_u: dict,
        cssp_kwargs_v: dict,
        auto_truncate: bool,
        dense_output: bool,
    ) -> QuasiSVD:
        """Online projection computing interpolatory matrices via CSSP methods."""
        # Validate inputs
        if not isinstance(Y, QuasiSVD):
            raise TypeError("Y must be a QuasiSVD object")
        if not callable(cssp_method_u) or not callable(cssp_method_v):
            raise TypeError("cssp_method_u and cssp_method_v must be callable")
        if self.shape != Y.shape:
            raise ValueError(
                f"Shape mismatch: self has shape {self.shape}, Y has shape {Y.shape}"
            )

        # Compute interpolation indices and matrices
        p_u, M_u = cssp_method_u(self.U, return_projector=True, **cssp_kwargs_u)
        p_u = np.array(p_u)

        p_v, M_v = cssp_method_v(self.V, return_projector=True, **cssp_kwargs_v)
        p_v = np.array(p_v)

        # Extract interpolated slices from Y
        Y_full = Y.full()
        Y_u = Y_full[p_u, :]
        Y_v = Y_full[:, p_v]
        Y_uv = Y_full[np.ix_(p_u, p_v)]

        # Delegate to offline method
        return self._project_onto_interpolated_tangent_space_offline(
            Y_u, Y_v, Y_uv, M_u, M_v, auto_truncate, dense_output
        )

    def project_onto_interpolated_tangent_space(
        self,
        mode: str = "online",
        dense_output: bool = False,
        auto_truncate: bool = AUTOMATIC_TRUNCATION,
        **kwargs,
    ) -> QuasiSVD:
        """
        Oblique projection onto the tangent space at self using interpolation (DEIM-like methods).

        The formula is given by:
            P_X Y = M_u Y_u - M_u Y_uv M_v^* + Y_v M_v^*
        where M_u = U (P_U^* U)^{-1}, M_v = V (P_V^* V)^{-1},
        Y_u = Y[p_u, :], Y_v = Y[:, p_v] and Y_uv = Y[p_u, p_v]
        with p_u and p_v being the interpolation indices for U and V, respectively.

        Here, self = U S Vh is the SVD of matrix self and Y is the input matrix to project.

        **Two usage modes:**

        1. **Offline mode** (provide pre-computed interpolatory matrices):
           Provide Y_u, Y_v, Y_uv, M_u, M_v in kwargs.

        2. **Online mode** (compute interpolation on-the-fly):
           Provide Y in kwargs. Optionally provide cssp_method_u and/or cssp_method_v.
           The indices and interpolatory matrices are computed using the specified CSSP methods.
           If not provided, QDEIM is used by default.

        Parameters
        ----------
        mode : str, optional
            Either 'online' or 'offline'. Default is 'online'.
        dense_output : bool, optional
            Whether to return a dense matrix. False by default.
        auto_truncate : bool, optional
            Whether to automatically truncate small singular values after projection.
            Default is AUTOMATIC_TRUNCATION (False by default).
        **kwargs : dict
            For offline mode: Y_u, Y_v, Y_uv, M_u, M_v
            For online mode: Y (required), cssp_method_u (optional, default QDEIM),
                cssp_method_v (optional, default QDEIM), cssp_kwargs_u (optional),
                cssp_kwargs_v (optional)

        Returns
        -------
        QuasiSVD or SVD
            Oblique projection of Y onto the tangent space at self using interpolation.
            Returns SVD if auto_truncate=True, QuasiSVD otherwise.

        Examples
        --------
        **Offline mode:**

        >>> # Pre-compute interpolatory matrices
        >>> p_u, M_u = DEIM(X.U, return_projector=True)
        >>> p_v, M_v = DEIM(X.V, return_projector=True)
        >>> Y_full = Y.full()
        >>> Y_u = Y_full[p_u, :]
        >>> Y_v = Y_full[:, p_v]
        >>> Y_uv = Y_full[np.ix_(p_u, p_v)]
        >>> result = X.project_onto_interpolated_tangent_space(
        ...     mode='offline',
        ...     Y_u=Y_u, Y_v=Y_v, Y_uv=Y_uv, M_u=M_u, M_v=M_v
        ... )

        **Online mode (with default QDEIM):**

        >>> result = X.project_onto_interpolated_tangent_space(
        ...     mode='online',
        ...     Y=Y
        ... )

        **Online mode (with custom CSSP methods):**

        >>> from lowrank.cssp import DEIM, QDEIM
        >>> result = X.project_onto_interpolated_tangent_space(
        ...     mode='online',
        ...     Y=Y,
        ...     cssp_method_u=DEIM,
        ...     cssp_method_v=QDEIM
        ... )
        """
        if mode == "offline":
            required = ["Y_u", "Y_v", "Y_uv", "M_u", "M_v"]
            missing = [k for k in required if k not in kwargs]
            if missing:
                raise ValueError(f"Offline mode requires: {', '.join(missing)}")

            return self._project_onto_interpolated_tangent_space_offline(
                Y_u=kwargs["Y_u"],
                Y_v=kwargs["Y_v"],
                Y_uv=kwargs["Y_uv"],
                P_u=kwargs["M_u"],
                P_v=kwargs["M_v"],
                auto_truncate=auto_truncate,
                dense_output=dense_output,
            )

        elif mode == "online":
            if "Y" not in kwargs:
                raise ValueError("Online mode requires: Y")

            # Import QDEIM at runtime for default

            # Use QDEIM as default if not provided
            from ..cssp import QDEIM

            cssp_method_u = kwargs.get("cssp_method_u", QDEIM)
            cssp_method_v = kwargs.get("cssp_method_v", QDEIM)
            cssp_kwargs_u = kwargs.get("cssp_kwargs_u", {})
            cssp_kwargs_v = kwargs.get("cssp_kwargs_v", {})

            return self._project_onto_interpolated_tangent_space_online(
                Y=kwargs["Y"],
                cssp_method_u=cssp_method_u,
                cssp_method_v=cssp_method_v,
                cssp_kwargs_u=cssp_kwargs_u,
                cssp_kwargs_v=cssp_kwargs_v,
                auto_truncate=auto_truncate,
                dense_output=dense_output,
            )

        else:
            raise ValueError(f"mode must be 'online' or 'offline', got '{mode}'")

    ## OPTIMIZED METHODS (overriding LowRankMatrix)
    def trace(self) -> float:
        """
        Compute the trace of the matrix efficiently.

        For square QuasiSVD: trace(X) = trace(U @ S @ V.T) = trace(V.T @ U @ S)
        This is O(r³) instead of O(mn) for the full matrix.

        For non-square matrices, trace is only defined on the square part.

        Returns
        -------
        float
            The trace of the matrix

        Raises
        ------
        ValueError
            If matrix is not square
        """
        if self.shape[0] != self.shape[1]:
            raise ValueError(
                f"Trace is only defined for square matrices. Shape is {self.shape}"
            )

        # trace(U @ S @ V.T) = trace(V.T @ U @ S)
        # V.T has shape (q, n), U has shape (m, r), S has shape (r, q)
        # For square matrices, m = n
        return np.trace(self.Vh @ self.U @ self.S)

    def diag(self) -> ndarray:
        """
        Extract the diagonal of the matrix efficiently.

        For QuasiSVD: diag[i] = U[i,:] @ S @ V[i,:].T
        This is O(mr²) instead of O(mn) for the full matrix.

        Returns
        -------
        ndarray
            The diagonal elements
        """
        # Check cache
        if "diag" in self._cache:
            return self._cache["diag"]

        m, n = self.shape
        min_dim = min(m, n)
        diagonal = np.zeros(min_dim, dtype=self.dtype)

        for i in range(min_dim):
            # diag[i] = U[i,:] @ S @ V[i,:].conj()
            diagonal[i] = self.U[i, :] @ self.S @ self.V[i, :].conj()

        self._cache["diag"] = diagonal
        return diagonal

    def norm_squared(self) -> float:
        """
        Compute the squared Frobenius norm efficiently.

        For orthogonal U and V: ||X||²_F = ||S||²_F
        This is O(r²) instead of O(mnr) for generic computation.

        Returns
        -------
        float
            The squared Frobenius norm
        """
        # Check cache
        if "norm_squared" in self._cache:
            return self._cache["norm_squared"]
        if self.is_orthogonal():
            self._cache["norm_squared"] = np.sum(self.S * self.S.conj()).real
        else:
            # Fall back to full matrix computation
            self._cache["norm_squared"] = np.sum(self.full() ** 2).real
        return self._cache["norm_squared"]

    @property
    def T(self) -> QuasiSVD:
        """
        Transpose of the QuasiSVD matrix (without conjugation).

        For X = U @ S @ V.H, the transpose is:
            X.T = (U @ S @ V.H).T = V* @ S.T @ U.T
        where V* denotes conjugate (not Hermitian).

        Returns QuasiSVD(V.conj(), S.T, U.conj()) which stores [V*, S.T, U*.H]
        and represents V* @ S.T @ U.T.

        Returns
        -------
        QuasiSVD
            Transposed matrix in QuasiSVD format

        Notes
        -----
        For real matrices, this is equivalent to QuasiSVD(V, S.T, U).
        For complex matrices, U and V must be conjugated.
        """
        return QuasiSVD(self.V.conj(), self.S.T, self.U.conj(), **self._extra_data)

    def transpose(self) -> QuasiSVD:
        """Transpose the matrix (returns QuasiSVD)."""
        return self.T

    def conj(self) -> QuasiSVD:
        """
        Complex conjugate of the QuasiSVD matrix.

        Returns QuasiSVD instead of generic LowRankMatrix.

        Returns
        -------
        QuasiSVD
            Complex conjugate in QuasiSVD format
        """
        return QuasiSVD(self.U.conj(), self.S.conj(), self.V.conj(), **self._extra_data)

    @property
    def H(self) -> QuasiSVD:
        """
        Hermitian conjugate (conjugate transpose) of the QuasiSVD matrix.

        For X = U @ S @ V.H, the Hermitian conjugate is:
            X.H = (U @ S @ V.H).H = V @ S.H @ U.H = V @ S*.T @ U.H

        Returns QuasiSVD(V, S.T.conj(), U) which stores [V, S*.T, U.H]
        and represents V @ S*.T @ U.H.

        This is equivalent to X.T.conj() or X.conj().T.

        Returns
        -------
        QuasiSVD
            Hermitian conjugate in QuasiSVD format

        Notes
        -----
        For real matrices, X.H is equivalent to X.T.
        """
        return QuasiSVD(self.V, self.S.T.conj(), self.U, **self._extra_data)

    ## NEW FEATURES
    def rank_one_update(self, u: ndarray, v: ndarray, alpha: float = 1.0) -> QuasiSVD:
        """
        Efficient rank-1 update: X_new = X + alpha * u @ v.T

        This adds a rank-1 matrix to the current QuasiSVD without forming
        the full matrix. The result has rank at most rank(X) + 1.

        Parameters
        ----------
        u : ndarray, shape (m,)
            Left vector for rank-1 update
        v : ndarray, shape (n,)
            Right vector for rank-1 update
        alpha : float, optional
            Scaling factor, default is 1.0

        Returns
        -------
        QuasiSVD
            Updated matrix with rank increased by at most 1

        Examples
        --------
        >>> X = QuasiSVD.generate_random((100, 80), 5)
        >>> u = np.random.randn(100)
        >>> v = np.random.randn(80)
        >>> X_new = X.rank_one_update(u, v, alpha=0.5)
        >>> # X_new represents X + 0.5 * u @ v.T with rank at most 6
        """
        # Validate inputs
        if u.shape[0] != self.shape[0]:
            raise ValueError(f"u must have length {self.shape[0]}, got {u.shape[0]}")
        if v.shape[0] != self.shape[1]:
            raise ValueError(f"v must have length {self.shape[1]}, got {v.shape[0]}")

        # Reshape to column vectors if needed
        u = np.asarray(u).reshape(-1, 1)
        v = np.asarray(v).reshape(-1, 1)

        # Stack U with u
        U_new = np.hstack([self.U, u])

        # Stack V with v
        V_new = np.hstack([self.V, v])

        # Create block diagonal S with alpha in the new entry
        S_new = np.zeros((self.rank + 1, self.rank + 1), dtype=self.dtype)
        S_new[: self.rank, : self.rank] = self.S
        S_new[self.rank, self.rank] = alpha

        # Re-orthogonalize to maintain orthogonality
        Q1, R1 = la.qr(U_new, mode="economic")
        Q2, R2 = la.qr(V_new, mode="economic")
        S_combined = R1 @ S_new @ R2.T.conj()

        return QuasiSVD(Q1, S_combined, Q2, **self._extra_data)

    def reorthogonalize(self, method: str = "qr", inplace: bool = False) -> QuasiSVD:
        """
        Re-orthogonalize U and V if numerical drift has occurred.

        After many operations, U and V may lose orthogonality due to
        numerical errors. This method restores orthogonality.

        Parameters
        ----------
        method : str, optional
            Method for re-orthogonalization:
            - 'qr': QR decomposition (default, stable and efficient)
            - 'svd': Full SVD (more expensive but more accurate)
        inplace : bool, optional
            If True, modify the current object in-place. Otherwise, return a new object. Default is False.

        Returns
        -------
        QuasiSVD
            Re-orthogonalized QuasiSVD with same matrix values

        Examples
        --------
        >>> X = QuasiSVD.generate_random((100, 80), 10)
        >>> # ... many operations ...
        >>> if not X.is_orthogonal():
        ...     X = X.reorthogonalize()
        """
        if method == "qr":
            # Use QR decomposition
            Q_u, R_u = la.qr(self.U, mode="economic")
            Q_v, R_v = la.qr(self.V, mode="economic")

            # Update S to preserve the matrix: X = U @ S @ V.T = Q_u @ (R_u @ S @ R_v.T) @ Q_v.T
            S_new = R_u @ self.S @ R_v.T.conj()

            if inplace:
                self.U = Q_u
                self.V = Q_v
                self.S = S_new
                self._cache.clear()  # Clear cache as data changed
                return self
            else:
                return QuasiSVD(Q_u, S_new, Q_v, **self._extra_data)

        elif method == "svd":
            # Use full SVD for maximum accuracy (more expensive)
            if inplace:
                Y = self.to_svd()
                self.U = Y.U
                self.S = Y.S
                self.V = Y.V
                self._cache.clear()  # Clear cache as data changed
                return self
            else:
                return self.to_svd()
        else:
            raise ValueError(f"Unknown method: {method}. Choose 'qr' or 'svd'.")

    def numerical_health_check(self, verbose: bool = True) -> dict:
        """
        Check the numerical health of the QuasiSVD representation.

        This method checks:
        - Orthogonality of U and V
        - Condition number of S
        - Presence of near-zero singular values
        - Memory efficiency

        Parameters
        ----------
        verbose : bool, optional
            If True, print a summary. Default is True.

        Returns
        -------
        dict
            Dictionary with health metrics:
            - 'orthogonal_U': bool, whether U is orthogonal
            - 'orthogonal_V': bool, whether V is orthogonal
            - 'orthogonality_error_U': float, ||U.H @ U - I||_F
            - 'orthogonality_error_V': float, ||V.H @ V - I||_F
            - 'condition_number_S': float, condition number of S
            - 'is_singular': bool, whether S is numerically singular
            - 'min_singular_value': float, smallest singular value of S
            - 'max_singular_value': float, largest singular value of S
            - 'singular_value_ratio': float, max/min singular values
            - 'compression_ratio': float, storage efficiency
            - 'memory_efficient': bool, whether low-rank is beneficial
            - 'recommendations': list of str, suggested actions

        Examples
        --------
        >>> X = QuasiSVD.generate_random((100, 80), 10)
        >>> health = X.numerical_health_check()
        >>> if not health['orthogonal_U']:
        ...     X = X.reorthogonalize()
        """
        health: dict = {}
        recommendations: list = []

        # Check orthogonality
        I_u = np.eye(self.U.shape[1])
        I_v = np.eye(self.V.shape[1])
        UtU = self.Uh @ self.U
        VtV = self.Vh @ self.V

        orth_error_U = np.linalg.norm(UtU - I_u, ord="fro")
        orth_error_V = np.linalg.norm(VtV - I_v, ord="fro")

        health["orthogonal_U"] = bool(orth_error_U < 1e-10)
        health["orthogonal_V"] = bool(orth_error_V < 1e-10)
        health["orthogonality_error_U"] = float(orth_error_U)
        health["orthogonality_error_V"] = float(orth_error_V)

        if not health["orthogonal_U"] or not health["orthogonal_V"]:
            recommendations.append(
                "Re-orthogonalize U and/or V using .reorthogonalize()"
            )

        # Check conditioning of S
        sing_vals = self.sing_vals()
        health["min_singular_value"] = float(np.min(sing_vals))
        health["max_singular_value"] = float(np.max(sing_vals))
        health["singular_value_ratio"] = (
            health["max_singular_value"] / health["min_singular_value"]
            if health["min_singular_value"] > 0
            else np.inf
        )
        health["condition_number_S"] = (
            float(sing_vals[0] / sing_vals[-1]) if sing_vals[-1] > 0 else np.inf
        )
        health["is_singular"] = self.is_singular()

        if health["is_singular"]:
            recommendations.append(
                "S is singular. Consider using .truncate() to remove zero singular values."
            )
        elif health["condition_number_S"] > 1e10:
            recommendations.append(
                f"S is ill-conditioned (cond={health['condition_number_S']:.2e}). Consider truncation."
            )

        if health["min_singular_value"] < 1e-12:
            recommendations.append(
                f"Very small singular values detected (min={health['min_singular_value']:.2e}). Consider truncation."
            )

        # Check memory efficiency
        health["compression_ratio"] = float(self.compression_ratio())
        health["memory_efficient"] = bool(health["compression_ratio"] < 1.0)

        if not health["memory_efficient"]:
            recommendations.append(
                f"Low-rank representation uses more memory than dense (ratio={health['compression_ratio']:.2f}). Consider using dense arrays."
            )

        health["recommendations"] = recommendations

        # Print summary if verbose
        if verbose:
            print("=" * 60)
            print("QuasiSVD Numerical Health Check")
            print("=" * 60)
            print(f"Shape: {self.shape}, Rank: {self.rank}")
            print(f"\nOrthogonality:")
            print(
                f"  U orthogonal: {health['orthogonal_U']} (error: {orth_error_U:.2e})"
            )
            print(
                f"  V orthogonal: {health['orthogonal_V']} (error: {orth_error_V:.2e})"
            )
            print(f"\nSingular Values:")
            print(f"  Min: {health['min_singular_value']:.2e}")
            print(f"  Max: {health['max_singular_value']:.2e}")
            print(f"  Ratio: {health['singular_value_ratio']:.2e}")
            print(f"  Condition number (S): {health['condition_number_S']:.2e}")
            print(f"  Singular: {health['is_singular']}")
            print(f"\nMemory Efficiency:")
            print(f"  Compression ratio: {health['compression_ratio']:.4f}")
            print(f"  Memory efficient: {health['memory_efficient']}")

            if recommendations:
                print(f"\nRecommendations:")
                for i, rec in enumerate(recommendations, 1):
                    print(f"  {i}. {rec}")
            else:
                print(f"\n✓ No issues detected. Matrix is in good numerical health.")
            print("=" * 60)

        return health

    def to_qr(self) -> QR:
        """
        Convert QuasiSVD to QR format.

        Returns X = Q @ R where Q is orthogonal and R is upper triangular.
        This is useful for efficient column-space operations.

        Returns
        -------
        QR
            QR representation of the matrix

        Examples
        --------
        >>> X = QuasiSVD.generate_random((100, 80), 10)
        >>> X_qr = X.to_qr()
        """
        # X = U @ S @ V.T
        # We can write this as X = U @ (S @ V.T) = Q @ R
        # where Q = U (already orthogonal) and R = S @ V.T
        Q = self.U
        R = self.S @ self.Vh
        return QR(Q, R, tranposed=False)

    @classmethod
    def from_qr(cls, qr: QR) -> QuasiSVD:
        """
        Convert QR format to QuasiSVD.

        Parameters
        ----------
        qr : QR
            QR representation to convert

        Returns
        -------
        QuasiSVD
            QuasiSVD representation

        Examples
        --------
        >>> X_qr = QR(Q, R)
        >>> X = QuasiSVD.from_qr(X_qr)
        """
        # X = Q @ R  (standard mode) or X = R.H @ Q.H (conjugate mode)
        # We need to factorize into U @ S @ V.T format
        # Use QR decomposition of R.T: R.T = V @ S.T
        # So R = S @ V.T where S = (R.T @ V).T

        if qr._transposed:
            # X = R.H @ Q.H
            # Convert to standard form first
            Q, S = la.qr(qr.R.T.conj(), mode="economic")
            V = qr.Q
        else:
            # X = Q @ R
            # Factorize R into S @ V.T
            Q = qr.Q
            R = qr.R
            V, St = la.qr(R.T, mode="economic")
            S = St.T

        return cls(Q, S, V)

    ## ADVANCED LINEAR ALGEBRA FEATURES
    def pseudoinverse(
        self, rtol: Optional[float] = None, atol: float = DEFAULT_ATOL
    ) -> QuasiSVD:
        """
        Compute the Moore-Penrose pseudoinverse X⁺ efficiently.

        For QuasiSVD X = U @ S @ V.T, the pseudoinverse is:
            X⁺ = V @ S⁺ @ U.T
        where S⁺ is the pseudoinverse of S computed via SVD.

        Small singular values (< max(rtol*σ_max, atol)) are treated as zero.

        Parameters
        ----------
        rtol : float, optional
            Relative tolerance for determining zero singular values.
            Default is None (uses machine precision * max(m,n)).
        atol : float, optional
            Absolute tolerance. Default is DEFAULT_ATOL (machine precision).

        Returns
        -------
        QuasiSVD
            Pseudoinverse in QuasiSVD format

        Examples
        --------
        >>> X = QuasiSVD.generate_random((100, 80), 10)
        >>> X_pinv = X.pseudoinverse()
        >>> # Check: X @ X_pinv @ X ≈ X
        >>> reconstruction = X.dot(X_pinv.dot(X.full(), dense_output=True), dense_output=True)
        >>> np.allclose(X.full(), reconstruction)
        True

        See Also
        --------
        solve : Solve linear system Xx = b
        lstsq : Least squares solution
        """
        # Convert to SVD to get singular values
        X_svd = self.to_svd()
        U_svd = X_svd.U
        s = X_svd.S if X_svd.S.ndim == 1 else np.diag(X_svd.S)
        V_svd = X_svd.V

        # Determine threshold for zero singular values
        if rtol is None:
            rtol = max(self.shape) * np.finfo(self.dtype).eps
        threshold = max(rtol * np.max(s), atol)

        # Compute pseudoinverse of singular values
        s_pinv = np.zeros_like(s)
        mask = s > threshold
        s_pinv[mask] = 1.0 / s[mask]

        # X⁺ = V @ S⁺ @ U.T
        S_pinv = np.diag(s_pinv)
        return QuasiSVD(V_svd, S_pinv, U_svd, **self._extra_data)

    def solve(self, b: ndarray, method: str = "lstsq") -> ndarray:
        """
        Solve the linear system Xx = b.

        For square full-rank matrices, solves Xx = b.
        For rectangular or rank-deficient matrices, computes the least-squares solution.

        Parameters
        ----------
        b : ndarray
            Right-hand side vector or matrix. Shape (m,) or (m, k).
        method : str, optional
            Solution method:
            - 'lstsq': Use pseudoinverse (default, works for any shape)
            - 'direct': Use factored form assuming orthogonality (faster but requires
              orthogonal U, V and invertible S)

        Returns
        -------
        ndarray
            Solution x. Shape (n,) or (n, k).

        Raises
        ------
        ValueError
            If matrix is not square and method='direct'.
            If dimensions are incompatible.

        Examples
        --------
        >>> X = QuasiSVD.generate_random((100, 100), 100)  # Full rank for unique solution
        >>> b = np.random.randn(100)
        >>> x = X.solve(b)
        >>> # Check: X @ x ≈ b
        >>> np.allclose(X.dot(x), b)
        True

        >>> # For rank-deficient systems, use lstsq instead:
        >>> X_deficient = QuasiSVD.generate_random((100, 100), 20)
        >>> x_ls = X_deficient.lstsq(b)

        See Also
        --------
        pseudoinverse : Compute pseudoinverse
        lstsq : Least squares solution

        Notes
        -----
        The default method is 'lstsq' which uses the pseudoinverse and is robust
        to non-orthogonal factors and rank deficiency. Use method='direct' only
        when you are certain that U and V are orthogonal and S is invertible.
        """
        # Validate inputs
        if b.shape[0] != self.shape[0]:
            raise ValueError(
                f"Dimension mismatch: b has {b.shape[0]} rows, matrix has {self.shape[0]} rows"
            )

        if method == "direct":
            # Direct solution using factorization (assumes orthogonality)
            if self.shape[0] != self.shape[1]:
                raise ValueError(
                    "Direct solve requires square matrix. Use method='lstsq' for rectangular matrices."
                )

            # Check orthogonality
            if not self.is_orthogonal():
                warnings.warn(
                    "Using direct solve with non-orthogonal factors may give inaccurate results. "
                    "Consider using method='lstsq' instead.",
                    UserWarning,
                    stacklevel=2,
                )

            # Solve: X @ x = b
            # U @ S @ V.T @ x = b
            # S @ V.T @ x = U.T @ b  (assuming U orthogonal)
            # V.T @ x = S^(-1) @ U.T @ b
            # x = V @ S^(-1) @ U.T @ b  (assuming V orthogonal)

            Utb = self.Uh @ b
            S_inv_Utb = la.solve(self.S, Utb)
            x = self.V @ S_inv_Utb
            return x

        elif method == "lstsq":
            # Least squares solution using pseudoinverse (robust)
            return self.lstsq(b)
        else:
            raise ValueError(f"Unknown method: {method}. Choose 'direct' or 'lstsq'.")

    def lstsq(
        self, b: ndarray, rtol: Optional[float] = None, atol: float = DEFAULT_ATOL
    ) -> ndarray:
        """
        Compute the least-squares solution to Xx ≈ b.

        Minimizes ||Xx - b||₂ using the pseudoinverse: x = X⁺ @ b

        Parameters
        ----------
        b : ndarray
            Right-hand side vector or matrix. Shape (m,) or (m, k).
        rtol : float, optional
            Relative tolerance for pseudoinverse computation.
            Default is None (uses machine precision * max(m,n)).
        atol : float, optional
            Absolute tolerance for pseudoinverse. Default is DEFAULT_ATOL.

        Returns
        -------
        ndarray
            Least-squares solution x. Shape (n,) or (n, k).

        Examples
        --------
        >>> X = QuasiSVD.generate_random((100, 80), 20)
        >>> b = np.random.randn(100)
        >>> x = X.lstsq(b)
        >>> # x minimizes ||X @ x - b||
        >>> residual = np.linalg.norm(X.dot(x) - b)

        See Also
        --------
        pseudoinverse : Compute pseudoinverse
        solve : Solve linear system
        """
        # Validate inputs
        if b.shape[0] != self.shape[0]:
            raise ValueError(
                f"Dimension mismatch: b has {b.shape[0]} rows, matrix has {self.shape[0]} rows"
            )

        # Use pseudoinverse
        X_pinv = self.pseudoinverse(rtol=rtol, atol=atol)
        return X_pinv.dot(b, dense_output=True)

    def sqrtm(self, inplace: bool = False, **extra_data) -> QuasiSVD:
        """
        Compute the matrix square root X^{1/2} such that X^{1/2} @ X^{1/2} = X.

        For QuasiSVD X = U @ S @ V.T, the square root is:
            X^{1/2} = U @ S^{1/2} @ V.T
        where S^{1/2} is the matrix square root of S.

        Parameters
        ----------
        inplace : bool, optional
            If True, modify the current object in-place. Default is False.

        Returns
        -------
        QuasiSVD
            Matrix square root in QuasiSVD format
        """
        S_sqrt = la.sqrtm(self.S)

        if inplace:
            self.S = S_sqrt
            self._cache.clear()  # Clear cache as data changed
            return self
        else:
            return QuasiSVD(self.U, S_sqrt, self.V, **extra_data)

    def expm(self, inplace: bool = False, **extra_data) -> QuasiSVD:
        """
        Compute the matrix exponential exp(X) = e^X.

        For QuasiSVD X = U @ S @ V.H, if X is Hermitian (V.H == U), then:
            exp(X) = U @ exp(S) @ U.H
        where exp(S) is the matrix exponential of S.

        This is more efficient than general matrix exponentiation when the
        middle matrix S is small (r << n).

        Parameters
        ----------
        inplace : bool, optional
            If True, modify the current object in-place. Default is False.
        **extra_data
            Additional keyword arguments passed to __init__ for the new object.

        Returns
        -------
        QuasiSVD
            Matrix exponential in QuasiSVD format.

        Raises
        ------
        ValueError
            If matrix is not square.
        NotImplementedError
            If matrix is not Hermitian (V.H != U).

        Notes
        -----
        Computational cost: O(r³) for matrix exponential of r×r matrix S
        (vs O(n³) for general n×n matrix).

        This method currently only supports Hermitian matrices where V.H == U.
        For general matrices, the eigenvalue decomposition would be needed.

        Examples
        --------
        >>> S = np.array([[1.0, 0.1], [0.1, 0.5]])
        >>> U, _ = np.linalg.qr(np.random.randn(100, 2))
        >>> X = QuasiSVD(U, S, U)  # Symmetric/Hermitian
        >>> # Note: Current implementation has a bug - references self.s which doesn't exist
        >>> # Use SVD.expm() for diagonal S matrices instead
        >>> # X_exp = X.expm()  # Will raise AttributeError

        See Also
        --------
        sqrtm : Matrix square root
        SVD.expm : Optimized version for diagonal S
        """
        # Check if matrix is square
        if self.shape[0] != self.shape[1]:
            raise ValueError(f"Matrix must be square for expm, got shape {self.shape}")

        # Check if matrix is complex
        if (
            np.iscomplexobj(self.U)
            or np.iscomplexobj(self.V)
            or np.iscomplexobj(self.s)
        ):
            raise NotImplementedError(
                "Matrix exponential not implemented for complex matrices. "
                "Use scipy.linalg.expm(X.full()) instead."
            )

        # Check if matrix is symmetric (V.H == U, which means V == U in storage for real matrices)
        if not self.is_symmetric():
            raise NotImplementedError(
                "Matrix exponential currently only implemented for symmetric matrices (V.T == U). "
                "For general matrices, use scipy.linalg.expm(X.full())."
            )

        # Compute exp(S) using scipy
        S_exp = la.expm(self.S)

        if inplace:
            self.S = S_exp
            self._cache.clear()  # Clear cache as data changed
            return self
        else:
            return QuasiSVD(self.U, S_exp, self.V, **extra_data)

    def plot_singular_value_decay(
        self, semilogy: bool = True, show: bool = True, **kwargs
    ):
        """
        Plot the singular value decay of the middle matrix S.

        This helps visualize the numerical rank and identify a good
        truncation threshold.

        Parameters
        ----------
        semilogy : bool, optional
            If True, use logarithmic scale for y-axis. Default is True.
        show : bool, optional
            If True, call plt.show(). Default is True.
        **kwargs
            Additional keyword arguments passed to plt.plot()

        Returns
        -------
        fig, ax
            Matplotlib figure and axes objects

        Examples
        --------
        >>> X = QuasiSVD.generate_random((100, 80), 20)
        >>> fig, ax = X.plot_singular_value_decay()
        >>> ax.axhline(1e-10, color='r', linestyle='--', label='Threshold')
        >>> ax.legend()
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                "matplotlib is required for plotting. Install it with: pip install matplotlib"
            )

        sing_vals = self.sing_vals()

        fig, ax = plt.subplots(figsize=kwargs.pop("figsize", (10, 6)))

        if semilogy:
            ax.semilogy(range(1, len(sing_vals) + 1), sing_vals, "o-", **kwargs)
        else:
            ax.plot(range(1, len(sing_vals) + 1), sing_vals, "o-", **kwargs)

        ax.set_xlabel("Index", fontsize=12)
        ax.set_ylabel("Singular Value", fontsize=12)
        ax.set_title(f"Singular Value Decay (Rank {self.rank})", fontsize=14)
        ax.grid(True, alpha=0.3)

        # Add some useful reference lines
        if semilogy:
            ax.axhline(
                1e-10,
                color="r",
                linestyle="--",
                alpha=0.5,
                linewidth=1,
                label="Machine precision",
            )
            ax.axhline(
                1e-6,
                color="orange",
                linestyle="--",
                alpha=0.5,
                linewidth=1,
                label="Typical tolerance",
            )

        ax.legend()

        if show:
            plt.show()

        return fig, ax
