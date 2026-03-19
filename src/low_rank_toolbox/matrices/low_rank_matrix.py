"""Generic low-rank matrix base class and utilities.

Authors: Benjamin Carrel and Rik Vorhaar
         University of Geneva, 2022-2025
"""

# Import packages
from __future__ import annotations

import warnings
from copy import deepcopy
from typing import List, Optional, Sequence

import numpy as np
import scipy.sparse.linalg as spala
from numpy import ndarray
from scipy.sparse import spmatrix
from scipy.sparse.linalg import LinearOperator


class LowRankEfficiencyWarning(Warning):
    """Warning for inefficient operations on low-rank matrices."""

    pass


class MemoryEfficiencyWarning(Warning):
    """Warning when low-rank representation uses more memory than dense storage."""

    pass


warnings.simplefilter("once", LowRankEfficiencyWarning)
warnings.simplefilter("once", MemoryEfficiencyWarning)


# %% Define class LowRankMatrix
class LowRankMatrix(LinearOperator):
    """
    Meta class for dealing with low rank matrices in different formats.

    Inherits from scipy.sparse.linalg.LinearOperator for seamless integration
    with iterative solvers and standard numerical linear algebra tools.

    Do not use this class directly, but rather use its subclasses.

    We always decompose a matrix as a product of smaller matrices. These smaller
    matrices are stored in ``self._matrices``.

    LinearOperator Integration
    ---------------------------
    As a subclass of scipy.sparse.linalg.LinearOperator, LowRankMatrix provides
    efficient matrix-vector and matrix-matrix products without forming the full
    dense matrix. This enables direct use with scipy's iterative solvers and
    other numerical algorithms.

    Key LinearOperator features:
    - Matrix-vector multiplication: A @ v or A.matvec(v)
    - Adjoint multiplication: A.H @ v or A.rmatvec(v)
    - Matrix-matrix multiplication: A @ B or A.matmat(B)
    - Lazy composition: A + B, A @ B (when B is LinearOperator)
    - Shape and dtype attributes: A.shape, A.dtype

    Examples
    --------
    Using with scipy iterative solvers:

    >>> from scipy.sparse.linalg import gmres, LinearOperator
    >>> from low_rank_toolbox import SVD
    >>> import numpy as np
    >>>
    >>> # Create a well-conditioned low-rank matrix
    >>> U, _ = np.linalg.qr(np.random.randn(1000, 10))
    >>> s = np.logspace(0, -1, 10)  # Better conditioned
    >>> V, _ = np.linalg.qr(np.random.randn(1000, 10))
    >>> # Add diagonal regularization for better conditioning
    >>> A = SVD(U, s, V)
    >>> A_reg = A + 0.1 * LinearOperator((1000, 1000), matvec=lambda x: x)
    >>>
    >>> # Solve Ax = b using GMRES (never forms the full matrix)
    >>> b = np.random.randn(1000)
    >>> x, info = gmres(A_reg, b, rtol=1e-6, atol=1e-6, maxiter=100)
    >>> assert info == 0  # Verify convergence

    Lazy composition with other operators:

    >>> from scipy.sparse import diags
    >>> from scipy.sparse.linalg import aslinearoperator
    >>>
    >>> # Create a diagonal operator for better conditioning
    >>> D = diags([0.5 for i in range(1000)])
    >>> D_op = aslinearoperator(D)
    >>>
    >>> # Lazy sum - doesn't form full matrix
    >>> B = A + D_op  # Returns _SumLinearOperator
    >>>
    >>> # Use in iterative solver
    >>> x2, info2 = gmres(B, b, rtol=1e-6, atol=1e-6, maxiter=100)
    >>> assert info2 == 0  # Verify convergence

    Matrix-vector products (efficient, no dense formation):

    >>> v = np.random.randn(1000)
    >>> y = A @ v  # Efficient: never forms full 1000x1000 matrix
    >>>
    >>> # Adjoint product
    >>> z = A.H @ v  # Hermitian transpose product

    Custom preconditioners:

    >>> # Use the regularized matrix from above
    >>> def precondition(v):
    ...     # Simple diagonal preconditioner
    ...     return v
    >>>
    >>> # Create LinearOperator from function
    >>> M = LinearOperator((1000, 1000), matvec=precondition)
    >>>
    >>> # Use as preconditioner
    >>> x3, info3 = gmres(A_reg, b, M=M, rtol=1e-6, atol=1e-6, maxiter=100)
    >>> assert info3 == 0  # Verify convergence

    Notes
    -----
    - All matrix-vector products are computed efficiently in O(rank) operations
    - Full matrix is never formed unless explicitly requested with .full()
    - Subclasses (SVD, QR, QuasiSVD) inherit all LinearOperator functionality
    - Compatible with all scipy.sparse.linalg iterative solvers (gmres, cg, bicgstab, etc.)
    """

    _format = "generic"
    # Set high array priority so numpy prefers our methods over its own for binary operations
    __array_priority__ = 1000

    def __init__(
        self,
        *matrices: ndarray,
        **extra_data,
    ):
        """Initialize a low-rank matrix from a sequence of factor matrices.

        Parameters
        ----------
        *matrices : ndarray
            Sequence of matrices whose product forms the low-rank matrix.
            At least two matrices must be provided, and their shapes must align
            for matrix multiplication (i.e., matrices[i].shape[1] == matrices[i+1].shape[0]).
        **extra_data : dict
            Additional data to store with the matrix (e.g., poles, residues).

        Raises
        ------
        ValueError
            If fewer than 2 matrices are provided or if matrix shapes do not align.
        """
        # Convert so values can be changed.
        self._matrices: List[ndarray] = list(matrices)
        self._extra_data = extra_data
        # Sanity check
        if len(self._matrices) < 2:
            raise ValueError(
                "At least two matrices must be provided for a low-rank factorization."
            )
        for i in range(len(self._matrices) - 1):
            if self._matrices[i].shape[1] != self._matrices[i + 1].shape[0]:
                raise ValueError(
                    f"Matrix shapes do not align: "
                    f"{self._matrices[i].shape} and {self._matrices[i + 1].shape}."
                )

        # Initialize parent LinearOperator
        shape = (self._matrices[0].shape[0], self._matrices[-1].shape[-1])
        dtype = self._matrices[0].dtype
        super().__init__(dtype=dtype, shape=shape)

        # Initialize cache dictionary BEFORE computing storage size (if not provided in extra_data)
        if "_cache" in extra_data:
            self._cache = extra_data["_cache"]
        else:
            self._cache = {}

        # Check memory efficiency after initialization (unless disabled)
        if not extra_data.get("_skip_memory_check", False):
            m, n = self.shape
            dense_size = m * n
            lowrank_size = self._compute_storage_size()

            # Only check if matrix is non-empty
            if dense_size > 0 and lowrank_size >= dense_size:
                compression_ratio = lowrank_size / dense_size
                warnings.warn(
                    f"Memory inefficiency: {self.__class__.__name__} format uses {compression_ratio:.2f}x "
                    f"the memory of dense storage ({lowrank_size:,} vs {dense_size:,} elements). "
                    f"Consider using dense array instead, or reduce the rank.",
                    MemoryEfficiencyWarning,
                    stacklevel=2,
                )

    ## PROPERTIES
    @property
    def rank(self) -> int:
        """Rank of the low-rank factorization.

        Returns
        -------
        int
            The maximum possible rank given the factorization structure,
            computed as the minimum dimension across all factor matrices.

        Notes
        -----
        This is an upper bound on the true numerical rank. The actual rank
        may be lower if the factors are rank-deficient. For the true numerical
        rank, compute SVD of the full matrix.
        """
        return min(min(M.shape) for M in self._matrices)

    @property
    def length(self) -> int:
        """Number of factor matrices in the factorization.

        Returns
        -------
        int
            The number of matrices in the product chain.
        """
        return len(self._matrices)

    @property
    def deepshape(self) -> tuple:
        """Shape tuple including all intermediate dimensions of the factorization.

        Returns
        -------
        tuple
            Tuple of dimensions showing the shape of each factor in the chain.
            For factors A (m×k), B (k×n), returns (m, k, n).
        """
        if not self._matrices:
            return ()
        return tuple(M.shape[0] for M in self._matrices) + (
            self._matrices[-1].shape[-1],
        )

    @property
    def ndim(self) -> int:
        """Number of dimensions (always 2 for matrices)."""
        return len(self.shape)

    @property
    def size(self) -> int:
        """Total number of elements stored in the factorization.

        Returns
        -------
        int
            Sum of elements across all factor matrices.

        Notes
        -----
        This returns the storage cost, not the matrix size (m×n).
        For matrix dimensions, use `.shape`. For compression ratio,
        use `.compression_ratio()`.
        """
        return np.sum([M.size for M in self._matrices])

    @property
    def T(self) -> LowRankMatrix:
        """Transpose of the matrix (reverse order and transpose each factor)."""
        # Create a new LowRankMatrix with transposed factors in reverse order
        transposed_matrices = [M.T for M in reversed(self._matrices)]
        new_matrix = type(self)(*transposed_matrices, **self._extra_data)
        return new_matrix

    def conj(self) -> LowRankMatrix:
        """Complex conjugate of the matrix."""
        new_matrix = self.copy()
        new_matrix._matrices = [M.conj() for M in self._matrices]
        return new_matrix

    @property
    def H(self):
        """Hermitian transpose (conjugate transpose) of the matrix."""
        # Create a new LowRankMatrix with conjugate-transposed factors in reverse order
        hermitian_matrices = [M.T.conj() for M in reversed(self._matrices)]
        new_matrix = type(self)(*hermitian_matrices, **self._extra_data)
        return new_matrix

    def is_symmetric(self) -> bool:
        """Check if the matrix is symmetric.

        Returns
        -------
        bool
            True if the matrix is square and symmetric, False otherwise.

        Notes
        -----
        This method forms the full matrix to check symmetry.
        For large matrices, this may be memory-intensive.
        """
        warnings.warn(
            "Checking symmetry requires forming the full dense matrix, which may be inefficient.",
            LowRankEfficiencyWarning,
        )
        if self.shape[0] != self.shape[1]:
            return False
        dense = self.full()
        return bool(np.allclose(dense, dense.T))

    def transpose(self) -> LowRankMatrix:
        """Transpose the matrix (alias for .T property)."""
        return self.T

    ## CLASS METHODS
    @classmethod
    def from_matrix(cls, matrix: ndarray) -> LowRankMatrix:
        """Create a low-rank matrix from a full matrix.

        This method must be implemented by subclasses (e.g., SVD, QR).
        The base LowRankMatrix class requires at least 2 matrices for factorization.
        """
        raise NotImplementedError(
            "from_matrix() must be implemented by subclasses. "
            "The base LowRankMatrix class requires at least 2 matrices for factorization."
        )

    @classmethod
    def from_full(cls, matrix: ndarray):
        """Alias for from_matrix(). Must be implemented by subclasses."""
        return cls.from_matrix(matrix)

    @classmethod
    def from_dense(cls, matrix: ndarray):
        """Alias for from_matrix(). Must be implemented by subclasses."""
        return cls.from_matrix(matrix)

    @classmethod
    def from_low_rank(cls, low_rank_matrix: LowRankMatrix) -> LowRankMatrix:
        """Convert a LowRankMatrix to this specific subclass format.

        Parameters
        ----------
        low_rank_matrix : LowRankMatrix
            Existing low-rank matrix to convert.

        Returns
        -------
        LowRankMatrix
            New matrix of the target subclass type.

        Notes
        -----
        Subclasses should override this to perform format-specific conversions.
        For example, SVD.from_low_rank() would compute the SVD of the input.
        The base class implementation creates a generic LowRankMatrix.

        Examples
        --------
        >>> X_generic = LowRankMatrix(A, B)
        >>> X_svd = SVD.from_low_rank(X_generic)  # Converts to SVD format
        """
        return LowRankMatrix(*low_rank_matrix._matrices)

    def norm(self, ord: str | int = "fro") -> float:
        """Default implementation, overload this for some subclasses"""
        if ord == "fro":
            if "fro" in self._cache:
                return self._cache["fro"]
            else:
                norm = np.sqrt(self.norm_squared())
                self._cache["fro"] = norm
                return norm
        else:
            if ord in self._cache:
                return self._cache[ord]
            else:
                norm = float(np.linalg.norm(self.full(), ord=ord))  # type: ignore[arg-type]
                self._cache[ord] = norm
                return norm

    def __repr__(self) -> str:
        """String representation of the low-rank matrix."""
        return (
            f"{self.shape} low-rank matrix rank {self.rank}"
            f" and type {self.__class__._format}."
        )

    def copy(self):
        """Create a deep copy of the matrix.

        Returns
        -------
        LowRankMatrix
            A deep copy of this matrix with independent factor matrices.
        """
        return deepcopy(self)

    def __add__(
        self, other: LowRankMatrix | ndarray | LinearOperator
    ) -> ndarray | LinearOperator:
        """Addition of matrices.

        Behavior depends on the type of `other`:
        - If `other` is a generic LinearOperator (not LowRankMatrix): returns lazy _SumLinearOperator
        - If `other` is LowRankMatrix or ndarray: returns dense array (with warning)

        Parameters
        ----------
        other : LowRankMatrix, ndarray, or LinearOperator
            Matrix to add.

        Returns
        -------
        ndarray or LinearOperator
            Result of addition. Type depends on input type.
        """
        # Check if other is a generic LinearOperator (but not our subclass)
        if isinstance(other, LinearOperator) and not isinstance(other, LowRankMatrix):
            # Use lazy composition from parent LinearOperator
            return super().__add__(other)
        else:
            # Keep current behavior for LowRankMatrix + LowRankMatrix or ndarray
            warnings.warn(
                "Addition of generic low-rank matrices returns a dense matrix, which may be inefficient.",
                LowRankEfficiencyWarning,
            )
            if isinstance(other, LowRankMatrix):
                return self.full() + other.full()
            else:
                return self.full() + other

    def __radd__(
        self, other: LowRankMatrix | ndarray | LinearOperator
    ) -> ndarray | LinearOperator:
        """Right-side addition: other + self (addition is commutative)."""
        return self.__add__(other)

    def __imul__(
        self, other: float | LowRankMatrix | ndarray
    ) -> LowRankMatrix | ndarray:
        """In-place multiplication (scalar or element-wise).

        Notes
        -----
        For scalar multiplication, modifies the matrix in-place.
        For element-wise multiplication, creates a new object (inefficient).
        """
        if isinstance(other, (float, int, complex, np.number)):
            self._matrices[0] *= other
            return self
        elif isinstance(other, LowRankMatrix | ndarray):
            warnings.warn(
                "In-place Hadamard multiplication creates a new object.",
                LowRankEfficiencyWarning,
            )
            return self.hadamard(other)

    def __mul__(
        self, other: float | LowRankMatrix | ndarray
    ) -> LowRankMatrix | ndarray:
        """Scalar or element-wise (Hadamard) multiplication."""
        if isinstance(other, (float, int, complex, np.number)):
            new_mat = self.copy()
            new_mat._matrices[0] *= other
            return new_mat
        elif isinstance(other, LowRankMatrix | ndarray):
            # Hadamard method will issue its own warning
            return self.hadamard(other)
        else:
            raise TypeError(
                f"Unsupported operand type(s) for *: 'LowRankMatrix' and '{type(other)}'"
            )

    __rmul__ = __mul__

    def __neg__(self) -> LowRankMatrix | ndarray:
        """Negation of the matrix."""
        return -1 * self

    def __sub__(
        self, other: LinearOperator | LowRankMatrix
    ) -> LinearOperator | ndarray:
        """Subtraction of two matrices (returns dense result)."""
        warnings.warn(
            "Subtraction of low-rank matrices returns a dense matrix, which may be inefficient.",
            LowRankEfficiencyWarning,
        )
        return self.full() - (
            other.full() if isinstance(other, LowRankMatrix) else other
        )

    def __rsub__(
        self, other: LinearOperator | LowRankMatrix | ndarray
    ) -> ndarray | LinearOperator:
        """Right-side subtraction: other - self (not commutative)."""
        return (-1) * self + other

    def full(self) -> ndarray:
        """Form the full dense matrix by multiplying all factors in optimal order."""
        return np.linalg.multi_dot(self._matrices)

    def todense(self) -> ndarray:
        """Convert to dense matrix (alias for full)."""
        return self.full()

    def to_dense(self) -> ndarray:
        """Convert to dense matrix (alias for full)."""
        return self.full()

    def to_full(self) -> ndarray:
        """Convert to dense matrix (alias for full)."""
        return self.full()

    def flatten(self) -> ndarray:
        """Flatten the matrix to a 1D array."""
        return self.full().flatten()

    def gather(self, indices: ndarray) -> ndarray:
        """Access specific matrix entries without forming the full matrix.

        Parameters
        ----------
        indices : ndarray or list
            Index specification. For single element: [row_idx, col_idx].
            For multiple elements (fancy indexing): [row_indices, col_indices].

        Returns
        -------
        float | ndarray
            The requested matrix element(s).

        Notes
        -----
        This is faster and more memory-efficient than forming the full matrix.
        Very useful for matrix completion tasks or estimating reconstruction error.
        """
        A = self._matrices[0][indices[0], :]
        Z = self._matrices[-1][:, indices[1]]
        return np.linalg.multi_dot([A, *self._matrices[1:-1], Z])

    ## STANDARD MATRIX MULTIPLICATION
    def dot(
        self,
        other: LowRankMatrix | ndarray | spmatrix,
        side: str = "right",
        dense_output: bool = False,
    ) -> ndarray | LowRankMatrix:
        """Matrix and vector multiplication

        Parameters
        ----------
        other : LowRankMatrix, ndarray, spmatrix
            Matrix or vector to multiply with.
        side : str, optional
            Whether to multiply on the left or right, by default 'right'.
        dense_output : bool, optional
            Whether to return a dense matrix or a low-rank matrix, by default False.
        """
        # MATRIX-VECTOR CASE
        if len(other.shape) == 1:
            dense_output = True

        # SPARSE MATRIX CASE
        if isinstance(other, spmatrix):
            return self.dot_sparse(other, side, dense_output)

        # DENSE OUTPUT
        if dense_output:
            if isinstance(other, LowRankMatrix):
                if side.lower() in [
                    "right",
                    "usual",
                ]:  # usual is for backwards compatibility
                    return np.linalg.multi_dot(self._matrices + other._matrices)
                elif side.lower() in [
                    "left",
                    "opposite",
                ]:  # opposite is for backwards compatibility
                    return np.linalg.multi_dot(other._matrices + self._matrices)
                else:
                    raise ValueError('Incorrect side. Choose "right" or "left".')
            else:
                if side.lower() in ["right", "usual"]:
                    return np.linalg.multi_dot(self._matrices + [other])
                elif side.lower() in ["left", "opposite"]:
                    return np.linalg.multi_dot([other] + self._matrices)
                else:
                    raise ValueError('Incorrect side. Choose "right" or "left".')

        # LOW RANK OUTPUT (default)
        if isinstance(other, LowRankMatrix):
            if side.lower() in ["right", "usual"]:
                return LowRankMatrix(
                    *self._matrices, *other._matrices, **self._extra_data
                )
            elif side.lower() in ["left", "opposite"]:
                return LowRankMatrix(
                    *other._matrices, *self._matrices, **self._extra_data
                )
            else:
                raise ValueError('Incorrect side. Choose "right" or "left".')
        else:
            if side.lower() in ["right", "usual"]:
                return LowRankMatrix(*self._matrices, other, **self._extra_data)
            elif side.lower() in ["left", "opposite"]:
                return LowRankMatrix(other, *self._matrices, **self._extra_data)
            else:
                raise ValueError('Incorrect side. Choose "right" or "left".')

    __matmul__ = dot

    def __rmatmul__(self, other: LowRankMatrix | ndarray) -> ndarray | LowRankMatrix:
        """Right-side matrix multiplication: other @ self (not commutative)."""
        return self.dot(other, side="left")

    def multi_dot(self, others: Sequence[LowRankMatrix | ndarray]) -> LowRankMatrix:
        """Matrix multiplication of a sequence of matrices.

        Parameters
        ----------
        others : Sequence[LowRankMatrix | ndarray]
            Sequence of matrices to multiply.

        Returns
        -------
        LowRankMatrix
            Low rank matrix representing the product.
        """
        output = self.copy()
        for other in others:
            output = output.dot(other)
        return output

    ## MULTIPLICATION WITH A SPARSE MATRIX
    def dot_sparse(
        self, sparse_other: spmatrix, side: str = "usual", dense_output: bool = False
    ) -> ndarray | LowRankMatrix:
        """Efficient multiplication with a sparse matrix.

        Parameters
        ----------
        sparse_other : spmatrix
            Sparse matrix to multiply with.
        side : str, optional
            'right' or 'usual': output = self @ sparse_other
            'left' or 'opposite': output = sparse_other @ self
        dense_output : bool, optional
            Whether to return a dense matrix, by default False.

        Returns
        -------
        ndarray | LowRankMatrix
            Result of the multiplication.
        """
        sparse_other = sparse_other.tocsc()
        new_mat = self.copy()
        if side == "right" or side == "usual":
            new_mat._matrices[-1] = (sparse_other.T.dot(new_mat._matrices[-1].T)).T
        elif side == "opposite" or side == "left":
            new_mat._matrices[0] = sparse_other.dot(new_mat._matrices[0])
        else:
            raise ValueError("incorrect side")
        if dense_output:
            return new_mat.full()
        return new_mat

    ## AUTOMATIC COMPRESSION
    def compress(self) -> LowRankMatrix:
        """Compress the factorization to maximize memory efficiency."""
        # Matrix is already optimal
        if self.length <= 2:
            return self
        # Find the minimum rank among the matrices
        rank = self.rank
        idx_rank = min(
            [i for i, M in enumerate(self._matrices) if min(M.shape) == rank]
        )
        # Special case: rank is at the final matrix
        if idx_rank == len(self._matrices) - 1:
            return self.todense()  # type: ignore[return-value]
        # Compress towards the left
        if idx_rank == 0:
            left_prod = self._matrices[0]
        else:
            left_prod = np.linalg.multi_dot(self._matrices[: idx_rank + 1])
        # Compress the right part
        if idx_rank == len(self._matrices) - 2:
            right_prod = self._matrices[-1]
        else:
            right_prod = np.linalg.multi_dot(self._matrices[idx_rank + 1 :])
        return LowRankMatrix(left_prod, right_prod, **self._extra_data)

    def compress_(self) -> LowRankMatrix:
        """Compress the factorization in-place to maximize memory efficiency.

        Returns
        -------
        LowRankMatrix
            Self (modified in-place).

        Notes
        -----
        This modifies the matrix in-place. Use compress() for a non-destructive version.
        """
        if self.length <= 2:
            return self

        compressed = self.compress()
        if isinstance(compressed, ndarray):
            # If compress returns dense array, we can't stay in low-rank format
            warnings.warn(
                "Compression would return a dense matrix. Matrix unchanged.",
                LowRankEfficiencyWarning,
            )
            return self

        self._matrices = compressed._matrices
        return self

    def scale_(self, scalar: float) -> LowRankMatrix:
        """Scale the matrix in-place by a scalar.

        Parameters
        ----------
        scalar : float
            Scalar to multiply the matrix by.

        Returns
        -------
        LowRankMatrix
            Self (modified in-place).
        """
        self._matrices[0] *= scalar
        return self

    ## EXPONENTIAL ACTION OF A SPARSE MATRIX ON THE LOW-RANK MATRIX
    def expm_multiply(
        self,
        A: spmatrix,
        h: float,
        side: str = "left",
        dense_output: bool = False,
        **extra_args,
    ) -> ndarray | LowRankMatrix:
        """Efficient action of sparse matrix exponential
        left: output = exp(h*A) @ self
        right: output = self @ exp(h*A)

        Parameters
        ----------
        A : spmatrix
            Sparse matrix in the exponential.
        h : float
            Time step.
        side : str, optional
            Whether to multiply on the left or right, by default 'left'.
        dense_output : bool, optional
            Whether to return a dense matrix or a low-rank matrix, by default False.
        extra_args : dict, optional
            Extra arguments to pass to scipy.sparse.linalg.expm_multiply.

        Returns
        -------
        ndarray | LowRankMatrix
            Resulting matrix after applying the exponential action.
        """
        if h <= 0:
            raise ValueError("h must be positive")
        A = A.tocsc()  # sanity check
        new_mat = self.copy()
        if side == "left":
            new_mat._matrices[0] = spala.expm_multiply(
                A,
                self._matrices[0],
                start=0,
                stop=h,
                num=2,
                endpoint=True,
                **extra_args,
            )[-1]
        elif side == "right":
            new_mat._matrices[-1] = spala.expm_multiply(
                A.T,
                self._matrices[-1].T,
                start=0,
                stop=h,
                num=2,
                endpoint=True,
                **extra_args,
            )[-1].T
        else:
            raise ValueError("incorrect side")
        if dense_output:
            return new_mat.to_dense()
        return new_mat

    ## ADDITION OF MULTIPLE LOW-RANK MATRICES
    def multi_add(self, others: Sequence[LowRankMatrix]) -> ndarray:
        """Add multiple low-rank matrices efficiently.

        Parameters
        ----------
        others : Sequence[LowRankMatrix]
            Sequence of low-rank matrices to add.

        Returns
        -------
        ndarray
            Dense matrix representing the sum.

        Notes
        -----
        While more efficient than repeated use of +, this still requires
        forming full dense matrices and may be memory-intensive.
        """
        warnings.warn(
            "multi_add() requires forming full dense matrices, which may be inefficient.",
            LowRankEfficiencyWarning,
        )
        result = self.full()
        for other in others:
            result = result + other.full()
        return result

    ## HADAMARD PRODUCT
    def hadamard(self, other: LowRankMatrix | ndarray) -> ndarray:
        """Element-wise (Hadamard) product with another matrix.

        Parameters
        ----------
        other : LowRankMatrix or ndarray
            Matrix to multiply element-wise.

        Returns
        -------
        ndarray
            Dense result of element-wise multiplication.
        """
        warnings.warn(
            "Hadamard product requires forming full dense matrices, which may be inefficient.",
            LowRankEfficiencyWarning,
        )
        other_dense = other.to_dense() if isinstance(other, LowRankMatrix) else other
        return np.multiply(self.to_dense(), other_dense)

    ## DIAGONAL AND TRACE OPERATIONS
    def diag(self) -> ndarray:
        """Extract diagonal without forming full matrix.

        Returns
        -------
        ndarray
            Diagonal elements of the matrix.

        Notes
        -----
        This is computed efficiently by extracting only the diagonal elements
        during the matrix chain multiplication, avoiding forming the full matrix.
        """
        if self.shape[0] != self.shape[1]:
            raise ValueError("Matrix must be square to extract diagonal.")

        if "diag" in self._cache:
            return self._cache["diag"]

        n = self.shape[0]
        diag_elements = np.zeros(n, dtype=self.dtype)

        for i in range(n):
            diag_elements[i] = self.gather(np.array([i, i]))

        return diag_elements

    def trace(self) -> float:
        """Compute trace efficiently using cyclic property: tr(ABC) = tr(CAB) = tr(BCA).

        Returns
        -------
        float
            Trace of the matrix.

        Notes
        -----
        For a product of matrices A₁A₂...Aₙ, this method uses the cyclic property
        of the trace to minimize computational cost. The trace is computed as
        tr(A₂...AₙA₁) by cyclically permuting to minimize the intermediate matrix sizes.
        """
        if self.shape[0] != self.shape[1]:
            raise ValueError("Matrix must be square to compute trace.")
        if "trace" in self._cache:
            return self._cache["trace"]

        # Use cyclic property: tr(A1 @ A2 @ ... @ An) = tr(A2 @ ... @ An @ A1)
        # Try to minimize computation by finding best cyclic permutation
        n_matrices = len(self._matrices)

        if n_matrices == 2:
            # tr(AB) = tr(BA) - choose smaller intermediate
            A, B = self._matrices
            if A.shape[1] * B.shape[0] <= A.shape[0] * B.shape[1]:
                tr = np.trace(A @ B)
                self._cache["trace"] = tr
                return tr
            else:
                tr = np.trace(B @ A)
                self._cache["trace"] = tr
                return tr

        # For more matrices, compute trace by cyclically permuting
        # to minimize the first multiplication
        costs = []
        for i in range(n_matrices):
            # Cost of first multiplication after cyclic shift by i
            M1 = self._matrices[i]
            M2 = self._matrices[(i + 1) % n_matrices]
            cost = M1.shape[0] * M1.shape[1] * M2.shape[1]
            costs.append(cost)

        best_shift = np.argmin(costs)

        # Cyclically shift matrices
        shifted_matrices = self._matrices[best_shift:] + self._matrices[:best_shift]

        # Compute the product and take trace
        product = np.linalg.multi_dot(shifted_matrices)
        tr = np.trace(product)
        self._cache["trace"] = tr
        return tr

    def norm_squared(self) -> float:
        """Compute squared Frobenius norm efficiently: ||X||²_F = tr(X^H X).

        Returns
        -------
        float
            Squared Frobenius norm of the matrix.

        Notes
        -----
        This is more efficient than computing the norm and squaring it,
        as it avoids the square root operation. For complex matrices,
        uses Hermitian transpose (X^H X) to ensure real result.
        """
        if "norm_squared" in self._cache:
            return self._cache["norm_squared"]

        # ||X||²_F = tr(X^H @ X)
        # For X = A₁A₂...Aₙ, we have X^H X = Aₙ^H...A₁^H A₁...Aₙ
        hermitian_matrices = [M.T.conj() for M in reversed(self._matrices)]
        combined_matrices = hermitian_matrices + self._matrices

        # Create temporary LowRankMatrix for trace computation
        XHX = LowRankMatrix(*combined_matrices)
        trace_val = XHX.trace()

        # For X^H X, trace should be real (up to numerical errors)
        if np.iscomplexobj(trace_val):
            return np.real(trace_val)

        self._cache["norm_squared"] = trace_val
        return trace_val

    ## MATRIX POWER
    def power(self, n: int) -> LowRankMatrix | ndarray:
        """Compute matrix power X^n efficiently using repeated squaring.

        Parameters
        ----------
        n : int
            Power to raise the matrix to. Must be non-negative.

        Returns
        -------
        LowRankMatrix | ndarray
            Matrix raised to the n-th power. Returns identity matrix (as ndarray)
            for n=0, self for n=1, and LowRankMatrix for n>1.

        Raises
        ------
        ValueError
            If matrix is not square or n is negative.

        Notes
        -----
        This uses the binary exponentiation algorithm for efficiency.
        Time complexity is O(log n) matrix multiplications.
        """
        if self.shape[0] != self.shape[1]:
            raise ValueError("Matrix must be square to compute powers.")
        if n < 0:
            raise ValueError("Negative powers not supported. Use matrix inverse first.")

        if n == 0:
            return np.eye(self.shape[0], dtype=self.dtype)
        if n == 1:
            return self.copy()

        # Binary exponentiation
        result: Optional[LowRankMatrix | ndarray] = None
        base = self.copy()

        while n > 0:
            if n % 2 == 1:
                result = base if result is None else result.dot(base)
            n //= 2
            if n > 0:
                base = base.dot(base)

        assert result is not None, "Result should not be None for n > 0"
        assert isinstance(result, LowRankMatrix), "Result should be LowRankMatrix"
        return result

    ## SLICING SUPPORT
    def __getitem__(self, key) -> float | ndarray:
        """Access matrix elements or slices.

        Parameters
        ----------
        key : tuple, int, slice, or ndarray
            Index specification. Can be:
            - (row_idx, col_idx): single element → returns float
            - (row_slice, col_slice): block submatrix → returns ndarray
            - (row_array, col_array): fancy indexing → returns ndarray
            - single int: row indexing → returns ndarray (1D)

        Returns
        -------
        float or ndarray
            The requested matrix element(s) or submatrix.
            - float: for single element access (row_idx, col_idx)
            - ndarray: for slices, fancy indexing, or row selection

        Notes
        -----
        Slicing operations form the full matrix, which may be inefficient.
        For single element access, gather() is used for efficiency.

        Examples
        --------
        >>> X = LowRankMatrix(A, B)
        >>> x_ij = X[2, 3]  # Single element (efficient via gather)
        >>> row = X[2, :]   # Row slice (forms full matrix)
        >>> block = X[0:5, 0:10]  # Block submatrix
        >>> fancy = X[[0, 2, 4], [1, 3, 5]]  # Fancy indexing (returns 1D array of selected elements)
        >>> row = X[2, :]   # Row slice (forms full matrix)
        >>> block = X[0:5, 0:10]  # Block submatrix
        >>> fancy = X[[0, 2, 4], [1, 3, 5]]  # Fancy indexing
        """
        # Handle single index (not a tuple)
        if not isinstance(key, tuple):
            key = (key, slice(None))

        if len(key) != 2:
            raise IndexError("Matrix indexing requires exactly 2 indices.")

        row_idx, col_idx = key

        # Single element access - use efficient gather
        if isinstance(row_idx, (int, np.integer)) and isinstance(
            col_idx, (int, np.integer)
        ):
            return self.gather(np.array([row_idx, col_idx]))

        # Slicing or fancy indexing - form full matrix
        warnings.warn(
            "Slicing operations require forming the full dense matrix, which may be inefficient.",
            LowRankEfficiencyWarning,
        )
        return self.full()[key]

    def get_block(self, rows: slice, cols: slice) -> ndarray:
        """Extract block submatrix.

        Parameters
        ----------
        rows : slice
            Row slice specification.
        cols : slice
            Column slice specification.

        Returns
        -------
        ndarray
            The requested block submatrix.

        Notes
        -----
        This method forms the full matrix, which may be inefficient.
        It's provided for convenience and compatibility.
        """
        warnings.warn(
            "Block extraction requires forming the full dense matrix, which may be inefficient.",
            LowRankEfficiencyWarning,
        )
        return self.full()[rows, cols]

    ## ITERATIVE SOLVERS INTERFACE (LinearOperator implementation)
    def _matvec(self, v: ndarray) -> ndarray:
        """Matrix-vector product (required by LinearOperator).

        Parameters
        ----------
        v : ndarray
            Vector to multiply with.

        Returns
        -------
        ndarray
            Result of matrix-vector multiplication.

        Notes
        -----
        This is the core method required by scipy.sparse.linalg.LinearOperator.
        """
        return self.dot(v, dense_output=True)

    def _rmatvec(self, v: ndarray) -> ndarray:
        """Adjoint matrix-vector product (required by LinearOperator).

        Parameters
        ----------
        v : ndarray
            Vector to multiply with.

        Returns
        -------
        ndarray
            Result of adjoint matrix-vector multiplication.

        Notes
        -----
        This computes v^H @ self for compatibility with iterative solvers.
        This is the core method for the Hermitian adjoint in LinearOperator.
        """
        return self.H.dot(v, dense_output=True)

    def _matmat(self, X: ndarray) -> ndarray:
        """Matrix-matrix product (optional, for efficiency).

        Parameters
        ----------
        X : ndarray
            Matrix to multiply with, shape (n, k).

        Returns
        -------
        ndarray
            Result of matrix-matrix multiplication, shape (m, k).

        Notes
        -----
        This method is optional but improves performance for matrix-matrix operations.
        """
        return self.dot(X, dense_output=True)

    def _rmatmat(self, X: ndarray) -> ndarray:
        """Adjoint matrix-matrix product (optional, for efficiency).

        Parameters
        ----------
        X : ndarray
            Matrix to multiply with, shape (m, k).

        Returns
        -------
        ndarray
            Result of adjoint matrix-matrix multiplication, shape (n, k).

        Notes
        -----
        This method is optional but improves performance for matrix-matrix operations.
        """
        return self.H.dot(X, dense_output=True)

    def _adjoint(self):
        """Return the adjoint (Hermitian transpose) as a LinearOperator.

        Returns
        -------
        LowRankMatrix
            The Hermitian transpose of this matrix.

        Notes
        -----
        This method is part of the LinearOperator interface. It enables
        expressions like `A.H` or `A.T` (for real matrices) to work properly
        with scipy's iterative solvers.
        """
        return self.H

    # Public wrappers for backward compatibility
    def matvec(self, v: ndarray) -> ndarray:
        """Matrix-vector product (public interface, backward compatibility).

        Parameters
        ----------
        v : ndarray
            Vector to multiply with.

        Returns
        -------
        ndarray
            Result of matrix-vector multiplication.
        """
        return self._matvec(v)

    def rmatvec(self, v: ndarray) -> ndarray:
        """Adjoint matrix-vector product (public interface, backward compatibility).

        Parameters
        ----------
        v : ndarray
            Vector to multiply with.

        Returns
        -------
        ndarray
            Result of adjoint matrix-vector multiplication.
        """
        return self._rmatvec(v)

    ## CONDITION NUMBER ESTIMATION
    def cond_estimate(self, method: str = "power_iteration", n_iter: int = 10) -> float:
        """Estimate condition number without computing full SVD.

        Parameters
        ----------
        method : str, optional
            Method to use: 'power_iteration' (default) or 'norm_ratio'.
        n_iter : int, optional
            Number of iterations for power iteration method, by default 10.

        Returns
        -------
        float
            Estimated condition number (ratio of largest to smallest singular value).

        Notes
        -----
        The 'power_iteration' method estimates the largest singular value
        using power iteration. The smallest is estimated using the inverse
        (via solving linear systems). This is approximate but avoids full SVD.

        The 'norm_ratio' method uses matrix norms as bounds.
        """
        if self.shape[0] != self.shape[1]:
            warnings.warn(
                "Condition number is typically defined for square matrices. "
                "Computing norm-based estimate.",
                UserWarning,
            )

        # Create cache key that includes method
        cache_key = f"cond_estimate_{method}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        if method == "norm_ratio":
            # Use Frobenius norm approximation
            norm_fro = self.norm("fro")
            norm_2 = self.norm(2)
            # This is a rough estimate
            cond = (norm_fro / norm_2) * np.sqrt(min(self.shape))
            self._cache[cache_key] = cond
            return cond

        elif method == "power_iteration":
            # Estimate largest singular value via power iteration on X^T X
            # σ_max^2 is the largest eigenvalue of X^T X
            v = np.random.randn(self.shape[1])
            v = v / np.linalg.norm(v)

            for _ in range(n_iter):
                v = self.T.matvec(self.matvec(v))
                norm_v = np.linalg.norm(v)
                if norm_v > 0:
                    v = v / norm_v

            sigma_max = np.sqrt(norm_v)

            # Estimate smallest singular value (inverse power iteration)
            # This is more complex and would require solving systems
            # For now, use a simpler bound
            sigma_min_estimate = 1.0 / self.norm(2)
            cond = sigma_max / sigma_min_estimate
            self._cache[cache_key] = cond

            return cond

        else:
            raise ValueError(
                f"Unknown method: {method}. Use 'norm_ratio' or 'power_iteration'."
            )

    ## MEMORY FOOTPRINT REPORTING
    def _compute_storage_size(self) -> int:
        """Compute total number of elements stored in the factorization.

        This method can be overloaded in subclasses for specialized storage schemes.
        For example, SVD stores only the diagonal of S, not the full matrix.

        Returns
        -------
        int
            Total number of elements stored.
        """
        size = sum(M.size for M in self._matrices)
        self._cache["storage_size"] = size
        return size

    def memory_usage(self, unit: str = "MB") -> float:
        """Report actual memory used by the factorization.

        Parameters
        ----------
        unit : str, optional
            Unit for memory size: 'B', 'KB', 'MB', 'GB', by default 'MB'.

        Returns
        -------
        float
            Memory usage in the specified unit.
        """
        bytes_per_element = self.dtype.itemsize
        total_elements = self._compute_storage_size()
        total_bytes = total_elements * bytes_per_element

        units = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3}
        if unit not in units:
            raise ValueError(f"Unknown unit: {unit}. Use 'B', 'KB', 'MB', or 'GB'.")

        return total_bytes / units[unit]

    def compression_ratio(self) -> float:
        """Compute compression ratio relative to dense storage.

        Returns
        -------
        float
            Ratio of low-rank storage to dense storage (< 1 means savings).

        Notes
        -----
        A ratio of 0.1 means the low-rank format uses 10% of the memory
        of the dense format, i.e., a 10x compression.
        """
        low_rank_size = self._compute_storage_size()
        dense_size = np.prod(self.shape)
        return float(low_rank_size / dense_size)

    @property
    def is_memory_efficient(self) -> bool:
        """Check if the low-rank representation uses less memory than dense storage.

        Returns
        -------
        bool
            True if low-rank format uses less memory than dense format, False otherwise.

        Notes
        -----
        This compares the total number of elements stored in the factorization
        versus the full m×n matrix. For empty matrices (zero size), returns True.

        Examples
        --------
        >>> A = LowRankMatrix(np.random.randn(1000, 10), np.random.randn(10, 1000))
        >>> A.is_memory_efficient  # True: 20,000 elements vs 1,000,000
        True
        >>> B = LowRankMatrix(np.random.randn(100, 90), np.random.randn(90, 100))
        >>> B.is_memory_efficient  # False: 18,000 elements vs 10,000
        False
        """
        dense_size = np.prod(self.shape)
        if dense_size == 0:
            return True
        low_rank_size = self._compute_storage_size()
        return bool(low_rank_size < dense_size)

    ## SERIALIZATION
    def save(self, filename: str):
        """Save low-rank matrix to disk efficiently.

        Parameters
        ----------
        filename : str
            Path to save the matrix. Extension '.npz' will be added if not present.

        Notes
        -----
        The matrix is saved in compressed NumPy format with all factor matrices
        and extra data preserved.
        """
        if not filename.endswith(".npz"):
            filename += ".npz"

        # Prepare data dictionary
        save_dict = {f"matrix_{i}": M for i, M in enumerate(self._matrices)}
        save_dict["n_matrices"] = len(self._matrices)  # type: ignore[assignment]
        save_dict["format"] = self._format  # type: ignore[assignment]

        # Save extra data
        for key, value in self._extra_data.items():
            save_dict[f"extra_{key}"] = value

        np.savez_compressed(filename, **save_dict)  # type: ignore[arg-type]

    @classmethod
    def load(cls, filename: str) -> LowRankMatrix:
        """Load low-rank matrix from disk.

        Parameters
        ----------
        filename : str
            Path to the saved matrix file.

        Returns
        -------
        LowRankMatrix
            Loaded low-rank matrix.
        """
        if not filename.endswith(".npz"):
            filename += ".npz"

        data = np.load(filename)
        n_matrices = int(data["n_matrices"])

        # Load factor matrices
        matrices = [data[f"matrix_{i}"] for i in range(n_matrices)]

        # Load extra data
        extra_data = {}
        for key in data.keys():
            if key.startswith("extra_"):
                extra_data[key[6:]] = data[key]

        return cls(*matrices, **extra_data)

    ## APPROXIMATION ERROR
    def approximation_error(self, reference: ndarray, ord: str = "fro") -> float:
        """Compute approximation error ||X - reference|| efficiently.

        Parameters
        ----------
        reference : ndarray
            Reference matrix to compare against.
        ord : str or int, optional
            Norm type, by default 'fro' (Frobenius).

        Returns
        -------
        float
            Approximation error in the specified norm.

        Notes
        -----
        This forms the full matrix, which may be inefficient for very large matrices.
        For Frobenius norm, consider using norm_squared() for efficiency.
        """
        warnings.warn(
            "Computing approximation error requires forming the full dense matrix.",
            LowRankEfficiencyWarning,
        )
        diff = self.full() - reference
        return np.linalg.norm(diff, ord=ord)  # type: ignore[call-overload]

    ## STABILITY ANALYSIS
    def is_well_conditioned(self, threshold: float = 1e10) -> bool:
        """Check if the factorization is numerically well-conditioned.

        Parameters
        ----------
        threshold : float, optional
            Condition number threshold, by default 1e10.

        Returns
        -------
        bool
            True if estimated condition number is below threshold.

        Notes
        -----
        This uses an approximate condition number estimate and may not be
        accurate for all cases. Consider this a heuristic check.
        """
        if "is_well_conditioned" in self._cache:
            return self._cache["is_well_conditioned"]
        if self.shape[0] != self.shape[1]:
            warnings.warn(
                "Well-conditioning check is designed for square matrices.", UserWarning
            )
            self._cache["is_well_conditioned"] = True
            return True

        cond = self.cond_estimate(method="norm_ratio")
        self._cache["is_well_conditioned"] = bool(cond < threshold)
        return bool(cond < threshold)

    ## SPARSE CONVERSION
    def to_sparse(self, format: str = "csr", threshold: float = 1e-10) -> spmatrix:
        """Convert to sparse format, zeroing small entries.

        Parameters
        ----------
        format : str, optional
            Sparse matrix format: 'csr', 'csc', 'coo', by default 'csr'.
        threshold : float, optional
            Entries with absolute value below threshold are set to zero, by default 1e-10.

        Returns
        -------
        spmatrix
            Sparse matrix representation.

        Notes
        -----
        This forms the full dense matrix first, which may be inefficient.
        Only use this if you expect the resulting matrix to be sparse.
        """
        from scipy import sparse as sp

        warnings.warn(
            "Converting to sparse format requires forming the full dense matrix.",
            LowRankEfficiencyWarning,
        )

        dense = self.full()
        # Zero out small entries
        dense[np.abs(dense) < threshold] = 0

        if format == "csr":
            return sp.csr_matrix(dense)
        elif format == "csc":
            return sp.csc_matrix(dense)
        elif format == "coo":
            return sp.coo_matrix(dense)
        else:
            raise ValueError(f"Unknown format: {format}. Use 'csr', 'csc', or 'coo'.")

    ## EQUALITY AND COMPARISON
    def __eq__(self, other) -> bool:
        """Check exact equality with another matrix.

        Parameters
        ----------
        other : LowRankMatrix or ndarray
            Matrix to compare with.

        Returns
        -------
        bool
            True if matrices are exactly equal.

        Notes
        -----
        For floating point comparisons, use allclose() instead.
        This forms the full matrix for comparison.
        """
        warnings.warn(
            "Equality check requires forming the full dense matrix.",
            LowRankEfficiencyWarning,
        )

        if isinstance(other, LowRankMatrix):
            if self.shape != other.shape:
                return False
            return np.array_equal(self.full(), other.full())
        elif isinstance(other, ndarray):
            if self.shape != other.shape:
                return False
            return np.array_equal(self.full(), other)
        else:
            return False

    def allclose(
        self, other: LowRankMatrix | ndarray, rtol: float = 1e-5, atol: float = 1e-8
    ) -> bool:
        """Check approximate equality with another matrix.

        Parameters
        ----------
        other : LowRankMatrix or ndarray
            Matrix to compare with.
        rtol : float, optional
            Relative tolerance, by default 1e-5.
        atol : float, optional
            Absolute tolerance, by default 1e-8.

        Returns
        -------
        bool
            True if matrices are approximately equal within tolerances.

        Notes
        -----
        This forms the full matrix for comparison, which may be inefficient.
        """
        if isinstance(other, LowRankMatrix):
            if self.shape != other.shape:
                return False
            return np.allclose(self.full(), other.full(), rtol=rtol, atol=atol)
        elif isinstance(other, ndarray):
            if self.shape != other.shape:
                return False
            return np.allclose(self.full(), other, rtol=rtol, atol=atol)
        else:
            return False

    @staticmethod
    def create_matrix_alias(index: int, transpose=False, conjugate=False) -> property:
        """Create a property that provides access to a factor matrix with optional transformations.

        Parameters
        ----------
        index : int
            Index of the factor matrix in self._matrices.
        transpose : bool, optional
            Whether to transpose the matrix when accessed, by default False.
        conjugate : bool, optional
            Whether to conjugate the matrix when accessed, by default False.

        Returns
        -------
        property
            Property object for accessing the transformed matrix.

        Examples
        --------
        Create a property to access the first factor matrix:

        >>> class MyLowRank(LowRankMatrix):
        ...     U = LowRankMatrix.create_matrix_alias(0)
        ...     V = LowRankMatrix.create_matrix_alias(1, transpose=True)

        Now `my_matrix.U` accesses `self._matrices[0]` and `my_matrix.V`
        accesses `self._matrices[1].T`.
        """
        if transpose and conjugate:

            def getter(self):
                return self._matrices[index].T.conj()

            def setter(self, value):
                self._matrices[index] = value.T.conj()

        elif transpose:

            def getter(self):
                return self._matrices[index].T

            def setter(self, value):
                self._matrices[index] = value.T

        elif conjugate:

            def getter(self):
                return self._matrices[index].conj()

            def setter(self, value):
                self._matrices[index] = value.conj()

        else:

            def getter(self):
                return self._matrices[index]

            def setter(self, value):
                self._matrices[index] = value

        return property(getter, setter)

    @staticmethod
    def create_data_alias(key: "str") -> property:
        """Create a property that provides access to an entry in extra_data.

        Parameters
        ----------
        key : str
            Key for accessing the data in self._extra_data.

        Returns
        -------
        property
            Property object for accessing the extra data.

        Examples
        --------
        Create a property to access extra data stored with the matrix:

        >>> class RationalKrylov(LowRankMatrix):
        ...     poles = LowRankMatrix.create_data_alias('poles')
        ...     residues = LowRankMatrix.create_data_alias('residues')

        Now `my_matrix.poles` accesses `self._extra_data['poles']`.
        """

        def getter(self) -> ndarray:
            return self._extra_data[key]

        def setter(self, value: ndarray):
            self._extra_data[key] = value

        return property(getter, setter)
