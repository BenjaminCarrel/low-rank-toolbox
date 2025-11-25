# Authors: Benjamin Carrel and Rik Vorhaar
#          University of Geneva, initiated in 2022
# File for SVD low-rank matrix class and functions
# Path: src/lowrank/matrices/svd.py

#%% Imports
from __future__ import annotations
import numpy as np
from numpy import ndarray
from scipy import linalg as la
from .quasi_svd import QuasiSVD
from .low_rank_matrix import LowRankMatrix
from ._svd_config import AUTOMATIC_TRUNCATION, DEFAULT_ATOL, DEFAULT_RTOL


#%% Define class SVD
class SVD(QuasiSVD):
    """
    Singular Value Decomposition (SVD) for low-rank matrices.
    
    The gold standard for low-rank matrix representation with guaranteed diagonal structure and non-negative singular values. Inherits from QuasiSVD but enforces diagonal middle matrix for optimal storage.
    
    Mathematical Representation
    ---------------------------
    X = U @ diag(s) @ V.H = U @ S @ V.H
    
    where:
    - U ∈ ℝ^(m×r) has orthonormal columns (U^H @ U = I_r)
    - s ∈ ℝ^r contains non-negative singular values in descending order (s[0] ≥ s[1] ≥ ... ≥ s[r-1] ≥ 0)
    - V ∈ ℝ^(n×r) has orthonormal columns (V^H @ V = I_r)
    - S = diag(s) is an r×r matrix with singular values on the diagonal
    
    Key Differences from QuasiSVD
    ------------------------------
    - **Diagonal structure**: S is always diagonal (guaranteed by construction)
    - **Singular values**: Non-negative by definition, stored in descending order
    - **Efficiency**: Optimized operations exploit diagonal structure
    - **Operations**: Addition, multiplication, and Hadamard preserve SVD format when possible
    
    Storage Efficiency
    ------------------
    Full matrix: O(mn) storage
    SVD: O(mr + r² + nr) storage (same as QuasiSVD)
    Memory savings when r << min(m, n)
    
    For 1000×1000 matrix with rank 10:
    - Dense: 1,000,000 elements
    - SVD: 20,100 elements (50x compression)
    - QuasiSVD: 20,100 elements
    
    Important Notes
    ---------------
    - CRITICAL: U and V MUST have orthonormal columns. 
        This is NOT verified at initialization for performance.
        Invalid inputs produce incorrect results without warning.
    - Use .is_orthogonal() to verify orthonormality if needed
    - Singular values are stored internally as a diagonal matrix S (2D array)
    - The property `s` extracts the diagonal as a 1D vector when needed
    - After operations, may return QuasiSVD or LowRankMatrix (orthogonality not preserved)
    - For non-diagonal S, use QuasiSVD instead
    
    Attributes
    ----------
    U : ndarray, shape (m, r)
        Left singular vectors (orthonormal columns)
    s : ndarray, shape (r,)
        Singular values (1D vector, non-negative, descending order)
    S : ndarray, shape (r, r)
        Diagonal matrix of singular values (property, computed on demand)
    V : ndarray, shape (n, r)
        Right singular vectors (orthonormal columns)
    Vh : ndarray, shape (r, n)
        Hermitian conjugate of V (V.T.conj())
    Vt : ndarray, shape (r, n)
        Transpose of V (without conjugate)
    Ut : ndarray, shape (r, m)
        Transpose of U (without conjugate)
    Uh : ndarray, shape (r, m)
        Hermitian conjugate of U (U.T.conj())
    
    Properties
    ----------
    shape : tuple
        Shape of the represented matrix (m, n)
    rank : int
        Number of singular values (length of s)
    sing_vals : ndarray
        Singular values (alias for s)
    T : SVD
        Transpose of the matrix (returns SVD)
    
    Methods Overview
    ----------------
    Core Operations:
        - __add__ : Addition preserving SVD structure when possible
        - __sub__ : Subtraction preserving SVD structure when possible
        - __mul__ : Scalar or Hadamard (element-wise) multiplication preserving SVD structure when possible
        - dot : Matrix-vector or matrix-matrix multiplication preserving SVD structure when possible
        - hadamard : Element-wise multiplication with another matrix preserving SVD structure when possible
    
    Truncation:
        - truncate() : Remove small singular values
        - truncate_perpendicular() : Remove large singular values and keep smallest singular values
    
    Class Methods:
        - from_matrix() : Convert any matrix to SVD
        - from_quasiSVD() : Convert QuasiSVD to SVD
        - from_low_rank() : Convert LowRankMatrix to SVD
        - reduced_svd() : Compute reduced/economic SVD
        - truncated_svd() : Compute truncated SVD with specified rank or (relative/absolute) tolerance
        - generate_random() : Generate random SVD with specified singular values
    
    Optimized Operations:
        - norm() : Compute norms efficiently using singular values
        - norm_squared() : Squared Frobenius norm from singular values
        - trace() : Efficient trace computation
        - diag() : Extract diagonal efficiently
    
    Configuration
    -------------
    Default behavior controlled by:
    - AUTOMATIC_TRUNCATION (default: False): Auto-truncate after operations
    - DEFAULT_ATOL (machine precision): Absolute tolerance for truncation (when AUTOMATIC_TRUNCATION is True)
    - DEFAULT_RTOL (None): Relative tolerance for truncation
    
    Examples
    --------
    **Creating an SVD from scratch:**
    
    >>> import numpy as np
    >>> from lowrank.matrices import SVD
    >>> 
    >>> # Create orthonormal matrices
    >>> m, n, r = 100, 80, 10
    >>> U, _ = np.linalg.qr(np.random.randn(m, r))
    >>> V, _ = np.linalg.qr(np.random.randn(n, r))
    >>> s = np.logspace(0, -2, r)  # Singular values (1D array)
    >>> 
    >>> # Create SVD
    >>> X = SVD(U, s, V)
    >>> print(X.shape)  # (100, 80)
    >>> print(X.rank)   # 10
    >>> print(X.s.shape)  # (10,) - 1D vector!
    >>> print(X.S.shape)  # (10, 10) - 2D diagonal matrix (property)
    
    **Computing SVD from a matrix:**
    
    >>> A = np.random.randn(100, 80)
    >>> X_reduced = SVD.reduced_svd(A)  # Reduced SVD
    >>> X_truncated = SVD.truncated_svd(A, r=10)  # Keep top 10 singular values
    >>> X_auto = SVD.truncated_svd(A, rtol=1e-6)  # Adaptive truncation
    
    **Efficient operations:**
    
    >>> # Operations exploit diagonal structure and orthogonality
    >>> norm_fro = X.norm('fro')  # sqrt(sum(s²)) - O(r)
    >>> norm_squared = X.norm_squared()  # sum(s²) - O(r)
    >>> norm_2 = X.norm(2)  # max(s) - instant!
    >>> norm_nuc = X.norm('nuc')  # sum(s) - O(r) instead of O(mnr)
    >>> 
    >>> # Addition preserves SVD structure
    >>> Y = X + X  # Returns SVD
    >>> Z = X @ X.T  # Matrix multiplication, returns SVD
    >>> 0 = X - X  # Subtraction, returns SVD of rank 2*rank(X) (if AUTOMATIC_TRUNCATION is False) or rank 0 (if AUTOMATIC_TRUNCATION is True)
    
    **Truncation:**
    
    >>> # Remove small singular values
    >>> X_trunc = X.truncate(r=5)  # Keep top 5
    >>> X_trunc = X.truncate(rtol=1e-10)  # Relative tolerance (based on max singular value)
    >>> X_trunc = X.truncate(atol=1e-12)  # Absolute tolerance (raw threshold, independent of max singular value)
    >>> 
    >>> # Get perpendicular component (residual)
    >>> X_perp = X.truncate_perpendicular(r=5)  # Keep last (r-5) singular values
    
    **Random matrices with controlled spectrum:**
    
    >>> # Generate test matrices
    >>> s_decay = np.logspace(0, -10, 20)  # Exponential decay
    >>> X_test = SVD.generate_random((100, 100), s_decay, is_symmetric=True)
    >>> # Perfect for testing algorithms!
    
    **Conversion between formats:**
    
    >>> from lowrank.matrices import QuasiSVD, LowRankMatrix
    >>> 
    >>> # From QuasiSVD (non-diagonal S)
    >>> X_quasi = QuasiSVD(U, S_full, V)  # S_full is general matrix
    >>> X_svd = SVD.from_quasiSVD(X_quasi)  # Diagonalize S
    >>> 
    >>> # From generic low-rank
    >>> X_lr = LowRankMatrix(A, B, C)
    >>> X_svd = SVD.from_low_rank(X_lr)  # Compute SVD
    
    **Memory efficiency:**
    
    >>> X.compression_ratio()  # < 1.0 means memory savings
    >>> X.memory_usage('MB')  # Actual memory used
    >>> X.is_memory_efficient  # True if saves memory
    
    See Also
    --------
    QuasiSVD : Generalized SVD with non-diagonal middle matrix
    LowRankMatrix : Base class for low-rank matrix representations
    QR : QR factorization format
    Randomized SVD methods for large-scale problems
    
    References
    ----------
    .. [1] Trefethen, L. N., & Bau III, D. (1997). Numerical Linear Algebra. SIAM.
    .. [2] Golub, G. H., & Van Loan, C. F. (2013). Matrix Computations (4th ed.). Johns Hopkins University Press.

    
    Notes
    -----
    - This class is designed for numerical linear algebra applications
    - Singular values are ALWAYS stored as a 1D array for memory efficiency, otherwise use QuasiSVD
    - The 2D diagonal matrix S is computed on-the-fly when needed (via property)
    - Operations exploit orthogonality and diagonal structure for efficiency
    - After general operations with non-SVD matrices (e.g., X + Y), may return QuasiSVD or LowRankMatrix
    - Use .truncate() to insure (numerical) invertibility or reduce rank
    """

    ## ATTRIBUTES
    _format = "SVD"

    def __init__(self, U: ndarray, s: ndarray, V: ndarray, **extra_data):
        """
        Create a low-rank matrix stored by its SVD: X = U @ diag(s) @ V.T
        
        This is the primary constructor for the SVD class. It efficiently stores
        singular values as a 1D vector rather than a full diagonal matrix.
        
        Parameters
        ----------
        U : ndarray
            Left singular vectors, shape (m, r).
            ASSUMED to have orthonormal columns: U.T @ U = I_r.
            Not verified at initialization for performance.
        s : ndarray
            Singular values. Accepts two formats:
            - **1D array** (shape (r,)): Vector of singular values (RECOMMENDED)
            - **2D array** (shape (r, r) or (r, k)): Diagonal or nearly-diagonal matrix
        V : ndarray
            Right singular vectors, shape (n, k).
            ASSUMED to have orthonormal columns: V.T @ V = I_k.
            Not verified at initialization for performance.
            NOTE: Provide V, not V.T or V.H (conjugate transpose).
            
        **extra_data : dict, optional
            Additional metadata to store with the matrix (e.g., poles, residues).
            Internal flags:
            - '_skip_diagonal_check': Skip diagonal verification (used internally)
            - '_skip_memory_check': Skip memory efficiency check
            
        Raises
        ------
        ValueError
            - If input dimensions are incompatible for multiplication
            - If number of singular values doesn't match min(r, k)
            - If U or V are not 2D arrays
        TypeError
            - If U, V, or s are not numpy arrays
            - If s is not 1D or 2D
            
        Notes
        -----
        **Storage**: Singular values are stored internally as a diagonal matrix S (2D array).
        The input parameter `s` can be provided in either 1D or 2D format for convenience.
        
        **Input format flexibility**: 
        - If s is 1D: converted to diagonal matrix S for storage
        - If s is 2D diagonal: stored directly as S
        - If s is 2D rectangular (r ≠ k): stored as rectangular matrix with filled diagonal
        
        **Orthogonality assumption**: U and V are ASSUMED to have orthonormal columns.
        This is NOT verified at initialization for performance. Use .is_orthogonal() to verify if needed.
        
        **Parent class**: Calls QuasiSVD.__init__ with a flag to suppress the "use SVD class" warning (since we ARE the SVD class).
        
        Examples
        --------
        **Recommended usage (1D singular values):**
        
        >>> import numpy as np
        >>> from lowrank.matrices import SVD
        >>> 
        >>> m, n, r = 100, 80, 10
        >>> U, _ = np.linalg.qr(np.random.randn(m, r))
        >>> s = np.logspace(0, -2, r)  # 1D vector
        >>> V, _ = np.linalg.qr(np.random.randn(n, r))
        >>> 
        >>> X = SVD(U, s, V)
        >>> print(X.s.shape)  # (10,) - extracted from diagonal of S
        >>> print(X.S.shape)  # (10, 10) - stored 2D matrix
        
        **Alternative usage (2D diagonal matrix):**
        
        >>> S_diag = np.diag(s)  # 2D diagonal matrix
        >>> X = SVD(U, S_diag, V)  # Also works
        >>> print(X.s.shape)  # (10,) - extracted from diagonal of S
        
        **From numpy.linalg.svd:**
        
        >>> A = np.random.randn(100, 80)
        >>> U, s, Vt = np.linalg.svd(A, full_matrices=False)
        >>> X = SVD(U, s, Vt.T)  # Note: Vt.T to get V!
        """
        # Input validation
        if not isinstance(U, ndarray) or not isinstance(V, ndarray):
            raise TypeError("U and V must be numpy arrays")
        
        if U.ndim != 2 or V.ndim != 2:
            raise ValueError("U and V must be 2D arrays")
        
        if not isinstance(s, ndarray):
            raise TypeError("s must be a numpy array")
        
        # Handle different input formats for s
        if s.ndim == 1:
            # s is a vector of singular values
            sing_vals = s
        elif s.ndim == 2:
            sing_vals = np.diag(s)
        else:
            raise ValueError(f"s must be 1D or 2D array, got {s.ndim}D")
        
        # Ensure singular values are float type (not integers)
        if not np.issubdtype(sing_vals.dtype, np.floating):
            sing_vals = sing_vals.astype(float)
        
        # Dimension checks
        r = U.shape[1]
        k = V.shape[1]
        
        if len(sing_vals) != min(r, k):
            raise ValueError(
                f"Number of singular values ({len(sing_vals)}) does not match "
                f"min(U.shape[1], V.shape[1]) = min({r}, {k}) = {min(r, k)}"
            )
        
        # Create a diagonal matrix S for storage in _matrices[1]
        # Square case
        if r == k:
            S = np.diag(sing_vals)
        # Rectangular case
        else:
            S = np.zeros((r, k), dtype=sing_vals.dtype)
            np.fill_diagonal(S, sing_vals)
        
        # Call parent constructor
        super().__init__(U, S, V, **extra_data)
        

    ## SPECIFIC PROPERTIES
    @property
    def s(self) -> ndarray:
        """Return the singular values as a 1D array.
        
        Extracts the diagonal from the stored diagonal matrix S.
        
        Returns
        -------
        ndarray, shape (r,)
            Singular values (diagonal of S). Returns a copy for writability.
            
        Notes
        -----
        This extracts the diagonal from the 2D matrix S stored in _matrices[1].
        For SVD, S is guaranteed to be diagonal.
        Returns a copy to allow modification.
        """
        return np.diagonal(self.S).copy()
    
    def sing_vals(self) -> ndarray:
        """Return the singular values as a 1D array.
        
        This is an alias for the `s` property, provided for API consistency with QuasiSVD (which computes singular values on demand).
        
        Returns
        -------
        ndarray, shape (r,)
            Singular values in descending order (non-negative).
            
        Notes
        -----
        For SVD, this is O(1) as it extracts the diagonal from S.
        For QuasiSVD, this is O(r³) as it computes svd(S).
        
        Examples
        --------
        >>> X = SVD.generate_random((100, 80), np.logspace(0, -5, 10))
        >>> s = X.sing_vals()
        >>> print(s)  # [1.0, 0.1, 0.01, ..., 1e-5]
        >>> assert np.array_equal(s, X.s)  # True (same values)
        """
        return self.s
    
    def full(self) -> ndarray:
        """Reconstruct the full dense matrix from its SVD representation.
        
        Computes X = (U * s)  @ V.H to obtain the full matrix.
        
        Returns
        -------
        ndarray, shape (m, n)
            The full dense matrix represented by the SVD.
            
        Notes
        -----
        This operation is O(mnr) and should be avoided for large matrices.
        Use low-rank operations whenever possible.
        
        Examples
        --------
        >>> A = np.random.randn(100, 80)
        >>> X = SVD.reduced_svd(A)
        >>> A_reconstructed = X.full()
        >>> assert np.allclose(A, A_reconstructed)  # True within numerical precision
        """
        # For rectangular matrices, use only first r columns of U and Vh
        r = self.rank
        return np.einsum('ik,k,kj->ij', self.U[:, :r], self.s, self.Vh[:r, :])
    
    @property
    def T(self) -> SVD:
        """Transpose of the SVD matrix, returning another SVD.
        
        For X = U @ diag(s) @ V.H (where V.H is Hermitian conjugate of V),
        the transpose is:
            X.T = (U @ diag(s) @ V.H).T
                = (V.H).T @ diag(s).T @ U.T
                = V* @ diag(s) @ U.T  (where V* is conjugate of V)
        
        Note: This is the transpose WITHOUT conjugation.
        For Hermitian conjugate (conjugate transpose), use X.H instead.
        
        Returns
        -------
        SVD
            Transposed matrix in SVD format (not QuasiSVD).
            
        Notes
        -----
        This is O(1) as it only swaps references, doesn't copy data.
        The result is guaranteed to be an SVD (diagonal structure preserved).
        
        Examples
        --------
        >>> X = SVD.generate_random((100, 80), np.ones(10))
        >>> print(X.shape)  # (100, 80)
        >>> print(X.T.shape)  # (80, 100)
        >>> assert isinstance(X.T, SVD)  # True
        """
        # Transpose: swap U and V, conjugate both
        # V* @ diag(s) @ U.T where V* is conjugate of V  
        return SVD(self.V.conj(), self.s, self.U.conj())
    
    @property
    def H(self) -> SVD:
        """Hermitian conjugate (conjugate transpose) of the SVD matrix.
        
        For X = U @ diag(s) @ V.H, the Hermitian conjugate is:
            X.H = (U @ diag(s) @ V.H).H
                = V @ diag(s) @ U.H
        
        This is equivalent to X.T.conj() or X.conj().T.
        
        Returns
        -------
        SVD
            Hermitian conjugate in SVD format.
            
        Notes
        -----
        This is O(1) as it only swaps references, doesn't copy data.
        For real matrices, X.H == X.T.
        
        Examples
        --------
        >>> # For complex matrix
        >>> A = np.random.randn(10, 8) + 1j * np.random.randn(10, 8)
        >>> X = SVD.reduced_svd(A)
        >>> X_H = X.H
        >>> assert np.allclose(X_H.full(), A.T.conj())
        """
        # Hermitian: swap U and V without conjugation
        # This works because V is already stored to represent V.H in the original matrix
        return SVD(self.V, self.s, self.U)
    
    
    def _compute_storage_size(self) -> int:
        """Compute storage size for SVD format.
        
        
        Storage breakdown:
        - U: m × r elements
        - S: r × r or r × k elements (diagonal matrix stored as 2D array)
        - V: n × k elements
        - Total: m*r + r*k + n*k
        
        Returns
        -------
        int
            Total number of elements stored.
        """
        # Count U, S (full 2D matrix), and V
        return self.U.size + self.S.size + self.V.size
    
    def norm(self, ord: str | int = 'fro') -> float:
        """Calculate matrix norm, optimized for SVD representation.
        
        For orthogonal U and V, common norms are computed directly from singular
        values without forming the full matrix. This is MUCH faster: O(r) vs O(mnr).
        
        Parameters
        ----------
        ord : str or int, optional
            Order of the norm. Default is 'fro' (Frobenius).
            Optimized norms (computed from singular values):
            - 'fro': Frobenius norm = ||s||₂
            - 2: Spectral norm (largest singular value) = max(s)
            - 'nuc': Nuclear norm (sum of singular values) = sum(s)
            Other norms fall back to full matrix computation.
        
        Returns
        -------
        float
            The requested norm of the matrix.
            
        Notes
        -----
        **Result caching**: Norms are cached after first computation for efficiency.
        Each norm type is cached separately.
        
        **Orthogonality check**: IMPORTANT: This method assumes U and V are orthogonal for efficient computation.
        If they are not, the computed norm is incorrect.
        """
        # Check if norm is already cached
        if ord in self._cache:
            return self._cache[ord]
        
        # Compute norm using singular values directly (assuming orthogonality)
        if ord == 'fro':
            self._cache[ord] = np.linalg.norm(self.s)
        elif ord == 2:
            self._cache[ord] = np.max(self.s) if len(self.s) > 0 else 0.0
        elif ord == 'nuc':
            self._cache[ord] = np.sum(self.s)
        else:
            # For other norms, compute from full matrix
            self._cache[ord] = np.linalg.norm(self.full(), ord=ord)
        
        return self._cache[ord]

    
    ## CLASS METHODS
    @classmethod
    def singular_values(cls, X: SVD | LowRankMatrix | ndarray) -> ndarray:
        "Extract singular values from any matrix type."
        if isinstance(X, SVD):
            return X.sing_vals()
        elif isinstance(X, LowRankMatrix):
            return SVD.from_low_rank(X).sing_vals()
        else:
            return np.linalg.svd(X, compute_uv=False)

    @classmethod
    def from_quasiSVD(cls, mat: QuasiSVD) -> SVD:
        """Convert QuasiSVD to SVD by diagonalizing the middle matrix.
        
        This computes the SVD of the non-diagonal middle matrix S to obtain a true SVD representation with diagonal singular values.
        
        Parameters
        ----------
        mat : QuasiSVD
            QuasiSVD matrix to convert.
        
        Returns
        -------
        SVD
            SVD representation with diagonal singular values.
            
        Notes
        -----
        Computational cost: O(r³) where r is the rank of S.
        This is equivalent to calling mat.to_svd().
        """
        return mat.to_svd()

    @classmethod
    def from_low_rank(cls, mat: LowRankMatrix, **extra_data) -> SVD:
        """Convert any LowRankMatrix to SVD format.
        
        This performs QR decompositions on the outer factors and SVD on the middle product to obtain an efficient SVD representation.
        
        Parameters
        ----------
        mat : LowRankMatrix
            Low-rank matrix in any format (e.g., product of multiple matrices).
        **extra_data : dict, optional
            Additional metadata to store with the result.
        
        Returns
        -------
        SVD
            SVD representation of the matrix.
        """
        mat = mat.compress()
        # QR decomposition of the first matrix
        Q1, R1 = la.qr(mat._matrices[0], mode='economic')
        # QR decomposition of the last matrix
        Q2, R2 = la.qr(mat._matrices[-1].T.conj(), mode='economic')
        # SVD of the middle matrix
        middle = np.linalg.multi_dot([R1] + mat._matrices[1:-1] + [R2.T.conj()])
        U, s, Vh = np.linalg.svd(middle, full_matrices=False)
        # Create the SVD
        U = Q1.dot(U)
        V = Q2.dot(Vh.T.conj())
        return cls(U, s, V, **extra_data)

    @classmethod
    def from_matrix(cls, mat: ndarray | LowRankMatrix, **extra_data) -> SVD:
        """Convert any matrix type to SVD format (automatic dispatch).
        
        This is the universal converter that automatically selects the best method based on input type. Use this when you don't know the input type.
        
        Parameters
        ----------
        mat : ndarray, LowRankMatrix, QuasiSVD, or SVD
            Matrix to convert to SVD format.
        **extra_data : dict, optional
            Additional metadata to store with the result.
        
        Returns
        -------
        SVD
            SVD representation of the matrix.
        """
        if isinstance(mat, SVD):
            return mat
        elif isinstance(mat, QuasiSVD):
            return mat.to_svd()
        elif isinstance(mat, LowRankMatrix):
            return cls.from_low_rank(mat, **extra_data)
        else:
            return cls.reduced_svd(mat, **extra_data)

    @classmethod
    def full_svd(cls, mat: ndarray, **extra_data) -> SVD:
        """Compute a full SVD"""
        u, s, vh = np.linalg.svd(mat, full_matrices=True)
        return cls(u, s, vh.T.conj(), **extra_data)

    @classmethod
    def reduced_svd(cls, mat: ndarray, **extra_data) -> SVD:
        """Compute a reduced SVD of rank r"""
        u, s, vh = np.linalg.svd(mat, full_matrices=False)
        return cls(u, s, vh.T.conj(), **extra_data)

    @classmethod
    def truncated_svd(cls, mat: SVD | LowRankMatrix | ndarray, r: int = None, rtol: float = DEFAULT_RTOL, atol: float = DEFAULT_ATOL, **extra_data) -> SVD:
        """Compute truncated SVD with automatic rank selection.

        First converts the input to SVD format (if needed), then truncates small singular values based on rank or tolerance criteria.
        
        Parameters
        ----------
        mat : SVD, LowRankMatrix, or ndarray
            Input matrix to decompose and truncate.
        r : int, optional
            Target rank. If specified, keep only the r largest singular values.
            Takes priority over tolerances. Default is None.
        rtol : float, optional
            Relative tolerance. Singular values < rtol * σ_max are removed.
            Takes priority over atol. Default is DEFAULT_RTOL (None).
        atol : float, optional
            Absolute tolerance. Singular values < atol are removed.
            Default is DEFAULT_ATOL (machine precision).
        **extra_data : dict, optional
            Additional metadata to store.

        Returns
        -------
        SVD
            Truncated SVD with reduced rank.
            
        Notes
        -----
        **Truncation priority** (from highest to lowest):
        1. r (explicit rank)
        2. rtol (relative tolerance)
        3. atol (absolute tolerance)
        
        Examples
        --------
        **Fixed rank truncation:**
        
        >>> A = np.random.randn(100, 80)
        >>> X_r10 = SVD.truncated_svd(A, r=10)
        >>> print(X_r10.rank)  # 10
        
        **Relative tolerance (adaptive):**
        
        >>> # Keep singular values > 1e-6 * max(singular values)
        >>> X_rel = SVD.truncated_svd(A, rtol=1e-6)
        >>> print(X_rel.rank)  # Depends on spectrum
        
        **Absolute tolerance:**
        
        >>> # Keep singular values > 1e-10
        >>> X_abs = SVD.truncated_svd(A, atol=1e-10)
        """
        X = cls.from_matrix(mat, **extra_data)
        return X.truncate(r=r, rtol=rtol, atol=atol)

    @classmethod
    def generate_random(cls, shape: tuple, sing_vals: ndarray, seed: int = 1234, is_symmetric: bool = False, **extra_data) -> SVD:
        """Generate a random SVD with given singular values.
        
        Parameters
        ----------
        shape : tuple
            Shape of the matrix
        sing_vals : ndarray
            Singular values
        seed : int, optional
            Random seed, by default 1234
        is_symmetric : bool, optional
            Whether the generated matrix is symmetric, by default False

        Returns
        -------
        SVD
            SVD matrix generated randomly
            """
        np.random.seed(seed) # for reproducibility
        (m, n) = shape
        r = len(sing_vals)
        if is_symmetric:
            A = np.random.rand(m, r)
            Q, _ = la.qr(A, mode='economic')
            return SVD(Q, sing_vals, Q, **extra_data)
        else:
            A = np.random.rand(m, r)
            Q1, _ = la.qr(A, mode='economic')
            B = np.random.rand(n, r)
            Q2, _ = la.qr(B, mode='economic')
            return SVD(Q1, sing_vals, Q2, **extra_data)
        
    #%% INSTANCE METHODS
    def truncate(self,
                 r: int = None,
                 rtol: float = DEFAULT_RTOL,
                 atol: float = DEFAULT_ATOL,
                 inplace: bool = False) -> SVD:
        """
        Truncate the SVD.
        The rank is prioritized over the tolerance.
        The relative tolerance is prioritized over absolute tolerance. 
        NOTE: By default, the truncation is done with respect to the machine precision (DEFAULT_ATOL).

        Parameters
        ----------
        r : int, optional
            Rank, by default None
        rtol : float, optional
            Relative tolerance, by default None.
            Uses the largest singular value as reference.
        atol : float, optional
            Absolute tolerance, by default DEFAULT_ATOL.
        inplace : bool, optional
            If True, modify the matrix inplace, by default False
        """
        # If all are None, do nothing
        if r is None and rtol is None and atol is None:
            return self
        
        # Compute the rank associated to the tolerance
        if r is None:
            if rtol is not None:
                r = np.sum(self.sing_vals() > self.sing_vals()[0] * rtol)
            else:
                r = np.sum(self.sing_vals() > atol)
        
        # Truncate
        (m,n) = self.shape
        if r == 0: # trivial case
            U = np.zeros((m, 0))
            s = np.zeros(0)
            V = np.zeros((n, 0))
        else: # general case
            U = self.U[:, :r]
            s = self.sing_vals()[:r]
            V = self.V[:, :r]
        if inplace:
            self.U = U
            self.S = np.diag(s)
            self.V = V
            self.r = r
            self._cache.clear()  # Invalidate cached values after modification
            return self
        else:
            return SVD(U, s, V)

    def truncate_perpendicular(self, 
                               r: int = None,
                               rtol: float = DEFAULT_RTOL,
                               atol: float = DEFAULT_ATOL,
                               inplace: bool = False) -> SVD:
        """Perpendicular truncation of SVD. (keep the minimal rank)

        Rank is prioritized over tolerance.
        Relative tolerance is prioritized over absolute tolerance.
        NOTE: By default, the truncation is done with respect to the machine precision (DEFAULT_ATOL).

        Parameters
        ----------
        r : int, optional
            Rank, by default None
        rtol : float, optional
            Relative tolerance, by default None.
            Uses the largest singular value as reference.
        atol : float, optional
            Absolute tolerance, by default DEFAULT_ATOL.
        inplace : bool, optional
            If True, modify the matrix inplace, by default False

        """
        # If all are None, do nothing
        if r is None and rtol is None and atol is None:
            return self
        
        # Compute the rank associated to the tolerance
        if r is None:
            if rtol is not None:
                r = np.sum(self.sing_vals() > self.sing_vals()[0] * rtol)
            else:
                r = np.sum(self.sing_vals() > atol)
        
        # Truncate perpendicular
        (m,n) = self.shape
        if r == min(m,n): # matrix is full rank -> perpendicular truncation is trivial
            U = np.zeros((m, 0))
            s = np.zeros(0)
            V = np.zeros((n, 0))
        else: # general case
            U = self.U[:, r:]
            s = self.sing_vals()[r:]
            V = self.V[:, r:]
        if inplace:
            self.U = U
            self.S = np.diag(s)
            self.V = V
            self.r = r
            self._cache.clear()  # Invalidate cached values after modification
            return self
        else:
            return SVD(U, s, V)
    
    def __imul__(self, other: float | LowRankMatrix | ndarray) -> LowRankMatrix | ndarray:
        """In-place scalar multiplication for SVD class.
        
        Overrides parent's __imul__ to properly handle the 1D singular values vector `s`
        instead of the 2D matrix `S`.
        
        Parameters
        ----------
        other : float, complex, LowRankMatrix, or ndarray
            Scalar: multiplies singular values in-place
            Matrix: computes Hadamard (element-wise) product (returns NEW object)
            
        Returns
        -------
        SVD or ndarray
            For scalars: returns self (modified in-place)
            For matrices: returns new Hadamard product object
        """
        if isinstance(other, (LowRankMatrix, ndarray)):
            # Not in-place for matrices - use parent's hadamard method
            return self.hadamard(other)
        else:
            # Scalar multiplication: modify s directly
            if isinstance(other, (complex, np.complexfloating)):
                self.S = self.S.astype(np.complex128)
            self.S = self.S * other
            # Clear cache since we modified the matrix
            self._cache.clear()
        return self
        
    def __add__(self, other: SVD | LowRankMatrix | ndarray, auto_truncate: bool = AUTOMATIC_TRUNCATION) -> SVD | LowRankMatrix | ndarray:
        "Specific addition for SVDs."
        if isinstance(other, SVD):
            U_tilde = np.hstack((self.U, other.U))
            V_tilde = np.hstack((self.V, other.V)) 
            S_tilde = la.block_diag(self.S, other.S)
            U_hat, R = la.qr(U_tilde, mode='economic')
            V_hat, S = la.qr(V_tilde, mode='economic')
            middle = np.linalg.multi_dot([R, S_tilde, S.T.conj()])
            U_m, s_m, Vh_m = np.linalg.svd(middle, full_matrices=False)
            new_U = np.dot(U_hat, U_m)
            new_V = np.dot(V_hat, Vh_m.T.conj())
            output = SVD(new_U, s_m, new_V)
            if auto_truncate:
                output = output.truncate(atol=DEFAULT_ATOL)
        else:
            output = super().__add__(other)
        return output
    
    def dot(self, other: SVD | LowRankMatrix | ndarray, auto_truncate: bool = AUTOMATIC_TRUNCATION, dense_output: bool = False) -> SVD | LowRankMatrix | ndarray:
        """Matrix multiplication optimized for SVD × SVD.
        
        Efficiently multiplies two SVD matrices while preserving the SVD structure.
        The resulting rank is min(rank(X), rank(Y)).
        
        Parameters
        ----------
        other : SVD, LowRankMatrix, or ndarray
            Matrix to multiply with.
        auto_truncate : bool, optional
            Whether to automatically truncate small singular values after multiplication.
            Default is AUTOMATIC_TRUNCATION (False by default).
        dense_output : bool, optional
            If True, return dense ndarray. If False, return SVD/LowRankMatrix.
            Default is False.
            
        Returns
        -------
        SVD, LowRankMatrix, or ndarray
            Result of matrix multiplication.
            
        Notes
        -----
        Algorithm for SVD @ SVD: Computes M = S1 @ V1.T @ U2 @ S2, then SVD of M.
        Computational cost: O(r³) where r = min(r1, r2).
        
        Examples
        --------
        >>> X = SVD.generate_random((100, 80), np.ones(20))
        >>> Y = SVD.generate_random((80, 60), np.ones(15))
        >>> Z = X.dot(Y)
        >>> print(Z.shape)  # (100, 60)
        >>> print(Z.rank)   # min(20, 15) = 15
        """
        if isinstance(other, SVD):
            # Compute the middle matrix
            middle = np.linalg.multi_dot([self.S, self.Vh, other.U, other.S])
            # SVD of the middle matrix
            U_m, s_m, Vh_m = np.linalg.svd(middle, full_matrices=False)
            # New U and V
            new_U = np.dot(self.U, U_m)
            new_V = np.dot(other.V, Vh_m.T.conj())
            output = SVD(new_U, s_m, new_V)
            if auto_truncate:
                output = output.truncate(atol=DEFAULT_ATOL)
        else:
            output = super().dot(other)
        if dense_output:
            return output.todense()
        else:
            return output
    
    def hadamard(self, other: SVD | LowRankMatrix | ndarray, auto_truncate: bool = AUTOMATIC_TRUNCATION) -> SVD | LowRankMatrix | ndarray:
        """Faster version of the Hadamard product for SVDs."""
        if isinstance(other, SVD) and not self.rank * other.rank >= min(self.shape):
            # The new matrices U and V are obtained from transposed Khatri-Rao products
            new_U = la.khatri_rao(self.Uh, other.Uh).T.conj()
            new_V = la.khatri_rao(self.Vh, other.Vh).T.conj()
            # The new singular values are obtained from the Kronecker product
            new_S = np.kron(self.sing_vals(), other.sing_vals())
            output = SVD(new_U, new_S, new_V)
            if auto_truncate:
                output = output.truncate(atol=DEFAULT_ATOL)
        else:
            output = super().hadamard(other, auto_truncate=auto_truncate)
        return output
    
    def pseudoinverse(self, rtol: float = None, atol: float = DEFAULT_ATOL) -> SVD:
        """
        Compute the Moore-Penrose pseudoinverse X⁺ efficiently.
        
        For SVD X = U @ diag(s) @ V.H, the pseudoinverse is:
            X⁺ = V @ diag(s⁺) @ U.H
        where s⁺[i] = 1/s[i] if s[i] > threshold, else 0.
        
        This is more efficient than QuasiSVD.pseudoinverse() since S is already diagonal.
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
        SVD
            Pseudoinverse in SVD format (guaranteed diagonal structure).
        
        Notes
        -----
        Computational cost: O(r) for computing s⁺ (vs O(r³) for QuasiSVD).
        The threshold is: max(rtol * max(s), atol)
        
        Examples
        --------
        >>> X = SVD.generate_random((100, 80), np.logspace(0, -5, 20))
        >>> X_pinv = X.pseudoinverse()
        >>> # Check: X @ X_pinv @ X ≈ X
        >>> reconstruction = X @ X_pinv @ X
        >>> np.allclose(X.full(), reconstruction.full())
        True
        
        See Also
        --------
        solve : Solve linear system Xx = b
        lstsq : Least squares solution
        """
        # Determine threshold for zero singular values
        if rtol is None:
            rtol = max(self.shape) * np.finfo(self.dtype).eps
        threshold = max(rtol * np.max(self.s), atol)
        
        # Compute pseudoinverse of singular values (element-wise)
        s_pinv = np.zeros_like(self.s)
        mask = self.s > threshold
        s_pinv[mask] = 1.0 / self.s[mask]
        
        # X⁺ = V @ diag(s⁺) @ U.H  (stored as SVD with V, s⁺, U)
        return SVD(self.V, s_pinv, self.U, **self._extra_data)
    
    def solve(self, b: ndarray, method: str = 'direct') -> ndarray:
        """
        Solve the linear system Xx = b efficiently using SVD structure.
        
        For square full-rank matrices, solves Xx = b directly.
        For rectangular or rank-deficient matrices, computes the least-squares solution.
        
        Parameters
        ----------
        b : ndarray
            Right-hand side vector or matrix. Shape (m,) or (m, k).
        method : str, optional
            Solution method:
            - 'direct': Use SVD factorization (default, very fast: O(mr + rk))
              x = V @ diag(1/s) @ U.T @ b
            - 'lstsq': Use pseudoinverse (handles rank deficiency)
        
        Returns
        -------
        ndarray
            Solution x. Shape (n,) or (n, k).
        
        Raises
        ------
        ValueError
            If dimensions are incompatible.
        LinAlgError
            If matrix is singular and method='direct'.
        
        Notes
        -----
        The 'direct' method is faster than QuasiSVD.solve() since S is diagonal:
        - SVD.solve: O(mr + rk) where k is the number of RHS vectors
        - QuasiSVD.solve: O(mr² + r²k) due to S inversion
        
        For rank-deficient matrices, use method='lstsq' or call lstsq() directly.
        
        Examples
        --------
        >>> X = SVD.generate_random((100, 100), np.ones(20))
        >>> b = np.random.randn(100)
        >>> x = X.solve(b)
        >>> # Check: X @ x ≈ b
        >>> np.allclose(X @ x, b)
        True
        
        See Also
        --------
        lstsq : Least squares solution (handles rank deficiency)
        pseudoinverse : Compute Moore-Penrose pseudoinverse
        """
        # Validate inputs
        if b.ndim == 1:
            if b.shape[0] != self.shape[0]:
                raise ValueError(f"Dimension mismatch: b has {b.shape[0]} elements, matrix has {self.shape[0]} rows")
        else:
            if b.shape[0] != self.shape[0]:
                raise ValueError(f"Dimension mismatch: b has {b.shape[0]} rows, matrix has {self.shape[0]} rows")
        
        if method == 'direct':
            # Check for zero singular values
            if np.any(self.s < DEFAULT_ATOL):
                raise np.linalg.LinAlgError(
                    "Matrix is singular or rank-deficient. Use method='lstsq' or lstsq() for least-squares solution."
                )
            # x = V @ diag(1/s) @ U.H @ b  (U.H for complex, U.T for real)
            # = V @ ((1/s) * (U.H @ b))
            Uh_b = self.Uh @ b  # Uses Uh which handles both real and complex correctly
            if b.ndim == 1:
                return self.V @ ((1.0 / self.s) * Uh_b)
            else:
                # For multiple RHS, need to broadcast correctly
                return self.V @ ((1.0 / self.s)[:, np.newaxis] * Uh_b)
        elif method == 'lstsq':
            return self.lstsq(b)
        else:
            raise ValueError(f"Unknown method '{method}'. Use 'direct' or 'lstsq'.")
    
    def lstsq(self, b: ndarray, rtol: float = None, atol: float = DEFAULT_ATOL) -> ndarray:
        """
        Compute the least-squares solution to Xx ≈ b.
        
        Minimizes ||Xx - b||₂ using the pseudoinverse: x = X⁺ @ b
        This is optimized for SVD with diagonal structure.
        
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
        
        Notes
        -----
        This is faster than QuasiSVD.lstsq() due to diagonal S:
        - SVD.lstsq: O(mr + rk) for computing x
        - QuasiSVD.lstsq: O(r³ + mr + rk) due to SVD(S)
        
        Examples
        --------
        >>> X = SVD.generate_random((100, 80), np.logspace(0, -10, 20))
        >>> b = np.random.randn(100)
        >>> x = X.lstsq(b)
        >>> # x minimizes ||X @ x - b||
        >>> residual = np.linalg.norm(X @ x - b)
        
        See Also
        --------
        pseudoinverse : Compute pseudoinverse
        solve : Solve linear system (for full-rank matrices)
        """
        # Validate inputs
        if b.ndim == 1:
            if b.shape[0] != self.shape[0]:
                raise ValueError(f"Dimension mismatch: b has {b.shape[0]} elements, matrix has {self.shape[0]} rows")
        else:
            if b.shape[0] != self.shape[0]:
                raise ValueError(f"Dimension mismatch: b has {b.shape[0]} rows, matrix has {self.shape[0]} rows")
        
        # Determine threshold for zero singular values
        if rtol is None:
            rtol = max(self.shape) * np.finfo(self.dtype).eps
        threshold = max(rtol * np.max(self.s), atol)
        
        # Compute pseudoinverse of singular values
        s_pinv = np.zeros_like(self.s)
        mask = self.s > threshold
        s_pinv[mask] = 1.0 / self.s[mask]
        
        # x = V @ diag(s⁺) @ U.H @ b
        U_T_b = self.Uh.dot(b)
        if b.ndim == 1:
            return self.V.dot((s_pinv * U_T_b))
        else:
            # For multiple RHS, need to broadcast correctly
            return self.V.dot((s_pinv[:, np.newaxis] * U_T_b))
    
    def sqrtm(self, inplace: bool = False) -> SVD:
        """
        Compute the matrix square root X^{1/2} such that X^{1/2} @ X^{1/2} = X.
        
        For SVD X = U @ diag(s) @ V.H, the square root is:
            X^{1/2} = U @ diag(√s) @ V.H
        where √s is the element-wise square root of singular values.
        
        This is MUCH simpler and faster than QuasiSVD.sqrtm() since S is diagonal.
            
        Parameters
        ----------
        inplace : bool, optional
            If True, modify the current object in-place. Default is False.
            
        Returns
        -------
        SVD
            Matrix square root in SVD format.
        
        Notes
        -----
        Computational cost: O(r) for computing √s (vs O(r³) for QuasiSVD).
        For matrices with negative eigenvalues, the square root may be complex.
        
        Examples
        --------
        >>> X = SVD.generate_random((100, 100), np.array([4.0, 9.0, 16.0]), is_symmetric=True)
        >>> X_sqrt = X.sqrtm()
        >>> # Check: X_sqrt @ X_sqrt ≈ X
        >>> reconstruction = X_sqrt @ X_sqrt
        >>> np.allclose(X.full(), reconstruction.full())
        True
        
        See Also
        --------
        expm : Matrix exponential
        """
        if not self.is_symmetric():
            raise NotImplementedError(
                "Matrix square root is defined only for symmetric matrices (U == V)."
            )
        s_sqrt = np.sqrt(self.s)
        
        if inplace:
            self.S = np.diag(s_sqrt)
            self._cache.clear()  # Invalidate cached values after modification
            return self
        else:
            return SVD(self.U, s_sqrt, self.V, **self._extra_data)
    
    def expm(self, inplace: bool = False) -> SVD:
        """
        Compute the matrix exponential exp(X) = e^X.
        
        For SVD X = U @ diag(s) @ V.H, if X is Hermitian (U == V for real, U == V.conj() for complex), then:
            exp(X) = U @ diag(exp(s)) @ U.H
        where exp(s) is the element-wise exponential of singular values.
        
        This is MUCH faster than matrix exponentiation for general matrices:
        - SVD.expm (symmetric): O(r) for computing exp(s)
        - General expm: O(n³) using Padé approximation
        
        Parameters
        ----------
        inplace : bool, optional
            If True, modify the current object in-place. Default is False.
            
        Returns
        -------
        SVD
            Matrix exponential in SVD format.
        
        Raises
        ------
        ValueError
            If matrix is not square.
        NotImplementedError
            If matrix is not symmetric (U != V).
        
        Notes
        -----
        This method currently only supports symmetric matrices where U == V.
        For general matrices, the eigenvalue decomposition would be needed.
        
        Examples
        --------
        >>> X = SVD.generate_random((100, 100), np.array([1.0, 0.5, 0.1]), is_symmetric=True)
        >>> X_exp = X.expm()
        >>> print(X_exp.s)  # [e^1.0, e^0.5, e^0.1] ≈ [2.718, 1.649, 1.105]
        
        See Also
        --------
        sqrtm : Matrix square root
        """
        # Check if matrix is square
        if self.shape[0] != self.shape[1]:
            raise ValueError(f"Matrix must be square for expm, got shape {self.shape}")
        
        # Check if matrix is complex
        if np.iscomplexobj(self.U) or np.iscomplexobj(self.V) or np.iscomplexobj(self.s):
            raise NotImplementedError(
                "Matrix exponential not implemented for complex matrices. "
                "Use scipy.linalg.expm(X.full()) instead."
            )
        
        # Check if matrix is symmetric (U == V)
        if not self.is_symmetric():
            raise NotImplementedError(
                "Matrix exponential is defined only for symmetric matrices (U == V). "
                "For general matrices, use scipy.linalg.expm(X.full())."
            )
        
        # Compute exp(s) element-wise
        s_exp = np.exp(self.s)
        
        if inplace:
            self.S = np.diag(s_exp)
            self._cache.clear()  # Invalidate cached values after modification
            return self
        else:
            return SVD(self.U, s_exp, self.V, **self._extra_data)
        
        
        
        