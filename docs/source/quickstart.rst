Quick Start
===========

This guide provides quick examples to get you started with the ``lowrank`` package.

Basic Import
------------

.. code-block:: python

   import numpy as np
   from lowrank import SVD, QR, LowRankMatrix
   from lowrank import QDEIM, DEIM
   from lowrank import solve_lyapunov, solve_sylvester
   from lowrank.randomized import randomized_svd, generalized_nystrom

Low-Rank Matrix Representations
--------------------------------

Create and manipulate matrices in SVD format:

.. code-block:: python

   # Create orthonormal matrices
   m, n, r = 1000, 800, 20
   U, _ = np.linalg.qr(np.random.randn(m, r))
   V, _ = np.linalg.qr(np.random.randn(n, r))
   s = np.logspace(0, -3, r)  # Singular values with exponential decay

   # Create SVD representation (only stores U, s, V - not the full matrix!)
   X = SVD(U, s, V)
   print(f"Matrix shape: {X.shape}, Rank: {X.rank}")
   print(f"Memory savings: {X.compression_ratio():.1%}")

   # Efficient operations exploiting low-rank structure
   norm = X.norm('fro')      # Frobenius norm: O(r) instead of O(mn)
   trace = X.trace()         # Trace: O(r²) instead of O(min(m,n))
   Y = X @ X.T              # Matrix multiplication returns SVD

**Memory Efficiency Example:**

For a 1000×1000 matrix with rank 10:

- Dense storage: 1,000,000 elements
- SVD storage: 20,100 elements (50× compression!)

Computing SVD from Matrices
----------------------------

.. code-block:: python

   # Full SVD
   A = np.random.randn(500, 400)
   X_full = SVD.reduced_svd(A)

   # Truncated SVD (keep top 10 singular values)
   X_trunc = SVD.truncated_svd(A, r=10)

   # Adaptive truncation (tolerance-based)
   X_adaptive = SVD.truncated_svd(A, rtol=1e-6)
   print(f"Adaptive rank: {X_adaptive.rank}")

Column Subset Selection (CSSP)
-------------------------------

Select representative columns for interpolation:

.. code-block:: python

   # Create basis matrix (e.g., POD modes, eigenvectors, etc.)
   U, _ = np.linalg.qr(np.random.randn(1000, 10))

   # Select 10 interpolation points with guaranteed bounds
   indices, projector = QDEIM(U, return_projector=True)
   print(f"Selected indices: {indices}")

   # Interpolation property: U ≈ Projector @ U[indices, :]
   interpolated = projector @ U[indices, :]
   error = np.linalg.norm(U - interpolated, 'fro')
   print(f"Interpolation error: {error:.2e}")

**Available Algorithms:**

- ``DEIM`` - Discrete Empirical Interpolation Method
- ``QDEIM`` - QR-based DEIM with better stability
- ``sQDEIM`` - Strong QDEIM with conditioning guarantees
- ``ARP`` - Adaptive Row Pivoting
- ``gpode`` - Greedy Point Selection (Oversampled DEIM Extension)
- ``gpodr`` - Greedy Point Selection with Reduced basis
- ``Osinsky`` - Alternative column selection method

Krylov Subspace Methods
------------------------

Solve large-scale Lyapunov equations efficiently:

.. code-block:: python

   from scipy.sparse import diags

   # Large sparse matrix
   n = 10000
   A = diags([-1, 2, -1], [-1, 0, 1], shape=(n, n), format='csc')

   # Low-rank right-hand side: AX + XA^T = C
   C_left = np.random.randn(n, 5)
   C_right = np.random.randn(n, 5)
   C = LowRankMatrix(C_left, C_right.T)

   # Solve using Krylov methods (never forms full n×n solution!)
   X = solve_lyapunov(A, C, tol=1e-8)
   print(f"Solution rank: {X.rank}")
   print(f"Solution shape: {X.shape}")

**Key Features:**

- Automatically detects symmetric matrices and uses Lanczos (faster)
- Extended Krylov spaces for better convergence
- Rational Krylov with optimal poles
- Never forms full matrix - keeps low-rank structure

Randomized Low-Rank Approximation
----------------------------------

Fast approximation for large matrices:

.. code-block:: python

   # Large matrix
   A = np.random.randn(5000, 4000)

   # Randomized SVD (much faster than full SVD)
   X_approx = randomized_svd(A, r=50, p=10, q=2)
   error = np.linalg.norm(A - X_approx.to_dense(), 'fro')
   print(f"Approximation error: {error:.2e}")

   # Generalized Nyström method (for symmetric/positive-semidefinite matrices)
   A_sym = A @ A.T
   X_nystrom = generalized_nystrom(A_sym, r=50, p=10)
   print(f"Nyström rank: {X_nystrom.rank}")

**Performance Benefits:**

For a 10000×10000 matrix, rank 100 approximation:

- Full SVD: ~5 minutes
- Randomized SVD: ~5 seconds (60× faster!)

**Parameters:**

- ``r`` - Target rank
- ``p`` - Oversampling parameter (typically 5-10)
- ``q`` - Power iterations for accuracy (0-3)

Common Operations
-----------------

**Matrix Operations:**

.. code-block:: python

   X = SVD.generate_random((100, 80), np.logspace(0, -2, 10))
   Y = SVD.generate_random((100, 80), np.logspace(0, -3, 10))

   # Addition/Subtraction
   Z = X + Y  # Returns SVD of rank 20
   W = X - Y

   # Scalar multiplication
   X2 = 2 * X  # Multiplies singular values by 2

   # Matrix multiplication
   XT = X @ X.T  # Returns SVD
   XY = X @ Y.T

   # Hadamard (element-wise) product
   H = X.hadamard(Y)

**Norms and Properties:**

.. code-block:: python

   # Efficient norm computation
   norm_fro = X.norm('fro')    # Frobenius: sqrt(sum(s²))
   norm_2 = X.norm(2)          # Spectral: max(s)
   norm_nuc = X.norm('nuc')    # Nuclear: sum(s)

   # Matrix properties
   trace = X.trace()
   diagonal = X.diag()
   is_sym = X.is_symmetric()
   is_orth = X.is_orthogonal()

**Conversion:**

.. code-block:: python

   # To dense array
   A_dense = X.to_dense()

   # To sparse (if sparsity threshold met)
   A_sparse = X.to_sparse(threshold=1e-10)

   # Between formats
   X_qr = X.to_qr()
   X_quasi = X.to_quasi_svd()

Advanced Topics
---------------

**Truncation Strategies:**

.. code-block:: python

   # By rank
   X_r5 = X.truncate(r=5)

   # By relative tolerance (relative to largest singular value)
   X_rtol = X.truncate(rtol=1e-6)

   # By absolute tolerance
   X_atol = X.truncate(atol=1e-10)

   # Keep perpendicular component (residual)
   X_perp = X.truncate_perpendicular(r=5)

**Linear System Solving:**

.. code-block:: python

   # Square system
   b = np.random.randn(X.shape[0])
   x = X.solve(b)

   # Least squares (overdetermined/underdetermined)
   x_ls = X.lstsq(b)

   # Pseudoinverse
   X_pinv = X.pseudoinverse()

**Custom Krylov Spaces:**

.. code-block:: python

   from lowrank.krylov import KrylovSpace, ExtendedKrylovSpace

   # Standard Krylov space
   K = KrylovSpace(A, b, is_symmetric=True)
   K.augment_basis()  # Add more vectors

   # Extended Krylov (uses A and A^-1)
   invA = lambda x: scipy.sparse.linalg.spsolve(A, x)
   EK = ExtendedKrylovSpace(A, b, invA, is_symmetric=True)

Next Steps
----------

- Explore the :doc:`api/index` for detailed documentation
- See :doc:`examples/index` for more comprehensive examples
- Check out the `test suite <https://github.com/BenjaminCarrel/low-rank-toolbox/tree/main/tests>`_ for additional usage patterns
