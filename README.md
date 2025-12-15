![Tests](https://github.com/BenjaminCarrel/low-rank-toolbox/actions/workflows/tests.yml/badge.svg)
![Documentation](https://github.com/BenjaminCarrel/low-rank-toolbox/actions/workflows/documentation.yml/badge.svg)
![Coverage](https://img.shields.io/badge/coverage-check%20actions-blue)
![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)

# **Low-Rank Toolbox**

`lowrank` is a Python library providing efficient data structures and algorithms for numerical linear algebra with low-rank matrices. The package is designed for researchers and practitioners who need performant and memory-conscious computations.

```python
import lowrank
```

## **Features**

* **Memory-efficient storage** for low-rank matrices with multiple formats (SVD, QR, QuasiSVD)
* **Fundamental operations** optimized for low-rank structures (addition, multiplication, norms, etc.)
* **Column Subset Selection** algorithms (DEIM, QDEIM, sQDEIM, adaptive randomized pivoting (ARP), Osinsky, gpode, gpodr)
* **Krylov subspace methods** for solving large-scale linear systems (Lyapunov, Sylvester matrix equations)
* **Randomized algorithms** for fast low-rank approximation (rangefinder, randomized SVD, Nyström method)
* Built on **NumPy and SciPy** for seamless integration with existing scientific Python tools
* **1000+ tests** ensuring reliability and correctness

## **Examples**

### **Low-Rank Matrix Representations**

Create and manipulate matrices in SVD format:

```python
import numpy as np
from lowrank import SVD

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
```

### **Computing SVD from Matrices**

```python
from lowrank.matrices import SVD

# Full SVD
A = np.random.randn(500, 400)
X_full = SVD.reduced_svd(A)

# Truncated SVD (keep top 10 singular values)
X_trunc = SVD.truncated_svd(A, r=10)

# Adaptive truncation (tolerance-based)
X_adaptive = SVD.truncated_svd(A, rtol=1e-6)
print(f"Adaptive rank: {X_adaptive.rank}")
```

### **Column Subset Selection (CSSP)**

Select representative columns for interpolation:

```python
from lowrank import QDEIM

# Create basis matrix (e.g., POD modes, eigenvectors, etc.)
U, _ = np.linalg.qr(np.random.randn(1000, 10))

# Select 10 interpolation points with guaranteed bounds
indices, projector = QDEIM(U, return_projector=True)
print(f"Selected indices: {indices}")

# Interpolation property: U ≈ Projector @ U[indices, :]
interpolated = projector @ U[indices, :]
error = np.linalg.norm(U - interpolated, 'fro')
print(f"Interpolation error: {error:.2e}")
```

### **Krylov Subspace Methods**

Solve large-scale Lyapunov equations:

```python
from scipy.sparse import diags
from lowrank import solve_lyapunov, LowRankMatrix

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
```

### **Randomized Low-Rank Approximation**

Fast approximation for large matrices:

```python
from lowrank.randomized import randomized_svd, generalized_nystrom

# Large matrix
A = np.random.randn(5000, 4000)

# Randomized SVD (much faster than full SVD)
X_approx = randomized_svd(A, r=50, p=10, q=2)
error = np.linalg.norm(A - X_approx.to_dense(), 'fro')
print(f"Approximation error: {error:.2e}")

# Generalized Nyström method (for symmetric/positive-semidefinite matrices)
A_sym = A @ A.T
X_nystrom = generalized_nystrom(A_sym, r=50, p=10)
print(f"Generalized Nyström rank: {X_nystrom.rank}")
print(f"Generalized Nyström error: {np.linalg.norm(A - X_nystrom.to_dense(), 'fro'):.2e}"
```

## **Installation**

### **For Users**

```bash
# Via pip
pip install lowrank
```

### **For Developers**

To contribute or modify the package:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/BenjaminCarrel/low-rank-toolbox.git
   cd low-rank-toolbox
   ```

2. **Create the conda environment:**
   ```bash
   conda env create -f environment.yml
   conda activate low-rank-dev
   ```

3. **Install in editable mode:**
   ```bash
   pip install -e .
   ```

## **Verifying the Installation**

Run the comprehensive test suite (1000+ tests):

```bash
pytest
```

All tests should pass. If you encounter any issues, please [open an issue](https://github.com/BenjaminCarrel/low-rank-toolbox/issues).

## **License**

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## **Citation**

If you use this package in your research, please cite:

```bibtex
@software{lowrank2025,
  author = {Carrel, Benjamin and Vorhaar, Rik},
  title = {Low-Rank Toolbox: Efficient Low-Rank Matrix Computations in Python},
  year = {2025},
  url = {https://github.com/BenjaminCarrel/low-rank-toolbox}
}
```

## **Contributing**

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## **Authors**

- **Benjamin Carrel** - Paul Scherrer Institute / University of Geneva
- **Rik Vorhaar** - University of Geneva

## **Acknowledgments**

This package implements algorithms from various academic papers. See individual function docstrings for specific references.
