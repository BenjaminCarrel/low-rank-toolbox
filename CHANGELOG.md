# Changelog

All notable changes to the Low-Rank Toolbox will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-12-15

### Initial Release

This is the first stable release of the Low-Rank Toolbox, a Python library for efficient low-rank matrix and tensor operations.

#### Features

**Core Data Structures**
- `LowRankMatrix`: Base class for low-rank matrix representations
- `SVD`: Singular Value Decomposition format with diagonal structure
- `QuasiSVD`: Generalized SVD with non-diagonal middle matrix
- `QR`: QR factorization format for efficient column-space operations

**Matrix Operations**
- Memory-efficient storage and manipulation of low-rank matrices
- Optimized arithmetic operations (addition, multiplication, scalar operations)
- Efficient norm computations (Frobenius, spectral, nuclear)
- Matrix-vector and matrix-matrix products
- Hadamard (element-wise) products
- Truncation and rank reduction with automatic or manual tolerance control

**Column Subset Selection (CSSP)**
- DEIM (Discrete Empirical Interpolation Method)
- QDEIM (QR-based DEIM)
- sQDEIM (Strong QDEIM with oversampling)
- ARP (Adaptive Randomized Pivoting)
- Osinsky's method
- GPODE and GPODR algorithms

**Krylov Subspace Methods**
- Extended Krylov spaces
- Rational Krylov spaces
- Inverted Krylov spaces
- Arnoldi and Lanczos algorithms
- Lyapunov equation solvers
- Sylvester equation solvers

**Randomized Algorithms**
- Randomized rangefinders
- Randomized SVD
- Generalized Nyström method for low-rank approximation

**Linear Algebra**
- Efficient QR and SVD decompositions
- Strong Rank-Revealing QR (sRRQR)
- Givens rotations
- Pseudoinverse computation
- Linear system solvers (direct and least-squares)
- Matrix square root and exponential (for symmetric matrices)

**Type Safety**
- Complete type annotations for all public APIs
- Full MyPy compliance with zero type errors
- Enhanced IDE support and autocomplete

**Testing & Quality**
- 1059+ comprehensive unit tests
- Full test coverage across all modules
- Continuous integration via GitHub Actions
- Automated documentation building

#### Documentation
- Comprehensive API documentation
- Tutorial notebooks for key features
- Examples for common use cases
- Mathematical background and references

#### Dependencies
- Python ≥ 3.10
- NumPy ≥ 1.21
- SciPy ≥ 1.7

#### Notes
- This release focuses on stability, correctness, and type safety
- All public APIs are considered stable
- Future releases will maintain backward compatibility following semantic versioning

---

## Future Releases

See [GitHub Releases](https://github.com/BenjaminCarrel/low-rank-toolbox/releases) for version history.
