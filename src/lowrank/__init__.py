# src/lowrank/__init__.py - Add comprehensive module docstring
"""
lowrank: Efficient Low-Rank Matrix Operations
=========================================================

A Python library for numerical linear algebra with low-rank matrices.

Main Modules
------------
- matrices: SVD, QR, QuasiSVD representations
- cssp: Column Subset Selection algorithms
- krylov: Krylov subspace methods
- randomized: Randomized low-rank approximations

Quick Start
-----------
>>> from lowrank import SVD
>>> import numpy as np
>>> U, _ = np.linalg.qr(np.random.randn(100, 10))
>>> s = np.logspace(0, -2, 10)
>>> V, _ = np.linalg.qr(np.random.randn(80, 10))
>>> X = SVD(U, s, V)
>>> print(X)
(100, 80) low-rank matrix rank 10 and type SVD.
"""
from .cssp import *
from .krylov import *
from .matrices import *
from .randomized import *
