"""Krylov Subspace Methods.

This submodule implements Krylov subspace methods for solving large-scale
linear systems, eigenvalue problems, and matrix equations. Krylov methods
are particularly effective for large sparse matrices.

Submodules
----------
spaces : Krylov, extended Krylov, and rational Krylov subspace constructions
solvers : Iterative solvers for Sylvester and Lyapunov equations
utils : Arnoldi and Lanczos iteration algorithms
data : Optimal pole selection and precomputed data

Author: Benjamin Carrel, University of Geneva
"""

from .spaces import *
from .solvers import *
from .utils import *
from .data import *