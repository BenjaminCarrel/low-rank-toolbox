"""Matrix Equation Solvers.

This submodule provides iterative solvers for large-scale matrix equations
using Krylov subspace methods. These solvers are efficient for large sparse
matrices where direct methods would be computationally prohibitive.

Functions
---------
solve_sylvester : Solve the Sylvester equation AX + XB = C
solve_lyapunov : Solve the Lyapunov equation AX + XA^T = C

Author: Benjamin Carrel, University of Geneva
"""

from .sylvester_solvers import solve_sylvester
from .lyapunov_solvers import solve_lyapunov

__all__ = ['solve_sylvester', 'solve_lyapunov']