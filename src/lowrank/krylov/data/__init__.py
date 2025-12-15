"""Precomputed Data and Optimal Parameters.

This submodule provides precomputed optimal parameters for Krylov
subspace methods, particularly optimal pole selections for rational
Krylov methods.

Functions
---------
extract_optimal_poles : Extract optimal poles for rational Krylov methods
"""

from .optimal_poles import extract_optimal_poles

__all__ = ["extract_optimal_poles"]
