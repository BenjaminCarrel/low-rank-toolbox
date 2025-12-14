"""Utility Functions for Column Subset Selection.

This submodule provides supporting numerical routines for CSSP algorithms,
including QR factorization variants and Givens rotations.

Functions
---------
givens : Compute Givens rotation matrix
sRRQR : Strong Rank-Revealing QR factorization
sRRQR_rank : sRRQR with prescribed rank
sRRQR_tol : sRRQR with tolerance-based rank selection

Author: Benjamin Carrel, University of Geneva
"""

from .givens import givens
from .sRRQR import sRRQR, sRRQR_rank, sRRQR_tol

__all__ = ['givens', 'sRRQR', 'sRRQR_rank', 'sRRQR_tol']