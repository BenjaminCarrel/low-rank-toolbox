"""Krylov Subspace Constructions.

This submodule provides classes for constructing and working with
various types of Krylov subspaces, which are fundamental for iterative
matrix computations.

Classes
-------
KrylovSpace : Standard Krylov subspace K_m(A, b)
InvertedKrylovSpace : Inverted Krylov subspace K_m(A^{-1}, b)
ExtendedKrylovSpace : Extended Krylov subspace EK_m(A, b)
RationalKrylovSpace : Rational Krylov subspace with arbitrary poles
"""

from .extended_krylov_space import ExtendedKrylovSpace
from .inverted_krylov_space import InvertedKrylovSpace
from .krylov_space import KrylovSpace
from .rational_krylov_space import RationalKrylovSpace

__all__ = [
    "KrylovSpace",
    "InvertedKrylovSpace",
    "ExtendedKrylovSpace",
    "RationalKrylovSpace",
]
