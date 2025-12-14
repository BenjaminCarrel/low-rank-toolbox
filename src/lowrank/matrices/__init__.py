"""Low-Rank Matrix Representations.

This submodule provides efficient representations for low-rank matrices,
enabling memory-efficient storage and fast operations on matrices that
can be expressed as products of thin factors.

Classes
-------
LowRankMatrix : Generic low-rank matrix base class
QuasiSVD : Quasi-SVD factorization (U @ S @ V.T) where S is dense
SVD : Singular Value Decomposition (U @ diag(s) @ V.T)
QR : QR factorization (Q @ R)

Authors: Benjamin Carrel and Rik Vorhaar, University of Geneva, 2022
"""

from .quasi_svd import QuasiSVD
from .svd import SVD
from .qr import QR
from .low_rank_matrix import LowRankMatrix

__all__ = ['QuasiSVD', 'SVD', 'QR', 'LowRankMatrix']