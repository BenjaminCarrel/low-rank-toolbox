# Authors: Benjamin Carrel and Rik Vorhaar
#          University of Geneva, 2022
# File for QR low-rank matrix class and functions
# Path: low_rank_toolbox/matrices/qr.py

# Imports
from __future__ import annotations
from .low_rank_matrix import LowRankMatrix
import numpy as np
from numpy import ndarray
from typing import Union
from scipy import linalg as la

#%% Define class QR
class QR(LowRankMatrix):
    """
    Class for QR decomposition
    Inherited from LowRankMatrix
    X = Q @ R
    where Q is an orthogonal matrix.
    The QR decomposition is commonly used in numerical linear algebra. 
    """

    ## ATTRIBUTES
    _format = "QR"
    Q = LowRankMatrix.create_matrix_alias(0)
    R = LowRankMatrix.create_matrix_alias(1)

    def __init__(self, Q: ndarray, R: ndarray, **extra_data):
        super().__init__(Q, R, **extra_data)

    ## PROPERTIES
    def norm(self, ord: str | int = 'fro') -> float:
        """Return the norm of the matrix"""
        return np.linalg.norm(self.R, ord)

    ## STANDARD OPERATIONS
    def __add__(self, other: QR | LowRankMatrix | ndarray) -> Union[QR, ndarray]:
        if isinstance(other, QR):
            new_Q = np.hstack((self.Q, other.Q))
            new_R = np.vstack((self.R, other.R))
            # compute QR of the new R
            Q, R = la.qr(new_R, mode='economic')
            # update Q
            new_Q = new_Q @ Q
            return QR(new_Q, R)
        else:
            return super().__add__(other)
        
    def __sub__(self, other: QR | LowRankMatrix | ndarray) -> Union[QR, ndarray]:
        if isinstance(other, QR):
            new_Q = np.hstack((self.Q, other.Q))
            new_R = np.vstack((self.R, -other.R))
            # compute QR of the new R
            Q, R = la.qr(new_R, mode='economic')
            # update Q
            new_Q = new_Q @ Q
            return QR(new_Q, R)
        else:
            return super().__sub__(other)

    def dot(self, other: Union[QR, LowRankMatrix, ndarray], side='right') -> Union[QR, ndarray]:
        # Multiply self @ other
        if side == 'right' or side=='usual':
            if isinstance(other, QR):
                M = np.linalg.multi_dot((self.R, other.Q, other.R))
                Q, R = la.qr(M)
                return QR(self.Q @ Q, R)
            else:
                return super().dot(other)
        # Multiply other @ self
        elif side == 'opposite' or side == 'left':
            if isinstance(other, QR):
                M = np.linalg.multi_dot((other.R, self.Q, self.R))
                Q, R = la.qr(M)
                return QR(other.Q @ Q, R)
            else:
                return super().dot(other, side='opposite')

    ## CLASS METHODS
    @classmethod
    def from_matrix(cls, matrix: ndarray, mode = 'economic', **extra_data):
        """Compute the (reduced) QR decomposition of a matrix"""
        Q, R = la.qr(matrix, mode=mode)
        return cls(Q, R, **extra_data)

    @classmethod
    def from_low_rank(cls, low_rank: LowRankMatrix, **extra_data):
        """Compute the (reduced) QR decomposition of a matrix from its low rank representation"""
        Q, S = la.qr(low_rank._matrices[0], mode='economic')
        R = np.linalg.multi_dot([S, *low_rank._matrices[1:]])
        return cls(Q, R, **extra_data)

    @classmethod
    def from_svd(cls, svd: SVD, **extra_data):
        """Compute the (reduced) QR decomposition of a matrix from its SVD"""
        Q, R = svd.U, svd.S @ svd.Vt
        return cls(Q, R, **extra_data)

    @classmethod
    def qr(cls, matrix: ndarray, mode = 'economic', **extra_data) -> QR:
        """Compute a QR decomposition (reduced by default)"""
        return cls.from_matrix(matrix, **extra_data)

    @classmethod
    def rrqr(cls, matrix: ndarray, **extra_data) -> QR:
        """Rank-revealing QR decomposition"""
        # TODO: implement RRQR
        return NotImplementedError

    @classmethod
    def generate_random(cls, shape, **extra_data) -> QR:
        """Generate a random QR decomposition"""
        A = np.random.randn(*shape)
        Q, R = la.qr(A, mode='economic')
        return cls(Q, R, **extra_data)
