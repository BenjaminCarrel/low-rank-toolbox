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
    
    Standard form: X = Q @ R
    Conjugate form: X = R.H @ Q.H (when conjugate=True)
    
    where Q is an orthogonal matrix.
    For real matrices, conjugate transpose is equivalent to regular transpose.
    The QR decomposition is commonly used in numerical linear algebra.
    
    Parameters
    ----------
    Q : ndarray
        Orthogonal matrix (m x r)
    R : ndarray
        Upper triangular matrix (r x n)
    conjugate : bool, optional
        If True, represents X = R.H @ Q.H instead of X = Q @ R, by default False
        For real matrices, this is equivalent to X = R.T @ Q.T
    **extra_data
        Additional data to store with the matrix
    """

    ## ATTRIBUTES
    _format = "QR"
    
    def __init__(self, Q: ndarray, R: ndarray, conjugate: bool = False, **extra_data):
        """
        Initialize a QR decomposition.
        
        Parameters
        ----------
        Q : ndarray
            Orthogonal matrix
        R : ndarray
            Upper triangular matrix
        conjugate : bool, optional
            If True, represents X = R.H @ Q.H, by default False
        **extra_data
            Additional data to store with the matrix
        """
        # Store conjugate flag
        self._conjugate = conjugate
        
        # Initialize based on whether we're in conjugate mode
        if self._conjugate:
            # X = R.H @ Q.H
            super().__init__(R.T.conj(), Q.T.conj(), **extra_data)
        else:
            # Standard: X = Q @ R
            super().__init__(Q, R, **extra_data)
    
    ## MATRIX ALIASES
    @property
    def Q(self) -> ndarray:
        """Return Q matrix (adjusted for conjugate mode)"""
        if self._conjugate:
            # In conjugate mode, Q is actually stored as second matrix (conjugated)
            return self._matrices[1].T.conj()
        else:
            # Standard mode: Q is first matrix
            return self._matrices[0]
    
    @property
    def R(self) -> ndarray:
        """Return R matrix (adjusted for conjugate mode)"""
        if self._conjugate:
            # In conjugate mode, R is actually stored as first matrix (conjugated)
            return self._matrices[0].T.conj()
        else:
            # Standard mode: R is second matrix
            return self._matrices[1]

    ## PROPERTIES
    @property
    def T(self) -> 'QR':
        """Return the transpose of the QR matrix"""
        if self._conjugate:
            # Was R.H @ Q.H, becomes Q.conj() @ R.conj() in conjugate mode
            # which represents (R.H @ Q.H).T = Q @ R
            return QR(self.Q.conj(), self.R.conj(), conjugate=False, **self._extra_data)
        else:
            # Was Q @ R, becomes R.T @ Q.T
            return QR(self.Q, self.R, conjugate=True, **self._extra_data)
    
    @property
    def H(self) -> 'QR':
        """Return the conjugate transpose (Hermitian) of the QR matrix"""
        if self._conjugate:
            # Was R.H @ Q.H, becomes Q @ R
            return QR(self.Q.conj(), self.R.conj(), conjugate=False, **self._extra_data)
        else:
            # Was Q @ R, becomes R.H @ Q.H
            return QR(self.Q, self.R, conjugate=True, **self._extra_data)
    
    def norm(self, ord: str | int = 'fro') -> float:
        """Return the norm of the matrix"""
        return np.linalg.norm(self.R, ord)
    
    def __repr__(self) -> str:
        """String representation of the QR matrix"""
        mode = " (conjugate)" if self._conjugate else ""
        return (
            f"{self.shape} QR decomposition with rank {self.rank}{mode}"
        )

    ## STANDARD OPERATIONS
    def __add__(self, other: QR | LowRankMatrix | ndarray) -> Union[QR, ndarray]:
        if isinstance(other, QR):
            # If either QR is in conjugate mode, fall back to parent's method
            if self._conjugate or other._conjugate:
                return super().__add__(other)
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
            # If either QR is in conjugate mode, fall back to parent's method
            if self._conjugate or other._conjugate:
                return super().__sub__(other)
            new_Q = np.hstack((self.Q, other.Q))
            new_R = np.vstack((self.R, -other.R))
            # compute QR of the new R
            Q, R = la.qr(new_R, mode='economic')
            # update Q
            new_Q = new_Q @ Q
            return QR(new_Q, R)
        else:
            return super().__sub__(other)

    def dot(self, other: Union[QR, LowRankMatrix, ndarray], side='right', dense_output: bool = False) -> Union[QR, LowRankMatrix, ndarray]:
        # Multiply self @ other
        if side == 'right' or side=='usual':
            # QR @ AB -> keep QR format
            if isinstance(other, LowRankMatrix) and not self._conjugate:
                M = other.dot(self.R, side='left').todense()
                output = QR(self.Q , M)
            # RQ @ AB -> generic low-rank format
            elif self._conjugate:
                output = super().dot(other, side='right')
            # QR @ other -> keep QR format
            else:
                M = self.R.dot(other)
                output = QR(self.Q , M)
        # Multiply other @ self
        elif side == 'opposite' or side == 'left':
            # AB @ RQ -> keep QR conjugate format
            if isinstance(other, LowRankMatrix) and self._conjugate:
                M = other.dot(self.R.T.conj())
                output = QR(self.Q, M.T.conj(), conjugate=True)
            # AB @ QR -> generic low-rank format
            elif not self._conjugate:
                output = super().dot(other, side='left')
            # other @ RQ -> keep QR conjugate format
            else:
                M = other.dot(self.R.T.conj())
                output = QR(self.Q, M.T.conj(), conjugate=True)
            
        if dense_output:
            return output.todense()
        else:
            return output
                

    ## CLASS METHODS
    @classmethod
    def from_matrix(cls, matrix: ndarray, mode = 'economic', conjugate: bool = False, **extra_data):
        """Compute the (reduced) QR decomposition of a matrix
        
        Parameters
        ----------
        matrix : ndarray
            Matrix to decompose
        mode : str, optional
            QR mode ('economic' or 'complete'), by default 'economic'
        conjugate : bool, optional
            If True, the returned QR represents matrix = R.H @ Q.H instead of Q @ R
            (i.e., compute QR of matrix.H)
        **extra_data
            Additional data to store
            
        Returns
        -------
        QR
            QR decomposition object where full() reconstructs the input matrix
        """
        # Compute QR decomposition of the input matrix
        Q, R = la.qr(matrix, mode=mode)
        # Return with conjugate flag to indicate how factors should be interpreted
        return cls(Q, R, conjugate=conjugate, **extra_data)

    @classmethod
    def from_low_rank(cls, low_rank: LowRankMatrix, conjugate: bool = False, **extra_data):
        """Compute the (reduced) QR decomposition of a matrix from its low rank representation
        
        Parameters
        ----------
        low_rank : LowRankMatrix
            Low rank matrix to decompose
        conjugate : bool, optional
            If True, compute conjugate transposed QR, by default False
        **extra_data
            Additional data to store
            
        Returns
        -------
        QR
            QR decomposition object
        """
        Q, S = la.qr(low_rank._matrices[0], mode='economic')
        R = np.linalg.multi_dot([S, *low_rank._matrices[1:]])
        return cls(Q, R, conjugate=conjugate, **extra_data)

    @classmethod
    def from_svd(cls, svd, conjugate: bool = False, **extra_data):
        """Compute the (reduced) QR decomposition of a matrix from its SVD
        
        Parameters
        ----------
        svd : SVD
            SVD object to convert from
        conjugate : bool, optional
            If True, compute conjugate transposed QR, by default False
        **extra_data
            Additional data to store
            
        Returns
        -------
        QR
            QR decomposition object
        """
        Q, R = svd.U, svd.S @ svd.Vt
        return cls(Q, R, conjugate=conjugate, **extra_data)

    @classmethod
    def qr(cls, matrix: ndarray, mode = 'economic', conjugate: bool = False, **extra_data) -> QR:
        """Compute a QR decomposition (reduced by default)
        
        Parameters
        ----------
        matrix : ndarray
            Matrix to decompose
        mode : str, optional
            QR mode, by default 'economic'
        conjugate : bool, optional
            If True, compute conjugate transposed QR, by default False
        **extra_data
            Additional data to store
            
        Returns
        -------
        QR
            QR decomposition object
        """
        return cls.from_matrix(matrix, mode=mode, conjugate=conjugate, **extra_data)

    @classmethod
    def rrqr(cls, matrix: ndarray, **extra_data) -> QR:
        """Rank-revealing QR decomposition"""
        # TODO: implement RRQR
        return NotImplementedError

    @classmethod
    def generate_random(cls, shape, conjugate: bool = False, **extra_data) -> QR:
        """Generate a random QR decomposition
        
        Parameters
        ----------
        shape : tuple
            Shape of the matrix to generate
        conjugate : bool, optional
            If True, generate conjugate transposed QR, by default False
        **extra_data
            Additional data to store
            
        Returns
        -------
        QR
            Random QR decomposition object
        """
        A = np.random.randn(*shape)
        Q, R = la.qr(A, mode='economic')
        return cls(Q, R, conjugate=conjugate, **extra_data)
