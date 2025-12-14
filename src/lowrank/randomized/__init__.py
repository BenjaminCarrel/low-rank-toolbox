"""Randomized algorithms for low-rank matrix approximation.

This submodule implements state-of-the-art randomized algorithms for computing
low-rank approximations of large matrices, based on the seminal work by
Halko, Martinsson, and Tropp (2010).

Functions
---------
rangefinder : Approximate the range of a dense matrix with a target rank
adaptive_rangefinder : Approximate the range of a dense matrix with a target tolerance
randomized_svd : Implementation of the randomized SVD algorithm with a prescribed rank
adaptive_randomized_svd : Adaptive randomized SVD with tolerance-based rank selection
generalized_nystrom : Generalized Nyström method for low-rank approximation

References
----------
.. [1] Halko, N., Martinsson, P. G., & Tropp, J. A. (2010).
       Finding structure with randomness: Probabilistic algorithms for
       constructing approximate matrix decompositions.
       SIAM Review, 53(2), 217-288.

.. [2] Nakatsukasa, Y. (2019).
       Fast and stable randomized low-rank matrix approximation.
       arXiv preprint arXiv:1902.02138.

Author: Benjamin Carrel, University of Geneva
"""

from .generalized_nystrom import generalized_nystrom
from .randomized_svd import randomized_svd, adaptive_randomized_svd
from .rangefinder import rangefinder, adaptive_rangefinder

__all__ = [
    'generalized_nystrom',
    'randomized_svd',
    'adaptive_randomized_svd',
    'rangefinder',
    'adaptive_rangefinder',
]


