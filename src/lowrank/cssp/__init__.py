"""Column Subset Selection Problem (CSSP) Algorithms.

This submodule implements various algorithms for selecting representative
columns from a matrix.

Algorithms
----------
DEIM : Discrete Empirical Interpolation Method
QDEIM : QR-based Discrete Empirical Interpolation Method
sQDEIM : Strong QDEIM
ARP : Adaptive Randomized Pivoting
gpode : Greedy POD with Energy constraint
gpodr : Greedy POD with Residual constraint
Osinsky : Osinsky's algorithm
oversampling_sQDEIM : Oversampling strong QDEIM

Author: Benjamin Carrel, Paul Scherrer Institute, 2025
"""

from .arp import ARP
from .deim import DEIM
from .gpode import gpode
from .gpodr import gpodr
from .osinsky import Osinsky
from .oversampling_sqdeim import oversampling_sQDEIM
from .qdeim import QDEIM
from .sqdeim import sQDEIM

__all__ = [
    "DEIM",
    "QDEIM",
    "sQDEIM",
    "ARP",
    "gpode",
    "gpodr",
    "Osinsky",
    "oversampling_sQDEIM",
]
