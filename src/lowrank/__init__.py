"""
Authors: Benjamin Carrel and Rik Vorhaar
        University of Geneva, 2022

Currently supported low-rank formats:

Matrices:
- Generic low-rank matrix format
- Quasi-SVD
- SVD
- QR


Techniques related to low-rank matrices:
- CSSP (Column Subset Selection Problem)
- Krylov subspace methods

"""
from .matrices import *
from .cssp import *
from .utils import *