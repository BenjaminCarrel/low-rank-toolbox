"""Configuration parameters for SVD computations.

Authors: Benjamin Carrel and Rik Vorhaar
         University of Geneva, 2022-2025
"""

import numpy as np

# Default absolute tolerance for truncation
# Singular values below this threshold are considered zero
DEFAULT_ATOL = 100 * np.finfo(float).eps  # ~2.22e-14 for float64

# Default relative tolerance for truncation (when used)
# Singular values below max(sing_vals) * DEFAULT_RTOL are truncated
DEFAULT_RTOL = None  # Disable by default, can be set to a float value like 1e-8
