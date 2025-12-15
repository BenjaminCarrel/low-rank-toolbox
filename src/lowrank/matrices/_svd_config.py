"""Configuration parameters for SVD computations.

Authors: Benjamin Carrel and Rik Vorhaar
         University of Geneva, 2022-2025
"""

import numpy as np

# Default behavior for automatic truncation in operations
# Set to False to maintain algebraic consistency (e.g., X - X is always exactly zero)
# Set to True for automatic memory management (removes near-zero singular values)
# Recommendation: Keep False and let users explicitly call .truncate() when needed
AUTOMATIC_TRUNCATION = False

# Default absolute tolerance for truncation
# Singular values below this threshold are considered zero
DEFAULT_ATOL = 100 * np.finfo(float).eps  # ~2.22e-14 for float64

# Default relative tolerance for truncation (when used)
# Singular values below max(sing_vals) * DEFAULT_RTOL are truncated
DEFAULT_RTOL = None  # Disable by default, can be set to a float value like 1e-8
