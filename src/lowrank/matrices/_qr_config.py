"""Configuration parameters for QR computations.

Authors: Benjamin Carrel and Rik Vorhaar
         University of Geneva, 2022-2025
"""

import numpy as np

# Default behavior for automatic truncation in operations
# Set to False to maintain algebraic consistency (e.g., X + Y uses full rank sum)
# Set to True for automatic memory management (removes columns with small R diagonal)
# Recommendation: Keep False and let users explicitly call .truncate() when needed
AUTOMATIC_TRUNCATION = False

# Default absolute tolerance for truncation
# R diagonal elements below this threshold are considered zero
DEFAULT_ATOL = 100 * np.finfo(float).eps  # ~2.22e-14 for float64

# Default relative tolerance for truncation (when used)
# R diagonal elements below max(|diag(R)|) * DEFAULT_RTOL are truncated
DEFAULT_RTOL = None  # Disable by default, can be set to a float value like 1e-8
