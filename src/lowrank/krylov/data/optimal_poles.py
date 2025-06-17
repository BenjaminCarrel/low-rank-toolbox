"""
Wrapper for the optimal poles data set.

Author: Benjamin Carrel, University of Geneva, 2023
"""

import scipy.io as sio
import os

# Path to the data
path = os.path.dirname(os.path.abspath(__file__)) + "/"

def extract_optimal_poles(nb_poles: int) -> list:
    """
    Extract the optimal poles precomputed in MATLAB.

    Parameters
    ----------
    nb_poles : int
        Number of poles to extract. Must be in [1, 15]
    """
    # Check input
    if nb_poles < 1 or nb_poles > 15:
        raise ValueError("nb_poles must be in [1, 15]")
    
    # Extract MATLAB .mat data
    data = sio.loadmat(path + f"optimal_poles_{nb_poles}.mat")
    poles = data["opti_poles"].squeeze().tolist()[-1].squeeze().tolist()
    if nb_poles == 1:
        poles = [poles]
    return poles