"""
Test file for DEIM.py

Author: Benjamin Carrel, University of Geneva, 2023
"""

#%% Importations
import numpy as np
from numpy import ndarray
from lowrank import DEIM

#%% Test DEIM
def test_DEIM():
    A = np.random.rand(20, 5)
    Q, R = np.linalg.qr(A, mode="reduced")
    indexes = DEIM(Q)
    assert len(indexes) == 5, "The number of DEIM indexes is not correct"
    print("DEIM test passed")

test_DEIM()

# %%
