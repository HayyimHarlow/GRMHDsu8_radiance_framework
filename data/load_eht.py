"""
load_eht.py
Minimal EHT loader for SU(8 pipeline.
"""

import numpy as np
import os

def load_eht_subset(data_root="eht_data"):
    """
    Minimal placeholder loader.
    Returns dict with visibilities, errors, design_matrix.
    """

    # For now, create synthetic placeholders.
    # Replace with your actual EHT loading logic later.

    # Example: 100 visibilities, 10 parameters
    n_vis = 100
    n_params = 10

    visibilities = np.random.normal(0, 1, n_vis)
    errors = np.ones(n_vis) * 0.1
    design_matrix = np.random.normal(0, 1, (n_vis, n_params))

    return {
        "visibilities": visibilities,
        "errors": errors,
        "design_matrix": design_matrix
    }
