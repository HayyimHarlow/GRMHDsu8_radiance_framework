"""
likelihood.py
Define misfit or likelihood between data and model features.
"""

import numpy as np

def feature_misfit(data_features: dict, model_features: dict) -> float:
    keys = ["evpa_amp", "helicity_flips"]
    diffs = []
    for k in keys:
        diffs.append(float(model_features[k] - data_features[k]))
    return float(np.sum(np.array(diffs) ** 2))


def likelihood(misfit: float) -> float:
    return float(np.exp(-misfit))

