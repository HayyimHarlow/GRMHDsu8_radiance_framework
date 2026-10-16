"""
diagnostics.py
Feature extraction: EVPA swings, helicity flips, variability spectra.
"""

import numpy as np

def evpa_swing_amplitude(evpa_series: np.ndarray) -> float:
    return float(np.max(evpa_series) - np.min(evpa_series))


def helicity_flip_count(helicity_series: np.ndarray) -> int:
    if len(helicity_series) < 2:
        return 0
    flips = np.diff(helicity_series)
    return int(np.sum(flips != 0))


def variability_spectrum(lightcurve: np.ndarray) -> np.ndarray:
    return np.abs(np.fft.rfft(lightcurve))

