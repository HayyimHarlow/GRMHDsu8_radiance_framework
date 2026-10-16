"""
feature_space.py
Compute comparable features for data and models.
"""

def extract_features(evpa_series, helicity_series, lightcurve):
    from grrt_synthetic.diagnostics import (
        evpa_swing_amplitude,
        helicity_flip_count,
        variability_spectrum,
    )
    return {
        "evpa_amp": evpa_swing_amplitude(evpa_series),
        "helicity_flips": helicity_flip_count(helicity_series),
        "var_spectrum": variability_spectrum(lightcurve),
    }

