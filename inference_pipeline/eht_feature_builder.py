import numpy as np

def build_helicity_series(visibilities, design_matrix):
    """
    Construct a synthetic helicity time series from visibilities + design matrix.
    This is a proxy: sign changes in a projected visibility combination
    are treated as helicity flips.

    Parameters
    ----------
    visibilities : np.ndarray
        Complex or real visibilities array of shape (N, M, ...) or (N,).
    design_matrix : np.ndarray
        Design matrix encoding baseline/time mapping, shape (N, K, ...).

    Returns
    -------
    helicity_series : np.ndarray
        1D array of length N with integer helicity state (+1 / -1).
    """
    # Flatten visibilities to (N,) amplitude
    amp = np.abs(visibilities.reshape(visibilities.shape[0], -1)).mean(axis=1)

    # Use a simple projection from design_matrix to define a "phase driver"
    proj = design_matrix.reshape(design_matrix.shape[0], -1).mean(axis=1)

    # Synthetic helicity: sign of (proj - median(proj)) modulated by amp
    center = np.median(proj)
    helicity_raw = proj - center

    helicity_series = np.sign(helicity_raw)
    helicity_series[helicity_series == 0] = 1  # avoid zeros

    return helicity_series


def build_lightcurve(visibilities):
    """
    Construct a synthetic lightcurve from visibility amplitudes.

    Parameters
    ----------
    visibilities : np.ndarray
        Complex or real visibilities array of shape (N, M, ...) or (N,).

    Returns
    -------
    lightcurve : np.ndarray
        1D array of length N with normalized total flux proxy.
    """
    amp = np.abs(visibilities.reshape(visibilities.shape[0], -1)).mean(axis=1)

    # Normalize to [0, 1]
    if amp.max() > 0:
        lightcurve = (amp - amp.min()) / (amp.max() - amp.min())
    else:
        lightcurve = np.zeros_like(amp)

    return lightcurve


def build_eht_feature_space(eht_raw, extract_features_fn):
    """
    High-level builder: takes eht_raw dict from load_eht_subset and
    returns a feature-space dict suitable for saving to eht_feature_space.npz.

    Parameters
    ----------
    eht_raw : dict
        Must contain keys: 'visibilities', 'errors', 'design_matrix'.
    extract_features_fn : callable
        Function with signature:
            extract_features(visibilities, helicity_series, lightcurve)

    Returns
    -------
    feature_space : dict
        Dictionary returned by extract_features_fn, ready to be saved.
    """
    vis = eht_raw["visibilities"]
    dm  = eht_raw["design_matrix"]

    helicity_series = build_helicity_series(vis, dm)
    lightcurve      = build_lightcurve(vis)

    feature_space = extract_features_fn(vis, helicity_series, lightcurve)
    return feature_space
