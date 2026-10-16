"""
parallel_grrt.py
Parallel GRRT imaging executor using multiprocessing.
Runs multiple parameter sets (e.g., κ_ac, γ_ac, T_prec) in parallel.
"""

import multiprocessing as mp
import numpy as np

from grrt_synthetic.radiative_transfer import compute_brightness_temperature
from grrt_synthetic.polarization import compute_evpa, polarization_fraction
from inference_pipeline.feature_space import extract_features

def run_single_grrt_job(params):
    """
    params: dict with keys:
        'I_series', 'nu', 'Q_series', 'U_series'
    Returns: model_features dict
    """
    I_series = params["I_series"]
    nu = params["nu"]
    Q_series = params["Q_series"]
    U_series = params["U_series"]

    # brightness temperature time series
    Tb_series = np.array([
        compute_brightness_temperature(I, nu) for I in I_series
    ])

    # EVPA series
    evpa_series = np.array([
        compute_evpa(Q_series[i], U_series[i]) for i in range(len(Q_series))
    ])

    # helicity series
    from grrt_synthetic.polarization import helicity
    helicity_series = helicity(evpa_series)

    # features
    features = extract_features(evpa_series, helicity_series, Tb_series)
    return features

def run_parallel_grrt(jobs_params_list, n_procs: int = 4):
    """
    jobs_params_list: list of dicts, each passed to run_single_grrt_job
    Returns: list of feature dicts
    """
    with mp.Pool(processes=n_procs) as pool:
        results = pool.map(run_single_grrt_job, jobs_params_list)
    return results
