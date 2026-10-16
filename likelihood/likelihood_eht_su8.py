# GRMHDsu8_radiance_framework/likelihood/likelihood_eht_su8.py

import numpy as np

def compute_logL_vis(eht_vis, model_vis):
    """
    eht_vis: dict with keys 'V', 'sigma'
    model_vis: dict with key 'V_model'
    """
    V_data = eht_vis["V"]          # complex array
    sigma = eht_vis["sigma"]       # real array
    V_model = model_vis["V_model"] # complex array

    resid2 = np.abs(V_data - V_model)**2
    return -0.5 * np.sum(resid2 / sigma**2)

def compute_logL_cp(eht_cp, model_cp):
    """
    eht_cp: dict with keys 'phi', 'sigma'
    model_cp: dict with key 'phi_model'
    """
    phi_data = eht_cp["phi"]
    sigma = eht_cp["sigma"]
    phi_model = model_cp["phi_model"]

    # wrap differences to [-pi, pi]
    dphi = phi_data - phi_model
    dphi = (dphi + np.pi) % (2*np.pi) - np.pi

    return -0.5 * np.sum(dphi**2 / sigma**2)

def compute_logL_total(eht_data, model_data):
    """
    eht_data: {'vis': {...}, 'cp': {...}}
    model_data: {'vis': {...}, 'cp': {...}}
    """
    logL_vis = compute_logL_vis(eht_data["vis"], model_data["vis"])
    logL_cp  = compute_logL_cp(eht_data["cp"],  model_data["cp"])
    return logL_vis + logL_cp
