"""
config.py
Runtime configuration for GRMHDsu8_radiance_framework.
Centralizes paths, physical constants, and simulation parameters.
"""

from dataclasses import dataclass

REPO_ROOT = "/content/GRMHDsu8_radiance_framework"

# Physical constants (cgs)
C_LIGHT = 2.99792458e10   # cm/s
K_BOLTZ = 1.380649e-16    # erg/K

@dataclass
class SimulationConfig:
    dx: float = 1.0
    c_mhd: float = 1.0
    c_ac: float = 0.5
    CFL: float = 0.3
    T_prec: float = 10.0
    n_steps: int = 200
    kappa_ac: float = 0.4
    J_sup0: float = 0.5

DEFAULT_SIM_CONFIG = SimulationConfig()
