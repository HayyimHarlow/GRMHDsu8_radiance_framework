"""
model_grid.py
Generate grid over (κ_ac, γ_ac, T_prec) and run GRRT.
"""

from dataclasses import dataclass

@dataclass
class ModelParams:
    kappa_ac: float
    gamma_ac: float
    T_prec: float

def generate_model_grid(kappa_vals, gamma_vals, T_vals):
    """
    Cartesian product of parameter ranges.
    """
    pass
