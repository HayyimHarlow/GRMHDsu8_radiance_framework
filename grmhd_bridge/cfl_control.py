"""
cfl_control.py
Combined GRMHD + acoustic CFL constraint.
"""

def compute_dt(dx: float, c: float, c_ac: float,
               tau_sup: float, tau_ac_load: float, tau_ac_damp: float,
               CFL: float = 0.4) -> float:
    candidates = [
        dx / max(c, 1e-12),
        dx / max(c_ac, 1e-12),
        tau_sup,
        tau_ac_load,
        tau_ac_damp
    ]
    return CFL * min(candidates)

