"""
stability.py — Full SU(8 stability architecture
Replace toy coefficients with monograph-derived partial derivatives.
"""

import numpy as np
from dataclasses import dataclass

@dataclass
class StabilityParams:
    a_su8: float   # ∂Γ_ac/∂ε_SU8
    a_ac: float    # ∂Γ_ac/∂ε_ac
    b_ac: float    # ∂Λ_ac/∂ε_ac


def jacobian_matrix(params: StabilityParams) -> np.ndarray:
    """
    J = [[-a_su8, -a_ac],
         [ a_su8,  a_ac - b_ac]]
    """
    return np.array([
        [-params.a_su8, -params.a_ac],
        [ params.a_su8,  params.a_ac - params.b_ac]
    ], dtype=float)


def eigenvalues(J: np.ndarray) -> np.ndarray:
    return np.linalg.eigvals(J)


def is_linearly_stable(params: StabilityParams) -> bool:
    """
    Stability Theorem:
        Re(λ) < 0 for all eigenvalues.
    """
    lam = eigenvalues(jacobian_matrix(params))
    return np.all(np.real(lam) < 0.0)


def z_pinch_excluded(epsilon_su8: float,
                     epsilon_ac: float,
                     epsilon_max_val: float,
                     gamma_ac: float) -> bool:
    """
    Z-Pinch Exclusion Theorem:
        ε_tot = ε_SU8 + ε_ac <= ε_max
        γ_ac > 0 ensures damping
    """
    epsilon_tot = epsilon_su8 + epsilon_ac
    return (epsilon_tot <= epsilon_max_val) and (gamma_ac > 0.0)

