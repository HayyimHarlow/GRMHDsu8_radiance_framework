"""
geometry.py — SU(8 radiance‑channel geometry
Full replacement of toy laws with monograph‑aligned structure.
"""

from dataclasses import dataclass
import numpy as np

@dataclass
class GeometryState:
    epsilon_su8: float      # SU(8 geometric energy density
    I_R: float              # curvature invariant ℐ_R


def compute_I_R(metric_tensor) -> float:
    """
    Compute curvature invariant ℐ_R.
    Replace this with your preferred invariant:
        ℐ_R = R, or R_{μν}R^{μν}, or R_{μνρσ}R^{μνρσ}.
    """
    g = np.array(metric_tensor)
    # placeholder: scalar curvature-like contraction
    return float(np.trace(g @ g))


def epsilon_max(I_R: float) -> float:
    """
    Saturation bound ε_max(ℐ_R).
    Monograph structure:
        ε_max = ε0 * (1 + α |ℐ_R|^p)
    """
    eps0 = 1.0
    alpha = 0.3
    p = 1.0
    return eps0 * (1.0 + alpha * abs(I_R)**p)


def update_epsilon_su8(state: GeometryState,
                       J_sup: float,
                       Gamma_ac: float,
                       dt: float) -> float:
    """
    dε_SU8/dt = J_sup(t) - Γ_ac
    with saturation enforcement:
        ε_SU8 <= ε_max(ℐ_R)
    """
    d_eps = J_sup - Gamma_ac
    state.epsilon_su8 += d_eps * dt

    # enforce saturation bound
    eps_max = epsilon_max(state.I_R)
    state.epsilon_su8 = min(state.epsilon_su8, eps_max)

    return state.epsilon_su8


