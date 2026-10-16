"""
acoustic.py — SU(8 acoustic bleed‑off channel
Full replacement of toy laws with monograph‑aligned structure.
"""

from dataclasses import dataclass
import math

@dataclass
class AcousticState:
    epsilon_ac: float
    rho_ac: float
    c_ac: float
    gamma_ac: float


def gamma_load(epsilon_su8: float,
               epsilon_ac: float,
               epsilon_max: float,
               kappa_ac: float) -> float:
    """
    Full SU(8 acoustic loading law:
        Γ_ac = κ_ac * max(ε_SU8 - ε_max, 0)
    i.e. bleed-off only when geometric energy exceeds saturation.
    """
    excess = max(epsilon_su8 - epsilon_max, 0.0)
    return kappa_ac * excess


def gamma_acoustic(epsilon_ac: float,
                   gamma_ac: float,
                   rho_ac: float = 1.0) -> float:
    """
    Full damping law:
        Λ_ac = γ_ac * ε_ac
    Optionally include density dependence:
        Λ_ac = γ_ac * ε_ac / ρ_ac
    """
    return gamma_ac * epsilon_ac / rho_ac


def update_epsilon_ac(state: AcousticState,
                      Gamma_ac: float,
                      Lambda_ac: float,
                      dt: float) -> float:
    """
    dε_ac/dt = Γ_ac - Λ_ac
    """
    d_eps = Gamma_ac - Lambda_ac
    state.epsilon_ac += d_eps * dt
    return state.epsilon_ac


def dispersion_relation(k: float, c_ac: float) -> float:
    """
    ω = c_ac * k
    (Appendix AF)
    """
    return c_ac * k

