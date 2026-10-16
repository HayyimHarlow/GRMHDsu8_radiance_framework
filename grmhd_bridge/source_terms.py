"""
source_terms.py
Inject SU(8–acoustic source terms into GRMHD energy/momentum equations.
"""

from dataclasses import dataclass

from GRMHDsu8_radiance_framework.su8_core.geometry import (
    GeometryState,
    epsilon_max,
    update_epsilon_su8,
)

from GRMHDsu8_radiance_framework.su8_core.acoustic import (
    AcousticState,
    gamma_load,
    gamma_acoustic,
    update_epsilon_ac,
)
@dataclass
class GRMHDState:
    epsilon: float
    S: float
    geom: GeometryState
    ac: AcousticState

@dataclass
class SourceParams:
    kappa_ac: float
    J_sup: float


def apply_su8_sources(state: GRMHDState, params: SourceParams, dt: float) -> GRMHDState:
    """
    Update SU(8, acoustic, and GRMHD energy consistently.
    """
    eps_max_val = epsilon_max(state.geom.I_R)

    # acoustic rates from full SU(8 laws
    Gamma_ac = gamma_load(
        epsilon_su8=state.geom.epsilon_su8,
        epsilon_ac=state.ac.epsilon_ac,
        epsilon_max=eps_max_val,
        kappa_ac=params.kappa_ac,
    )
    Lambda_ac = gamma_acoustic(
        epsilon_ac=state.ac.epsilon_ac,
        gamma_ac=state.ac.gamma_ac,
        rho_ac=state.ac.rho_ac,
    )

    # update SU(8 and acoustic energies
    update_epsilon_su8(state.geom, J_sup=params.J_sup, Gamma_ac=Gamma_ac, dt=dt)
    update_epsilon_ac(state.ac, Gamma_ac=Gamma_ac, Lambda_ac=Lambda_ac, dt=dt)

    # GRMHD total energy gets net source
    d_eps = params.J_sup - Gamma_ac + Lambda_ac
    state.epsilon += d_eps * dt

    return state


