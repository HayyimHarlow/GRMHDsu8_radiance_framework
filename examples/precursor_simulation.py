""""
precursor_simulation.py
Real precursor evolution simulation (SU(8 + acoustic + GRMHD)
""""
import numpy as np

from su8_core.geometry import GeometryState, epsilon_max
from su8_core.acoustic import AcousticState
from su8_core.stability import StabilityParams, is_linearly_stable, z_pinch_excluded
from grmhd_bridge.source_terms import GRMHDState, SourceParams, apply_su8_sources
from grmhd_bridge.cfl_control import compute_dt

def main():
    # initial states
    geom = GeometryState(epsilon_su8=0.5, I_R=0.1)
    ac = AcousticState(epsilon_ac=0.1, rho_ac=1.0, c_ac=0.5, gamma_ac=0.3)
    grmhd = GRMHDState(epsilon=geom.epsilon_su8 + ac.epsilon_ac, S=0.0, geom=geom, ac=ac)

    # parameters
    kappa_ac = 0.4
    J_sup0 = 0.5
    T_prec = 10.0
    n_steps = 200

    times = []
    eps_su8_series = []
    eps_ac_series = []

    for i in range(n_steps):
        t = i * (T_prec / n_steps)
        # simple suppression current decay
        J_sup = J_sup0 * np.exp(-t / T_prec)

        # CFL-based dt (toy c=1)
        dt = compute_dt(
            dx=1.0,
            c=1.0,
            c_ac=ac.c_ac,
            tau_sup=T_prec,
            tau_ac_load=T_prec,
            tau_ac_damp=1.0 / ac.gamma_ac,
            CFL=0.3,
        )

        src = SourceParams(kappa_ac=kappa_ac, J_sup=J_sup)
        apply_su8_sources(grmhd, src, dt)

        times.append(t)
        eps_su8_series.append(geom.epsilon_su8)
        eps_ac_series.append(ac.epsilon_ac)

        # stability + Z-pinch check
        params = StabilityParams(
            a_su8=kappa_ac,          # ∂Γ_ac/∂ε_SU8 ~ κ_ac
            a_ac=0.0,                # simple model: no dependence on ε_ac
            b_ac=ac.gamma_ac,        # ∂Λ_ac/∂ε_ac = γ_ac
        )
        stable = is_linearly_stable(params)
        bounded = z_pinch_excluded(
            epsilon_su8=geom.epsilon_su8,
            epsilon_ac=ac.epsilon_ac,
            epsilon_max_val=epsilon_max(geom.I_R),
            gamma_ac=ac.gamma_ac,
        )

        if not stable or not bounded:
            print(f"Instability or Z-pinch risk at t={t:.3f}")
            break

    print("Final ε_SU8:", geom.epsilon_su8)
    print("Final ε_ac:", ac.epsilon_ac)
    print("Stable throughout:", stable and bounded)

if __name__ == "__main__":
    main()
