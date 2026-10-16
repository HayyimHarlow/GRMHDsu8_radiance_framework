import numpy as np

from su8_core.geometry import GeometryState, epsilon_max, update_epsilon_su8
from su8_core.acoustic import AcousticState, gamma_acoustic, gamma_load, update_epsilon_ac
from su8_core.stability import StabilityParams, is_linearly_stable, z_pinch_excluded

from grmhd_bridge.source_terms import GRMHDState, SourceParams, apply_su8_sources
from grmhd_bridge.cfl_control import compute_dt

from grrt_synthetic.polarization import compute_evpa, helicity
from grrt_synthetic.diagnostics import evpa_swing_amplitude, helicity_flip_count

from inference_pipeline.feature_space import extract_features
from inference_pipeline.likelihood import feature_misfit, likelihood


def main():
    # 1) Initialize states
    geom = GeometryState(epsilon_su8=1.0, I_R=0.0)
    ac = AcousticState(epsilon_ac=0.5, rho_ac=1.0, c_ac=0.5, gamma_ac=0.2)
    grmhd = GRMHDState(epsilon=1.5, S=0.0, epsilon_su8=geom.epsilon_su8, epsilon_ac=ac.epsilon_ac)

    # 2) Time step from CFL
    dt = compute_dt(dx=1.0, c=1.0, c_ac=ac.c_ac,
                    tau_sup=10.0, tau_ac_load=10.0, tau_ac_damp=1.0)

    # 3) Compute acoustic rates
    Gamma_ac = gamma_load(geom.epsilon_su8, ac.epsilon_ac, kappa_ac=0.3)
    Lambda_ac = gamma_acoustic(ac.epsilon_ac, ac.gamma_ac)

    # 4) Update SU(8 and acoustic energies
    update_epsilon_su8(geom, J_sup=0.2, Gamma_ac=Gamma_ac, dt=dt)
    update_epsilon_ac(ac, Gamma_ac=Gamma_ac, Lambda_ac=Lambda_ac, dt=dt)

    # 5) Apply sources to GRMHD
    src = SourceParams(J_sup=0.2, Gamma_ac=Gamma_ac, Lambda_ac=Lambda_ac)
    apply_su8_sources(grmhd, src, dt)

    # 6) Check stability and Z-pinch exclusion
    params = StabilityParams(a_su8=1.0, a_ac=0.5, b_ac=ac.gamma_ac)
    stable = is_linearly_stable(params)
    bounded = z_pinch_excluded(geom.epsilon_su8, ac.epsilon_ac, epsilon_max(geom.I_R), ac.gamma_ac)

    print("Stable:", stable, "Z-pinch excluded:", bounded)

    # 7) Toy EVPA series and helicity
    Q = np.array([1.0, 0.8, 0.6, 0.4])
    U = np.array([0.0, 0.2, 0.4, 0.6])
    evpa_series = np.array([compute_evpa(Q[i], U[i]) for i in range(len(Q))])
    helicity_series = helicity(evpa_series)

    print("EVPA swing:", evpa_swing_amplitude(evpa_series))
    print("Helicity flips:", helicity_flip_count(helicity_series))

    # 8) Feature extraction and simple likelihood
    lightcurve = np.array([1.0, 1.1, 0.9, 1.05])
    model_features = extract_features(evpa_series, helicity_series, lightcurve)

    # toy data features: slightly different
    data_features = {
        "evpa_amp": model_features["evpa_amp"] * 1.1,
        "helicity_flips": model_features["helicity_flips"],
        "var_spectrum": model_features["var_spectrum"],
    }

    mis = feature_misfit(data_features, model_features)
    L = likelihood(mis)

    print("Misfit:", mis, "Likelihood:", L)


if __name__ == "__main__":
    main()
