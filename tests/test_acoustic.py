from su8_core.geometry import GeometryState, compute_I_R, epsilon_max, update_epsilon_su8
import numpy as np

def test_compute_I_R():
    g = np.eye(4)
    I_R = compute_I_R(g)
    assert I_R == 4.0

def test_epsilon_max_monotonic():
    assert epsilon_max(0.0) == 1.0
    assert epsilon_max(2.0) >= epsilon_max(0.0)

def test_update_epsilon_su8():
    state = GeometryState(epsilon_su8=1.0, I_R=0.0)
    new_eps = update_epsilon_su8(state, J_sup=0.5, Gamma_ac=0.2, dt=1.0)
    assert new_eps == 1.3
