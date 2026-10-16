"""
coupling_schemes.py
Explicit / semi-implicit integration of SU(8–acoustic source terms.
"""

def explicit_step(value: float, source: float, dt: float) -> float:
    return value + source * dt


def semi_implicit_damp(epsilon_ac: float, gamma_ac: float, dt: float) -> float:
    return epsilon_ac / (1.0 + gamma_ac * dt)

