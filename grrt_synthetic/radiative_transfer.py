"""
radiative_transfer.py
Brightness temperature, optical depth, and ray propagation.
"""

import math

# physical constants (cgs)
c = 2.99792458e10      # cm/s
k_B = 1.380649e-16     # erg/K

def compute_brightness_temperature(I: float, nu: float) -> float:
    """
    T_b = c^2 / (2 k_B ν^2) * I
    I: specific intensity [erg s^-1 cm^-2 Hz^-1 sr^-1]
    nu: frequency [Hz]
    """
    factor = (c**2) / (2.0 * k_B)
    return factor * I / (nu**2)


def optical_depth(kappa: float, ds: float) -> float:
    """
    τ = κ ds
    """
    return kappa * ds


