"""
visualization.py
Basic plotting utilities for EVPA, ε_SU8, ε_ac time series.
"""

import matplotlib.pyplot as plt
import numpy as np

def plot_evpa(times: np.ndarray, evpa_series: np.ndarray):
    plt.figure(figsize=(6, 4))
    plt.plot(times, evpa_series, label="EVPA")
    plt.xlabel("t")
    plt.ylabel("EVPA [rad]")
    plt.title("EVPA evolution")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_energies(times: np.ndarray,
                  eps_su8_series: np.ndarray,
                  eps_ac_series: np.ndarray):
    plt.figure(figsize=(6, 4))
    plt.plot(times, eps_su8_series, label="ε_SU8")
    plt.plot(times, eps_ac_series, label="ε_ac")
    plt.xlabel("t")
    plt.ylabel("Energy density")
    plt.title("SU(8 and acoustic energy evolution")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
