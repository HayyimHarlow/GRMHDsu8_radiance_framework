"""
stability_sweep.py
"""



import numpy as np
from su8_core.stability import StabilityParams, is_linearly_stable

def main():
    a_su8_vals = np.linspace(0.1, 2.0, 10)
    b_ac_vals = np.linspace(0.1, 2.0, 10)
    a_ac = 0.5  # fixed for now

    stable_points = []
    unstable_points = []

    for a_su8 in a_su8_vals:
        for b_ac in b_ac_vals:
            params = StabilityParams(a_su8=a_su8, a_ac=a_ac, b_ac=b_ac)
            if is_linearly_stable(params):
                stable_points.append((a_su8, b_ac))
            else:
                unstable_points.append((a_su8, b_ac))

    print("Stable points (a_su8, b_ac):", stable_points[:10])
    print("Unstable points (a_su8, b_ac):", unstable_points[:10])

if __name__ == "__main__":
    main()
