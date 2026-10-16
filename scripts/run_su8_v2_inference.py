# scripts/run_su8_v2_inference.py

import numpy as np
from GRMHDsu8_radiance_framework.data.load_eht import load_eht_subset
from GRMHDsu8_radiance_framework.inference.inference_hmc_nuts import run_su8_hmc

def main():
    # 1) Load EHT visibilities + closure phases (subset for laptop)
    eht_data = load_eht_subset()  # you implement this to return {'vis': {...}, 'cp': {...}}

    # 2) Run HMC/NUTS
    samples = run_su8_hmc(eht_data, num_warmup=500, num_samples=1000, num_chains=1)

    # 3) Save posterior samples
    np.savez("archive_v2/posterior_samples_su8.npz", **samples)
    print("Saved posterior samples to archive_v2/posterior_samples_su8.npz")

if __name__ == "__main__":
    main()
