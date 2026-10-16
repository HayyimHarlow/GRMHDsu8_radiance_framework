import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

def load_image(path):
    with fits.open(path) as hdul:
        return hdul[0].data

def plot_comparison(img_baseline, img_su8, title_baseline="Baseline",
                    title_su8="SU(8", cmap="inferno"):
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    im0 = axes[0].imshow(img_baseline, origin="lower", cmap=cmap)
    axes[0].set_title(title_baseline)
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(img_su8, origin="lower", cmap=cmap)
    axes[1].set_title(title_su8)
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()

def plot_evpa_timeseries(t, evpa_baseline, evpa_su8):
    plt.figure(figsize=(6, 4))
    plt.plot(t, evpa_baseline, label="Baseline EVPA")
    plt.plot(t, evpa_su8, label="SU(8 EVPA")
    plt.xlabel("t")
    plt.ylabel("EVPA [rad]")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
