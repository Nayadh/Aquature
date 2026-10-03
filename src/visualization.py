"""
Plotting helpers using matplotlib only.
Four-class, six-panel dashboard.
"""

import numpy as np
import matplotlib.pyplot as plt


CLASS_NAMES = ["clean", "turbid", "organic", "industrial"]


def plot_health_dashboard(pred_map, whi, prob_map):
    """
    Six-panel dashboard:
      1. Classification map
      2. Water Health Index
      3. Clean probability
      4. Turbid probability
      5. Organic probability
      6. Industrial probability
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    im1 = axes[0, 0].imshow(pred_map, cmap="tab10", vmin=0, vmax=3)
    axes[0, 0].set_title("Pollution Classification (4 classes)")
    axes[0, 0].axis("off")
    plt.colorbar(im1, ax=axes[0, 0], fraction=0.046)

    im2 = axes[0, 1].imshow(whi, cmap="RdYlGn", vmin=0, vmax=100)
    axes[0, 1].set_title("Water Health Index (0-100)")
    axes[0, 1].axis("off")
    plt.colorbar(im2, ax=axes[0, 1], fraction=0.046)

    im3 = axes[0, 2].imshow(prob_map[..., 0], cmap="Blues", vmin=0, vmax=1)
    axes[0, 2].set_title("Clean Water Probability")
    axes[0, 2].axis("off")
    plt.colorbar(im3, ax=axes[0, 2], fraction=0.046)

    im4 = axes[1, 0].imshow(prob_map[..., 1], cmap="YlOrBr", vmin=0, vmax=1)
    axes[1, 0].set_title("Turbidity Probability")
    axes[1, 0].axis("off")
    plt.colorbar(im4, ax=axes[1, 0], fraction=0.046)

    im5 = axes[1, 1].imshow(prob_map[..., 2], cmap="YlGn", vmin=0, vmax=1)
    axes[1, 1].set_title("Organic / Algae Probability")
    axes[1, 1].axis("off")
    plt.colorbar(im5, ax=axes[1, 1], fraction=0.046)

    im6 = axes[1, 2].imshow(prob_map[..., 3], cmap="coolwarm", vmin=0, vmax=1)
    axes[1, 2].set_title("Industrial Effluent Probability")
    axes[1, 2].axis("off")
    plt.colorbar(im6, ax=axes[1, 2], fraction=0.046)

    plt.tight_layout()
    return fig


def plot_mean_spectrum(wavelengths_nm, mean_spectrum, title="Mean Spectrum"):
    plt.figure(figsize=(10, 4))
    plt.plot(wavelengths_nm, mean_spectrum, lw=1.4, color="darkblue")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Surface Reflectance")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    return plt.gcf()


def plot_sentinel2_rgb(bands, title="Sentinel-2 RGB"):
    def stretch(a, p1=2, p99=98):
        lo, hi = np.nanpercentile(a, p1), np.nanpercentile(a, p99)
        return np.clip((a - lo) / (hi - lo + 1e-6), 0, 1)

    rgb = np.stack([
        stretch(bands["B04"]),
        stretch(bands["B03"]),
        stretch(bands["B02"]),
    ], axis=-1)
    rgb = np.nan_to_num(rgb, nan=0.0)

    plt.figure(figsize=(8, 8))
    plt.imshow(rgb)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    return plt.gcf()