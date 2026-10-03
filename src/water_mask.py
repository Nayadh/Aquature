"""
Water mask and index computation for a Tanager scene.
"""

import os
import json
import h5py
import numpy as np
import matplotlib.pyplot as plt


SCENE_ID = "20250511_074311_00_4001"
H5_PATH = f"data/raw/tanager/{SCENE_ID}_ortho_sr_hdf5.h5"
JSON_PATH = f"data/raw/tanager/{SCENE_ID}_stac.json"
OUTPUT_DIR = "outputs/figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

NDWI_WATER_THRESHOLD = 0.0
EPS = 1e-6


def nearest_band(wavelengths_nm, target_nm):
    return int(np.argmin(np.abs(wavelengths_nm - target_nm)))


def normalized_difference(a, b):
    return (a - b) / (a + b + EPS)


def main():
    with open(JSON_PATH) as f:
        item = json.load(f)

    bands_meta = item["assets"]["ortho_sr_hdf5"].get("bands", [])
    spectral = [b for b in bands_meta if "eo:center_wavelength" in b]
    wavelengths_nm = np.array(
        [b["eo:center_wavelength"] * 1000 for b in spectral]
    )

    print("Loading HDF5 cube...")
    with h5py.File(H5_PATH, "r") as f:
        root = "HDFEOS/GRIDS/HYP/Data Fields"
        cube = f[f"{root}/surface_reflectance"][:].astype("float32")
        cloud = f[f"{root}/beta_cloud_mask"][:]
        nodata = f[f"{root}/nodata_pixels"][:]
        cirrus = f[f"{root}/beta_cirrus_mask"][:]

    valid = (cloud == 0) & (nodata == 0) & (cirrus == 0)

    idx_443 = nearest_band(wavelengths_nm, 443)
    idx_560 = nearest_band(wavelengths_nm, 560)
    idx_665 = nearest_band(wavelengths_nm, 665)
    idx_708 = nearest_band(wavelengths_nm, 708)
    idx_860 = nearest_band(wavelengths_nm, 860)

    R_443 = cube[idx_443]
    R_560 = cube[idx_560]
    R_665 = cube[idx_665]
    R_708 = cube[idx_708]
    R_860 = cube[idx_860]

    NDWI = normalized_difference(R_560, R_860)
    NDCI = normalized_difference(R_708, R_665)
    Turbidity = normalized_difference(R_665, R_560)
    CDOM = normalized_difference(R_560, R_443)

    water_mask = (NDWI > NDWI_WATER_THRESHOLD) & valid
    print(f"Water pixels: {water_mask.mean()*100:.1f}%")

    NDWI_w = np.where(water_mask, NDWI, np.nan)
    NDCI_w = np.where(water_mask, NDCI, np.nan)
    Turbidity_w = np.where(water_mask, Turbidity, np.nan)
    CDOM_w = np.where(water_mask, CDOM, np.nan)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    def stretch(a, p1=2, p99=98):
        v = a[valid & np.isfinite(a)]
        if v.size == 0:
            return np.zeros_like(a)
        lo, hi = np.percentile(v, p1), np.percentile(v, p99)
        return np.clip((a - lo) / (hi - lo + 1e-6), 0, 1)

    rgb = np.stack([
        stretch(R_665),
        stretch(R_560),
        stretch(R_443),
    ], axis=-1)
    rgb = np.nan_to_num(rgb, nan=0.0)

    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title("True Color RGB")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(water_mask, cmap="Blues")
    axes[0, 1].set_title(f"Water Mask (NDWI > {NDWI_WATER_THRESHOLD})")
    axes[0, 1].axis("off")

    im = axes[0, 2].imshow(NDWI_w, cmap="RdYlBu_r", vmin=-0.5, vmax=0.8)
    axes[0, 2].set_title("NDWI (water only)")
    axes[0, 2].axis("off")
    plt.colorbar(im, ax=axes[0, 2], fraction=0.046)

    im = axes[1, 0].imshow(NDCI_w, cmap="YlGn", vmin=-0.2, vmax=0.4)
    axes[1, 0].set_title("NDCI - Chlorophyll / Algae")
    axes[1, 0].axis("off")
    plt.colorbar(im, ax=axes[1, 0], fraction=0.046)

    im = axes[1, 1].imshow(Turbidity_w, cmap="inferno", vmin=-0.3, vmax=0.3)
    axes[1, 1].set_title("Turbidity")
    axes[1, 1].axis("off")
    plt.colorbar(im, ax=axes[1, 1], fraction=0.046)

    im = axes[1, 2].imshow(CDOM_w, cmap="PuBuGn", vmin=-0.3, vmax=0.5)
    axes[1, 2].set_title("CDOM - Dissolved Organics")
    axes[1, 2].axis("off")
    plt.colorbar(im, ax=axes[1, 2], fraction=0.046)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, f"{SCENE_ID}_water_mask.png")
    plt.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")
    plt.show()


if __name__ == "__main__":
    main()