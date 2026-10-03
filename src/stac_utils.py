"""
Utilities to read STAC item metadata and extract wavelengths.
"""

import json
import requests
import numpy as np


def fetch_stac_item(item_url):
    """Download a STAC item JSON from a URL."""
    r = requests.get(item_url, timeout=60)
    r.raise_for_status()
    return r.json()


def load_stac_item_from_file(path):
    """Load a STAC item JSON from disk."""
    with open(path, "r") as f:
        return json.load(f)


def extract_wavelengths(stac_item, asset_key="ortho_sr_hdf5"):
    """Extract center wavelengths (nm) from STAC band metadata."""
    bands_meta = stac_item["assets"][asset_key].get("bands", [])
    spectral = [b for b in bands_meta if "eo:center_wavelength" in b]
    wavelengths_um = np.array(
        [b["eo:center_wavelength"] for b in spectral], dtype=float
    )
    return wavelengths_um * 1000.0