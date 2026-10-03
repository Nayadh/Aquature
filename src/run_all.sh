#!/bin/bash
# Run the full download pipeline for Gulf of Oman

echo "=========================================="
echo "Gulf of Oman Water Quality - Data Pipeline"
echo "=========================================="

# 1. Install dependencies
echo "[1/4] Installing dependencies..."
pip install -r requirements.txt

# 2. Download Sentinel-2 (always works)
echo "[2/4] Downloading Sentinel-2..."
python src/download_sentinel2.py

# 3. Try Tanager (may not be available)
echo "[3/4] Checking Tanager..."
python src/download_tanager.py || echo "Tanager not available"

# 4. EnMAP (requires credentials)
echo "[4/4] EnMAP (optional)..."
echo "Set ENMAP_USERNAME and ENMAP_PASSWORD env vars, then run:"
echo "  python src/download_enmap.py"

echo "=========================================="
echo "Done. Check data/raw/"
echo "=========================================="