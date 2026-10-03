"""
Export WHI overlay PNG + grid JSON for the website.
Run this once after main.py finishes.
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RESULTS_DIR = "outputs/results"

# ============================================================
# 1. Export WHI color overlay PNG
# ============================================================
print("[1/2] Exporting WHI overlay PNG...")

whi = np.load(os.path.join(RESULTS_DIR, "whi.npy"))
rows, cols = whi.shape

cmap = plt.cm.RdYlGn.copy()
cmap.set_bad(alpha=0)

masked = np.ma.masked_invalid(whi)

fig = plt.figure(figsize=(cols / 100, rows / 100), dpi=100)
ax = fig.add_axes([0, 0, 1, 1])
ax.axis('off')
ax.imshow(masked, cmap=cmap, vmin=0, vmax=100, aspect='equal')
fig.savefig(
    os.path.join(RESULTS_DIR, "whi_overlay.png"),
    transparent=True, dpi=100, pad_inches=0
)
plt.close(fig)
print(f"  → Saved whi_overlay.png ({cols}x{rows})")

# ============================================================
# 2. Export grid JSON for zone sampling
# ============================================================
print("[2/2] Exporting grid JSON...")

prob_map = np.load(os.path.join(RESULTS_DIR, "prob_map.npy"))

GRID = 100  # 100x100 grid

row_edges = np.linspace(0, rows, GRID + 1).astype(int)
col_edges = np.linspace(0, cols, GRID + 1).astype(int)

whi_grid = np.full((GRID, GRID), np.nan)
prob_grid = np.full((GRID, GRID, 4), np.nan)

for i in range(GRID):
    for j in range(GRID):
        r0, r1 = row_edges[i], row_edges[i + 1]
        c0, c1 = col_edges[j], col_edges[j + 1]

        block = whi[r0:r1, c0:c1]
        v = block[~np.isnan(block)]
        if len(v) > 0:
            whi_grid[i, j] = float(np.mean(v))

        pb = prob_map[r0:r1, c0:c1, :].reshape(-1, 4)
        vp = pb[~np.isnan(pb[:, 0])]
        if len(vp) > 0:
            prob_grid[i, j] = np.mean(vp, axis=0)

# bbox = [west_lon, south_lat, east_lon, north_lat]
bbox = [53.63, 23.99, 53.89, 24.19]

out = {
    "bbox": bbox,
    "grid": GRID,
    "whi": whi_grid.tolist(),
    "prob": prob_grid.tolist()
}

with open(os.path.join(RESULTS_DIR, "whi_grid.json"), "w") as f:
    json.dump(out, f)

valid = whi_grid[~np.isnan(whi_grid)]
print(f"  → Grid: {GRID}x{GRID}")
print(f"  → WHI range: {valid.min():.1f} - {valid.max():.1f}")
print(f"  → Mean: {valid.mean():.1f}")

print("\n✅ Done. Files saved to outputs/results/")