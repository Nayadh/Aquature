"""
Aquature Backend — Water Intelligence Platform
Real-time Sentinel-2 water quality analysis via Microsoft Planetary Computer.
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import requests
import io
import base64
import time
import threading
from datetime import datetime, timedelta, timezone
from PIL import Image

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

STAC_SEARCH = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
SAS_TOKEN_URL = "https://planetarycomputer.microsoft.com/api/sas/v1/token"

MAX_SIZE_DEG = 5.0
MAX_AREA_DEG2 = 15.0

_sas_cache = {}
_sas_lock = threading.Lock()
SAS_TTL = 300


# ============================================================
# SIGN URL
# ============================================================
def get_sas_token(collection="sentinel-2-l2a"):
    with _sas_lock:
        now = time.time()
        if collection in _sas_cache:
            token, expiry = _sas_cache[collection]
            if now < expiry:
                return token
        try:
            r = requests.get(f"{SAS_TOKEN_URL}/{collection}", timeout=20)
            if r.status_code != 200:
                return None
            token = r.json().get("token", "")
            if token:
                _sas_cache[collection] = (token, now + SAS_TTL)
            return token
        except Exception:
            return None


def sign_url(url, collection="sentinel-2-l2a"):
    token = get_sas_token(collection)
    if not token:
        return url
    sep = "&" if "?" in url else "?"
    return f"{url}{sep}{token}"


# ============================================================
# SEARCH
# ============================================================
def search_stac_all(bbox, days_back=120, max_items=10):
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days_back)
    body = {
        "collections": ["sentinel-2-l2a"],
        "bbox": bbox,
        "datetime": f"{start.date().isoformat()}T00:00:00Z/{end.date().isoformat()}T23:59:59Z",
        "query": {"eo:cloud_cover": {"lt": 50}},
        "limit": 30,
    }
    try:
        r = requests.post(STAC_SEARCH, json=body, timeout=60)
        if r.status_code != 200:
            print(f"[STAC] HTTP {r.status_code}")
            return []
        data = r.json()
    except Exception as e:
        print(f"[STAC] {e}")
        return []
    features = data.get("features", [])
    print(f"[STAC] found {len(features)} scenes for bbox")
    features.sort(key=lambda f: f["properties"].get("eo:cloud_cover", 100))
    return features[:max_items]


# ============================================================
# LOAD BAND
# ============================================================
def load_band(url, bbox, size=512, max_retries=4):
    import rasterio
    from rasterio.windows import from_bounds
    from rasterio.warp import transform_bounds
    from rasterio.enums import Resampling

    signed = sign_url(url)

    for attempt in range(1, max_retries + 1):
        try:
            with rasterio.open(signed) as src:
                try:
                    bbox_src = transform_bounds('EPSG:4326', src.crs, *bbox)
                except Exception:
                    bbox_src = bbox
                window = from_bounds(
                    bbox_src[0], bbox_src[1],
                    bbox_src[2], bbox_src[3],
                    transform=src.transform
                )
                arr = src.read(
                    1,
                    window=window,
                    out_shape=(size, size),
                    resampling=Resampling.bilinear,
                    boundless=True,
                    fill_value=0,
                ).astype("float32")
            return arr / 10000.0
        except Exception as e:
            err = str(e)
            if any(x in err for x in ["409", "429", "Timeout", "timeout"]):
                time.sleep(2 ** attempt)
                with _sas_lock:
                    _sas_cache.clear()
                signed = sign_url(url)
            else:
                return None
    return None


# ============================================================
# OVERLAY
# ============================================================
def make_overlay(clean_m, turbid_m, organic_m, industrial_m, oil_m):
    h, w = clean_m.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[clean_m] = [0, 212, 255, 150]
    rgba[turbid_m] = [255, 180, 68, 170]
    rgba[organic_m] = [0, 255, 136, 160]
    rgba[industrial_m] = [255, 60, 60, 180]
    rgba[oil_m] = [120, 40, 200, 200]
    img = Image.fromarray(rgba, 'RGBA')
    buf = io.BytesIO()
    img.save(buf, format='PNG', optimize=True)
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"


# ============================================================
# ADAPTIVE RESOLUTION
# ============================================================
def pick_resolution(size_w, size_h):
    m = max(size_w, size_h)
    if m <= 0.3:
        return 512, "ultra"
    elif m <= 0.8:
        return 384, "high"
    elif m <= 1.5:
        return 320, "medium"
    elif m <= 3.0:
        return 256, "low"
    else:
        return 192, "overview"


def nd(a, b, eps=1e-6):
    return (a - b) / (a + b + eps)


# ============================================================
# QUICK WATER CHECK
# ============================================================
def check_scene_has_water(feature, bbox, resolution=128):
    """Quick check: does this scene have water in the bbox?"""
    assets = feature.get("assets", {})
    if "B03" not in assets or "B08" not in assets:
        return False, 0

    try:
        green = load_band(assets["B03"]["href"], bbox, size=resolution, max_retries=2)
        if green is None:
            return False, 0
        nir = load_band(assets["B08"]["href"], bbox, size=resolution, max_retries=2)
        if nir is None:
            return False, 0

        if green.max() < 0.01:
            return False, 0

        NDWI = nd(green, nir)
        water_px = int((NDWI > 0.0).sum())
        return water_px > 20, water_px
    except Exception as e:
        print(f"[check_scene] error: {e}")
        return False, 0


# ============================================================
# ANALYZE SINGLE FEATURE
# ============================================================
def analyze_with_feature(feature, bbox, resolution=512):
    """Run full analysis on a specific scene."""
    item_id = feature["id"]
    scene_date = feature["properties"]["datetime"][:10]
    cloud = feature["properties"].get("eo:cloud_cover", "?")
    print(f"[analyze] scene={item_id} date={scene_date} cloud={cloud}%")

    assets = feature.get("assets", {})
    bands_needed = ["B03", "B04", "B08", "B11", "B12"]
    has_b05 = "B05" in assets

    bands = {}
    for i, band in enumerate(bands_needed):
        if band not in assets:
            return None, f"Band {band} missing"
        print(f"[analyze] loading {band} ({i+1}/{len(bands_needed)})...")
        arr = load_band(assets[band]["href"], bbox, size=resolution)
        if arr is None:
            return None, f"Failed to load {band}"
        bands[band] = arr
        if i < len(bands_needed) - 1:
            time.sleep(0.3)

    if has_b05:
        time.sleep(0.3)
        arr = load_band(assets["B05"]["href"], bbox, size=resolution)
        if arr is not None:
            bands["B05"] = arr

    green = bands["B03"]
    red = bands["B04"]
    nir = bands["B08"]
    swir1 = bands["B11"]
    swir2 = bands["B12"]
    red_edge = bands.get("B05", None)

    h = min(green.shape[0], red.shape[0], nir.shape[0], swir1.shape[0], swir2.shape[0])
    w = min(green.shape[1], red.shape[1], nir.shape[1], swir1.shape[1], swir2.shape[1])
    green = green[:h, :w]
    red = red[:h, :w]
    nir = nir[:h, :w]
    swir1 = swir1[:h, :w]
    swir2 = swir2[:h, :w]
    if red_edge is not None:
        red_edge = red_edge[:h, :w]

    print(f"[validate] B03 max={green.max():.4f}  B08 max={nir.max():.4f}")
    if green.max() < 0.01:
        return None, "Scene is empty (no data for this bbox)"

    # ============================================================
    # INDICES
    # ============================================================
    NDWI = nd(green, nir)
    Turbidity = nd(red, green)
    MNDWI = nd(green, swir1)

    if red_edge is not None:
        NDCI = nd(red_edge, red)
        ndci_source = "B05-B04 (Red-Edge)"
    else:
        NDCI = nd(nir, red)
        ndci_source = "B08-B04 (proxy)"

    NDSI = nd(red, swir1)
    NDMI = nd(nir, swir1)

    # ============================================================
    # ADAPTIVE WATER MASK
    # ============================================================
    array_pixels = green.size

    water = (NDWI > 0.20) & (MNDWI > 0.10) & (nir < 0.05)
    print(f"[mask] STRICT: {water.sum()} px")

    if water.sum() < 100:
        water = (NDWI > 0.10) & (MNDWI > 0.05)
        print(f"[mask] MODERATE: {water.sum()} px")

    if water.sum() < 100:
        water = (NDWI > 0.05) & (MNDWI > 0.0)
        print(f"[mask] LOOSE: {water.sum()} px")

    if water.sum() < 100:
        water = NDWI > 0.0
        print(f"[mask] NDWI-ONLY: {water.sum()} px")

    water_count = int(water.sum())
    water_pct = 100 * water_count / array_pixels if array_pixels > 0 else 0

    print(f"[analyze] final water: {water_count} / {array_pixels} ({water_pct:.1f}%)")

    if water_count < 30:
        return None, f"Only {water_count} water pixels"

    # ============================================================
    # STATISTICS
    # ============================================================
    turb_w = Turbidity[water]
    ndci_w = NDCI[water]
    ndwi_w = NDWI[water]
    mndwi_w = MNDWI[water]
    ndsi_w = NDSI[water]
    ndmi_w = NDMI[water]
    swir1_w = swir1[water]
    swir2_w = swir2[water]

    swir1_med = float(np.median(swir1_w))
    swir2_med = float(np.median(swir2_w))
    turb_med = float(np.median(turb_w))
    ndci_med = float(np.median(ndci_w))
    mndwi_med = float(np.median(mndwi_w))

    stats = {
        "ndwi_mean": float(np.mean(ndwi_w)),
        "ndwi_min": float(np.min(ndwi_w)),
        "ndwi_max": float(np.max(ndwi_w)),
        "turbidity_mean": float(np.mean(turb_w)),
        "turbidity_min": float(np.min(turb_w)),
        "turbidity_max": float(np.max(turb_w)),
        "ndci_mean": float(np.mean(ndci_w)),
        "ndci_min": float(np.min(ndci_w)),
        "ndci_max": float(np.max(ndci_w)),
        "mndwi_mean": float(np.mean(mndwi_w)),
        "mndwi_median": mndwi_med,
        "ndmi_mean": float(np.mean(ndmi_w)),
        "ndsi_mean": float(np.mean(ndsi_w)),
        "swir1_mean": float(np.mean(swir1_w)),
        "swir1_median": swir1_med,
        "swir2_mean": float(np.mean(swir2_w)),
        "swir2_median": swir2_med,
        "ndci_source": ndci_source,
    }

    print(f"[stats] Turbidity mean={stats['turbidity_mean']:.3f} median={turb_med:.3f}")
    print(f"[stats] NDCI mean={stats['ndci_mean']:.3f} median={ndci_med:.3f}")
    print(f"[stats] MNDWI mean={stats['mndwi_mean']:.3f} median={mndwi_med:.3f}")
    print(f"[stats] SWIR1 med={swir1_med:.4f}  SWIR2 med={swir2_med:.4f}")

    # ============================================================
    # CLASSIFICATION
    # ============================================================
    # Oil — strict (real oil slicks only)
    oil_m = (
        water &
        (swir2 > swir2_med + 0.10) &
        (swir1 > swir1_med + 0.08) &
        (Turbidity < turb_med + 0.02) &
        (NDWI > 0.15)
    )

    # Industrial
    industrial_m = water & ~oil_m & (
        ((Turbidity > turb_med + 0.15) & (swir2 > swir2_med + 0.03)) |
        ((MNDWI < mndwi_med - 0.20) & (swir2 > swir2_med + 0.04))
    )

    # Organic — only with real chlorophyll
    organic_m = water & ~oil_m & ~industrial_m & (NDCI > max(0.02, ndci_med + 0.06))

    # Turbid
    turbid_m = water & ~oil_m & ~industrial_m & ~organic_m & (
        (Turbidity > turb_med + 0.05) |
        (MNDWI < mndwi_med - 0.15)
    )

    # Clean
    clean_m = water & ~oil_m & ~industrial_m & ~organic_m & ~turbid_m

    counts = {
        "clean": int(clean_m.sum()),
        "turbid": int(turbid_m.sum()),
        "organic": int(organic_m.sum()),
        "industrial": int(industrial_m.sum()),
        "oil": int(oil_m.sum()),
    }
    print(f"[analyze] counts: {counts}")

    total = sum(counts.values())
    if total == 0:
        return None, "No classified pixels"

    classes_pct = {k: (v / total) * 100 for k, v in counts.items()}

    scores = {"clean": 100, "turbid": 55, "organic": 45, "industrial": 20, "oil": 10}
    whi = sum(scores[k] * classes_pct[k] / 100 for k in scores)

    overlay = make_overlay(clean_m, turbid_m, organic_m, industrial_m, oil_m)

    bbox_w_deg = bbox[2] - bbox[0]
    bbox_h_deg = bbox[3] - bbox[1]
    mid_lat = (bbox[1] + bbox[3]) / 2
    bbox_w_m = bbox_w_deg * 111320 * np.cos(np.radians(mid_lat))
    bbox_h_m = bbox_h_deg * 110570
    real_pixels = max(1, int(bbox_w_m * bbox_h_m / 100))
    real_water = int(real_pixels * water_pct / 100)

    result = {
        "success": True,
        "scene_id": item_id,
        "scene_date": scene_date,
        "cloud_cover": cloud,
        "bbox": bbox,
        "resolution": resolution,
        "water_pixels": real_water,
        "total_pixels": real_pixels,
        "water_pct": round(water_pct, 1),
        "whi": round(whi, 1),
        "classes": {k: round(v, 1) for k, v in classes_pct.items()},
        "stats": stats,
        "overlay": overlay,
    }
    return result, None


# ============================================================
# ANALYZE — MULTI-TILE SEARCH
# ============================================================
def analyze(bbox, resolution=512):
    """Try multiple scenes until one has water."""
    print(f"\n{'='*60}")
    print(f"[analyze] bbox={bbox}")

    features = search_stac_all(bbox, max_items=8)
    if not features:
        return {"error": "No Sentinel-2 scenes found for this location."}

    tried = []
    for idx, feature in enumerate(features):
        item_id = feature["id"]
        print(f"\n[trial {idx+1}/{len(features)}] {item_id}")
        tried.append(item_id)

        has_water, water_px = check_scene_has_water(feature, bbox, resolution=128)
        if not has_water:
            print(f"[trial] no water found in this scene ({water_px} px) — skipping")
            continue

        print(f"[trial] water check passed ({water_px} px at 128×128)")

        result, err = analyze_with_feature(feature, bbox, resolution=resolution)
        if result is not None:
            print(f"[trial] SUCCESS with {item_id}")
            print(f"{'='*60}\n")
            return result
        else:
            print(f"[trial] failed: {err}")

    print(f"[analyze] all {len(features)} scenes failed")
    print(f"{'='*60}\n")
    return {
        "error": f"No water found in any of the {len(features)} available scenes for this location. "
                 f"The area may be entirely on land, or all recent scenes have cloud/data issues. "
                 f"Try drawing directly over the water body."
    }


# ============================================================
# ROUTES
# ============================================================
@app.route('/')
def index():
    return send_from_directory('.', 'index.html')


@app.route('/<path:path>')
def static_files(path):
    return send_from_directory('.', path)


@app.route('/api/test')
def api_test():
    return jsonify({
        "status": "ok",
        "service": "aquature",
        "bands": ["B03", "B04", "B05", "B08", "B11", "B12"],
        "classes": ["clean", "turbid", "organic", "industrial", "oil"],
        "time": datetime.now(timezone.utc).isoformat()
    })


@app.route('/api/analyze')
def api_analyze():
    try:
        min_lon = float(request.args.get('min_lon'))
        min_lat = float(request.args.get('min_lat'))
        max_lon = float(request.args.get('max_lon'))
        max_lat = float(request.args.get('max_lat'))
    except (TypeError, ValueError) as e:
        return jsonify({"error": f"Invalid bbox: {e}"}), 400

    if min_lon >= max_lon or min_lat >= max_lat:
        return jsonify({"error": "Invalid bbox"}), 400

    size_w = max_lon - min_lon
    size_h = max_lat - min_lat
    if size_w > MAX_SIZE_DEG or size_h > MAX_SIZE_DEG:
        return jsonify({"error": f"Zone too large (max {MAX_SIZE_DEG}°)"}), 400
    if size_w * size_h > MAX_AREA_DEG2:
        return jsonify({"error": "Area too large"}), 400

    resolution, quality = pick_resolution(size_w, size_h)
    print(f"[analyze] zone={size_w:.3f}°×{size_h:.3f}° → {quality} ({resolution})")

    try:
        result = analyze([min_lon, min_lat, max_lon, max_lat], resolution=resolution)
        return jsonify(result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Server error: {str(e)}"}), 500


if __name__ == '__main__':
    print("=" * 60)
    print("Aquature Backend — Water Intelligence Platform")
    print("=" * 60)
    print("Bands: B03, B04, B05, B08, B11, B12")
    print("Classes: Clean, Turbid, Organic, Industrial, Oil")
    print("Water mask: STRICT → MODERATE → LOOSE → NDWI-ONLY")
    print("=" * 60)
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)