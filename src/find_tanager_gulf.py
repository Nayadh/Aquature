"""
Search all Tanager scenes and find the closest one to the Gulf of Oman.
"""

import requests
import json

COLLECTION = "coastal-water-bodies"
BASE_URL = (
    f"https://www.planet.com/data/stac/tanager-core-imagery/{COLLECTION}"
)

# Gulf of Oman target bbox
TARGET = [58.20, 23.40, 58.80, 23.90]   # [min_lon, min_lat, max_lon, max_lat]


def bbox_center(bbox):
    return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)


def distance_to_target(bbox):
    """Approximate distance in degrees from target center."""
    c_lon, c_lat = bbox_center(bbox)
    t_lon, t_lat = bbox_center(TARGET)
    return ((c_lon - t_lon) ** 2 + (c_lat - t_lat) ** 2) ** 0.5


def main():
    print("Fetching Tanager collection...")
    r = requests.get(f"{BASE_URL}/collection.json", timeout=60)
    r.raise_for_status()
    col = r.json()

    item_links = [l for l in col.get("links", []) if l.get("rel") == "item"]
    item_ids = [l["href"].split("/")[-1].replace(".json", "") for l in item_links]

    print(f"Found {len(item_ids)} scenes. Checking each one...\n")

    results = []
    for i, iid in enumerate(item_ids):
        try:
            url = f"{BASE_URL}/{iid}/{iid}.json"
            item = requests.get(url, timeout=30).json()
            bbox = item["bbox"]
            dist = distance_to_target(bbox)
            date = item["properties"]["datetime"][:10]
            results.append((dist, iid, bbox, date))
            if (i + 1) % 10 == 0:
                print(f"  Checked {i+1}/{len(item_ids)}")
        except Exception as e:
            print(f"  [!] {iid}: {e}")

    results.sort(key=lambda x: x[0])

    print("\n" + "=" * 70)
    print("Closest scenes to Gulf of Oman:")
    print("=" * 70)
    for dist, iid, bbox, date in results[:10]:
        print(f"\nDistance: {dist:.2f}°")
        print(f"  ID  : {iid}")
        print(f"  Date: {date}")
        print(f"  BBox: {bbox}")


if __name__ == "__main__":
    main()