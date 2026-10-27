"""Builds data/landcover_<lat>_<lon>.npz -- the textured ground's land-cover
raster -- from the USGS National Land Cover Database (NLCD 2021, 30 m,
public domain). Run on the PC, not the Pi.

NLCD comes from MRLC's map server as palette PNG tiles in lat/lon; every
pixel's color is one of the official NLCD legend colors, so the class
decodes exactly with no GIS library. The tiles are mosaicked, then resampled
(nearest) onto the sim's own local grid -- meters east/north of the home
spot (geo.Projection), M_PER_PX per pixel, the same +-112 km square the road
scenery covers -- and NLCD's 16 classes merged into scenery.LAND_CLASSES.

Tiles are cached (default tools/nlcd_cache/, gitignored), so a re-run only
re-resamples.

Usage: python tools/build_landcover.py [--cache DIR]
"""
import argparse
import io
import math
import os
import sys
import time
import urllib.request

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import config  # noqa: E402
import scenery  # noqa: E402
from geo import M_PER_DEG_LAT, Projection  # noqa: E402

WMS = ("https://www.mrlc.gov/geoserver/mrlc_display/wms?service=WMS&version=1.1.1&request=GetMap"
       "&layers=NLCD_2021_Land_Cover_L48&styles=&srs=EPSG:4326&format=image/png"
       "&bbox={w},{s},{e},{n}&width={px}&height={px}")
TILE_PX = 1024
TILE_DEG = 0.25  # ~24 m/px north-south at 1024 px -- finer than NLCD's 30 m
HALF_M = scenery.GRID_HALF * scenery.CELL_M
M_PER_PX = scenery.LAND_M_PER_PX
USER_AGENT = "AIRMETAL/0.1 (Raspberry Pi CRT flight sim; land cover build)"

# Official NLCD legend: class code -> RGB.
NLCD_COLORS = {
    11: (70, 107, 159), 12: (209, 222, 248),
    21: (222, 197, 197), 22: (217, 146, 130), 23: (235, 0, 0), 24: (171, 0, 0),
    31: (179, 172, 159),
    41: (104, 171, 95), 42: (28, 95, 44), 43: (181, 197, 143),
    52: (204, 184, 121), 71: (223, 223, 194),
    81: (220, 217, 57), 82: (171, 108, 40),
    90: (184, 217, 235), 95: (108, 159, 184),
}
# NLCD code -> index into scenery.LAND_CLASSES.
NLCD_TO_CLASS = {
    11: "water", 12: "water",
    21: "developed_open", 22: "developed_low", 23: "developed_medium", 24: "developed_high",
    31: "barren",
    41: "forest", 43: "forest", 42: "evergreen",
    52: "grass", 71: "grass",
    81: "pasture", 82: "crops",
    90: "wetland", 95: "wetland",
}


def fetch_tile(cache, s, w):
    path = os.path.join(cache, f"nlcd_{s:.3f}_{w:.3f}.png")
    if not os.path.exists(path):
        url = WMS.format(w=w, s=s, e=w + TILE_DEG, n=s + TILE_DEG, px=TILE_PX)
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        for attempt in range(4):
            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    data = resp.read()
                Image.open(io.BytesIO(data)).verify()
                break
            except Exception as exc:  # flaky server: retry, then give up loudly
                if attempt == 3:
                    raise
                print(f"  retry {s:.3f},{w:.3f}: {exc}")
                time.sleep(5)
        with open(path, "wb") as f:
            f.write(data)
    return np.array(Image.open(path).convert("RGB"))


def decode(rgb):
    """RGB tile -> LAND_CLASSES index per pixel (nearest legend color; 255 = no data)."""
    codes = list(NLCD_COLORS)
    legend = np.array([NLCD_COLORS[c] for c in codes], np.int32)
    names = list(scenery.LAND_CLASSES)
    packed = (rgb[..., 0].astype(np.int32) << 16) | (rgb[..., 1].astype(np.int32) << 8) | rgb[..., 2]
    # A tile holds a dozen-odd distinct colors: match those, not every pixel.
    uniq, inverse = np.unique(packed.ravel(), return_inverse=True)
    u_rgb = np.stack([uniq >> 16, (uniq >> 8) & 255, uniq & 255], -1)
    d = ((u_rgb[:, None, :] - legend[None, :, :]) ** 2).sum(-1)
    lut = np.array([names.index(NLCD_TO_CLASS[codes[k]]) for k in d.argmin(1)], np.uint8)
    lut[d.min(1) > 300] = 255  # white/transparent outside the data, or a stray color
    return lut[inverse].reshape(rgb.shape[:2])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache", default=os.path.join(ROOT, "tools", "nlcd_cache"))
    args = ap.parse_args()
    os.makedirs(args.cache, exist_ok=True)
    s = config.load()
    proj = Projection(s["lat"], s["lon"])
    lat_span = HALF_M / M_PER_DEG_LAT
    lon_span = HALF_M / proj.m_per_deg_lon
    # Tile grid on round TILE_DEG boundaries covering the square.
    s0 = math.floor((s["lat"] - lat_span) / TILE_DEG) * TILE_DEG
    w0 = math.floor((s["lon"] - lon_span) / TILE_DEG) * TILE_DEG
    n_rows = math.ceil((s["lat"] + lat_span - s0) / TILE_DEG)
    n_cols = math.ceil((s["lon"] + lon_span - w0) / TILE_DEG)
    print(f"{n_rows}x{n_cols} tiles of {TILE_DEG} deg")
    mosaic = np.full((n_rows * TILE_PX, n_cols * TILE_PX), 255, np.uint8)  # row 0 = north edge
    for r in range(n_rows):
        for c in range(n_cols):
            south = s0 + (n_rows - 1 - r) * TILE_DEG
            west = w0 + c * TILE_DEG
            mosaic[r * TILE_PX:(r + 1) * TILE_PX, c * TILE_PX:(c + 1) * TILE_PX] = \
                decode(fetch_tile(args.cache, south, west))
            print(f"  tile {r * n_cols + c + 1}/{n_rows * n_cols}", end="\r", flush=True)
    print()
    # Resample onto the local grid: row 0 = south edge (z = -HALF_M), col 0 = west.
    n = int(round(2 * HALF_M / M_PER_PX))
    centers = -HALF_M + (np.arange(n) + 0.5) * M_PER_PX
    lat = s["lat"] + centers / M_PER_DEG_LAT
    lon = s["lon"] + centers / proj.m_per_deg_lon
    deg_per_px = TILE_DEG / TILE_PX
    north = s0 + n_rows * TILE_DEG
    rows = np.clip(((north - lat) / deg_per_px).astype(np.int64), 0, mosaic.shape[0] - 1)
    cols = np.clip(((lon - w0) / deg_per_px).astype(np.int64), 0, mosaic.shape[1] - 1)
    land = mosaic[rows[:, None], cols[None, :]]
    missing = land == 255
    if missing.any():
        print(f"{missing.mean() * 100:.2f}% no data -> grass")
        land[missing] = list(scenery.LAND_CLASSES).index("grass")
    out = os.path.join(config.DATA_DIR, f"landcover_{s['lat']:.4f}_{s['lon']:.4f}.npz")
    np.savez_compressed(out, land=land, m_per_px=M_PER_PX, half_m=HALF_M,
                        classes=np.array(list(scenery.LAND_CLASSES)),
                        attribution="NLCD 2021 Land Cover, USGS / MRLC (public domain)")
    counts = np.bincount(land.ravel(), minlength=len(scenery.LAND_CLASSES)) / land.size * 100
    for name, pct in zip(scenery.LAND_CLASSES, counts):
        print(f"  {name:17s} {pct:5.1f}%")
    print(f"wrote {out} ({os.path.getsize(out) / 1e6:.1f} MB, {n}x{n})")


if __name__ == "__main__":
    main()
