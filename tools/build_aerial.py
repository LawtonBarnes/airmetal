"""Builds data/aerial_<lat>_<lon>.jpg + .npz -- the aerial-photo ground near
the Metal Shop: USDA NAIP orthoimagery (public domain), via USGS's map
server, AERIAL_M_PER_PX per pixel over a square +-AERIAL_HALF_M around the
shop. Run on the PC, not the Pi.

The photo is a JPEG (row 0 = north, as photographed); the .npz holds the
geometry and `emit`, the night-light map at the same resolution (row 0 =
south, like the land-cover raster): pavement and roofs -- bright and not
green -- weighted by how developed the land cover says each spot is, so
streets, parking lots and shopping strips glow at night and fields don't.

Tiles are cached in tools/naip_cache/ (gitignored); --refresh re-fetches.

Usage: python tools/build_aerial.py [--refresh]
"""
import argparse
import io
import math
import os
import sys
import urllib.request

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import config  # noqa: E402
import scenery  # noqa: E402
from geo import M_PER_DEG_LAT, Projection  # noqa: E402

SERVER = "https://imagery.nationalmap.gov/arcgis/rest/services/USGSNAIPImagery/ImageServer/exportImage"
USER_AGENT = "AIRMETAL/0.1 (Raspberry Pi CRT flight sim; aerial ground build)"
TILE_PX = 2000
# NAIP is flown in high summer and processed bright: this per-channel gain
# matches it to the drone shots from the shop (2026-10-27, measured at
# saturation 1.0 -- the app's saturation setting then applies on top).
PHOTO_GAIN = np.array((0.81, 0.77, 0.745), np.float32)
JPEG_QUALITY = 90
# Night light by land class: how much of the photo's pavement/roof light
# shows (fields and bare ground are bright too, but unlit).
LIT_BY_CLASS = {"developed_high": 1.0, "developed_medium": 0.9, "developed_low": 0.7, "developed_open": 0.35}
LIT_OTHER = 0.04


def mercator(lat, lon):
    r = 6378137.0
    return r * math.radians(lon), r * math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))


def fetch_tile(proj, x0, z0, size_m, px, path):
    """The square [x0, x0+size_m] x [z0, z0+size_m] (local meters) as px x px."""
    s, n = proj.lat0 + z0 / M_PER_DEG_LAT, proj.lat0 + (z0 + size_m) / M_PER_DEG_LAT
    w, e = proj.lon0 + x0 / proj.m_per_deg_lon, proj.lon0 + (x0 + size_m) / proj.m_per_deg_lon
    # In Web Mercator, where this square stays square: in degrees it's
    # wider than tall, and the server would stretch it to fit the image.
    (mw, ms), (me, mn) = mercator(s, w), mercator(n, e)
    url = (f"{SERVER}?bbox={mw},{ms},{me},{mn}&bboxSR=3857&imageSR=3857&size={px},{px}"
           f"&format=png&f=image")
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=300) as resp:
        data = resp.read()
    Image.open(io.BytesIO(data)).convert("RGB").save(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    s = config.load()
    proj = Projection(s["lat"], s["lon"])
    half, mpp = scenery.AERIAL_HALF_M, scenery.AERIAL_M_PER_PX
    n = int(round(2 * half / mpp))
    tiles = n // TILE_PX
    assert tiles * TILE_PX == n, "AERIAL_HALF_M must be a whole number of tiles"
    cache = os.path.join(ROOT, "tools", "naip_cache")
    os.makedirs(cache, exist_ok=True)
    photo = Image.new("RGB", (n, n))
    tile_m = TILE_PX * mpp
    for ty in range(tiles):  # ty = 0: northmost row of tiles
        for tx in range(tiles):
            x0 = -half + tx * tile_m
            z0 = half - (ty + 1) * tile_m
            path = os.path.join(cache, f"naip_{s['lat']:.4f}_{s['lon']:.4f}_{mpp:g}m_{ty}_{tx}.png")
            if args.refresh or not os.path.exists(path):
                print(f"fetching tile {ty},{tx}...")
                fetch_tile(proj, x0, z0, tile_m, TILE_PX, path)
            photo.paste(Image.open(path), (tx * TILE_PX, ty * TILE_PX))

    base = os.path.join(config.DATA_DIR, f"aerial_{s['lat']:.4f}_{s['lon']:.4f}")
    rgb = np.asarray(photo, np.float32)[::-1] * PHOTO_GAIN  # row 0 = south

    # The land class under each photo pixel (nearest land-cover texel), and
    # the land cover's water, upsampled smoothly so shores aren't 32 m steps.
    lc = np.load(scenery.Scenery._find(config.DATA_DIR, s["lat"], s["lon"], "landcover"))
    land, lhalf, lmpp = lc["land"], float(lc["half_m"]), float(lc["m_per_px"])
    centers = -half + (np.arange(n) + 0.5) * mpp
    idx = ((centers + lhalf) / lmpp).astype(np.int64)
    cls = land[idx[:, None], idx[None, :]]
    water_cls = list(scenery.LAND_CLASSES).index("water")
    lo, hi = idx.min() - 1, idx.max() + 2
    wsmall = Image.fromarray(((land[lo:hi, lo:hi] == water_cls) * 255).astype(np.uint8))
    wsize = int(round((hi - lo) * lmpp / mpp))
    wbig = np.asarray(wsmall.resize((wsize, wsize), Image.BILINEAR), np.float32) / 255
    off = int(round(((lo * lmpp - lhalf) - (-half)) / -mpp))
    water = np.clip((wbig[off:off + n, off:off + n] - 0.35) / 0.3, 0, 1)
    # Lakes in NAIP are glare-white or a different color per flight: paint
    # them the scene's water color, keeping a little of the photo's texture.
    lum = rgb.mean(-1, keepdims=True)
    painted = np.array(scenery.WATER_COLOR, np.float32) * (0.85 + 0.3 * np.clip(lum / max(lum.mean(), 1), 0, 1))
    rgb = rgb * (1 - water[..., None]) + painted * water[..., None]
    Image.fromarray(rgb[::-1].astype(np.uint8)).save(base + ".jpg", quality=JPEG_QUALITY)

    # Night lights: pavement and roofs are bright and not green.
    lum = lum[..., 0]
    green = rgb[..., 1] - (rgb[..., 0] + rgb[..., 2]) / 2
    paved = np.clip((lum - 70) / 90, 0, 1) * np.clip(1 - np.maximum(green, 0) / 12, 0, 1) * (1 - water)
    del rgb, lum, green
    lit_lut = np.array([LIT_BY_CLASS.get(k, LIT_OTHER) for k in scenery.LAND_CLASSES], np.float32)
    lit = lit_lut[cls]
    emit = (paved * lit * 255).astype(np.uint8)
    np.savez_compressed(base + ".npz", emit=emit, m_per_px=mpp, half_m=half,
                        attribution="Aerial imagery: USDA NAIP via USGS The National Map (public domain)")
    print(f"{n}x{n} at {mpp:g} m/px; photo {os.path.getsize(base + '.jpg') / 1e6:.1f} MB, "
          f"night map {os.path.getsize(base + '.npz') / 1e6:.1f} MB, lit {np.mean(emit > 64) * 100:.0f}%")


if __name__ == "__main__":
    main()
