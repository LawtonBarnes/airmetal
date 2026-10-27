"""Builds data/buildings_<lat>_<lon>.npz -- the skyline: every tall building
(>= 60 m, or >= 18 floors) in the scenery square, from OpenStreetMap via
Overpass. Run on the PC, not the Pi.

Each building -- or each `building:part`, where a tower is mapped as parts
of different heights (Bank of America Plaza has eight) -- becomes a "piece":
a ring of footprint points with a bottom and top height, extruded. Pieces
can taper (the top ring differs from the bottom): a part with
roof:shape=dome (Reunion Tower's ball) is built as a stack of tapered bands
around its center.

The Overpass result is cached (tools/osm_cache/, gitignored); --refresh
re-fetches.

Usage: python tools/build_buildings.py [--refresh]
"""
import argparse
import json
import math
import os
import sys
import urllib.parse
import urllib.request
import zlib

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

import config  # noqa: E402
import scenery  # noqa: E402
from build_scenery import rdp  # noqa: E402
from geo import M_PER_DEG_LAT, Projection  # noqa: E402

OVERPASS = "https://overpass-api.de/api/interpreter"
USER_AGENT = "AIRMETAL/0.1 (Raspberry Pi CRT flight sim; skyline build)"
MIN_HEIGHT_M = 60.0
METERS_PER_LEVEL = 3.6  # when only building:levels is tagged
FOOTPRINT_TOL_M = 2.5
MAX_RING = 16  # more corners than this get simplified harder
# Mapping errors: OSM way id -> why it's skipped.
SKIP_IDS = {
    654513276: "192 m on a low mosque in Arlington",
}

QUERY = """[out:json][timeout:180];
(
  way["building"]["height"~"^([6-9][0-9]|[1-9][0-9][0-9])(\\\\.[0-9]+)?( ?m)?$"]({bbox});
  way["building"]["building:levels"~"^(1[8-9]|[2-9][0-9]|1[0-9][0-9])$"]({bbox});
  way["building:part"]["height"~"^([6-9][0-9]|[1-9][0-9][0-9])(\\\\.[0-9]+)?( ?m)?$"]({bbox});
);
out tags geom;
"""

# Untagged facades: a deterministic pick (by OSM id) from glass and stone.
FACADES = [
    (112, 128, 140),  # blue-gray glass
    (98, 116, 112),  # green glass
    (150, 150, 146),  # concrete
    (164, 156, 140),  # limestone
    (128, 132, 138),  # steel
    (86, 98, 112),  # dark glass
]
NAMED_COLORS = {
    "white": (196, 196, 192), "grey": (140, 140, 140), "gray": (140, 140, 140),
    "silver": (168, 170, 172), "black": (54, 56, 60), "blue": (92, 112, 150),
    "green": (96, 128, 108), "brown": (120, 96, 78), "beige": (176, 166, 140),
    "tan": (170, 150, 120), "red": (150, 80, 66), "yellow": (190, 176, 110),
}

# roof:shape=dome parts, as bands around the center: (fraction of the part's
# height, fraction of its footprint radius) -- a ball, flattened underneath.
DOME_PROFILE = [(0.0, 0.45), (0.15, 0.85), (0.45, 1.0), (0.75, 0.85), (1.0, 0.3)]
DOME_SIDES = 12


def parse_m(v):
    if v is None:
        return None
    try:
        return float(v.strip().lower().replace("m", "").strip().split()[0].split(";")[0])
    except (ValueError, IndexError):
        return None


def color_for(tags, osm_id):
    c = (tags.get("building:colour") or tags.get("building:color") or "").strip().lower()
    if c.startswith("#") and len(c) == 7:
        try:
            return tuple(int(c[i:i + 2], 16) for i in (1, 3, 5))
        except ValueError:
            pass
    if c in NAMED_COLORS:
        return NAMED_COLORS[c]
    return FACADES[zlib.crc32(str(osm_id).encode()) % len(FACADES)]


def fetch(bbox, path):
    q = QUERY.format(bbox=bbox)
    req = urllib.request.Request(OVERPASS, data=urllib.parse.urlencode({"data": q}).encode(),
                                 headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=240) as resp:
        data = resp.read()
    with open(path, "wb") as f:
        f.write(data)


def ring_of(geom, proj):
    pts = np.array([proj.to_xz(p["lat"], p["lon"]) for p in geom], np.float64)
    if len(pts) > 2 and np.allclose(pts[0], pts[-1]):
        pts = pts[:-1]
    tol = FOOTPRINT_TOL_M
    out = pts
    while True:
        closed = np.vstack([out, out[:1]])
        out = rdp(closed, tol)[:-1]
        if len(out) <= MAX_RING:
            break
        tol *= 1.5
    # Counter-clockwise seen from above (x east, z north), so wall normals point out.
    area = 0.5 * np.sum(out[:, 0] * np.roll(out[:, 1], -1) - np.roll(out[:, 0], -1) * out[:, 1])
    return (out if area > 0 else out[::-1]), abs(area)


def point_in(poly, p):
    x, z = p
    inside = False
    n = len(poly)
    for i in range(n):
        x1, z1 = poly[i]
        x2, z2 = poly[(i + 1) % n]
        if (z1 > z) != (z2 > z) and x < x1 + (z - z1) * (x2 - x1) / (z2 - z1):
            inside = not inside
    return inside


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    s = config.load()
    proj = Projection(s["lat"], s["lon"])
    half = scenery.GRID_HALF * scenery.CELL_M
    dlat, dlon = half / M_PER_DEG_LAT, half / proj.m_per_deg_lon
    bbox = f"{s['lat'] - dlat:.4f},{s['lon'] - dlon:.4f},{s['lat'] + dlat:.4f},{s['lon'] + dlon:.4f}"
    cache = os.path.join(ROOT, "tools", "osm_cache")
    os.makedirs(cache, exist_ok=True)
    path = os.path.join(cache, f"tall_{s['lat']:.4f}_{s['lon']:.4f}.json")
    if args.refresh or not os.path.exists(path):
        print("fetching from Overpass...")
        fetch(bbox, path)
    elements = json.load(open(path, encoding="utf-8"))["elements"]

    pieces = []  # (bottom ring, top ring, y0, y1, color, osm id)
    parts, outlines = [], []
    for el in elements:
        if el["type"] != "way" or el["id"] in SKIP_IDS or len(el.get("geometry", [])) < 4:
            continue
        t = el.get("tags", {})
        h = parse_m(t.get("height"))
        if h is None and t.get("building:levels", "").isdigit():
            h = int(t["building:levels"]) * METERS_PER_LEVEL
        if h is None or h < MIN_HEIGHT_M:
            continue
        y0 = parse_m(t.get("min_height")) or 0.0
        ring, area = ring_of(el["geometry"], proj)
        if len(ring) < 3 or area < 20:
            continue
        rec = (ring, y0, h, color_for(t, el["id"]), el["id"], t)
        (parts if "building:part" in t else outlines).append(rec)

    # A building mapped as parts: draw the parts, not the outline around them.
    centroids = [r[0].mean(0) for r in parts]
    for rec in outlines:
        ring = rec[0]
        lo, hi = ring.min(0), ring.max(0)
        if any(lo[0] <= c[0] <= hi[0] and lo[1] <= c[1] <= hi[1] and point_in(ring, c) for c in centroids):
            continue
        parts.append(rec)

    ang = np.arange(DOME_SIDES) * 2 * math.pi / DOME_SIDES
    unit = np.stack([np.cos(ang), np.sin(ang)], -1)
    domes = 0
    for ring, y0, h, color, osm_id, t in parts:
        if t.get("roof:shape") != "dome":
            pieces.append((ring, ring, y0, h, color, osm_id))
            continue
        center = ring.mean(0)
        radius = np.hypot(*(ring - center).T).mean()
        for (fa, ra), (fb, rb) in zip(DOME_PROFILE, DOME_PROFILE[1:]):
            pieces.append((center + unit * radius * ra, center + unit * radius * rb,
                           y0 + (h - y0) * fa, y0 + (h - y0) * fb, color, osm_id))
        domes += 1
    print(f"{domes} domes")

    n = len(pieces)
    starts = np.zeros(n + 1, np.int64)
    starts[1:] = np.cumsum([len(p[0]) for p in pieces])
    np.savez_compressed(
        os.path.join(config.DATA_DIR, f"buildings_{s['lat']:.4f}_{s['lon']:.4f}.npz"),
        bottom=np.concatenate([p[0] for p in pieces]).astype(np.float32),
        top=np.concatenate([p[1] for p in pieces]).astype(np.float32),
        starts=starts,
        y0=np.array([p[2] for p in pieces], np.float32),
        y1=np.array([p[3] for p in pieces], np.float32),
        color=np.array([p[4] for p in pieces], np.uint8),
        osm_id=np.array([p[5] for p in pieces], np.int64),
        attribution="Buildings (c) OpenStreetMap contributors, ODbL",
    )
    walls = int(starts[-1])
    print(f"{n} pieces, {walls} wall quads, tallest {max(p[3] for p in pieces):.0f} m")


if __name__ == "__main__":
    main()
