"""Builds data/scenery_<lat>_<lon>.npz from JOAN JET's prebuilt basemap -- run
on the PC, not the Pi.

The basemap (joanjett/tools/build_basemap.py: OpenStreetMap roads, rail,
rivers, lakes, runways out to 60 NM, NM offsets from the center) carries
far more detail than a 512x192 windshield can show, and the Pi pays per
point. This converts it to meters and thins it:

  * every line feature simplified (Ramer-Douglas-Peucker) to LINE_TOL_M,
    then split into segments;
  * lake outlines simplified to LAKE_TOL_M, islands dropped, and lakes
    smaller than MIN_LAKE_M2 dropped.

Usage: python tools/build_scenery.py [BASEMAP.json.gz]
(default: the basemap in data/ matching settings' location)
"""
import base64
import glob
import gzip
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import config  # noqa: E402
from geo import M_PER_NM  # noqa: E402

LINE_TOL_M = 12.0
LINE_TOL_OVERRIDE_M = {"river": 30.0, "canal": 30.0}  # meanders aren't worth the points
LAKE_TOL_M = 25.0
MIN_LAKE_M2 = 20000.0  # ~2 hectares; smaller ponds aren't worth a polygon
LINE_LAYERS = ("river", "canal", "tertiary", "secondary", "rail", "taxiway", "primary", "trunk",
               "motorway", "runway")


def _int32(b64):
    return np.frombuffer(base64.b64decode(b64), dtype="<i4")


def rdp(pts, tol):
    """Ramer-Douglas-Peucker on an (N, 2) polyline -> kept points."""
    if len(pts) < 3:
        return pts
    keep = np.zeros(len(pts), bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(pts) - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        a, b = pts[i], pts[j]
        ab = b - a
        seg = pts[i + 1:j] - a
        n = np.hypot(*ab)
        if n == 0:
            d = np.hypot(seg[:, 0], seg[:, 1])
        else:
            d = np.abs(seg[:, 0] * ab[1] - seg[:, 1] * ab[0]) / n
        k = int(np.argmax(d))
        if d[k] > tol:
            m = i + 1 + k
            keep[m] = True
            stack += [(i, m), (m, j)]
    return pts[keep]


def chain(parts):
    """Joins line fragments that share endpoints into longer polylines (the
    basemap stores most features as separate ~2-point pieces). Walks through
    points where exactly two fragments meet; stops at junctions and ends."""
    key = lambda p: (round(p[0], 1), round(p[1], 1))
    ends = {}
    for i, p in enumerate(parts):
        ends.setdefault(key(p[0]), []).append(i)
        ends.setdefault(key(p[-1]), []).append(i)
    used = np.zeros(len(parts), bool)
    out = []

    def extend(line):
        while True:
            k = key(line[-1])
            nxt = [j for j in ends.get(k, []) if not used[j]]
            if len(ends.get(k, [])) != 2 or len(nxt) != 1:
                return line
            j = nxt[0]
            used[j] = True
            p = parts[j]
            line = np.vstack([line, (p if key(p[0]) == k else p[::-1])[1:]])

    for i, p in enumerate(parts):
        if used[i]:
            continue
        used[i] = True
        line = extend(p)
        line = extend(line[::-1])
        out.append(line)
    return out


def area(ring):
    x, y = ring[:, 0], ring[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def main():
    s = config.load()
    if len(sys.argv) > 1:
        src = sys.argv[1]
    else:
        src = next(p for p in sorted(glob.glob(os.path.join(config.DATA_DIR, "basemap_*.json.gz")))
                   if all(abs(float(v) - c) < 0.0005 for v, c in
                          zip(os.path.basename(p)[8:-8].split("_"), (s["lat"], s["lon"]))))
    with gzip.open(src, "rt", encoding="utf-8") as f:
        raw = json.load(f)
    q = float(raw.get("quant", 1000))
    out = {}
    for name, packed in raw["layers"].items():
        coords = (_int32(packed["coords"]).reshape(-1, 2) / q * M_PER_NM).astype(np.float64)
        offsets = _int32(packed["offsets"]).astype(np.int64)
        if name == "lake":
            rings = _int32(packed["rings"]).astype(np.int64)
            pts, starts, part, before = [], [0], 0, 0
            for count in rings:
                ring = coords[offsets[part]:offsets[part + 1]]
                part += count
                before += 1
                if len(ring) < 4 or area(ring) < MIN_LAKE_M2:
                    continue
                ring = rdp(ring, LAKE_TOL_M)
                if len(ring) >= 4:
                    pts.append(ring)
                    starts.append(starts[-1] + len(ring))
            out["lake_pts"] = np.concatenate(pts).astype(np.float32)
            out["lake_starts"] = np.array(starts, np.int64)
            print(f"lake: {before} -> {len(pts)} lakes, {len(coords)} -> {len(out['lake_pts'])} points")
        elif name in LINE_LAYERS:
            parts = [coords[offsets[i]:offsets[i + 1]] for i in range(len(offsets) - 1)]
            tol = LINE_TOL_OVERRIDE_M.get(name, LINE_TOL_M)
            lines = [rdp(line, tol) for line in chain(parts)]
            lines = [line for line in lines if len(line) >= 2]
            out[f"lines_{name}_pts"] = np.concatenate(lines).astype(np.float32)
            out[f"lines_{name}_starts"] = np.cumsum([0] + [len(line) for line in lines]).astype(np.int64)
            n_segs = sum(len(line) - 1 for line in lines)
            print(f"{name}: {len(coords) - len(parts)} segments in {len(parts)} pieces -> "
                  f"{n_segs} segments in {len(lines)} polylines")
    dst = os.path.join(config.DATA_DIR, f"scenery_{s['lat']:.4f}_{s['lon']:.4f}.npz")
    np.savez_compressed(dst, center=np.array([s["lat"], s["lon"]]),
                        attribution=np.array(raw.get("attribution", "")), **out)
    print(f"-> {dst} ({os.path.getsize(dst) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
