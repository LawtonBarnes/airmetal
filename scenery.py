"""The world AIR METAL flies over, loaded once at startup from
data/scenery_<lat>_<lon>.npz (built on the PC by tools/build_scenery.py from
JOAN JET's OpenStreetMap basemap -- roads, rail, rivers, lakes, runways out
to 60 NM -- in local meters, simplified for a 512x192 view).

  * line layers -> polylines, cut into short chunks and bucketed by a
    coarse grid, so a frame touches only the chunks near the camera and
    draws each with one pygame.draw.lines call;
  * lakes -> outline rings with bounding boxes.
"""
import glob
import math
import os

import numpy as np

CELL_M = 2000.0
GRID_HALF = 56  # cells each side of the origin: +-112 km covers the 60 NM basemap
CHUNK_POINTS = 12  # max points per drawn chunk

WATER_COLOR = (52, 92, 172)  # lakes, rivers and canals alike

# Line layers: (color, width px at 512x192, draw distance m). Draw order is
# this order -- rivers under roads, big roads last so they win at crossings.
LINE_STYLES = {
    "river": (WATER_COLOR, 2, 12000),
    "canal": (WATER_COLOR, 1, 6000),
    "tertiary": ((128, 128, 120), 1, 2500),
    "secondary": ((138, 138, 130), 1, 4500),
    "rail": ((132, 114, 102), 1, 8000),
    "taxiway": ((95, 95, 90), 1, 4000),
    "primary": ((150, 150, 142), 1, 10000),
    "trunk": ((165, 165, 158), 2, 20000),
    "motorway": ((175, 175, 168), 2, 25000),
    "runway": ((85, 85, 85), 3, 30000),
}
LAKE_COLOR = WATER_COLOR
LAKE_DISTANCE_M = 40000


class Lines:
    """Every line layer together, so a frame does one gather/project pass.
    pts: (N, 2) float32; chunk_start: start index of each chunk in pts plus
    a final end; chunk_style: index into LINE_STYLES; chunks sorted by grid
    cell, cell_first[c] = first chunk in cell c."""

    def __init__(self, layers):
        chunk_pts, chunk_start, chunk_cell, chunk_style = [], [0], [], []
        side = 2 * GRID_HALF
        for style, (pts, starts) in enumerate(layers):
            for i in range(len(starts) - 1):
                line = pts[starts[i]:starts[i + 1]]
                for a in range(0, len(line) - 1, CHUNK_POINTS - 1):
                    piece = line[a:a + CHUNK_POINTS]
                    mid = piece.mean(axis=0)
                    cx = int(math.floor(mid[0] / CELL_M)) + GRID_HALF
                    cz = int(math.floor(mid[1] / CELL_M)) + GRID_HALF
                    if not (0 <= cx < side and 0 <= cz < side):
                        continue
                    chunk_pts.append(piece)
                    chunk_start.append(chunk_start[-1] + len(piece))
                    chunk_cell.append(cz * side + cx)
                    chunk_style.append(style)
        order = np.argsort(np.array(chunk_cell, np.int64), kind="stable")
        self.pts = np.concatenate([chunk_pts[i] for i in order]).astype(np.float32)
        lens = np.diff(chunk_start)[order]
        self.chunk_start = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
        self.chunk_style = np.array(chunk_style, np.int64)[order]
        cells = np.array(chunk_cell, np.int64)[order]
        self.cell_first = np.searchsorted(cells, np.arange(side * side + 1))
        # Each chunk's center and radius, to cull by distance before touching its points.
        first = self.chunk_start[:-1]
        lo = np.stack([np.minimum.reduceat(self.pts[:, 0], first), np.minimum.reduceat(self.pts[:, 1], first)], 1)
        hi = np.stack([np.maximum.reduceat(self.pts[:, 0], first), np.maximum.reduceat(self.pts[:, 1], first)], 1)
        self.chunk_center = (lo + hi) / 2
        self.chunk_radius = np.hypot(*(hi - lo).T) / 2

    def chunks_in(self, cells):
        """Chunk index ranges for the given flat cell ids -> 1-D array of chunk ids."""
        ranges = [np.arange(self.cell_first[c], self.cell_first[c + 1]) for c in cells
                  if self.cell_first[c + 1] > self.cell_first[c]]
        return np.concatenate(ranges) if ranges else np.zeros(0, np.int64)


class Scenery:
    def __init__(self, data_dir, lat, lon):
        data = np.load(self._find(data_dir, lat, lon))
        empty = (np.zeros((0, 2), np.float32), np.zeros(1, np.int64))
        self.lines = Lines([(data[f"lines_{n}_pts"], data[f"lines_{n}_starts"]) if f"lines_{n}_pts" in data
                            else empty for n in LINE_STYLES])
        lake_pts, lake_starts = data["lake_pts"], data["lake_starts"]
        self.lake_pts = lake_pts.astype(np.float32)
        self.lake_start = lake_starts.astype(np.int64)
        rings = [lake_pts[lake_starts[i]:lake_starts[i + 1]] for i in range(len(lake_starts) - 1)]
        self.lake_bboxes = np.array([[*r.min(0), *r.max(0)] for r in rings], np.float32).reshape(-1, 4)
        self.attribution = str(data["attribution"])

    @staticmethod
    def _find(data_dir, lat, lon):
        for path in sorted(glob.glob(os.path.join(data_dir, "scenery_*.npz"))):
            name = os.path.basename(path)[len("scenery_"):-len(".npz")]
            blat, blon = (float(v) for v in name.split("_"))
            if abs(blat - lat) < 0.0005 and abs(blon - lon) < 0.0005:
                return path
        raise FileNotFoundError(f"no scenery for {lat},{lon} in {data_dir} -- run tools/build_scenery.py")

    @staticmethod
    def cells_ahead(x, z, heading_deg, radius_m):
        """Flat cell ids within radius_m of (x, z), skipping cells well behind the camera."""
        side = 2 * GRID_HALF
        r = int(math.ceil(radius_m / CELL_M)) + 1
        cx0, cz0 = int(math.floor(x / CELL_M)), int(math.floor(z / CELL_M))
        fx, fz = math.sin(math.radians(heading_deg)), math.cos(math.radians(heading_deg))
        ii, jj = np.meshgrid(np.arange(-r, r + 1), np.arange(-r, r + 1))
        ccx, ccz = (cx0 + ii + 0.5) * CELL_M - x, (cz0 + jj + 0.5) * CELL_M - z
        keep = (np.hypot(ccx, ccz) < radius_m + CELL_M) & (ccx * fx + ccz * fz > -1.5 * CELL_M)
        gx, gz = (cx0 + ii)[keep] + GRID_HALF, (cz0 + jj)[keep] + GRID_HALF
        ok = (gx >= 0) & (gx < side) & (gz >= 0) & (gz < side)
        return (gz[ok] * side + gx[ok]).tolist()
