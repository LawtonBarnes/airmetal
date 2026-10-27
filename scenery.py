"""The world AIR METAL flies over, loaded once at startup from
data/scenery_<lat>_<lon>.npz (built on the PC by tools/build_scenery.py from
JOAN JET's OpenStreetMap basemap -- roads, rail, rivers, lakes, runways out
to 60 NM -- in local meters, simplified for a 512x192 view).

  * line layers -> polylines, cut into short chunks and bucketed by a
    coarse grid, so a frame touches only the chunks near the camera and
    draws each with one pygame.draw.lines call;
  * lakes -> outline rings with bounding boxes.

Plus, if they exist:
  * data/landcover_<lat>_<lon>.npz (tools/build_landcover.py) -- the
    land-cover raster the textured ground is looked up in;
  * data/buildings_<lat>_<lon>.npz (tools/build_buildings.py) -- the
    skyline: tall buildings as extruded (sometimes tapered) footprints.
"""
import glob
import math
import os

import numpy as np

CELL_M = 2000.0
GRID_HALF = 56  # cells each side of the origin: +-112 km covers the 60 NM basemap
CHUNK_POINTS = 12  # max points per drawn chunk

WATER_COLOR = (74, 104, 138)  # lakes, rivers and canals alike: steel blue, per the drone shots

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
LAND_LAKE_DISTANCE_M = 10000  # over textured ground: only lakes wholly this close get outlines

# Textured ground (data/landcover_*.npz, tools/build_landcover.py from NLCD):
# land class -> base color, before haze. Order = the index stored in the raster.
# Skyline: sunlight comes from here (horizontal unit vector, x east / z
# north) -- walls facing it are lit, the rest in shade.
SUN_DIR = (-0.42, -0.91)  # from the south-southwest
WALL_SHADE = (0.58, 1.0)  # brightness facing away from the sun .. facing it
BUILDING_DISTANCE_M = 45000

# Aerial-photo ground near the shop (data/aerial_*, tools/build_aerial.py):
# this many meters per pixel over +-AERIAL_HALF_M; the land cover takes
# over beyond it, blended across AERIAL_BLEND_M at the edge.
AERIAL_M_PER_PX = 8.0
AERIAL_HALF_M = 16000.0
AERIAL_BLEND_M = 2000.0

LAND_M_PER_PX = 32.0
LAND_CLASSES = {
    "water": WATER_COLOR,
    # From the drone shots: neighborhoods read as tree canopy, commercial
    # areas as pale roofs and parking.
    # A shade darker than first tried, for separation from the sky; then
    # vegetation matched to the drone shots (2026-10-27): real canopy is a
    # dark olive, much less saturated than the first greens.
    "developed_open": (81, 95, 65),  # parks, golf courses, big lawns
    "developed_low": (57, 68, 49),  # tree-shaded neighborhoods
    "developed_medium": (77, 80, 70),  # canopy + roofs
    "developed_high": (143, 139, 131),  # downtown, malls, the airports' aprons
    "barren": (145, 133, 107),
    "forest": (43, 58, 39),
    "evergreen": (34, 50, 35),
    "grass": (103, 104, 78),
    "pasture": (88, 98, 68),
    "crops": (101, 93, 74),
    "wetland": (62, 75, 62),
}
# Night: each land class's own light at full dark (0..255; see render3d's
# NIGHT_GLOW) -- per NASA's astronaut photo of Dallas at night: commercial
# and dense areas blaze yellow-white, medium ones orange-gold, tree-shaded
# neighborhoods stay mostly dark between the lit arterials.
LAND_EMIT = {"developed_high": 190, "developed_medium": 105, "developed_low": 24, "developed_open": 6}
# ...and lines' (rgb at full dark): street lights, runway and taxiway edge lights.
LINE_EMIT = {
    "motorway": (255, 172, 64), "trunk": (250, 164, 58), "primary": (240, 150, 48),
    "secondary": (220, 134, 40), "tertiary": (185, 110, 32),
    "runway": (220, 220, 200), "taxiway": (50, 80, 230),
}
WINDOW_GLOW = (150, 126, 76)  # lit windows: a tower wall's added light at full dark

# Trees: low-altitude rides get pyramid trees on the wooded land classes,
# one chance per land texel (a deterministic hash picks which, and where in
# the texel), so they stay put frame to frame.
TREE_CHANCE = {
    "forest": 0.9, "evergreen": 0.9, "wetland": 0.5,
    "developed_low": 0.45, "developed_open": 0.25, "developed_medium": 0.15,
    "pasture": 0.04, "grass": 0.04,
}
TREE_COLOR = {"evergreen": (36, 74, 40)}  # anything else: TREE_COLOR_DEFAULT
TREE_COLOR_DEFAULT = (60, 98, 44)
TREE_HEIGHT_M = (11.0, 20.0)
TREE_WIDTH = 0.75  # base width / height
# Range grows with height (the nearest ground in view is farther away):
# TREE_RANGE_PER_AGL x the camera's height, within these limits. Past
# TREE_THIN x the range, trees thin out with distance squared, so they're
# about as dense on screen far as near.
TREE_RANGE_M = (1200.0, 2500.0)
TREE_RANGE_PER_AGL = 8.0
TREE_THIN = 0.35
TREE_MAX_AGL_M = 500.0  # ~1,650 ft AGL, ~2,100 ft on the altimeter: higher, specks -- none drawn
TREE_CAP = 400  # at most this many (each costs ~2 polygons); past it, thinned evenly


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


class Buildings:
    """Pieces: a footprint ring (`bottom` at y0, `top` at y1 -- the same ring
    unless the piece tapers), counter-clockwise from above. Wall i runs from
    ring point i to i+1."""

    def __init__(self, data):
        self.bottom, self.top = data["bottom"], data["top"]
        self.start = data["starts"]
        self.y0, self.y1 = data["y0"], data["y1"]
        self.color = data["color"].astype(np.float32)
        self.attribution = str(data["attribution"])
        n = len(self.y0)
        self.center = np.array([self.bottom[self.start[k]:self.start[k + 1]].mean(0) for k in range(n)],
                               np.float32).reshape(-1, 2)
        self.radius = np.array([np.hypot(*(np.vstack([self.bottom[self.start[k]:self.start[k + 1]],
                                                       self.top[self.start[k]:self.start[k + 1]]])
                                           - self.center[k]).T).max() for k in range(n)], np.float32)
        # Per wall: outward unit normal (horizontal) and its sun shading.
        nxt = np.arange(len(self.bottom)) + 1
        for k in range(n):
            nxt[self.start[k + 1] - 1] = self.start[k]
        self.next = nxt
        edge = self.bottom[nxt] - self.bottom
        normal = np.stack([edge[:, 1], -edge[:, 0]], -1)
        normal /= np.maximum(np.hypot(normal[:, 0], normal[:, 1]), 1e-6)[:, None]
        self.normal = normal.astype(np.float32)
        self.relight(SUN_DIR)
        # How lit each piece's windows are at night (some offices empty).
        self.windows = (0.45 + 0.55 * ((np.arange(n) * 2654435761) % 1000) / 1000).astype(np.float32)
        self.mid = ((self.bottom + self.bottom[nxt]) / 2).astype(np.float32)
        # Every wall as a world-space quad: bottom i, bottom i+1, top i+1, top i.
        self.piece = np.repeat(np.arange(n), np.diff(self.start))
        y0, y1 = self.y0[self.piece], self.y1[self.piece]
        q = np.empty((len(self.bottom), 4, 3), np.float32)
        for j, (ring, idx, y) in enumerate(((self.bottom, None, y0), (self.bottom, nxt, y0),
                                            (self.top, nxt, y1), (self.top, None, y1))):
            pts = ring if idx is None else ring[idx]
            q[:, j, 0], q[:, j, 1], q[:, j, 2] = pts[:, 0], y, pts[:, 1]
        self.wall_quads = q


    def relight(self, sun_dir):
        """Wall shading for sunlight from `sun_dir` (horizontal, toward the sun)."""
        lit = np.clip(self.normal @ np.asarray(sun_dir, np.float32), 0, 1)
        self.shade = (WALL_SHADE[0] + (WALL_SHADE[1] - WALL_SHADE[0]) * lit).astype(np.float32)


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
        self.land = None  # (N, N) uint8 LAND_CLASSES index; row 0 = south edge, col 0 = west
        path = self._find(data_dir, lat, lon, "landcover", required=False)
        if path:
            lc = np.load(path)
            self.land = lc["land"]
            self.land_m_per_px = float(lc["m_per_px"])
            self.land_half_m = float(lc["half_m"])
            self.attribution += "; " + str(lc["attribution"])
        # Aerial photo near the shop (tools/build_aerial.py). Only its
        # geometry here: the renderer keeps a packed, mipmapped copy cached
        # (see render3d.View._init_aerial) and calls aerial_pixels() only to
        # (re)build that.
        self.aerial_path = None
        path = self._find(data_dir, lat, lon, "aerial", required=False)
        if path and self.land is not None:
            meta = np.load(path)
            self.aerial_path = path
            self.aerial_m_per_px = float(meta["m_per_px"])
            self.aerial_half_m = float(meta["half_m"])
            self.attribution += "; " + str(meta["attribution"])
        self.buildings = None
        path = self._find(data_dir, lat, lon, "buildings", required=False)
        if path:
            self.buildings = Buildings(np.load(path))
            self.attribution += "; " + self.buildings.attribution

    def aerial_pixels(self):
        """-> (N, N, 3) rgb, (N, N) night light; row 0 = south edge, col 0 = west."""
        import pygame  # here, not at the top: the build tools import scenery without a display
        img = pygame.image.load(self.aerial_path[:-len(".npz")] + ".jpg")
        rgb = pygame.surfarray.array3d(img).transpose(1, 0, 2)[::-1]
        return rgb, np.load(self.aerial_path)["emit"]

    @staticmethod
    def _find(data_dir, lat, lon, kind="scenery", required=True):
        for path in sorted(glob.glob(os.path.join(data_dir, f"{kind}_*.npz"))):
            name = os.path.basename(path)[len(kind) + 1:-len(".npz")]
            blat, blon = (float(v) for v in name.split("_"))
            if abs(blat - lat) < 0.0005 and abs(blon - lon) < 0.0005:
                return path
        if not required:
            return None
        raise FileNotFoundError(f"no {kind} for {lat},{lon} in {data_dir} -- run tools/build_{kind}.py")

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
