"""Software flat-shaded 3D for the windshield: 512x192, pixel-doubled to the
top two tubes. No GPU -- numpy does the projection, pygame.draw the
polygons and lines, painter's-algorithm style.

Sky and flat ground are one per-pixel lookup: each pixel's view ray has a
fixed elevation angle, and on flat ground a ray that far below the horizon
lands a known distance away (camera height / tan(angle)) -- so the sky
gradient and the distance haze on the ground are both just that angle,
mapped into a 256-color palette (96 haze steps of ground, 160 of sky:
smooth on a CRT, and no more costly than a few hard bands). Ground features (lakes,
roads) are then hazed by their own distance to match.

Coordinates: x east, y up, z north (meters); heading clockwise from north,
pitch nose-up positive, bank right-wing-down positive.
"""
import math

import numpy as np
import pygame

import scenery as sc

NEAR = 5.0  # m -- anything closer is clipped away

SKY_BANDS = [  # (degrees above the horizon, color) -- gradient anchors
    (0.0, (176, 204, 232)), (1.5, (160, 194, 230)), (3.5, (140, 182, 228)),
    (6.0, (120, 168, 226)), (9.0, (100, 154, 224)), (13.0, (82, 140, 222)),
    (18.0, (66, 126, 218)), (25.0, (52, 112, 212)), (35.0, (42, 100, 206)),
]
HAZE = np.array(SKY_BANDS[0][1], np.float32)
GROUND_NEAR = np.array((88, 132, 62), np.float32)
VISIBILITY_M = 40000.0  # distance where things have faded fully into the haze
GROUND_STEPS, SKY_STEPS = 96, 160  # palette entries; together, all 256
SKY_TOP_DEG = 40.0  # the gradient's last color, and everything above it
MAX_LINE_M = max(d for _, _, d in sc.LINE_STYLES.values())
MIN_LAKE_PX = 2.0  # lakes narrower than this on screen are skipped


def haze_amount(d):
    return np.clip(np.asarray(d, np.float32) / VISIBILITY_M, 0.0, 1.0) ** 0.75


def hazed(color, d):
    f = haze_amount(d)[..., None]
    return np.asarray(color, np.float32) * (1 - f) + HAZE * f


def _rgb(c):
    return tuple(int(v) for v in c)


def feature_colors(base, slant, horiz, max_dist):
    """Colors for ground features (N, 3 base) seen `slant` meters away along
    the line of sight -- the same haze the ground behind them gets -- and
    blended into that hazed ground as `horiz` (the horizontal distance)
    nears the feature's draw distance, so it fades out instead of popping."""
    fade = np.clip(horiz / max_dist, 0, 1)[:, None] ** 2
    own, ground = hazed(base, slant), hazed(GROUND_NEAR, slant)
    return (own * (1 - fade) + ground * fade).astype(int).tolist()


STYLE_BASE = np.array([c for c, _, _ in sc.LINE_STYLES.values()], np.float32)
STYLE_WIDTH = np.array([w for _, w, _ in sc.LINE_STYLES.values()])
STYLE_DIST = np.array([d for _, _, d in sc.LINE_STYLES.values()], np.float32)


class View:
    def __init__(self, width, height, fov_deg):
        self.w, self.h = width, height
        self.focal = width / 2 / math.tan(math.radians(fov_deg) / 2)
        self.surface = pygame.Surface((width, height))
        # Every pixel's unit view ray in camera space (x right, y up, z forward).
        px, py = np.meshgrid(np.arange(width) + 0.5, np.arange(height) + 0.5)
        rays = np.stack([(px - width / 2) / self.focal, (height / 2 - py) / self.focal, np.ones_like(px)], -1)
        self.rays = (rays / np.linalg.norm(rays, axis=-1, keepdims=True)).astype(np.float32)
        self.backdrop = pygame.Surface((width, height), depth=8)
        # Palette: 0..GROUND_STEPS-1 = ground, haze fraction 0 -> 1 (the index is
        # linear in distance / VISIBILITY_M, so each entry carries haze_amount's
        # curve); then the sky, linear in sin(elevation) from the horizon up.
        u = np.arange(GROUND_STEPS) / (GROUND_STEPS - 1)
        pal = [_rgb(hazed(GROUND_NEAR, k * VISIBILITY_M)) for k in u]
        sin_top = math.sin(math.radians(SKY_TOP_DEG))
        elev = np.degrees(np.arcsin(np.arange(SKY_STEPS) / (SKY_STEPS - 1) * sin_top))
        anchors = np.array([e for e, _ in SKY_BANDS])
        colors = np.array([c for _, c in SKY_BANDS], np.float32)
        pal += [_rgb([np.interp(e, anchors, colors[:, k]) for k in range(3)]) for e in elev]
        self.backdrop.set_palette(pal)
        self.sky_scale = (SKY_STEPS - 1) / sin_top

    # -- camera ------------------------------------------------------------

    def set_camera(self, x, y, z, heading, pitch, bank):
        h, p, b = (math.radians(a) for a in (heading, pitch, bank))
        f = np.array([math.sin(h) * math.cos(p), math.sin(p), math.cos(h) * math.cos(p)])
        r0 = np.array([math.cos(h), 0.0, -math.sin(h)])
        u0 = np.array([-math.sin(h) * math.sin(p), math.cos(p), -math.cos(h) * math.sin(p)])
        r = r0 * math.cos(b) - u0 * math.sin(b)
        u = u0 * math.cos(b) + r0 * math.sin(b)
        self.rot = np.stack([r, u, f]).astype(np.float32)  # rows: camera axes in world space
        self.pos = np.array([x, y, z], np.float32)
        self.heading, self.cam_height = heading, max(y, 1.0)

    def to_camera(self, pts):
        """(..., 3) world points -> (..., 3) camera space (x right, y up, z forward)."""
        return (pts - self.pos) @ self.rot.T

    def ground_to_camera(self, xz):
        """(..., 2) ground points (x, z) at height 0 -> camera space."""
        rel_x, rel_z = xz[..., 0] - self.pos[0], xz[..., 1] - self.pos[2]
        rel_y = -self.pos[1]
        r = self.rot
        return np.stack([r[0, 0] * rel_x + r[0, 1] * rel_y + r[0, 2] * rel_z,
                         r[1, 0] * rel_x + r[1, 1] * rel_y + r[1, 2] * rel_z,
                         r[2, 0] * rel_x + r[2, 1] * rel_y + r[2, 2] * rel_z], -1)

    def to_screen(self, cam):
        z = np.maximum(cam[..., 2], NEAR)
        return np.stack([self.w / 2 + self.focal * cam[..., 0] / z,
                         self.h / 2 - self.focal * cam[..., 1] / z], -1)

    # -- sky and ground ----------------------------------------------------

    def draw_sky_and_ground(self):
        # sin(elevation) of every pixel's ray = ray . world-up (in camera space).
        sin_e = self.rays @ self.rot[:, 1]
        # Ground: a ray at elevation -e meets flat ground ~h / sin(e) away.
        dist_frac = self.cam_height / (VISIBILITY_M * np.maximum(-sin_e, 1e-6))
        ground = np.minimum(dist_frac, 1.0) * (GROUND_STEPS - 1)
        sky = GROUND_STEPS + np.minimum(sin_e * self.sky_scale, SKY_STEPS - 1)
        idx = np.where(sin_e < 0, ground, sky).astype(np.uint8)
        pygame.surfarray.blit_array(self.backdrop, idx.T)
        self.surface.blit(self.backdrop, (0, 0))

    # -- ground features ---------------------------------------------------

    def draw_lakes(self, world):
        cx, cz = self.pos[0], self.pos[2]
        bb = world.lake_bboxes
        # Distance from the camera to each lake's bounding box (0 if inside it).
        dx = np.maximum(np.maximum(bb[:, 0] - cx, 0), cx - bb[:, 2])
        dz = np.maximum(np.maximum(bb[:, 1] - cz, 0), cz - bb[:, 3])
        dists = np.hypot(dx, dz)
        size = np.maximum(bb[:, 2] - bb[:, 0], bb[:, 3] - bb[:, 1])
        px_wide = self.focal * size / np.maximum(dists, 1.0)
        # Skip lakes wholly behind the camera: bbox center's forward distance plus its size.
        fx, fz = math.sin(math.radians(self.heading)), math.cos(math.radians(self.heading))
        ahead = ((bb[:, 0] + bb[:, 2]) / 2 - cx) * fx + ((bb[:, 1] + bb[:, 3]) / 2 - cz) * fz > -size
        lakes = np.nonzero((dists < sc.LAKE_DISTANCE_M) & (px_wide > MIN_LAKE_PX) & ahead)[0]
        if not len(lakes):
            return
        # Every candidate lake's outline in one transform, like draw_lines.
        starts, ends = world.lake_start[lakes], world.lake_start[lakes + 1]
        lens = ends - starts
        idx = np.repeat(starts - np.concatenate([[0], np.cumsum(lens)[:-1]]), lens) + np.arange(lens.sum())
        cam = self.ground_to_camera(world.lake_pts[idx])
        scr = self.to_screen(cam)
        bounds = np.concatenate([[0], np.cumsum(lens)])
        first = bounds[:-1]
        zmin = np.minimum.reduceat(cam[:, 2], first)
        zmax = np.maximum.reduceat(cam[:, 2], first)
        lo_x, hi_x = np.minimum.reduceat(scr[:, 0], first), np.maximum.reduceat(scr[:, 0], first)
        lo_y, hi_y = np.minimum.reduceat(scr[:, 1], first), np.maximum.reduceat(scr[:, 1], first)
        on = (zmax > NEAR) & (hi_x >= 0) & (lo_x <= self.w) & (hi_y >= 0) & (lo_y <= self.h)
        simple = on & (zmin >= NEAR) & (lo_x > -self.w) & (hi_x < 2 * self.w) & (lo_y > -self.h) & (hi_y < 2 * self.h)
        horiz = dists[lakes]
        slant = np.hypot(horiz, self.cam_height)
        colors = feature_colors(np.array(sc.LAKE_COLOR, np.float32), slant, horiz, sc.LAKE_DISTANCE_M)
        draw = pygame.draw.polygon
        for k in np.nonzero(on)[0]:
            a, b = bounds[k], bounds[k + 1]
            if simple[k]:
                pts = scr[a:b]
            else:  # under the camera, or reaching far off-screen: clip properly
                poly = _clip_near(cam[a:b])
                if len(poly) < 3:
                    continue
                pts = _clip_rect(self.to_screen(poly), -2, -2, self.w + 2, self.h + 2)
            if len(pts) >= 3:
                draw(self.surface, colors[k], pts.tolist())

    def draw_lines(self, lines):
        chunks = lines.chunks_in(sc.Scenery.cells_ahead(self.pos[0], self.pos[2], self.heading, MAX_LINE_M))
        # Each style has its own draw distance -- cull on chunk centers first.
        rel = lines.chunk_center[chunks] - (self.pos[0], self.pos[2])
        fx, fz = math.sin(math.radians(self.heading)), math.cos(math.radians(self.heading))
        r = lines.chunk_radius[chunks]
        chunks = chunks[(np.hypot(rel[:, 0], rel[:, 1]) - r < STYLE_DIST[lines.chunk_style[chunks]])
                        & (rel[:, 0] * fx + rel[:, 1] * fz > -r)]
        if not len(chunks):
            return 0
        starts, ends = lines.chunk_start[chunks], lines.chunk_start[chunks + 1]
        style = lines.chunk_style[chunks]
        # All chunk points in one go: gather, transform, project.
        lens = ends - starts
        idx = np.repeat(starts - np.concatenate([[0], np.cumsum(lens)[:-1]]), lens) + np.arange(lens.sum())
        cam = self.ground_to_camera(lines.pts[idx])
        scr = self.to_screen(cam)
        bounds = np.concatenate([[0], np.cumsum(lens)])
        first = bounds[:-1]
        zmin = np.minimum.reduceat(cam[:, 2], first)
        zmax = np.maximum.reduceat(cam[:, 2], first)
        sxmin = np.minimum.reduceat(scr[:, 0], first)
        sxmax = np.maximum.reduceat(scr[:, 0], first)
        symin = np.minimum.reduceat(scr[:, 1], first)
        symax = np.maximum.reduceat(scr[:, 1], first)
        max_dist = STYLE_DIST[style]
        # Line-of-sight distance to each chunk's nearest point, and the horizontal part of it.
        slant = np.sqrt(np.minimum.reduceat((cam * cam).sum(axis=1), first))
        horiz = np.sqrt(np.maximum(slant * slant - self.cam_height ** 2, 0))
        on = ((zmax > NEAR) & (sxmax >= 0) & (sxmin <= self.w) & (symax >= 0) & (symin <= self.h)
              & (horiz < max_dist))
        draw_lines, draw_line = pygame.draw.lines, pygame.draw.line
        surf = self.surface
        visible = np.nonzero(on)[0]
        visible = visible[np.argsort(style[visible], kind="stable")]  # LINE_STYLES order = draw order
        # Plain lists for the per-chunk loop: indexing numpy one scalar at a
        # time costs more than the drawing on the Pi.
        widths = STYLE_WIDTH.tolist()
        colors = feature_colors(STYLE_BASE[style[visible]], slant[visible], horiz[visible], max_dist[visible])
        for k, a, b, st, c, near_ok in zip(visible.tolist(), bounds[visible].tolist(),
                                           bounds[visible + 1].tolist(), style[visible].tolist(),
                                           colors, (zmin[visible] >= NEAR).tolist()):
            c = tuple(c)
            width = widths[st]
            if near_ok:
                draw_lines(surf, c, False, scr[a:b].tolist(), width)
            else:  # passes under the camera: clip segment by segment
                for p, q in zip(cam[a:b - 1], cam[a + 1:b]):
                    if p[2] < NEAR and q[2] < NEAR:
                        continue
                    if p[2] < NEAR:
                        p = p + (q - p) * ((NEAR - p[2]) / (q[2] - p[2]))
                    elif q[2] < NEAR:
                        q = q + (p - q) * ((NEAR - q[2]) / (p[2] - q[2]))
                    sp, sq = self.to_screen(np.stack([p, q]))
                    draw_line(surf, c, sp.tolist(), sq.tolist(), width)
        return len(visible)

    def render(self, world):
        self.draw_sky_and_ground()
        self.draw_lakes(world)
        return self.draw_lines(world.lines)


def _clip_rect(poly, x0, y0, x1, y1):
    """Sutherland-Hodgman clip of a screen polygon (N, 2) to a rectangle --
    pygame 2.1 on the Pi takes ~0.3s to fill a polygon that reaches far
    off-screen."""
    for axis, bound, keep_ge in ((0, x0, True), (0, x1, False), (1, y0, True), (1, y1, False)):
        if not len(poly):
            return poly
        v = poly[:, axis] - bound
        inside = v >= 0 if keep_ge else v <= 0
        if inside.all():
            continue
        nxt, vn, inside_n = np.roll(poly, -1, axis=0), np.roll(v, -1), np.roll(inside, -1)
        out = []
        for k in range(len(poly)):
            if inside[k]:
                out.append(poly[k])
            if inside[k] != inside_n[k]:
                out.append(poly[k] + (nxt[k] - poly[k]) * (v[k] / (v[k] - vn[k])))
        poly = np.array(out, np.float32) if out else np.zeros((0, 2), np.float32)
    return poly


def _clip_near(poly):
    """Sutherland-Hodgman clip of a camera-space polygon (N, 3) to z >= NEAR."""
    if poly[:, 2].min() >= NEAR:
        return poly
    out = []
    n = len(poly)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        ina, inb = a[2] >= NEAR, b[2] >= NEAR
        if ina:
            out.append(a)
        if ina != inb:
            out.append(a + (b - a) * ((NEAR - a[2]) / (b[2] - a[2])))
    return np.array(out, np.float32) if out else np.zeros((0, 3), np.float32)
