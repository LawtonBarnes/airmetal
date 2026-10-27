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

With a land-cover raster (scenery.Scenery.land) the ground is textured, SNES
"Mode 7" style: the same per-pixel ray also gives where on the ground it
lands, and the color comes from that spot's land class -- palette = 12
classes x 16 haze steps (even steps of haze_amount, so each is the same
small change) + 64 of sky. That pass runs at 1/ground_scale
resolution and is scaled up (at 2, 256x96: no slower than the flat ground on
a Pi 3B+), so the ground reads as chunky texture while roads and lakes stay
sharp on top. A 4x4 ordered dither hides the steps between haze levels.

That "nearest" texture shimmers in motion -- each sample takes one texel,
and jumps to the next as the camera moves. ground_filter = "smooth" (the
default) instead blends the four nearest texels (bilinear), from a
pre-averaged copy of the raster at 2x/4x/8x the texel size once texels get
smaller than a sample (mipmaps, built at startup from the class colors), in
full color with no palette, and scales the result up smoothly.

Coordinates: x east, y up, z north (meters); heading clockwise from north,
pitch nose-up positive, bank right-wing-down positive.
"""
import math

import numpy as np
import pygame

import daylight
import scenery as sc

NEAR = 5.0  # m -- anything closer is clipped away

SKY_BANDS = [  # (degrees above the horizon, color) -- gradient anchors, matched
    # to the drone shots (2026-10-27): a pale, grayish horizon, blue only higher up
    (0.0, (176, 196, 218)), (3.5, (162, 188, 216)), (6.0, (150, 181, 215)),
    (12.0, (132, 169, 212)), (20.0, (106, 151, 208)), (30.0, (82, 133, 204)),
    (40.0, (66, 120, 200)),
]
HAZE = np.array(SKY_BANDS[0][1], np.float32)
GROUND_NEAR = np.array((88, 132, 62), np.float32)
VISIBILITY_M = 40000.0  # distance where things have faded fully into the haze
GROUND_STEPS, SKY_STEPS = 96, 160  # palette entries; together, all 256
SKY_TOP_DEG = 40.0  # the gradient's last color, and everything above it
MAX_LINE_M = max(d for _, _, d in sc.LINE_STYLES.values())
MIN_LAKE_PX = 2.0  # lakes narrower than this on screen are skipped
LAND_HAZE_STEPS = 16
LAND_SKY_STEPS = 256 - len(sc.LAND_CLASSES) * LAND_HAZE_STEPS
LAND_MIP_LEVELS = 4  # 32, 64, 128, 256 m
TREE_TILE = 256  # per-texel tree randoms repeat every this many texels (8 km)
BAYER4 = np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]], np.float32) / 16 - 0.5


def haze_amount(d):
    return np.clip(np.asarray(d, np.float32) / VISIBILITY_M, 0.0, 1.0) ** 0.75


# Time of day (set by View.set_lighting; daytime = no change): every lit
# color is multiplied by LIGHT, and LIGHTS_ON (0..1) brings up the scene's
# own lights -- land-cover glow, street/runway lights, windows.
LIGHT = np.ones(3, np.float32)
LIGHTS_ON = np.zeros(1, np.float32)
# City light, per unit of LAND_EMIT: sodium orange where it's faint,
# warming toward yellow-white where it's brightest.
NIGHT_GLOW = np.array((1.0, 0.50, 0.10), np.float32)
NIGHT_GLOW_HOT = np.array((1.0, 0.80, 0.40), np.float32)
STAR_COUNT = 260
SKY_AZ_STEPS = 129  # the sky's glow: directions from away from the sun to toward it (fine enough to hide the steps)


def _glow(e):
    """City light rgb (k, 3) for land emission e (k, 1), 0..255."""
    return e * (NIGHT_GLOW + (NIGHT_GLOW_HOT - NIGHT_GLOW) * (e / 255.0))


def hazed(color, d, emit=None):
    """Lit (and, with `emit`, self-lit) color(s) seen d meters away."""
    f = haze_amount(d)[..., None]
    c = np.asarray(color, np.float32) * LIGHT
    if emit is not None:
        c = c + np.asarray(emit, np.float32) * LIGHTS_ON
    return c * (1 - f) + HAZE * f


def _vivid(c, s):
    """Color(s) with saturation scaled by s around their own luma."""
    c = np.asarray(c, np.float32)
    luma = (c * np.array([0.299, 0.587, 0.114], np.float32)).sum(-1, keepdims=True)
    return np.clip(luma + (c - luma) * s, 0, 255)


def set_saturation(s, world=None):
    """Scales the saturation of every scene color -- sky, haze, land cover,
    water, lines, trees and (given the Scenery) buildings -- from the
    drone-matched originals. Call once, before building a View."""
    global SKY_BANDS
    if s == 1.0:
        return
    SKY_BANDS = [(e, tuple(_vivid(c, s))) for e, c in SKY_BANDS]
    HAZE[:] = SKY_BANDS[0][1]
    GROUND_NEAR[:] = _vivid(GROUND_NEAR, s)
    STYLE_BASE[:] = _vivid(STYLE_BASE, s)
    for k, c in sc.LAND_CLASSES.items():
        sc.LAND_CLASSES[k] = tuple(_vivid(c, s))
    sc.LAKE_COLOR = tuple(_vivid(sc.LAKE_COLOR, s))
    for k, c in sc.TREE_COLOR.items():
        sc.TREE_COLOR[k] = tuple(_vivid(c, s))
    sc.TREE_COLOR_DEFAULT = tuple(_vivid(sc.TREE_COLOR_DEFAULT, s))
    if world is not None and world.buildings is not None:
        world.buildings.color = _vivid(world.buildings.color, s)


def _rgb(c):
    return tuple(int(v) for v in c)


def feature_colors(base, slant, horiz, max_dist, ground=GROUND_NEAR, emit=None):
    """Colors for ground features (N, 3 base) seen `slant` meters away along
    the line of sight -- the same haze the ground behind them gets -- and
    blended into that hazed ground (`ground`, its typical color) as `horiz`
    (the horizontal distance) nears the feature's draw distance, so it
    fades out instead of popping."""
    fade = np.clip(horiz / max_dist, 0, 1)[:, None] ** 2
    own, ground = hazed(base, slant, emit), hazed(ground, slant)
    return np.minimum(own * (1 - fade) + ground * fade, 255).astype(int).tolist()


STYLE_BASE = np.array([c for c, _, _ in sc.LINE_STYLES.values()], np.float32)
STYLE_WIDTH = np.array([w for _, w, _ in sc.LINE_STYLES.values()])
STYLE_DIST = np.array([d for _, _, d in sc.LINE_STYLES.values()], np.float32)
STYLE_EMIT = np.array([sc.LINE_EMIT.get(k, (0, 0, 0)) for k in sc.LINE_STYLES], np.float32)


def _sky_palette(steps):
    """`steps` sky colors, linear in sin(elevation) from the horizon to SKY_TOP_DEG."""
    sin_top = math.sin(math.radians(SKY_TOP_DEG))
    elev = np.degrees(np.arcsin(np.arange(steps) / (steps - 1) * sin_top))
    anchors = np.array([e for e, _ in SKY_BANDS])
    colors = np.array([c for _, c in SKY_BANDS], np.float32)
    return [_rgb([np.interp(e, anchors, colors[:, k]) for k in range(3)]) for e in elev]


class View:
    def __init__(self, width, height, fov_deg, land=None, ground_scale=2, ground_filter="smooth", trees=True):
        """`land`: a Scenery with a land-cover raster, for the textured ground
        (None, or a Scenery without one: flat ground)."""
        self.w, self.h = width, height
        self.focal = width / 2 / math.tan(math.radians(fov_deg) / 2)
        self.surface = pygame.Surface((width, height))
        # Every pixel's unit view ray in camera space (x right, y up, z forward).
        self.rays = self._rays(width, height, 1)
        self.backdrop = pygame.Surface((width, height), depth=8)
        self.land = land if land is not None and land.land is not None else None
        self.ground_smooth = ground_filter == "smooth"
        self.lighting = daylight.lighting_at(90.0, 0.0)  # plain day until set_lighting
        self.sun_dir = (0.0, -1.0)
        self.day_haze = HAZE.copy()
        # Stars: fixed directions on the sky (world space), and brightnesses.
        rng = np.random.default_rng(7)
        z = rng.uniform(math.sin(math.radians(4)), 1.0, STAR_COUNT)  # sin(elevation)
        a = rng.uniform(0, 2 * math.pi, STAR_COUNT)
        r = np.sqrt(1 - z * z)
        self.star_dirs = np.stack([r * np.sin(a), z, r * np.cos(a)], -1).astype(np.float32)
        # Few bright, many faint; dimmer low down, in the haze.
        self.star_bright = (130 + 125 * rng.random(STAR_COUNT) ** 2) * np.clip(z / math.sin(math.radians(20)), 0.25, 1)
        self.ground_color = GROUND_NEAR  # what features fade into
        self.tree_chance = None  # per land class, or None: no trees
        if self.land is not None:
            # The land cover's average color, by area.
            counts = np.bincount(self.land.land.ravel(), minlength=len(sc.LAND_CLASSES))
            self.ground_color = (counts[:, None] * np.array(list(sc.LAND_CLASSES.values()), np.float32)).sum(0) / counts.sum()
            if trees:
                self.tree_chance = np.array([sc.TREE_CHANCE.get(c, 0.0) for c in sc.LAND_CLASSES], np.float32)
                self.tree_rand = np.random.default_rng(1).random((5, TREE_TILE, TREE_TILE), np.float32)
                self.tree_color = np.array([sc.TREE_COLOR.get(c, sc.TREE_COLOR_DEFAULT) for c in sc.LAND_CLASSES],
                                           np.float32)
            self._init_land(ground_scale)
            if self.ground_smooth:
                self._init_smooth(ground_scale)
            return
        self._flat_palette()
        self.sky_scale = (SKY_STEPS - 1) / math.sin(math.radians(SKY_TOP_DEG))

    def _sky_1d(self, steps):
        """The sky gradient for the 8-bit palettes: no sun-side glow, the
        directions away from and toward the sun averaged."""
        lut = self.lighting.sky_lut(np.array(_sky_palette(steps), np.float32), SKY_AZ_STEPS, steps,
                                    math.sin(math.radians(SKY_TOP_DEG)))
        return [_rgb(np.clip(c, 0, 255)) for c in lut.mean(0)]

    def _flat_palette(self):
        # Palette: 0..GROUND_STEPS-1 = ground, haze fraction 0 -> 1 (the index is
        # linear in distance / VISIBILITY_M, so each entry carries haze_amount's
        # curve); then the sky, linear in sin(elevation) from the horizon up.
        u = np.arange(GROUND_STEPS) / (GROUND_STEPS - 1)
        pal = [_rgb(hazed(GROUND_NEAR, k * VISIBILITY_M)) for k in u]
        pal += self._sky_1d(SKY_STEPS)
        self.backdrop.set_palette(pal)

    def _rays(self, w, h, scale):
        """Unit view rays (h, w, 3) through the centers of a w x h grid of
        scale x scale pixel blocks."""
        px, py = np.meshgrid((np.arange(w) + 0.5) * scale, (np.arange(h) + 0.5) * scale)
        rays = np.stack([(px - self.w / 2) / self.focal, (self.h / 2 - py) / self.focal, np.ones_like(px)], -1)
        return (rays / np.linalg.norm(rays, axis=-1, keepdims=True)).astype(np.float32)

    def _init_land(self, scale):
        lw, lh = self.w // scale, self.h // scale
        self.land_rays = self._rays(lw, lh, scale)
        self.land_surface = pygame.Surface((lw, lh), depth=8)
        pal = self._land_palette()
        self.sky_scale = (LAND_SKY_STEPS - 1) / math.sin(math.radians(SKY_TOP_DEG))
        self.land_dither = np.tile(BAYER4, (lh // 4 + 1, lw // 4 + 1))[:lh, :lw]
        self.land_sky0 = len(pal) - LAND_SKY_STEPS
        self.land_inv_m = 1.0 / self.land.land_m_per_px
        self.land_n = self.land.land.shape[0]

    def _land_palette(self):
        pal = []
        for name, base in sc.LAND_CLASSES.items():
            f = np.arange(LAND_HAZE_STEPS)[:, None] / (LAND_HAZE_STEPS - 1)
            lit = np.asarray(base, np.float32) * LIGHT + _glow(np.float32(sc.LAND_EMIT.get(name, 0))) * LIGHTS_ON
            pal += [_rgb(np.minimum(c, 255)) for c in lit * (1 - f) + HAZE * f]
        pal += self._sky_1d(LAND_SKY_STEPS)
        self.land_surface.set_palette(pal)
        self.backdrop.set_palette(pal)
        return pal

    def _init_smooth(self, scale):
        lw, lh = self.w // scale, self.h // scale
        self.land_rgb_surface = pygame.Surface((lw, lh), depth=32)
        # Per class: color, then its own night light (scenery.LAND_EMIT).
        self.land_lut = np.array([tuple(c) + (sc.LAND_EMIT.get(k, 0),) for k, c in sc.LAND_CLASSES.items()],
                                 np.float32)
        self.day_sky = np.array(_sky_palette(256), np.float32)
        self.sky_lut = self.lighting.sky_lut(self.day_sky, SKY_AZ_STEPS, 256, math.sin(math.radians(SKY_TOP_DEG)))
        self.sky_scale_rgb = 255 / math.sin(math.radians(SKY_TOP_DEG))
        # Ground meters covered by one sample, per meter of ray length: the
        # sample's angular size (scale pixels).
        self.land_pix_angle = scale / self.focal
        # Mip levels 1..: box-averaged class colors + night light, (size*size, 4) uint8.
        # Level 0 stays the class raster (colors looked up on the fly).
        # Level 1 straight from the classes: each 2x2 block's four colors,
        # looked up through a (class, class) -> summed-color table so it's two
        # gathers, not four. Each later level averages the one before.
        land = self.land.land
        m = land.shape[0] // 2
        lut16 = self.land_lut.astype(np.uint16)
        k = len(lut16)
        pair = (lut16[:, None, :] + lut16[None, :, :]).reshape(k * k, 4)
        a = land[0:2 * m:2, 0:2 * m:2].astype(np.uint16) * k + land[0:2 * m:2, 1:2 * m:2]
        b = land[1:2 * m:2, 0:2 * m:2].astype(np.uint16) * k + land[1:2 * m:2, 1:2 * m:2]
        sums = pair[a]
        sums += pair[b]
        mip = (sums >> 2).astype(np.uint8)
        del a, b, sums
        self.land_mips = [None, (mip.reshape(-1, 4), m)]
        for _ in range(2, LAND_MIP_LEVELS):
            m //= 2
            src = mip[:2 * m, :2 * m].astype(np.uint16)
            mip = ((src[0::2, 0::2] + src[1::2, 0::2] + src[0::2, 1::2] + src[1::2, 1::2]) >> 2).astype(np.uint8)
            self.land_mips.append((mip.reshape(-1, 4), m))

    # -- time of day ------------------------------------------------------

    def set_lighting(self, lighting, world=None):
        """A daylight.Lighting for everything drawn from now on; with the
        Scenery, buildings are also re-shaded for the sun's direction."""
        self.lighting = lighting
        az = math.radians(lighting.sun_az)
        self.sun_dir = (math.sin(az), math.cos(az))
        LIGHT[:] = lighting.light
        LIGHTS_ON[:] = lighting.lights_on
        sin_top = math.sin(math.radians(SKY_TOP_DEG))
        day_sky = getattr(self, "day_sky", None)
        if day_sky is None:
            day_sky = np.array(_sky_palette(256), np.float32)
        lut = lighting.sky_lut(day_sky, SKY_AZ_STEPS, 256, sin_top)
        # Haze: the horizon's color, averaged around (plain day: as before).
        HAZE[:] = self.day_haze if lighting.day >= 1.0 else lut[:, 0].mean(0)
        if self.land is not None:
            self._land_palette()
            if self.ground_smooth:
                self.sky_lut = lut
        else:
            self._flat_palette()
        if world is not None and world.buildings is not None:
            world.buildings.relight(self.sun_dir)

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

    def draw_textured(self):
        world = self.land_rays @ self.rot  # world-space ray directions
        sin_e = world[..., 1]
        ground = sin_e < -1e-4
        # Distance along the ray to the ground, and where it lands.
        t = self.cam_height / np.where(ground, -sin_e, 1.0)
        half, inv, n = self.land.land_half_m, self.land_inv_m, self.land_n
        col = ((self.pos[0] + world[..., 0] * t + half) * inv).astype(np.int32)
        row = ((self.pos[2] + world[..., 2] * t + half) * inv).astype(np.int32)
        np.clip(col, 0, n - 1, out=col)
        np.clip(row, 0, n - 1, out=row)
        cls = self.land.land[row, col]
        haze = haze_amount(t) * (LAND_HAZE_STEPS - 1) + self.land_dither + 0.5
        haze = np.clip(haze, 0, LAND_HAZE_STEPS - 1).astype(np.uint8)
        sky = self.land_sky0 + np.minimum(sin_e * self.sky_scale, LAND_SKY_STEPS - 1).astype(np.uint8)
        idx = np.where(ground, cls * LAND_HAZE_STEPS + haze, sky).astype(np.uint8)
        pygame.surfarray.blit_array(self.land_surface, idx.T)
        pygame.transform.scale(self.land_surface, (self.w, self.h), self.backdrop)
        self.surface.blit(self.backdrop, (0, 0))

    def _land_texels(self, level, row, col):
        """Colors + night light (k, 4) float of texels at integer (row, col) on a mip level."""
        if level == 0:
            return self.land_lut[self.land.land[row, col]]
        mip, m = self.land_mips[level]
        return mip[row * m + col].astype(np.float32)

    def draw_smooth(self):
        rays = self.land_rays
        lh, lw = rays.shape[:2]
        world = (rays @ self.rot).reshape(-1, 3)
        sin_e = world[:, 1]
        sky_i = np.clip((sin_e * self.sky_scale_rgb).astype(np.int32), 0, 255)
        if self.lighting.day >= 1.0:
            out = self.sky_lut[0][sky_i]
        else:  # which way each ray looks, relative to the sun (for the sunset glow)
            sx, sz = self.sun_dir
            toward = world[:, 0] * sx + world[:, 2] * sz
            az_i = np.clip(((toward + 1) * ((SKY_AZ_STEPS - 1) / 2) + 0.5).astype(np.int32), 0, SKY_AZ_STEPS - 1)
            out = self.sky_lut[az_i, sky_i]
        g = np.nonzero(sin_e < -1e-4)[0]
        if len(g):
            w = world[g]
            down = -w[:, 1]
            t = self.cam_height / down
            inv = self.land_inv_m
            half = self.land.land_half_m
            # Ground position in level-0 texel units.
            gx = (self.pos[0] + w[:, 0] * t + half) * inv
            gz = (self.pos[2] + w[:, 2] * t + half) * inv
            # Texels per sample: grows with distance and as the ray grazes
            # the ground (geometric mean of the along- and across-view stretch).
            texels = t * self.land_pix_angle * inv / np.sqrt(down)
            level = np.clip(np.log2(np.maximum(texels, 1.0)), 0, LAND_MIP_LEVELS - 1).astype(np.int32)
            col = np.empty((len(g), 4), np.float32)
            for lv in range(LAND_MIP_LEVELS):
                sel = np.nonzero(level == lv)[0]
                if not len(sel):
                    continue
                size = self.land_n if lv == 0 else self.land_mips[lv][1]
                u = gx[sel] * (1.0 / 2 ** lv) - 0.5
                v = gz[sel] * (1.0 / 2 ** lv) - 0.5
                c0 = np.floor(u)
                r0 = np.floor(v)
                fu = (u - c0)[:, None]
                fv = (v - r0)[:, None]
                c0 = np.clip(c0.astype(np.int32), 0, size - 2)
                r0 = np.clip(r0.astype(np.int32), 0, size - 2)
                top = self._land_texels(lv, r0, c0) * (1 - fu) + self._land_texels(lv, r0, c0 + 1) * fu
                bot = self._land_texels(lv, r0 + 1, c0) * (1 - fu) + self._land_texels(lv, r0 + 1, c0 + 1) * fu
                col[sel] = top * (1 - fv) + bot * fv
            rgb = col[:, :3]
            if self.lighting.day < 1.0:
                rgb = rgb * LIGHT + _glow(col[:, 3:]) * LIGHTS_ON
            f = haze_amount(t)[:, None]
            out[g] = np.minimum(rgb * (1 - f) + HAZE * f, 255)
        pygame.surfarray.blit_array(self.land_rgb_surface,
                                    out.reshape(lh, lw, 3).transpose(1, 0, 2).astype(np.uint8))
        if (lw, lh) == (self.w, self.h):
            self.surface.blit(self.land_rgb_surface, (0, 0))
        else:
            pygame.transform.smoothscale(self.land_rgb_surface, (self.w, self.h), self.surface)

    def draw_sky_and_ground(self):
        if self.land is not None:
            return self.draw_smooth() if self.ground_smooth else self.draw_textured()
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
        # One flat color per lake, hazed for its middle distance, is only
        # right for a lake that's all about the same distance away. Over the
        # textured ground (which has the lakes in it, hazed per pixel), only
        # lakes wholly within LAND_LAKE_DISTANCE_M get a crisp outline drawn.
        far = np.hypot(np.maximum(np.abs(bb[:, 0] - cx), np.abs(bb[:, 2] - cx)),
                       np.maximum(np.abs(bb[:, 1] - cz), np.abs(bb[:, 3] - cz)))
        reach = far if self.land is not None else dists
        max_dist = sc.LAND_LAKE_DISTANCE_M if self.land is not None else sc.LAKE_DISTANCE_M
        lakes = np.nonzero((reach < max_dist) & (px_wide > MIN_LAKE_PX) & ahead)[0]
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
        horiz = (dists[lakes] + far[lakes]) / 2 if self.land is not None else dists[lakes]
        slant = np.hypot(horiz, self.cam_height)
        if self.land is not None:  # no fade: the ground under it is the same hazed water
            colors = hazed(np.array(sc.LAKE_COLOR, np.float32), slant).astype(int).tolist()
        else:
            colors = feature_colors(np.array(sc.LAKE_COLOR, np.float32), slant, horiz, max_dist)
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
        # Line-of-sight distance to each chunk's nearest point (for culling),
        # and its average (for the one haze color the chunk is drawn in).
        d2 = (cam * cam).sum(axis=1)
        horiz = np.sqrt(np.maximum(np.minimum.reduceat(d2, first) - self.cam_height ** 2, 0))
        slant = np.add.reduceat(np.sqrt(d2), first) / lens
        on = ((zmax > NEAR) & (sxmax >= 0) & (sxmin <= self.w) & (symax >= 0) & (symin <= self.h)
              & (horiz < max_dist))
        draw_lines, draw_line = pygame.draw.lines, pygame.draw.line
        surf = self.surface
        visible = np.nonzero(on)[0]
        visible = visible[np.argsort(style[visible], kind="stable")]  # LINE_STYLES order = draw order
        # Plain lists for the per-chunk loop: indexing numpy one scalar at a
        # time costs more than the drawing on the Pi.
        widths = STYLE_WIDTH.tolist()
        mid = np.sqrt(np.maximum(slant[visible] ** 2 - self.cam_height ** 2, 0))
        colors = feature_colors(STYLE_BASE[style[visible]], slant[visible], mid, max_dist[visible],
                                self.ground_color, STYLE_EMIT[style[visible]])
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

    def draw_buildings(self, world):
        """The skyline, painter's algorithm: pieces far to near, each piece's
        camera-facing walls far to near, then its roof if seen from above.
        Culling, transform, shading and the draw order are all done in one
        numpy pass over every wall; only the pygame calls loop."""
        b = world.buildings
        if b is None:
            return 0
        cx, cy, cz = (float(v) for v in self.pos)
        rel = b.center - (cx, cz)
        dist = np.hypot(rel[:, 0], rel[:, 1])
        fx, fz = math.sin(math.radians(self.heading)), math.cos(math.radians(self.heading))
        ahead = rel[:, 0] * fx + rel[:, 1] * fz > -b.radius - 50.0
        tall_px = self.focal * b.y1 / np.maximum(dist, 1.0)
        keep = (dist - b.radius < sc.BUILDING_DISTANCE_M) & ahead & (tall_px > 1.0)
        if not keep.any():
            return 0
        slant = np.hypot(dist, cy - (b.y0 + b.y1) / 2)
        haze = haze_amount(slant)

        # Walls: in a kept piece and facing the camera.
        to_cam = np.stack([cx - b.mid[:, 0], cz - b.mid[:, 1]], -1)
        walls = np.nonzero(keep[b.piece] & ((b.normal * to_cam).sum(1) > 0))[0]
        pc = b.piece[walls]
        cam = self.to_camera(b.wall_quads[walls])  # (k, 4, 3)
        zmin = cam[..., 2].min(1)
        scr = self.to_screen(cam)
        sx, sy = scr[..., 0], scr[..., 1]
        on = ((cam[..., 2].max(1) > NEAR) & (sx.max(1) >= 0) & (sx.min(1) <= self.w)
              & (sy.max(1) >= 0) & (sy.min(1) <= self.h))
        # Thinner than a pixel on screen (far, edge-on): not worth a polygon.
        on &= (zmin < NEAR) | (sx.max(1) - sx.min(1) > 0.8)
        f = haze[pc][:, None]
        colors = (b.color[pc] * b.shade[walls][:, None] * LIGHT
                  + np.array(sc.WINDOW_GLOW, np.float32) * (b.windows[pc] * LIGHTS_ON)[:, None]) * (1 - f) + HAZE * f
        wall_d = np.hypot(to_cam[walls, 0], to_cam[walls, 1])

        # Roofs of kept pieces seen from above, drawn after their own walls:
        # every candidate's ring in one transform, like draw_lakes.
        roofs = np.nonzero(keep & (b.y1 < cy))[0]
        starts, ends = b.start[roofs], b.start[roofs + 1]
        lens = ends - starts
        idx = np.repeat(starts - np.concatenate([[0], np.cumsum(lens)[:-1]]), lens) + np.arange(lens.sum())
        rp = np.empty((len(idx), 3), np.float32)
        rp[:, 0], rp[:, 1], rp[:, 2] = b.top[idx, 0], np.repeat(b.y1[roofs], lens), b.top[idx, 1]
        rcam = self.to_camera(rp)
        rscr = self.to_screen(rcam)
        bounds = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
        if len(roofs):
            first = bounds[:-1]
            r_near = np.minimum.reduceat(rcam[:, 2], first) < NEAR
            r_on = ((np.maximum.reduceat(rcam[:, 2], first) > NEAR)
                    & (np.maximum.reduceat(rscr[:, 0], first) >= 0) & (np.minimum.reduceat(rscr[:, 0], first) <= self.w)
                    & (np.maximum.reduceat(rscr[:, 1], first) >= 0) & (np.minimum.reduceat(rscr[:, 1], first) <= self.h))
            keep_r = np.nonzero(r_on)[0]
        else:
            r_near = np.zeros(0, bool)
            keep_r = np.zeros(0, np.int64)
        f = haze[roofs][:, None]
        roof_colors = np.minimum(b.color[roofs] * 1.06, 255) * LIGHT * (1 - f) + HAZE * f

        # One draw order: piece far -> near, then within a piece walls far ->
        # near, roof last.
        piece_key = np.concatenate([dist[pc[on]], dist[roofs[keep_r]]])
        within = np.concatenate([wall_d[on], np.full(len(keep_r), -1.0)])
        order = np.lexsort((-within, -piece_key))
        n_on = int(on.sum())
        wall_idx = np.nonzero(on)[0]

        draw = pygame.draw.polygon
        surf = self.surface
        w_pts = scr[on].tolist()
        w_near = (zmin[on] < NEAR).tolist()
        w_colors = colors[on].astype(int).tolist()
        r_colors = roof_colors.astype(int).tolist()
        r_pts = rscr.tolist()
        r_bounds = bounds.tolist()
        r_near_l = r_near.tolist()
        keep_r_l = keep_r.tolist()
        drawn = 0
        for o in order.tolist():
            if o < n_on:
                if w_near[o]:
                    poly = _clip_near(cam[wall_idx[o]])
                    if len(poly) < 3:
                        continue
                    pts = _clip_rect(self.to_screen(poly), -2, -2, self.w + 2, self.h + 2)
                    if len(pts) < 3:
                        continue
                    pts = pts.tolist()
                else:
                    pts = w_pts[o]
                draw(surf, w_colors[o], pts)
            else:
                r = keep_r_l[o - n_on]
                a, e = r_bounds[r], r_bounds[r + 1]
                if r_near_l[r]:  # the camera is over this roof's edge
                    poly = _clip_near(rcam[a:e])
                    if len(poly) < 3:
                        continue
                    pts = _clip_rect(self.to_screen(poly), -2, -2, self.w + 2, self.h + 2)
                    if len(pts) < 3:
                        continue
                    pts = pts.tolist()
                else:
                    pts = r_pts[a:e]
                draw(surf, r_colors[r], pts)
            drawn += 1
        return drawn

    def draw_trees(self):
        """Pyramid trees near the camera, on low rides. Positions come from
        the land-cover raster around the camera, hashed per texel; trees
        shrink to nothing toward TREE_RANGE_M so they grow in, not pop in."""
        if self.tree_chance is None or self.cam_height > sc.TREE_MAX_AGL_M:
            return 0
        cx, cz = float(self.pos[0]), float(self.pos[2])
        half, inv, n = self.land.land_half_m, self.land_inv_m, self.land_n
        rng = min(max(self.cam_height * sc.TREE_RANGE_PER_AGL, sc.TREE_RANGE_M[0]), sc.TREE_RANGE_M[1])
        # The texels under the view's wedge: the box around the camera and
        # the far arc, a little wider than the field of view.
        hd = math.radians(self.heading)
        spread = math.atan(self.w / 2 / self.focal) + math.radians(10)
        xs = [cx] + [cx + rng * math.sin(hd + a) for a in (-spread, 0.0, spread)]
        zs = [cz] + [cz + rng * math.cos(hd + a) for a in (-spread, 0.0, spread)]
        c0, c1 = (max(int((min(xs) + half) * inv), 0), min(int((max(xs) + half) * inv) + 1, n))
        r0, r1 = (max(int((min(zs) + half) * inv), 0), min(int((max(zs) + half) * inv) + 1, n))
        if r1 <= r0 or c1 <= c0:
            return 0
        # Per-texel randoms from a tiled table (repeats every TREE_TILE texels).
        # The pick channel for the whole window; the rest only where it hits.
        tr = np.arange(r0, r1) % TREE_TILE
        tc = np.arange(c0, c1) % TREE_TILE
        cls = self.land.land[r0:r1, c0:c1]
        # Distance thinning on the texel grid (texel centers), before any
        # per-tree work: most of the window is far, and mostly thinned away.
        dz2 = ((np.arange(r0, r1, dtype=np.float32) + 0.5) / inv - half - cz) ** 2
        dx2 = ((np.arange(c0, c1, dtype=np.float32) + 0.5) / inv - half - cx) ** 2
        d2 = np.maximum(dz2[:, None] + dx2[None, :], 1.0)
        odds = self.tree_chance[cls] * np.minimum(1.0, (sc.TREE_THIN * rng) ** 2 / d2)
        odds[d2 > rng * rng] = 0.0
        hit = np.nonzero((self.tree_rand[0][tr[:, None], tc[None, :]] < odds).ravel())[0]
        if not len(hit):
            return 0
        ncol = c1 - c0
        hr, hc = hit // ncol, hit % ncol
        rr, cc = hr + r0, hc + c0
        cls, odds = cls.ravel()[hit], odds.ravel()[hit]
        rnd = self.tree_rand[:, tr[hr], tc[hc]]  # (5, hits): pick, x, z, size, turn
        x = (cc + 0.15 + 0.7 * rnd[1]) / inv - half
        z = (rr + 0.15 + 0.7 * rnd[2]) / inv - half
        dist = np.hypot(x - cx, z - cz)
        keep = dist < rng
        # In front of the camera and within the view's width (plus a margin).
        base = self.ground_to_camera(np.stack([x, z], -1))
        keep &= (base[:, 2] > 3 * NEAR) & (np.abs(base[:, 0]) < base[:, 2] * self.w / 2 / self.focal + 30)
        idx = np.nonzero(keep)[0]
        if len(idx) > sc.TREE_CAP:  # thin evenly: the ones that only just made it go first
            idx = idx[np.argpartition(rnd[0, idx] / odds[idx], sc.TREE_CAP)[:sc.TREE_CAP]]
        if not len(idx):
            return 0
        grow = np.clip((rng - dist[idx]) / (0.4 * rng), 0, 1)
        lo, hi = sc.TREE_HEIGHT_M
        height = (lo + (hi - lo) * rnd[3, idx]) * grow
        tall_px = self.focal * height / np.maximum(base[idx, 2], 1.0)
        sel = tall_px > 1.0
        idx, height = idx[sel], height[sel]
        if not len(idx):
            return 0
        order = np.argsort(-dist[idx])  # far to near
        idx, height = idx[order], height[order]
        k = len(idx)
        # Four base corners, turned per tree, plus the apex: (k, 5) points.
        ang = rnd[4, idx, None] * (math.pi / 2) + np.arange(4) * (math.pi / 2)  # (k, 4)
        rad = (height * sc.TREE_WIDTH / 2)[:, None]
        pts = np.empty((k, 5, 3), np.float32)
        pts[:, :4, 0] = x[idx, None] + rad * np.cos(ang)
        pts[:, :4, 1] = 0.0
        pts[:, :4, 2] = z[idx, None] + rad * np.sin(ang)
        pts[:, 4, 0], pts[:, 4, 1], pts[:, 4, 2] = x[idx], height, z[idx]
        scr = self.to_screen(self.to_camera(pts))  # (k, 5, 2)
        # Face j: corners j, j+1 and the apex. Its outward horizontal
        # direction is halfway between the corners; normal ~ (height *
        # outward, apothem up), apothem = rad / sqrt 2.
        mid_ang = ang + math.pi / 4
        nx, nz = np.cos(mid_ang), np.sin(mid_ang)
        slope = rad * math.sqrt(0.5) / height[:, None]
        to_x, to_z = self.pos[0] - pts[:, :4, 0], self.pos[2] - pts[:, :4, 2]
        facing = nx * to_x + nz * to_z + slope * self.pos[1] > 0  # (k, 4); corners sit at y = 0
        lit = np.clip(nx * sc.SUN_DIR[0] + nz * sc.SUN_DIR[1], 0, 1)
        f = haze_amount(np.hypot(dist[idx], self.cam_height))[:, None, None]
        colors = self.tree_color[cls[idx]][:, None, :] * (0.62 + 0.5 * lit)[..., None] * LIGHT * (1 - f) + HAZE * f
        # Only the faces that show, already far to near (a tree's own
        # visible faces never overlap, so their order doesn't matter).
        tri = np.empty((k, 4, 3, 2), np.float32)
        tri[:, :, 0] = scr[:, :4]
        tri[:, :, 1] = np.roll(scr[:, :4], -1, axis=1)
        tri[:, :, 2] = scr[:, 4, None]
        draw = pygame.draw.polygon
        surf = self.surface
        for c, p in zip(np.minimum(colors[facing], 255).astype(int).tolist(), tri[facing].tolist()):
            draw(surf, c, p)
        return int(facing.sum())

    def draw_stars(self):
        k = self.lighting.stars
        if k <= 0:
            return
        cam = self.star_dirs @ self.rot.T
        on = cam[:, 2] > 0.05
        x = (self.w / 2 + self.focal * cam[on, 0] / cam[on, 2]).astype(np.int32)
        y = (self.h / 2 - self.focal * cam[on, 1] / cam[on, 2]).astype(np.int32)
        b = (self.star_bright[on] * k).astype(np.int32)
        inside = (x >= 0) & (x < self.w) & (y >= 0) & (y < self.h)
        surf = self.surface
        for px, py, v in zip(x[inside].tolist(), y[inside].tolist(), b[inside].tolist()):
            surf.set_at((px, py), (v, v, min(v + 12, 255)))

    def render(self, world):
        self.draw_sky_and_ground()
        self.draw_stars()
        self.draw_lakes(world)
        n = self.draw_lines(world.lines)
        self.draw_buildings(world)
        self.draw_trees()
        return n


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
