"""Prices out sky/edge-quality options on the Pi: smooth vs dithered sky
gradients, smoothscale vs nearest upscaling, anti-aliased vs plain lines.

Usage: python3 tools/bench_options.py
"""
import os
import sys
import time

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pygame  # noqa: E402

import config  # noqa: E402
import render3d  # noqa: E402
import scenery  # noqa: E402


def timeit(label, fn, n=10):
    fn()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    print(f"{label:44s} {1000 * (time.perf_counter() - t) / n:6.1f} ms", flush=True)


pygame.init()
s = config.load()
world = scenery.Scenery(config.DATA_DIR, s["lat"], s["lon"])
v = render3d.View(config.VIEW_LOW_W, config.VIEW_LOW_H, s["fov_deg"])
v.set_camera(0, 600, 0, 252, -4, 12)
W, H = config.VIEW_LOW_W, config.VIEW_LOW_H
surf24 = pygame.Surface((W, H))
big = pygame.Surface((1024, 384))

timeit("current: banded sky+ground (palette)", v.draw_sky_and_ground)

# Smooth gradient: per-pixel color from elevation, interpolated in float.
sky_e = np.radians([e for e, _ in render3d.SKY_BANDS]).astype(np.float32)
sky_c = np.array([c for _, c in render3d.SKY_BANDS], np.float32)


def smooth():
    sin_e = v.rays @ v.rot[:, 1]
    e = np.arcsin(np.clip(sin_e, -1, 1))
    rgb = np.empty(e.shape + (3,), np.float32)
    for k in range(3):
        rgb[..., k] = np.interp(e, sky_e, sky_c[:, k])
    pygame.surfarray.blit_array(surf24, rgb.astype(np.uint8).transpose(1, 0, 2))


timeit("smooth sky gradient (float, 24-bit)", smooth)

# Dithered: fractional band index + 4x4 Bayer threshold -> palette index.
bayer = (np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]], np.float32) + 0.5) / 16
thresh = np.tile(bayer, (H // 4 + 1, W // 4 + 1))[:H, :W]
band_sin = np.sin(sky_e)


def dithered():
    sin_e = v.rays @ v.rot[:, 1]
    pos = np.interp(sin_e, band_sin, np.arange(len(band_sin), dtype=np.float32))
    idx = (np.floor(pos) + (pos % 1 > thresh)).astype(np.uint8)
    pygame.surfarray.blit_array(v.backdrop, idx.T)
    surf24.blit(v.backdrop, (0, 0))


timeit("dithered sky bands (Bayer 4x4, palette)", dithered)

v.render(world)
timeit("upscale 512x192->1024x384 nearest (current)", lambda: pygame.transform.scale(v.surface, (1024, 384), big))
timeit("upscale smoothscale (bilinear)", lambda: pygame.transform.smoothscale(v.surface, (1024, 384), big))

pts = [(np.random.rand(12, 2) * (W, H)).tolist() for _ in range(150)]
timeit("150 polylines draw.lines (current)", lambda: [pygame.draw.lines(surf24, (150, 150, 140), False, p, 1) for p in pts])
timeit("150 polylines draw.aalines", lambda: [pygame.draw.aalines(surf24, (150, 150, 140), False, p) for p in pts])
