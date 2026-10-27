"""Phase 3 spike: what does a per-pixel textured ("Mode 7") ground cost?

Each ground pixel's ray hits flat ground at a known (x, z); look its color up
in a palette-quantized land-cover raster instead of flat green. Times, on the
same 512x192 view:

  A  today's sky + flat hazed ground (render3d.draw_sky_and_ground)
  B  textured ground: 16 m/px raster near, 64 m/px far, 8-bit palette of
     (land class x haze step) + sky
  C  B at 2x2 supersampling, averaged down (less far-field shimmer)

The raster is synthetic smooth noise (the cost doesn't depend on content).
Saves a PNG of B for a look. Usage: python tools/bench_mode7.py [OUT.png]
"""
import os
import sys
import time

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pygame  # noqa: E402

import render3d  # noqa: E402

W, H = 512, 192
CLASSES, HAZE_STEPS = 12, 16  # 192 palette entries of ground, 64 of sky
NEAR_M, NEAR_N = 16.0, 4096  # 65 km square around home
FAR_M, FAR_N = 64.0, 4096  # 262 km square
NEAR_HALF = NEAR_M * NEAR_N / 2

pygame.init()
rng = np.random.default_rng(1)


def noise_classes(n, cells):
    """Blocky-smooth class map: random classes on a coarse grid, upsampled."""
    small = rng.integers(0, CLASSES, (cells, cells), dtype=np.uint8)
    return np.repeat(np.repeat(small, n // cells, 0), n // cells, 1)


near_tex = noise_classes(NEAR_N, 256)  # 256 m blocks
far_tex = noise_classes(FAR_N, 512)
print(f"rasters: near {near_tex.nbytes / 1e6:.0f} MB, far {far_tex.nbytes / 1e6:.0f} MB")

view = render3d.View(W, H, 70)
base = rng.integers(40, 200, (CLASSES, 3)).astype(np.float32)
pal = []
for c in range(CLASSES):
    for k in range(HAZE_STEPS):
        pal.append(render3d._rgb(render3d.hazed(base[c], k / (HAZE_STEPS - 1) * render3d.VISIBILITY_M)))
sky_n = 256 - len(pal)
for k in range(sky_n):
    e = np.degrees(np.arcsin(k / (sky_n - 1) * np.sin(np.radians(40))))
    anchors = np.array([a for a, _ in render3d.SKY_BANDS])
    cols = np.array([c for _, c in render3d.SKY_BANDS], np.float32)
    pal.append(render3d._rgb([np.interp(e, anchors, cols[:, j]) for j in range(3)]))
surf8 = pygame.Surface((W, H), depth=8)
surf8.set_palette(pal)
surf8_ss = pygame.Surface((W * 2, H * 2), depth=8)
surf8_ss.set_palette(pal)
out = pygame.Surface((W, H))
pal_rgb = np.array(pal, np.uint16)


def rays_for(scale):
    if scale == 1:
        return view.rays
    px, py = np.meshgrid((np.arange(W * scale) + 0.5) / scale, (np.arange(H * scale) + 0.5) / scale)
    r = np.stack([(px - W / 2) / view.focal, (H / 2 - py) / view.focal, np.ones_like(px)], -1)
    return (r / np.linalg.norm(r, axis=-1, keepdims=True)).astype(np.float32)


RAYS = {1: rays_for(1), 2: rays_for(2)}


def textured(scale):
    rays = RAYS[scale]
    world = rays @ view.rot  # (h, w, 3) world-space ray directions
    sin_e = world[..., 1]
    ground = sin_e < -1e-4
    t = view.cam_height / np.where(ground, -sin_e, 1.0)
    gx = view.pos[0] + world[..., 0] * t
    gz = view.pos[2] + world[..., 2] * t
    near = (np.abs(gx) < NEAR_HALF) & (np.abs(gz) < NEAR_HALF)
    ni = np.clip(((gz + NEAR_HALF) / NEAR_M).astype(np.int32), 0, NEAR_N - 1)
    nj = np.clip(((gx + NEAR_HALF) / NEAR_M).astype(np.int32), 0, NEAR_N - 1)
    fi = np.clip(((gz + FAR_M * FAR_N / 2) / FAR_M).astype(np.int32), 0, FAR_N - 1)
    fj = np.clip(((gx + FAR_M * FAR_N / 2) / FAR_M).astype(np.int32), 0, FAR_N - 1)
    cls = np.where(near, near_tex[ni, nj], far_tex[fi, fj])
    slant = np.minimum(t / render3d.VISIBILITY_M, 1.0)
    haze = (slant * (HAZE_STEPS - 1)).astype(np.uint8)
    sky = len(pal) - sky_n + np.minimum(sin_e * ((sky_n - 1) / np.sin(np.radians(40))), sky_n - 1).astype(np.uint8)
    return np.where(ground, cls * HAZE_STEPS + haze, sky).astype(np.uint8)


def run_b():
    idx = textured(1)
    pygame.surfarray.blit_array(surf8, idx.T)
    out.blit(surf8, (0, 0))


def run_c():
    idx = textured(2)
    rgb = pal_rgb[idx]  # (2h, 2w, 3)
    avg = (rgb[0::2, 0::2] + rgb[1::2, 0::2] + rgb[0::2, 1::2] + rgb[1::2, 1::2]) >> 2
    pygame.surfarray.blit_array(out, avg.transpose(1, 0, 2).astype(np.uint8))


ONE_M, ONE_N = 32.0, 8192  # 262 km square, one raster
one_tex = noise_classes(ONE_N, 512)
HALF_RAYS = None


def textured_one(rays):
    world = rays @ view.rot
    sin_e = world[..., 1]
    ground = sin_e < -1e-4
    t = view.cam_height / np.where(ground, -sin_e, 1.0)
    half = ONE_M * ONE_N / 2
    i = np.clip(((view.pos[2] + world[..., 2] * t + half) * (1 / ONE_M)).astype(np.int32), 0, ONE_N - 1)
    j = np.clip(((view.pos[0] + world[..., 0] * t + half) * (1 / ONE_M)).astype(np.int32), 0, ONE_N - 1)
    cls = one_tex[i, j]
    haze = (np.minimum(t * (1 / render3d.VISIBILITY_M), 1.0) * (HAZE_STEPS - 1)).astype(np.uint8)
    sky = len(pal) - sky_n + np.minimum(sin_e * ((sky_n - 1) / np.sin(np.radians(40))), sky_n - 1).astype(np.uint8)
    return np.where(ground, cls * HAZE_STEPS + haze, sky).astype(np.uint8)


surf8_half = pygame.Surface((W // 2, H // 2), depth=8)
surf8_half.set_palette(pal)


def run_b1():
    pygame.surfarray.blit_array(surf8, textured_one(RAYS[1]).T)
    out.blit(surf8, (0, 0))


def run_b2():
    global HALF_RAYS
    if HALF_RAYS is None:
        HALF_RAYS = np.ascontiguousarray(RAYS[2][1::4, 1::4])  # 256x96 ray grid
    pygame.surfarray.blit_array(surf8_half, textured_one(HALF_RAYS).T)
    pygame.transform.scale(surf8_half, (W, H), surf8)
    out.blit(surf8, (0, 0))


def run_a():
    view.draw_sky_and_ground()


def bench(name, fn, n=20):
    fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    print(f"{name}: {1000 * (time.perf_counter() - t0) / n:6.1f} ms")


for alt_ft, pitch, bank in ((2000, -3, 0), (8000, -2, 15), (400, -6, 0)):
    view.set_camera(1000.0, alt_ft * 0.3048, -3000.0, 252.0, pitch, bank)
    print(f"-- {alt_ft} ft, pitch {pitch}, bank {bank}")
    bench("A today   ", run_a)
    bench("B mode7   ", run_b)
    bench("B1 one tex", run_b1)
    bench("B2 half   ", run_b2)

if len(sys.argv) > 1:
    view.set_camera(1000.0, 2000 * 0.3048, -3000.0, 252.0, -3, 0)
    run_b()
    pygame.image.save(pygame.transform.scale(out, (1024, 384)), sys.argv[1])
