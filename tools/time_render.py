"""Times each windshield render stage from a few fixed cameras -- run on the
Pi to check the 10fps budget.

Usage: python3 tools/time_render.py
"""
import os
import sys
import time

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pygame  # noqa: E402

import config  # noqa: E402
import render3d  # noqa: E402
import scenery  # noqa: E402

CAMERAS = [  # x, y (m above ground), z, heading, pitch, bank
    (0, 122, 0, 252, -6, 0),  # the drone shot
    (0, 122, 0, 90, -6, 15),
    (0, 1200, -15000, 252, -8, -20),
    (-3000, 300, 0, 270, -3, 0),  # over White Rock Lake
    (-9000, 600, -5000, 0, -4, 0),
]

pygame.init()
s = config.load()
t = time.perf_counter()
world = scenery.Scenery(config.DATA_DIR, s["lat"], s["lon"])
print(f"load {time.perf_counter() - t:.2f}s")
view = render3d.View(config.VIEW_LOW_W, config.VIEW_LOW_H, s["fov_deg"])
for cam in CAMERAS:
    view.set_camera(*cam)
    best = {}
    for _ in range(3):
        for name, fn in (("sky", view.draw_sky_and_ground), ("lakes", lambda: view.draw_lakes(world)),
                         ("lines", lambda: view.draw_lines(world.lines))):
            t = time.perf_counter()
            n = fn()
            best[name] = min(best.get(name, 9), time.perf_counter() - t)
            if name == "lines":
                chunks = n
    total = sum(best.values())
    print(f"{str(cam):36s} " + "  ".join(f"{k} {1000 * v:5.1f}ms" for k, v in best.items())
          + f"  = {1000 * total:5.1f}ms  ({chunks} chunks)")
