"""Renders one windshield frame from a fixed camera to a PNG, for checking the
scenery against the drone photos without the app or traffic.

Usage: python tools/snapshot.py OUT.png [heading pitch alt_ft east_m north_m bank]
Defaults: the drone's spot -- 400 ft over the Metal Shop, looking at downtown.
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

args = [float(a) for a in sys.argv[2:]]
heading, pitch, alt_ft, east, north, bank = (args + [252, -6, 400, 0, 0, 0][len(args):])
pygame.init()
s = config.load()
t0 = time.perf_counter()
world = scenery.Scenery(config.DATA_DIR, s["lat"], s["lon"])
render3d.set_saturation(s["saturation"], world)
t1 = time.perf_counter()
view = render3d.View(config.VIEW_LOW_W, config.VIEW_LOW_H, s["fov_deg"], world, s["ground_scale"],
                     s["ground_filter"])
view.set_camera(east, alt_ft * 0.3048, north, heading, pitch, bank)
drawn = view.render(world)
t2 = time.perf_counter()
pygame.image.save(pygame.transform.scale(view.surface, (1024, 384)), sys.argv[1])
print(f"load {1000 * (t1 - t0):.0f}ms, render {1000 * (t2 - t1):.1f}ms, {drawn} line segments")
