"""cProfile of the windshield render from one camera -- for finding hot spots on the Pi.

Usage: python3 tools/profile_render.py [x y z heading pitch bank]
"""
import cProfile
import os
import pstats
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pygame  # noqa: E402

import config  # noqa: E402
import render3d  # noqa: E402
import scenery  # noqa: E402

cam = [float(a) for a in sys.argv[1:]] or [0, 122, 0, 90, -6, 15]
pygame.init()
s = config.load()
world = scenery.Scenery(config.DATA_DIR, s["lat"], s["lon"])
view = render3d.View(config.VIEW_LOW_W, config.VIEW_LOW_H, s["fov_deg"])
view.set_camera(*cam)
view.render(world)
prof = cProfile.Profile()
prof.enable()
for _ in range(5):
    view.render(world)
prof.disable()
pstats.Stats(prof).sort_stats("tottime").print_stats(18)
