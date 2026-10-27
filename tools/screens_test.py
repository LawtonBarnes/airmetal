"""Renders the splash, ABOUT and power-dialog screens to PNGs, headless, from
a recording -- for checking their layout against the tube seams without
the wall.

Usage: python tools/screens_test.py OUTDIR [RECORDING]
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
out_dir = sys.argv[1]
rec = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "recordings", "feed_20261026.jsonl.gz")
sys.argv = [sys.argv[0], "--headless", "--replay", rec, "--offset", "600"]
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import pygame  # noqa: E402

import main  # noqa: E402
import screens  # noqa: E402

main.screens.SPLASH_SECONDS = 0
os.makedirs(out_dir, exist_ok=True)
SEAMS = (255, 0, 255)


def save(name, top, bottom):
    frame = pygame.Surface((1024, 768))
    frame.blit(top, (0, 0))
    frame.blit(bottom, (0, 384))
    # Tube seams in magenta, so text crossing a bezel stands out.
    pygame.draw.line(frame, SEAMS, (512, 0), (512, 767))
    pygame.draw.line(frame, SEAMS, (0, 384), (1023, 384))
    pygame.image.save(frame, os.path.join(out_dir, name))


class Capture:
    """Stands in for the window: records the last frame, feeds scripted keys."""

    def __init__(self, script):
        self.script = list(script)
        self.last = None

    def actions(self):
        return [self.script.pop(0)] if self.script else ["quit"]

    def write(self, top, bottom):
        self.last = (top.copy(), bottom.copy())

    def close(self):
        pass


pygame.init()
splash = screens.build_splash(main.config.SPLASH_PATH, main.config.VERSION)
save("1_splash.png", splash.subsurface((0, 0, 1024, 384)), splash.subsurface((0, 384, 1024, 384)))

for name, script in (("2_live.png", [None] * 30),
                     ("3_about.png", [None] * 30 + ["back"]),
                     ("4_about_power.png", [None] * 30 + ["back", "power", "right"]),
                     ("5_live_power.png", [None] * 30 + ["power"])):
    app = main.App()
    cap = Capture(script)
    app.out = cap
    orig = app.handle
    app.handle = lambda a, orig=orig: True if a is None else orig(a)
    # Stop right after the last scripted key's frame.
    real_actions = cap.actions

    def actions(cap=cap, real=real_actions, app=app):
        if not cap.script and cap.last is not None:
            save(name, *cap.last)
            app.quit_requested = True
            return []
        return real()
    cap.actions = actions
    app.run()
print("saved to", out_dir)
