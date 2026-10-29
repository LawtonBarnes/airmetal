"""The windshield's last step: every pixel snapped to one fixed palette
through a 4x4 Bayer ordered dither, so the textured ground, the sharp
roads and buildings, and the sky gradient all come out of the same
"engine" -- one pixel grid, one palette, one pattern -- instead of a soft
photo with hard lines on top and a banded sky (RGB565 has only 32 levels
of red and blue; a slow gradient bands).

The palette is a fixed 6x7x6 color cube (252 colors, VGA-style: no bias
toward any scene or time of day) plus 4 sky blues from the day sky's own
gradient, without which the pale sky dithers between lavender cube
entries. Picked on the wall 2026-10-29 from side-by-sides against
256x96, an adaptive 256-color palette and EGA-16: "looks much better up
close on the CRT and smooths out nicely when I sit farther away".

Ordered, not error diffusion: the pattern is fixed to the screen, so it
doesn't crawl from frame to frame (same reason as TV DINNER's PIXEL
mode). Nearest color is one lookup in a table indexed by the dithered
color quantized to RGB565, built once at startup (~0.5 s).
"""
import numpy as np
import pygame

BAYER4 = np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]], np.float32) / 16 - 0.5
AMPLITUDE = 20.0  # dither strength, in 0..255 color units
CUBE_LEVELS = (6, 7, 6)  # R, G, B -- green gets the extra level, as the eye wants
SKY_ELEVATIONS = (1.0, 3.0, 6.0, 10.0)  # degrees: the sky the windshield actually shows
# Nearest-color distance weights (R, G, B) -- roughly how strongly the eye
# notices each channel's difference (TV DINNER's).
COLOR_WEIGHTS = np.array([2.0, 4.0, 3.0], np.float32)


def palette(sky_colors):
    """The cube plus the given sky colors, (n, 3) float."""
    lv = [np.linspace(0, 255, n) for n in CUBE_LEVELS]
    cube = np.array(np.meshgrid(*lv, indexing="ij"), np.float32).reshape(3, -1).T
    return np.concatenate([cube, np.asarray(sky_colors, np.float32).reshape(-1, 3)])


class Dither:
    def __init__(self, surface, sky_colors):
        """`surface`: the windshield (32-bit), dithered in place each frame."""
        self.surface = surface
        w, h = surface.get_size()
        pal = palette(sky_colors)
        # Every RGB565 color (bit-expanded back to 8 bits/channel) -> its
        # nearest palette color, as the surface's own mapped pixel value.
        idx = np.arange(65536, dtype=np.uint32)
        c = np.stack([((idx >> 11) & 31) * (255 / 31), ((idx >> 5) & 63) * (255 / 63), (idx & 31) * (255 / 31)],
                     -1).astype(np.float32)
        best = np.zeros(65536, np.intp)
        best_d = np.full(65536, np.inf, np.float32)
        for k, p in enumerate(pal):
            d = (np.square(c - p) * COLOR_WEIGHTS).sum(1)
            closer = d < best_d
            best[closer], best_d[closer] = k, d[closer]
        mapped = np.array([surface.map_rgb(tuple(int(v) for v in p)) for p in pal], np.uint32)
        self.lut = mapped[best]
        # The threshold pattern, (w, h, 1) like surfarray's arrays (x first).
        self.offset = (np.tile(BAYER4.T, (w // 4 + 1, h // 4 + 1))[:w, :h, None] * AMPLITUDE).astype(np.float32)

    def apply(self):
        rgb = pygame.surfarray.array3d(self.surface).astype(np.float32)
        rgb += self.offset
        np.clip(rgb, 0, 255, out=rgb)
        q = rgb.astype(np.uint16)
        index = (q[..., 0] >> 3) << 11
        index |= (q[..., 1] >> 2) << 5
        index |= q[..., 2] >> 3
        pixels = pygame.surfarray.pixels2d(self.surface)
        pixels[...] = self.lut[index]
        del pixels  # unlocks the surface
