"""The windshield's optional softening, so the wall's 21" tubes look more
like EXITROW's 14" ones (user, 2026-11-25: "a mild blur on the upper
half"). Applied while pixel-doubling the 512x192 render up to 1024x384,
after the dither, before the windshield frame and its text -- so they stay sharp.

  off   plain pixel-doubling (as before)
  soft  a smooth (bilinear) 2x upscale instead: edges and the dither
        pattern melt a little, about what composite video does. Free.
  more  soft, plus a 1-2-1 horizontal pass -- composite smears sideways
        far more than up and down. A few ms on QUAD's Pi 5.
"""
import numpy as np
import pygame

MODES = ("off", "soft", "more")


def upscale(src, dst, mode):
    if mode == "off":
        pygame.transform.scale(src, dst.get_size(), dst)
        return
    pygame.transform.smoothscale(src, dst.get_size(), dst)
    if mode == "more":
        px = pygame.surfarray.pixels3d(dst)  # (w, h, 3), a view into dst
        a = px.astype(np.uint16)
        px[1:-1] = ((a[:-2] + 2 * a[1:-1] + a[2:]) >> 2).astype(np.uint8)
        del px  # unlock dst
