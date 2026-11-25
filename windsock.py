"""The Metal Shop's windsock: a landmark planted on the shop so it can be
picked out from the air, in AIR METAL's windshield and EXITROW's windows.

Modeled on the user's art (Desktop/AIR METAL/WINDSOCK.png, 2026-11-25):
a slate pole with a five-band orange/white sock, proportions measured off
the drawing (pole height H = 1):
  sock mouth center 0.735 H up, 0.16 H out from the pole
  mouth radius 0.19 H, tail radius 0.10 H, length 0.72 H
Flat-shaded polygons with the art's dark outline, drawn by
render3d.View.draw_props. The sock streams downwind: `wind_from` is the
compass direction the wind blows FROM (a sock points the other way).
"""
import math

import numpy as np

POLE = (72, 72, 109)
ORANGE = (255, 145, 72)
WHITE = (218, 218, 255)
INSIDE = (70, 40, 30)  # looking into the mouth
SIDES = 10
# At night the shop floodlights it, like an airfield's lit windsock: a warm
# light on the paint, 255 = daylight.
FLOOD = (190, 165, 130)


def _ring(center, axis, up, side, radius, n=SIDES):
    """n points around `center` in the plane normal to `axis`."""
    out = []
    for k in range(n):
        a = 2 * math.pi * k / n
        out.append(center + radius * (math.cos(a) * up + math.sin(a) * side))
    return np.array(out, np.float32)


def build(height_m, wind_from_deg, x=0.0, z=0.0):
    """-> list of props for render3d.View.draw_props: (points (n, 3) world
    x/y/z, rgb, outward normal (3,) or None for a two-sided face, night
    floodlight or None)."""
    H = float(height_m)
    props = []
    # Pole: an n-sided column, plus its cap.
    r = 0.025 * H
    base = np.array([x, 0.0, z], np.float32)
    up = np.array([0, 1, 0], np.float32)
    for k in range(8):
        a0, a1 = 2 * math.pi * k / 8, 2 * math.pi * (k + 1) / 8
        p0 = base + r * np.array([math.cos(a0), 0, math.sin(a0)], np.float32)
        p1 = base + r * np.array([math.cos(a1), 0, math.sin(a1)], np.float32)
        quad = np.array([p0, p1, p1 + up * H, p0 + up * H], np.float32)
        am = (a0 + a1) / 2
        props.append((quad, POLE, np.array([math.cos(am), 0, math.sin(am)], np.float32), FLOOD))
    # The sock, streaming downwind from an arm off the pole.
    to = math.radians((wind_from_deg + 180.0) % 360.0)
    axis = np.array([math.sin(to), 0, math.cos(to)], np.float32)
    side = np.array([math.cos(to), 0, -math.sin(to)], np.float32)
    mouth = base + up * (0.735 * H) + axis * (0.16 * H)
    length, r0, r1 = 0.72 * H, 0.19 * H, 0.10 * H
    # Arm and braces from the pole to the mouth (the art's bracket).
    for dy in (0.0, 0.19, -0.19):
        a = base + up * ((0.735 + 0.12 * (1 if dy > 0 else -1 if dy < 0 else 0)) * H)
        b = mouth + up * (dy * H)
        w = up * (0.012 * H) if dy == 0 else side * (0.012 * H)
        props.append((np.array([a - w, b - w, b + w, a + w], np.float32), POLE, None, FLOOD))
    bands = 5
    rings = [_ring(mouth + axis * (length * i / bands), axis, up, side, r0 + (r1 - r0) * i / bands)
             for i in range(bands + 1)]
    for i in range(bands):
        color = ORANGE if i % 2 == 0 else WHITE
        a, b = rings[i], rings[i + 1]
        for k in range(SIDES):
            k1 = (k + 1) % SIDES
            quad = np.array([a[k], a[k1], b[k1], b[k]], np.float32)
            mid = quad.mean(0)
            center = mouth + axis * (length * (i + 0.5) / bands)
            n = mid - center
            n -= axis * float(n @ axis)
            n /= max(float(np.linalg.norm(n)), 1e-6)
            props.append((quad, color, n.astype(np.float32), FLOOD))
    # The open mouth: its dark inside, seen from upwind.
    props.append((rings[0], INSIDE, -axis, None))
    return props
