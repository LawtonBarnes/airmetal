"""EXITROW's standby: parked at a gate, seen from the cabin window.

The aerial photo that makes the ground in flight is 8 m a pixel, which
from a window 3.6 m off the apron is brown mush -- so the gate is modeled:
concrete apron with expansion joints and painted lines, the concourse
(a long box with a glass band), jet bridges, and a parked jet two gates
down each way (the next gate over filled the window like a gray wall).
Flat polygons in the same View as everything else, so the dither and the
time-of-day light treat it the same.

Gate coordinates: a = meters to the aircraft's right, b = meters forward,
y = up, from the middle of our aircraft; its nose points at the concourse.
Picked on the tubes 2026-11-25 (user: "Render looks great!").
"""
import math

import numpy as np
import pygame

import render3d as r3

WINDOW_HEIGHT_M = 3.6  # a narrowbody's cabin windows, above the apron

CONCRETE = (178, 174, 164)
JOINT = (140, 137, 130)
YELLOW = (230, 190, 40)
TERM_FRONT, TERM_TOP, TERM_SIDE = (196, 188, 170), (150, 146, 140), (170, 162, 148)
GLASS = (70, 96, 120)
GLASS_NIGHT = (255, 214, 140)  # the concourse lit from inside
MULLION = (96, 100, 106)
PARAPET = (128, 122, 112)
MULLION_SPACING = 4.0
MULLION_SPAN = 150.0  # m either side of our gate (AIR METAL's cockpit looks straight at them)
BRIDGE = (150, 152, 158)
FUSE, FUSE_SIDE, BELLY, TAIL, WING = (232, 232, 236), (200, 200, 206), (120, 124, 134), (60, 70, 140), (180, 182, 190)
APRON_FLOOD = np.array((60, 52, 34), np.float32)  # sodium floodlights on the concrete at night
GATE_SPACING = 48.0
GATES = 3  # concourse gates each side of ours
PARKED = (-2, 2)  # which gates have a jet in them (ours is 0)
TERMINAL_B = 40.0  # concourse face, m ahead of our aircraft's middle


def _box(a0, a1, b0, b1, y0, y1, front, side, top):
    """A box's faces in gate coordinates."""
    return [
        ([(a0, b0, y0), (a1, b0, y0), (a1, b0, y1), (a0, b0, y1)], front),
        ([(a0, b1, y0), (a1, b1, y0), (a1, b1, y1), (a0, b1, y1)], front),
        ([(a0, b0, y0), (a0, b1, y0), (a0, b1, y1), (a0, b0, y1)], side),
        ([(a1, b0, y0), (a1, b1, y0), (a1, b1, y1), (a1, b0, y1)], side),
        ([(a0, b0, y1), (a1, b0, y1), (a1, b1, y1), (a0, b1, y1)], top),
    ]


def _jet(ac):
    """A parked narrowbody, nose toward the concourse, centered on a = ac."""
    f = []
    f += _box(ac - 1.9, ac + 1.9, -24, 12, 1.2, 2.4, BELLY, BELLY, BELLY)
    f += _box(ac - 1.9, ac + 1.9, -24, 12, 2.4, 5.0, FUSE, FUSE_SIDE, FUSE)
    f.append(([(ac - 1.9, 12, 1.2), (ac + 1.9, 12, 1.2), (ac, 15, 3.2)], FUSE))  # nose
    f.append(([(ac - 1.9, 12, 5.0), (ac + 1.9, 12, 5.0), (ac, 15, 3.2)], FUSE_SIDE))
    f.append(([(ac, -24, 5.0), (ac, -17, 5.0), (ac, -27, 12.0), (ac, -30, 12.0)], TAIL))  # fin
    for s in (-1, 1):
        f.append(([(ac + 1.9 * s, 0, 2.2), (ac + 1.9 * s, -6, 2.2), (ac + 17 * s, -14, 3.0), (ac + 17 * s, -11, 3.0)],
                  WING))
        f.append(([(ac + 1.9 * s, -21, 4.2), (ac + 1.9 * s, -24, 4.2), (ac + 7 * s, -27, 4.4), (ac + 7 * s, -25.5, 4.4)],
                  WING))
        f += _box(ac + 5 * s - 1, ac + 5 * s + 1, -5, -1, 0.6, 2.2, BELLY, BELLY, BELLY)  # engine
    return f


def _backdrop():
    """The concourse and what's on its face, in drawing order. Every camera
    (cockpit, cabin window) is on the apron side, below the roof, so this
    always goes first and needs no depth sort -- sorting the 1 km wall and
    its small mullions by their centers buried the mullions."""
    out = _box(-500, 500, TERMINAL_B, TERMINAL_B + 22, 0, 12, TERM_FRONT, TERM_SIDE, TERM_TOP)[::-1]
    out.append(([(-500, TERMINAL_B - 0.1, 4), (500, TERMINAL_B - 0.1, 4), (500, TERMINAL_B - 0.1, 9),
                 (-500, TERMINAL_B - 0.1, 9)], GLASS))
    out.append(([(-500, TERMINAL_B - 0.1, 10.6), (500, TERMINAL_B - 0.1, 10.6), (500, TERMINAL_B - 0.1, 12),
                 (-500, TERMINAL_B - 0.1, 12)], PARAPET))
    for a in np.arange(-MULLION_SPAN, MULLION_SPAN + 0.1, MULLION_SPACING):
        out.append(([(a - 0.15, TERMINAL_B - 0.2, 4), (a + 0.15, TERMINAL_B - 0.2, 4),
                     (a + 0.15, TERMINAL_B - 0.2, 9), (a - 0.15, TERMINAL_B - 0.2, 9)], MULLION))
    return out


def _faces():
    """Jet bridges and parked jets, depth-sorted at draw time."""
    out = []
    for k in range(-GATES, GATES + 1):
        ac = k * GATE_SPACING
        out += _box(ac - 5.5, ac - 2.5, 14, TERMINAL_B, 3.2, 6.2, BRIDGE, BRIDGE, BRIDGE)  # jet bridge
        if k in PARKED:
            out += _jet(ac)
    return out


BACKDROP = _backdrop()
FACES = _faces()


def _lit(color):
    """A day color under the current light (render3d.LIGHT); the concourse
    glass glows as the lights come on."""
    if color == GLASS:
        on = float(r3.LIGHTS_ON[0])
        c = np.array(GLASS, np.float32) * r3.LIGHT * (1 - on) + np.array(GLASS_NIGHT, np.float32) * on
    else:
        c = np.array(color, np.float32) * r3.LIGHT
        if color in (CONCRETE, JOINT, YELLOW, TERM_FRONT, PARAPET, MULLION, BRIDGE):
            c = c + APRON_FLOOD * float(r3.LIGHTS_ON[0])
    return tuple(int(v) for v in np.clip(c, 0, 255))


def draw(view, gx, gz, heading):
    """The gate, over whatever view.render() drew, on view.surface. Our
    aircraft's middle at (gx, gz), its nose toward `heading`."""
    surf = view.surface
    h = math.radians(heading)
    fx, fz = math.sin(h), math.cos(h)
    rx, rz = math.cos(h), -math.sin(h)

    def world(pts):
        p = np.asarray(pts, np.float32)
        x = gx + p[:, 0] * rx + p[:, 1] * fx
        z = gz + p[:, 0] * rz + p[:, 1] * fz
        return np.stack([x, p[:, 2], z], 1)

    def poly(pts, color):
        cam = r3._clip_near(view.to_camera(world(pts)))
        if len(cam) < 3:
            return
        scr = r3._clip_rect(view.to_screen(cam), -2, -2, view.w + 2, view.h + 2)
        if len(scr) >= 3:
            pygame.draw.polygon(surf, color, scr.tolist())

    def line(a0, b0, a1, b1, color, width=1):
        cam = view.to_camera(world([(a0, b0, 0.02), (a1, b1, 0.02)]))
        z0, z1 = cam[0, 2], cam[1, 2]
        if z0 < r3.NEAR and z1 < r3.NEAR:
            return
        if z0 < r3.NEAR or z1 < r3.NEAR:
            k = 0 if z0 < r3.NEAR else 1
            cam[k] = cam[0] + (cam[1] - cam[0]) * ((r3.NEAR - z0) / (z1 - z0))
        s = view.to_screen(cam)
        pygame.draw.line(surf, color, s[0].tolist(), s[1].tolist(), width)

    poly([(-500, -260, 0), (500, -260, 0), (500, TERMINAL_B, 0), (-500, TERMINAL_B, 0)], _lit(CONCRETE))
    joint = _lit(JOINT)
    for a in np.arange(-150, 151, 15):
        line(a, -150, a, TERMINAL_B, joint)
    for b in np.arange(-150, TERMINAL_B, 15):
        line(-150, b, 150, b, joint)
    yellow = _lit(YELLOW)
    for k in range(-6, 7):
        line(k * GATE_SPACING, -45, k * GATE_SPACING, 14, yellow, 2)  # lead-in lines
    line(-500, -60, 500, -60, yellow, 2)  # taxilane
    for pts, color in BACKDROP:
        poly(pts, _lit(color))
    # Painter's order, farthest first.
    cam_pos = view.pos
    depth = [float(np.linalg.norm(world(pts).mean(0) - cam_pos)) for pts, _ in FACES]
    for i in np.argsort(depth)[::-1]:
        pts, color = FACES[i]
        poly(pts, _lit(color))
