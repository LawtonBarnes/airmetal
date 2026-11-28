#!/usr/bin/env python3
"""EXITROW -- AIR METAL's side windows, on the puppet tubes either side of
the wall: P1 looks out the port side, P3 starboard (settings.ini
[exitrow] side).

It flies the aircraft AIR METAL is riding (asked of AIR METAL's ride link
on QUAD), from the same ADS-B feed through the same flight smoothing, the
same `delay_sec` behind live. Positions are stamped with readsb's own
receive times, so every window lands on the same moment without
streaming the camera around. The camera sits a little aft of the
cockpit, turned 90 degrees, looking down past the wing: the aircraft's
bank tips the window up or down, its pitch tilts the horizon.

Over the 3D view goes the user's window art (exitrow_art/, built from
the PSD by tools/build_exitrow_art.py): a jet wing, a Cessna's high wing
and strut for light aircraft, or for helicopters rotor blades
flickering through three frames under the user's chopper window matte
(heli_window.png, from CHOPPER-WINDOW-MATTE-W-SLIDE.png, alpha cut at
50% for a colorkey blit). The art is drawn as the port window;
starboard is the same art mirrored. The wings are tinted with the sky's
color as the light changes, down to dark gray at night (tint_wing); from
sunset on, the jet wing's red lights (wing_jet_lights.png) are lit over
it, untinted.

With nothing to ride (AIR METAL in standby, or not running), the window
shows the aircraft parked at a gate (gate.py).

Usage:
  exitrow.py                        Pi: live, straight to /dev/fb0
  exitrow.py --window               PC: in a window (dev)
  exitrow.py --side starboard       override settings.ini
  exitrow.py --replay FILE          ride a recording, choosing aircraft itself
  exitrow.py --standalone           live, but choose aircraft itself (no AIR METAL)
"""
import argparse
import json
import math
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.request

import config
import daylight
import dither as dithermod

VERSION = "0.9"  # EXITROW's own (STRINGS/SCRUTE read it from this file)


def parse_args():
    ap = argparse.ArgumentParser(description="EXITROW")
    ap.add_argument("--window", action="store_true", help="draw in a desktop window instead of /dev/fb0")
    ap.add_argument("--headless", action="store_true", help="window mode with no display (testing; use --snap)")
    ap.add_argument("--side", choices=("port", "starboard"), help="override settings.ini [exitrow] side")
    ap.add_argument("--replay", help="ride a recorded feed (implies --standalone)")
    ap.add_argument("--speed", type=float, default=1.0, help="replay speed")
    ap.add_argument("--offset", type=float, default=0.0, help="start this many seconds into the replay")
    ap.add_argument("--standalone", action="store_true", help="choose aircraft here instead of following AIR METAL")
    ap.add_argument("--gate", action="store_true", help="always show the gate (testing)")
    ap.add_argument("--seconds", type=float, default=0.0, help="quit after this long (testing)")
    ap.add_argument("--stats", action="store_true", help="print frame timings every 10s")
    ap.add_argument("--snap", help="save a frame to this directory every --snap-every seconds")
    ap.add_argument("--snap-every", type=float, default=10.0)
    ap.add_argument("--clock", help="light the scene as at this local time, HH:MM (overrides settings.ini)")
    return ap.parse_args()


ARGS = parse_args()
if ARGS.headless:
    ARGS.window = True
if not ARGS.window or ARGS.headless:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np  # noqa: E402
import pygame  # noqa: E402

import director as directormod  # noqa: E402
import exitrow_screens  # noqa: E402
import feed as feedmod  # noqa: E402
import flight  # noqa: E402
import gate  # noqa: E402
import output  # noqa: E402
import render3d  # noqa: E402
import scenery  # noqa: E402
import windsock  # noqa: E402
from geo import Projection  # noqa: E402

AFT_M = 12.0  # the window, behind the cockpit
SIDE_M = 1.9  # ...and out at the fuselage side
GATE_LOOK_DOWN = 2.0  # at the gate the window looks nearly level
LIGHTING_SEC = 10.0
GATE_REDRAW_SEC = 10.0  # the gate is still; redraw it only for the light
LINK_POLL_SEC = 1.0
LINK_TIMEOUT_SEC = 6.0  # no answer from AIR METAL this long: it's not there
EXIT_GOTO_HOME = 42  # see STRINGS/SCRUTE: Home hands off to the app menu
COLORKEY = (255, 0, 255)
TINTED_ART = ("wing_jet", "wing_cessna")
WING_LIGHTS_ON = 0.1  # lighting.lights_on above this (sun under ~3 deg): the jet's red lights are lit
WING_SKY_TINT = 0.5  # how far the wings take on the sky's hue (0 = none, 1 = multiplied by it)
WING_NIGHT_GRAY = 0.3  # the darkest the wings get: dark gray, not black
NOTE_FG, NOTE_EDGE = (255, 220, 0), (255, 165, 0)


def wing_kind(category):
    if category in directormod.HELICOPTER:
        return "heli"
    if category in directormod.SMALL:
        return "cessna"
    return "jet"  # A2 (regional/bizjets) and up, and unknown


class AirMetalLink:
    """Polls AIR METAL's GET /riding on a background thread: which aircraft
    it's riding (hex or None), and how far behind live it flies."""

    def __init__(self, url):
        self.url = url
        self.hex = None
        self.delay_sec = None
        self.last_ok = 0.0
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while True:
            try:
                with urllib.request.urlopen(self.url, timeout=3) as resp:
                    msg = json.load(resp)
                self.hex = msg.get("hex")
                self.delay_sec = msg.get("delay_sec")
                self.last_ok = time.time()
            except (OSError, ValueError):
                pass
            time.sleep(LINK_POLL_SEC)

    @property
    def up(self):
        return time.time() - self.last_ok < LINK_TIMEOUT_SEC


def load_overlay(name, flip):
    """An art PNG as an opaque colorkeyed surface (its alpha is 0/255 only;
    a colorkey blit is a fraction of the cost of per-pixel alpha on a Pi 3B+)."""
    img = pygame.image.load(os.path.join(config.EXIT_ART_DIR, name + ".png"))
    if flip:
        img = pygame.transform.flip(img, True, False)
    surf = pygame.Surface(img.get_size())
    surf.fill(COLORKEY)
    surf.blit(img, (0, 0))
    surf.set_colorkey(COLORKEY)
    return surf


def load_art_source(name, flip):
    """An art PNG as loaded (per-pixel alpha), for tint_wing."""
    img = pygame.image.load(os.path.join(config.EXIT_ART_DIR, name + ".png"))
    return pygame.transform.flip(img, True, False) if flip else img


def wing_tint(lighting):
    """RGB multiplier for the wings: the sky's average hue (at WING_SKY_TINT
    strength) times the scene's light, floored at WING_NIGHT_GRAY -- and
    going neutral toward that floor, so night is a dark gray, not a dark blue."""
    day_sky = np.array(render3d._sky_palette(64), np.float32)
    sky = lighting.sky_lut(day_sky, 16, 64, math.sin(math.radians(render3d.SKY_TOP_DEG))).mean(axis=(0, 1))
    hue = sky / max(float(sky.max()), 1.0)
    level = float(np.mean(lighting.light))
    bright = np.maximum(np.asarray(lighting.light, np.float32), WING_NIGHT_GRAY)  # warm at sunset
    strength = WING_SKY_TINT * min(max((level - 0.15) / 0.45, 0.0), 1.0)  # full until dusk, none at night
    return (1 - strength + strength * hue) * bright


def tint_wing(src, tint):
    """A colorkeyed overlay surface: the art's pixels times `tint` (SDL's
    multiply blend -- ~10x faster than numpy on a Pi 3B+)."""
    img = src.copy()
    r, g, b = (int(round(min(max(v, 0.0), 1.0) * 255)) for v in tint)
    img.fill((r, g, b, 255), special_flags=pygame.BLEND_RGBA_MULT)
    surf = pygame.Surface(img.get_size())
    surf.fill(COLORKEY)
    surf.blit(img, (0, 0))
    surf.set_colorkey(COLORKEY)
    return surf


def hidden_mask(names, size):
    """(h, w) bool at the 3D view's size: True where these art layers are
    opaque over every frame pixel the view pixel becomes."""
    covered = None
    for name in names:
        a = pygame.surfarray.array_alpha(pygame.image.load(os.path.join(config.EXIT_ART_DIR, name + ".png"))).T > 0
        covered = a if covered is None else covered | a
    w, h = size
    fh, fw = covered.shape
    return covered.reshape(h, fh // h, w, fw // w).all(axis=(1, 3))


class Presenter:
    """The back half of each frame -- dither, pixel-double, window art, the
    standby note, out to the tube -- on a thread of its own, so it runs on
    a second core while the main thread renders the next frame (numpy and
    SDL let go of the GIL for the heavy parts). About 25 ms a frame on a
    Pi 3B+ that would otherwise come on top of the render."""

    def __init__(self, out, view_size, frame_size, dither, font, threaded=True):
        self.out = out
        self.stage = pygame.Surface(view_size, 0, 32)
        self.dither = (dithermod.Dither(self.stage, render3d.sky_colors(dithermod.SKY_ELEVATIONS))
                       if dither else None)
        self.frame = pygame.Surface(frame_size)
        self.cache = None  # a composed frame kept for reuse (the gate), without its note
        self.font = font
        self.ms = 0.0  # time the last compose took
        self.threaded = threaded
        self._cv = threading.Condition()
        self._job = None
        self._busy = False
        self._error = None
        if threaded:
            threading.Thread(target=self._run, daemon=True).start()

    def submit(self, view_surface, overlays, note=None, keep=False):
        """Hand over a rendered view (copied, so rendering can carry on)."""
        self.wait()
        self.stage.blit(view_surface, (0, 0))
        self._post(("view", overlays, note, keep))

    def submit_cached(self, note=None):
        self.wait()
        self._post(("cached", None, note, False))

    def show(self, frame):
        """A finished frame as is (ABOUT, the shutdown dialog)."""
        self.wait()
        self._post(("frame", frame, None, False))

    def wait(self):
        with self._cv:
            while self._busy:
                self._cv.wait()
        if self._error is not None:
            err, self._error = self._error, None
            raise err

    def _post(self, job):
        if not self.threaded:
            self._compose(*job)
            return
        with self._cv:
            self._job, self._busy = job, True
            self._cv.notify_all()

    def _run(self):
        while True:
            with self._cv:
                while self._job is None:
                    self._cv.wait()
                job, self._job = self._job, None
            try:
                self._compose(*job)
            except Exception as exc:  # noqa: BLE001 -- handed to the main thread
                self._error = exc
            finally:
                with self._cv:
                    self._busy = False
                    self._cv.notify_all()

    def _compose(self, kind, overlays, note, keep):
        t0 = time.perf_counter()
        if kind == "view":
            if self.dither is not None:
                self.dither.apply()
            pygame.transform.scale(self.stage, self.frame.get_size(), self.frame)
            for ov in overlays:
                self.frame.blit(ov, (0, 0))
            if keep:
                self.cache = self.frame.copy()
        elif kind == "frame":
            self.frame.blit(overlays, (0, 0))  # (the frame rides in the overlays slot)
        else:
            self.frame.blit(self.cache, (0, 0))
        if note:
            img = self.font.render(note, False, NOTE_FG)
            box = img.get_rect(midtop=(self.frame.get_width() // 2, 55)).inflate(28, 18)
            pygame.draw.rect(self.frame, (0, 0, 0), box)
            pygame.draw.rect(self.frame, NOTE_EDGE, box, 3)
            self.frame.blit(img, img.get_rect(center=box.center))
        self.out.write_frame(self.frame)
        self.ms = 1000 * (time.perf_counter() - t0)


class App:
    def __init__(self):
        self.s = config.load()
        self.side = ARGS.side or self.s["exit_side"]
        if self.side not in ("port", "starboard"):
            raise SystemExit(f"settings.ini [exitrow] side must be port or starboard, not {self.side!r}")
        self.sgn = -1 if self.side == "port" else 1
        pygame.init()
        self.quit_requested = False
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, self._on_signal)
        self.exit_code = 0
        frame_size = (config.EXIT_FRAME_W, config.EXIT_FRAME_H)
        self.out = output.open_output(ARGS.window, frame_size, "EXITROW")
        # Splash first, so it's up while the scenery loads.
        self.screens = exitrow_screens.Screens(frame_size, config.LOGO_PATH,
                                               os.path.join(config.EXIT_ART_DIR, "splash.png"), self.side)
        self.out.write_frame(self.screens.build_splash(VERSION))
        splash_until = time.time() + exitrow_screens.SPLASH_SECONDS
        self.about = None  # the ABOUT canvas while it's up
        self.power_dialog = None  # selected POWER_OPTIONS index while it's up
        self.power_action = None
        # No waiting for vsync: at ~10 fps on a 60 Hz tube a torn frame is
        # rare and brief, and the wait cost up to a sixth of the frame.
        self.out.vsync = False
        self.world = scenery.Scenery(config.DATA_DIR, self.s["lat"], self.s["lon"])
        render3d.set_saturation(self.s["saturation"], self.world)
        self.view = render3d.View(config.EXIT_VIEW_W, config.EXIT_VIEW_H, self.s["exit_fov_deg"], self.world,
                                  self.s["exit_ground_scale"], self.s["ground_filter"], False,
                                  pixel_aspect=config.EXIT_PIXEL_ASPECT)
        self.view.line_range = self.s["exit_line_range"]
        if self.s["windsock_height_m"] > 0:
            self.view.set_props(windsock.build(self.s["windsock_height_m"], self.s["windsock_wind_from"]))
        flip = self.side == "starboard"
        self.art = {k: load_overlay(k, flip) for k in
                    ("wing_jet", "wing_jet_lights", "wing_cessna", "heli_window",
                     "blades1", "blades2", "blades3")}
        self.wing_lights = False
        self.art_src = {k: load_art_source(k, flip) for k in TINTED_ART}
        # What each kind of window hides of the view (blades flicker, so only
        # the window matte counts for helicopters).
        view_size = (config.EXIT_VIEW_W, config.EXIT_VIEW_H)
        self.hidden = {kind: hidden_mask(names, view_size) for kind, names in
                       (("jet", ["wing_jet"]), ("cessna", ["wing_cessna"]), ("heli", ["heli_window"]))}
        if flip:
            self.hidden = {k: m[:, ::-1].copy() for k, m in self.hidden.items()}
        self.hidden_kind = None
        self.blade_i = 0
        self.presenter = Presenter(self.out, view_size, (config.EXIT_FRAME_W, config.EXIT_FRAME_H),
                                   self.s["dither"], pygame.font.Font(config.FONT_PATH, 22),
                                   threaded=not (ARGS.window and not ARGS.headless))
        self.clock = ARGS.clock or self.s["clock"]
        self.next_lighting = 0.0
        self.update_lighting()
        if ARGS.replay:
            self.feed = feedmod.ReplayFeed(ARGS.replay, ARGS.speed, ARGS.offset)
        else:
            self.feed = feedmod.LiveFeed(self.s["feed_url"], self.s["poll_sec"])
        self.standalone = ARGS.standalone or bool(ARGS.replay)
        self.link = None if self.standalone else AirMetalLink(self.s["exit_airmetal_url"])
        self.director = directormod.Director() if self.standalone else None
        self.traffic = flight.Traffic(self.s["lat"], self.s["lon"])
        self.cam = flight.Camera(self.s["ground_msl_m"])
        self.riding = None
        gx, gz = Projection(self.s["lat"], self.s["lon"]).to_xz(self.s["exit_gate_lat"], self.s["exit_gate_lon"])
        self.gate_xz = (gx, gz)
        self.gate_ready = False  # the presenter holds a composed gate frame
        self.gate_drawn_at = 0.0
        self.fps = 0.0
        self.kind = None  # window art in use: jet / cessna / heli
        while time.time() < splash_until and not self.quit_requested:
            self.out.actions()  # keys during the splash are dropped, not queued
            time.sleep(0.05)

    def _on_signal(self, signum, frame):
        self.quit_requested = True

    def handle(self, action):
        """One remote action; returns False to quit. The fleet key map: Home
        back to SCRUTE's menu, Q/Esc/Menu quit, BACK toggles ABOUT, Power
        opens the shutdown dialog."""
        if self.power_dialog is not None:
            return self._handle_power_dialog(action)
        if action == "home":
            self.exit_code = EXIT_GOTO_HOME
            return False
        if action == "quit":
            return False
        if action == "back":
            self.about = None if self.about is not None else self.screens.build_about(
                f"EXITROW {VERSION}", self.about_rows())
        elif action == "power":
            self.power_dialog = 0
        return True

    def _handle_power_dialog(self, action):
        n = len(exitrow_screens.POWER_OPTIONS)
        if action in ("left", "up"):
            self.power_dialog = (self.power_dialog - 1) % n
        elif action in ("right", "down"):
            self.power_dialog = (self.power_dialog + 1) % n
        elif action == "enter":
            choice = exitrow_screens.POWER_OPTIONS[self.power_dialog]
            if choice != "NO":
                self.power_action = "shutdown" if choice == "YES" else "restart"
                return False
            self.power_dialog = None
        elif action in ("quit", "back", "power"):
            self.power_dialog = None
        return True

    def about_rows(self):
        host = lambda url: url.split("//", 1)[-1].split("/", 1)[0].split(":", 1)[0]
        if ARGS.replay:
            follow = "REPLAY"
        elif self.standalone:
            follow = "OWN PICKS"
        else:
            follow = f"AIR METAL {host(self.s['exit_airmetal_url'])}" + ("" if self.link.up else " (DOWN)")
        track = self.traffic.tracks.get(self.riding) if self.riding else None
        if track is not None and track.info is not None:
            riding = track.info.callsign or track.info.hex.upper()
        else:
            riding = "AT THE GATE"
        return [
            ("SIDE", self.side.upper()),
            ("FOLLOWING", follow),
            ("FEED", f"READSB {host(self.s['feed_url'])}"),
            ("RIDING", riding),
            ("WINDOW", {"jet": "JET WING", "cessna": "HIGH WING", "heli": "ROTOR"}.get(self.kind, "--")),
            ("FRAME RATE", f"{self.fps:.1f} FPS" if self.fps else "--"),
        ]

    def update_lighting(self):
        now = time.time()
        if now < self.next_lighting:
            return
        self.next_lighting = now + LIGHTING_SEC
        t = now
        if self.clock != "auto":
            hh, mm = (int(v) for v in self.clock.split(":"))
            lt = time.localtime(now)
            t = time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday, hh, mm, 0, 0, 0, -1))
        elev, az = daylight.sun_position(t, self.s["lat"], self.s["lon"])
        lighting = daylight.lighting_at(elev, az)
        self.view.set_lighting(lighting, self.world)
        self.wing_lights = lighting.lights_on > WING_LIGHTS_ON
        tint = wing_tint(lighting)
        last = getattr(self, "wing_tint", None)
        if last is None or np.abs(tint - last).max() > 0.004:  # all day long it doesn't change
            self.wing_tint = tint
            self.wing_tinted = {}  # retinted on next use, only the wing on screen (~75 ms each on a 3B+)

    def wing(self, name):
        """A wing's overlays: the art in the current sky tint (a fresh surface
        swapped in whole, so the presenter thread never blits a half-tinted
        wing), plus the jet's lit red lights from sunset on."""
        if not self.wing_tinted.get(name):
            self.art[name] = tint_wing(self.art_src[name], self.wing_tint)
            self.wing_tinted[name] = True
        if name == "wing_jet" and self.wing_lights:
            return [self.art[name], self.art["wing_jet_lights"]]
        return [self.art[name]]

    # -- what to show ------------------------------------------------------

    def delay(self):
        if self.link is not None and self.link.delay_sec is not None:
            return float(self.link.delay_sec)
        return self.s["delay_sec"]

    def update(self, dt):
        """Feeds the traffic, follows the ride. -> the track being flown, or None (gate)."""
        now = self.feed.clock()
        self.traffic.update(self.feed.poll(), now)
        t_view = now - self.delay()
        if ARGS.gate:
            hex_ = None
        elif self.director is not None:
            hex_ = self.director.update(self.traffic, now, t_view)
        else:
            hex_ = self.link.hex if self.link.up else None
        if hex_ != self.riding:
            self.riding = hex_
            self.cam.reset()
        track = self.traffic.tracks.get(hex_) if hex_ else None
        state = track.state_at(t_view) if track else None
        if state is None:
            return None
        self.cam.follow(state, dt)
        return track

    def note(self):
        if self.link is not None and not self.link.up:
            return "WAITING FOR AIR METAL"
        if getattr(self.feed, "error", None):
            return "NO ADS-B DATA"
        return None  # no traffic: just the terminal or the sky, no text

    # -- drawing -----------------------------------------------------------

    def _hide_for(self, kind):
        if kind != self.hidden_kind:
            self.view.set_hidden(self.hidden[kind])
            self.hidden_kind = kind

    def draw_flight(self, track):
        c = self.cam
        h = math.radians(c.heading)
        fx, fz, rx, rz = math.sin(h), math.cos(h), math.cos(h), -math.sin(h)
        x = c.x - AFT_M * fx + self.sgn * SIDE_M * rx
        z = c.z - AFT_M * fz + self.sgn * SIDE_M * rz
        kind = self.kind = wing_kind(track.info.category if track.info else "")
        self._hide_for(kind)
        # Bank right (positive) dips the starboard window and lifts the port one;
        # nose up raises the forward edge of the view (port: right, starboard: left).
        self.view.set_camera(x, c.y, z, (c.heading + self.sgn * 90) % 360,
                             -self.s["exit_look_down"] - self.sgn * c.bank, self.sgn * c.pitch)
        self.view.render(self.world)
        if kind == "heli":
            overlays = [self.art[f"blades{self.blade_i + 1}"], self.art["heli_window"]]
            self.blade_i = (self.blade_i + 1) % 3
        else:
            overlays = self.wing("wing_cessna" if kind == "cessna" else "wing_jet")
        self.presenter.submit(self.view.surface, overlays)
        self.gate_ready = False

    def draw_gate(self):
        now = time.time()
        if self.gate_ready and now - self.gate_drawn_at < GATE_REDRAW_SEC:
            self.presenter.submit_cached(self.note())
            return
        gx, gz = self.gate_xz
        heading = self.s["exit_gate_heading"]
        h = math.radians(heading)
        x = gx + self.sgn * SIDE_M * math.cos(h)
        z = gz - self.sgn * SIDE_M * math.sin(h)
        self._hide_for("jet")
        self.kind = "jet"
        self.view.set_camera(x, gate.WINDOW_HEIGHT_M, z, (heading + self.sgn * 90) % 360, -GATE_LOOK_DOWN, 0.0)
        self.view.render(self.world)
        gate.draw(self.view, gx, gz, heading)
        self.presenter.submit(self.view.surface, self.wing("wing_jet"), self.note(), keep=True)
        self.gate_ready = True
        self.gate_drawn_at = now

    def draw_screens(self):
        """ABOUT and/or the shutdown dialog, over the last window frame."""
        self.presenter.wait()
        canvas = self.about.copy() if self.about is not None else self.presenter.frame.copy()
        if self.power_dialog is not None:
            self.screens.draw_power_dialog(canvas, self.power_dialog)
        self.presenter.show(canvas)
        self.gate_ready = False  # the presenter's cached gate may be gone

    def run(self):
        clock = pygame.time.Clock()
        t_end = time.time() + ARGS.seconds if ARGS.seconds else None
        stats, last_print = {"draw": 0.0, "out": 0.0, "n": 0}, time.time()
        prev = time.time()
        next_snap, snap_n = time.time() + 3, 0
        if ARGS.snap:
            os.makedirs(ARGS.snap, exist_ok=True)
        fps_n, fps_t0 = 0, time.time()
        running = True
        while running and not self.quit_requested:
            for action in self.out.actions():
                if not self.handle(action):
                    running = False
                    break
            if not running:
                break
            now_wall = time.time()
            dt, prev = now_wall - prev, now_wall
            t0 = time.perf_counter()
            # The ride and the traffic keep going under ABOUT, so it comes
            # back to the same moment; only the 3D render stops.
            track = self.update(dt)
            self.update_lighting()
            if self.about is not None or self.power_dialog is not None:
                self.draw_screens()
            elif track is not None:
                self.draw_flight(track)
            else:
                self.draw_gate()
            t1 = time.perf_counter()
            stats["draw"] += t1 - t0
            stats["out"] += self.presenter.ms / 1000
            stats["n"] += 1
            fps_n += 1
            if now_wall - fps_t0 >= 5:
                self.fps, fps_n, fps_t0 = fps_n / (now_wall - fps_t0), 0, now_wall
            if ARGS.stats and now_wall - last_print > 10:
                n = stats["n"]
                info = track.info if track is not None else None
                ride = f"{info.callsign or info.hex} {info.category or '--'}" if info else "gate"
                print(f"{n / (now_wall - last_print):4.1f} fps  draw {1000 * stats['draw'] / n:5.1f}ms  "
                      f"present {1000 * stats['out'] / n:4.1f}ms  {self.side}  riding {ride}", flush=True)
                stats, last_print = {"draw": 0.0, "out": 0.0, "n": 0}, now_wall
            if ARGS.snap and now_wall >= next_snap:
                self.presenter.wait()
                pygame.image.save(self.presenter.frame, os.path.join(ARGS.snap, f"exitrow_{snap_n:03d}.png"))
                snap_n += 1
                next_snap = now_wall + ARGS.snap_every
            if t_end and now_wall > t_end:
                break
            clock.tick(config.TARGET_FPS)
        self.presenter.wait()
        self.out.close()
        pygame.quit()
        if self.power_action == "shutdown":
            subprocess.run(["sudo", "shutdown", "-h", "now"])
        elif self.power_action == "restart":
            subprocess.run(["sudo", "shutdown", "-r", "now"])
        return self.exit_code


def main():
    sys.exit(App().run())


if __name__ == "__main__":
    main()
