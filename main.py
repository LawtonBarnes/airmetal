#!/usr/bin/env python3
"""AIR METAL -- a retro flight sim on the 2x2 CRT wall that rides real
aircraft, live from P4's ADS-B receiver (the one JOAN JET shows).

Top two tubes: the windshield, a flat-shaded 1990-style view from the
aircraft being ridden. Bottom two: an analog six-pack panel. One XGA frame;
the OREI splits it across the wall.

Usage:
  main.py                         Pi: live feed, straight to /dev/fb0
  main.py --window                PC: in a window (dev)
  main.py --replay FILE [--speed N] [--offset SEC]   ride a recording
"""
import argparse
import math
import os
import signal
import subprocess
import sys
import time

import config
import daylight


def parse_args():
    ap = argparse.ArgumentParser(description="AIR METAL")
    ap.add_argument("--window", action="store_true", help="draw in a desktop window instead of /dev/fb0")
    ap.add_argument("--replay", help="ride a recorded feed (tools/record_feed.py output)")
    ap.add_argument("--speed", type=float, default=1.0, help="replay speed")
    ap.add_argument("--offset", type=float, default=0.0, help="start this many seconds into the replay")
    ap.add_argument("--seconds", type=float, default=0.0, help="quit after this long (testing)")
    ap.add_argument("--stats", action="store_true", help="print frame timings every 10s")
    ap.add_argument("--headless", action="store_true", help="window mode with no display (testing; use --snap)")
    ap.add_argument("--snap", help="save a frame to this directory every --snap-every seconds")
    ap.add_argument("--snap-every", type=float, default=10.0)
    ap.add_argument("--clock", help="light the scene as at this local time, HH:MM (overrides settings.ini)")
    return ap.parse_args()


ARGS = parse_args()
if ARGS.headless:
    ARGS.window = True
if not ARGS.window or ARGS.headless:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame  # noqa: E402

import director as directormod  # noqa: E402
import feed as feedmod  # noqa: E402
import flight  # noqa: E402
import link  # noqa: E402
import output  # noqa: E402
import panel as panelmod  # noqa: E402
import render3d  # noqa: E402
import scenery  # noqa: E402
import screens  # noqa: E402
from geo import M_PER_NM  # noqa: E402

# Standby, and the scenery check against the drone photos: 400 ft over the
# Metal Shop, panning slowly from White Rock Lake toward downtown.
STANDBY_CAM = {"alt_m": 122.0, "pitch": -6.0, "heading0": 252.0, "pan_deg": 25.0, "pan_sec": 60.0}
HANDOFF_SEC = 6.0
LIGHTING_SEC = 10.0  # how often the sun is recomputed
EXIT_GOTO_HOME = 42  # see STRINGS/SCRUTE: Home hands off to the app menu


class App:
    def __init__(self):
        self.s = config.load()
        pygame.init()
        # After pygame.init(), which otherwise turns SIGTERM (how STRINGS stops
        # apps) into a QUIT event that nothing in fb mode ever reads.
        self.quit_requested = False
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, self._on_signal)
        self.out = output.open_output(ARGS.window)
        # Splash first, so it's up while the scenery loads.
        splash = screens.build_splash(config.SPLASH_PATH, config.VERSION)
        self.out.write(splash.subsurface((0, 0, config.FRAME_W, config.VIEW_H)),
                       splash.subsurface((0, config.VIEW_H, config.FRAME_W, config.VIEW_H)))
        splash_until = time.time() + screens.SPLASH_SECONDS
        self.screens = screens.Screens(config.LOGO_PATH)
        self.about = None  # the ABOUT windshield, built once per BACK press
        self.power_dialog = None  # selected index into screens.POWER_OPTIONS while it's up
        self.power_action = None  # "shutdown" or "restart", run after the display is released
        self.exit_code = 0
        self.fps = 0.0
        self.world = scenery.Scenery(config.DATA_DIR, self.s["lat"], self.s["lon"])
        render3d.set_saturation(self.s["saturation"], self.world)
        self.view = render3d.View(config.VIEW_LOW_W, config.VIEW_LOW_H, self.s["fov_deg"],
                                  self.world, self.s["ground_scale"], self.s["ground_filter"],
                                  self.s["trees"])
        self.view_big = pygame.Surface((config.FRAME_W, config.VIEW_H))
        self.clock = ARGS.clock or self.s["clock"]
        self.next_lighting = 0.0
        self.update_lighting()
        self.panel = panelmod.Panel()
        if ARGS.replay:
            self.feed = feedmod.ReplayFeed(ARGS.replay, ARGS.speed, ARGS.offset)
        else:
            self.feed = feedmod.LiveFeed(self.s["feed_url"], self.s["poll_sec"])
        self.traffic = flight.Traffic(self.s["lat"], self.s["lon"])
        self.cam = flight.Camera(self.s["ground_msl_m"])
        self.director = directormod.Director()
        self.link = link.RideServer(self.s["link_port"])
        self.riding = None  # hex of the aircraft being ridden
        self.handoff_until = 0.0
        self.f_card_big = pygame.font.Font(config.FONT_PATH, 30)
        self.f_card = pygame.font.Font(config.FONT_PATH, 20)
        self.last_report_wall = time.time()
        while time.time() < splash_until and not self.quit_requested:
            self.out.actions()  # keys during the splash are dropped, not queued
            time.sleep(0.05)

    def _on_signal(self, signum, frame):
        self.quit_requested = True

    # -- remote ----------------------------------------------------------------

    def handle(self, action):
        """One remote action. Returns False to quit. The fleet key map: Home
        back to SCRUTE, Q/Esc/Menu quit, BACK toggles ABOUT, Power opens the
        shutdown dialog; Left/Right cycle the nearest aircraft."""
        if self.power_dialog is not None:
            return self._handle_power_dialog(action)
        if action == "home":
            self.exit_code = EXIT_GOTO_HOME
            return False
        if action == "quit":
            return False
        if action == "back":
            self.about = None if self.about is not None else self.screens.build_about(
                f"AIR METAL {config.VERSION}", self.about_rows())
        elif action == "power":
            self.power_dialog = 0
        elif self.about is not None:
            pass  # only the keys above act while ABOUT is up
        elif action in ("left", "right"):
            self.pick(self.feed.clock(), +1 if action == "right" else -1)
        return True

    def _handle_power_dialog(self, action):
        n = len(screens.POWER_OPTIONS)
        if action in ("left", "up"):
            self.power_dialog = (self.power_dialog - 1) % n
        elif action in ("right", "down"):
            self.power_dialog = (self.power_dialog + 1) % n
        elif action == "enter":
            choice = screens.POWER_OPTIONS[self.power_dialog]
            if choice != "NO":
                self.power_action = "shutdown" if choice == "YES" else "restart"
                return False
            self.power_dialog = None
        elif action in ("quit", "back", "power"):
            self.power_dialog = None
        return True

    def about_rows(self):
        if ARGS.replay:
            feed = ("FEED", "REPLAY")
        else:
            host = self.s["feed_url"].split("//", 1)[-1].split("/", 1)[0]
            feed = ("FEED", f"READSB {host}")
        status = "NO DATA" if getattr(self.feed, "error", None) else "ONLINE"
        track = self.traffic.tracks.get(self.riding) if self.riding else None
        riding = (track.info.callsign or track.info.hex.upper()) if track else "STANDBY"
        return [
            feed,
            ("STATUS", status),
            ("TRACKING", f"{len(self.traffic.tracks)} AIRCRAFT"),
            ("RIDING", riding),
            ("HOME", self.s["location_name"].upper()),
            ("DELAY", f"{self.s['delay_sec']:g} SEC"),
            ("FOV", f"{self.s['fov_deg']:g} DEG"),
            ("FRAME RATE", f"{self.fps:.1f} FPS" if self.fps else "--"),
        ]

    # -- choosing an aircraft ----------------------------------------------

    def pick(self, now, step):
        """Remote Left/Right: previous/next of the nearest aircraft (JOAN JET's NEARBY)."""
        self._switched_to(self.director.step(self.traffic, now, now - self.s["delay_sec"], step))

    def _switched_to(self, hex_):
        if hex_ == self.riding:
            return
        self.riding = hex_
        self.cam.reset()
        if hex_:
            self.handoff_until = time.time() + HANDOFF_SEC

    # -- frame ---------------------------------------------------------------

    def update_lighting(self):
        """The sun for the real time (or the fixed clock setting), into the view."""
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
        self.view.set_lighting(daylight.lighting_at(elev, az), self.world)

    def update(self, dt):
        now = self.feed.clock()
        reports = self.feed.poll()
        if reports:
            self.last_report_wall = time.time()
        self.traffic.update(reports, now)
        t_view = now - self.s["delay_sec"]
        self._switched_to(self.director.update(self.traffic, now, t_view))
        track = self.traffic.tracks.get(self.riding) if self.riding else None
        state = track.state_at(t_view) if track else None
        if state is not None:
            self.link.riding = (track.info.hex, track.info.callsign)
            self.cam.follow(state, dt)
            return track
        self.link.riding = None
        # Standby: hover over the Metal Shop, panning.
        c = self.cam
        c.reset()
        c.x = c.z = 0.0
        c.y = STANDBY_CAM["alt_m"]
        c.heading = STANDBY_CAM["heading0"] + STANDBY_CAM["pan_deg"] * math.sin(
            2 * math.pi * time.time() / STANDBY_CAM["pan_sec"])
        c.pitch, c.bank, c.turn_rate, c.gs_kt, c.vs_fpm = STANDBY_CAM["pitch"], 0.0, 0.0, 0.0, 0.0
        c.view_pitch = c.pitch
        c.msl_ft = (self.s["ground_msl_m"] + c.y) / 0.3048
        return None

    def draw_card(self, surf, track):
        """The handoff card: who we're riding now, top-left of the windshield's safe area."""
        info = track.info
        lines = [(self.f_card_big, info.callsign or info.hex.upper(), panelmod.YELLOW),
                 (self.f_card, panelmod.CATEGORY_NAMES.get(info.category, "AIRCRAFT"), panelmod.WHITE),
                 (self.f_card, f"{self.cam.msl_ft:,.0f} FT  {self.cam.gs_kt:.0f} KTS", panelmod.WHITE),
                 (self.f_card, f"{math.hypot(self.cam.x, self.cam.z) / M_PER_NM:.1f} NM FROM SHOP", panelmod.WHITE)]
        imgs = [(f.render(t, False, c)) for f, t, c in lines]
        w = max(i.get_width() for i in imgs) + 28
        h = sum(i.get_height() + 4 for i in imgs) + 20
        box = pygame.Rect(64, 50, w, h)
        pygame.draw.rect(surf, (0, 0, 0), box)
        pygame.draw.rect(surf, panelmod.ORANGE, box, 3)
        y = box.y + 10
        for img in imgs:
            surf.blit(img, (box.x + 14, y))
            y += img.get_height() + 4

    def draw_standby(self, surf):
        msg = "STANDBY - WAITING FOR TRAFFIC" if not getattr(self.feed, "error", None) else "NO ADS-B DATA"
        img = self.f_card.render(msg, False, panelmod.YELLOW)
        box = img.get_rect(midbottom=(config.FRAME_W // 2, config.VIEW_H - 50)).inflate(28, 16)
        pygame.draw.rect(surf, (0, 0, 0), box)
        pygame.draw.rect(surf, panelmod.ORANGE, box, 3)
        surf.blit(img, img.get_rect(center=box.center))

    def run(self):
        clock = pygame.time.Clock()
        t_end = time.time() + ARGS.seconds if ARGS.seconds else None
        stats, last_print = {"view": 0.0, "panel": 0.0, "out": 0.0, "n": 0}, time.time()
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
            # Traffic, the director and the camera keep going under ABOUT,
            # so it comes back to the same ride; only the 3D render stops.
            track = self.update(dt)
            self.update_lighting()
            c = self.cam
            if self.about is not None:
                top = self.about.copy() if self.power_dialog is not None else self.about
            else:
                self.view.set_camera(c.x, c.y, c.z, c.heading, c.view_pitch, c.bank)
                self.view.render(self.world)
                pygame.transform.scale(self.view.surface, (config.FRAME_W, config.VIEW_H), self.view_big)
                if track is not None and time.time() < self.handoff_until:
                    self.draw_card(self.view_big, track)
                elif track is None:
                    self.draw_standby(self.view_big)
                top = self.view_big
            if self.power_dialog is not None:
                self.screens.draw_power_dialog(top, self.power_dialog)
            t1 = time.perf_counter()
            dist_nm = math.hypot(c.x, c.z) / M_PER_NM
            ok = time.time() - self.last_report_wall < 30 and not getattr(self.feed, "error", None)
            panel_surf = self.panel.draw(c, track.info if track else None, dist_nm, ok)
            t2 = time.perf_counter()
            self.out.write(top, panel_surf)
            t3 = time.perf_counter()
            stats["view"] += t1 - t0
            stats["panel"] += t2 - t1
            stats["out"] += t3 - t2
            stats["n"] += 1
            fps_n += 1
            if now_wall - fps_t0 >= 5:
                self.fps, fps_n, fps_t0 = fps_n / (now_wall - fps_t0), 0, now_wall
            if ARGS.stats and now_wall - last_print > 10:
                n = stats["n"]
                print(f"{n / (now_wall - last_print):4.1f} fps  view {1000 * stats['view'] / n:5.1f}ms  "
                      f"panel {1000 * stats['panel'] / n:5.1f}ms  out {1000 * stats['out'] / n:5.1f}ms  "
                      f"riding {self.riding} ({len(self.traffic.tracks)} tracked)", flush=True)
                stats, last_print = {"view": 0.0, "panel": 0.0, "out": 0.0, "n": 0}, now_wall
            if ARGS.snap and now_wall >= next_snap:
                frame = pygame.Surface((config.FRAME_W, config.FRAME_H))
                frame.blit(top, (0, 0))
                frame.blit(panel_surf, (0, config.VIEW_H))
                pygame.image.save(frame, os.path.join(ARGS.snap, f"frame_{snap_n:03d}.png"))
                snap_n += 1
                next_snap = now_wall + ARGS.snap_every
            if t_end and now_wall > t_end:
                break
            clock.tick(config.TARGET_FPS)
        self.link.close()
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
