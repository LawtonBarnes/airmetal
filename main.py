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
import sys
import time

import config


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
    return ap.parse_args()


ARGS = parse_args()
if ARGS.headless:
    ARGS.window = True
if not ARGS.window or ARGS.headless:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame  # noqa: E402

import feed as feedmod  # noqa: E402
import flight  # noqa: E402
import output  # noqa: E402
import panel as panelmod  # noqa: E402
import render3d  # noqa: E402
import scenery  # noqa: E402
from geo import M_PER_NM  # noqa: E402

# Standby, and the scenery check against the drone photos: 400 ft over the
# Metal Shop, panning slowly from White Rock Lake toward downtown.
STANDBY_CAM = {"alt_m": 122.0, "pitch": -6.0, "heading0": 252.0, "pan_deg": 25.0, "pan_sec": 60.0}
HANDOFF_SEC = 6.0
MAX_PICK_ALT_FT = 25000  # higher than this, there's not much to see but haze


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
        self.world = scenery.Scenery(config.DATA_DIR, self.s["lat"], self.s["lon"])
        self.view = render3d.View(config.VIEW_LOW_W, config.VIEW_LOW_H, self.s["fov_deg"])
        self.view_big = pygame.Surface((config.FRAME_W, config.VIEW_H))
        self.panel = panelmod.Panel()
        if ARGS.replay:
            self.feed = feedmod.ReplayFeed(ARGS.replay, ARGS.speed, ARGS.offset)
        else:
            self.feed = feedmod.LiveFeed(self.s["feed_url"], self.s["poll_sec"])
        self.traffic = flight.Traffic(self.s["lat"], self.s["lon"])
        self.cam = flight.Camera(self.s["ground_msl_m"])
        self.riding = None  # hex of the aircraft being ridden
        self.ride_start = 0.0
        self.recent = {}  # hex -> when we last stopped riding it
        self.handoff_until = 0.0
        self.f_card_big = pygame.font.Font(config.FONT_PATH, 30)
        self.f_card = pygame.font.Font(config.FONT_PATH, 20)
        self.last_report_wall = time.time()

    def _on_signal(self, signum, frame):
        self.quit_requested = True

    # -- choosing an aircraft ----------------------------------------------

    def candidates(self, now):
        t = now - self.s["delay_sec"]
        out = []
        for tr in self.traffic.rideable(now):
            st = tr.state_at(t)
            if st is None or tr.info.alt_baro_ft is not None and tr.info.alt_baro_ft > MAX_PICK_ALT_FT:
                continue
            dist = math.hypot(st[0], st[1])
            out.append((dist + 20000 * (tr.hex in self.recent), tr.hex))
        return sorted(out)

    def pick(self, now, step=0):
        """Ride the best candidate (step=0), or the next/previous one in
        that order relative to the current one (remote Left/Right)."""
        cands = [h for _, h in self.candidates(now)]
        if not cands:
            self._ride(None, now)
            return
        if step and self.riding in cands:
            i = (cands.index(self.riding) + step) % len(cands)
            self._ride(cands[i], now)
        else:
            self._ride(cands[0], now)

    def _ride(self, hex_, now):
        if hex_ == self.riding:
            return
        if self.riding:
            self.recent[self.riding] = now
        self.riding = hex_
        self.ride_start = now
        self.cam.reset()
        if hex_:
            self.handoff_until = time.time() + HANDOFF_SEC

    # -- frame ---------------------------------------------------------------

    def update(self, dt):
        now = self.feed.clock()
        reports = self.feed.poll()
        if reports:
            self.last_report_wall = time.time()
        self.traffic.update(reports, now)
        for h in [h for h, t in self.recent.items() if t < now - 600]:
            del self.recent[h]
        track = self.traffic.tracks.get(self.riding) if self.riding else None
        state = track.state_at(now - self.s["delay_sec"]) if track else None
        if state is None or now - self.ride_start > self.s["ride_sec"]:
            self.pick(now)
            track = self.traffic.tracks.get(self.riding) if self.riding else None
            state = track.state_at(now - self.s["delay_sec"]) if track else None
        if state is not None:
            self.cam.follow(state, dt)
            return track
        # Standby: hover over the Metal Shop, panning.
        c = self.cam
        c.reset()
        c.x = c.z = 0.0
        c.y = STANDBY_CAM["alt_m"]
        c.heading = STANDBY_CAM["heading0"] + STANDBY_CAM["pan_deg"] * math.sin(
            2 * math.pi * time.time() / STANDBY_CAM["pan_sec"])
        c.pitch, c.bank, c.turn_rate, c.gs_kt, c.vs_fpm = STANDBY_CAM["pitch"], 0.0, 0.0, 0.0, 0.0
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
        while not self.quit_requested and self.out.handle_input(self):
            now_wall = time.time()
            dt, prev = now_wall - prev, now_wall
            t0 = time.perf_counter()
            track = self.update(dt)
            c = self.cam
            self.view.set_camera(c.x, c.y, c.z, c.heading, c.pitch, c.bank)
            self.view.render(self.world)
            pygame.transform.scale(self.view.surface, (config.FRAME_W, config.VIEW_H), self.view_big)
            if track is not None and time.time() < self.handoff_until:
                self.draw_card(self.view_big, track)
            elif track is None:
                self.draw_standby(self.view_big)
            t1 = time.perf_counter()
            dist_nm = math.hypot(c.x, c.z) / M_PER_NM
            ok = time.time() - self.last_report_wall < 30 and not getattr(self.feed, "error", None)
            panel_surf = self.panel.draw(c, track.info if track else None, dist_nm, ok)
            t2 = time.perf_counter()
            self.out.write(self.view_big, panel_surf)
            t3 = time.perf_counter()
            stats["view"] += t1 - t0
            stats["panel"] += t2 - t1
            stats["out"] += t3 - t2
            stats["n"] += 1
            if ARGS.stats and now_wall - last_print > 10:
                n = stats["n"]
                print(f"{n / (now_wall - last_print):4.1f} fps  view {1000 * stats['view'] / n:5.1f}ms  "
                      f"panel {1000 * stats['panel'] / n:5.1f}ms  out {1000 * stats['out'] / n:5.1f}ms  "
                      f"riding {self.riding} ({len(self.traffic.tracks)} tracked)", flush=True)
                stats, last_print = {"view": 0.0, "panel": 0.0, "out": 0.0, "n": 0}, now_wall
            if ARGS.snap and now_wall >= next_snap:
                frame = pygame.Surface((config.FRAME_W, config.FRAME_H))
                frame.blit(self.view_big, (0, 0))
                frame.blit(panel_surf, (0, config.VIEW_H))
                pygame.image.save(frame, os.path.join(ARGS.snap, f"frame_{snap_n:03d}.png"))
                snap_n += 1
                next_snap = now_wall + ARGS.snap_every
            if t_end and now_wall > t_end:
                break
            clock.tick(config.TARGET_FPS)
        self.out.close()
        return self.out.exit_code


def main():
    sys.exit(App().run())


if __name__ == "__main__":
    main()
