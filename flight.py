"""Turns sparse ADS-B position reports (about one a second, sometimes with
gaps) into a smooth flight the camera can ride.

The camera flies a few seconds behind live (config delay_sec), so almost
always there's a report on each side of the moment being drawn and the
path can be *interpolated* -- a Hermite curve through the two positions,
using each report's own ground speed/track as its tangent, so turns come
out as curves rather than corners. Past the newest report it falls back to
dead reckoning for a little while before calling the track stale.

ADS-B carries no attitude, so pitch and bank are estimated: pitch from
climb rate over ground speed (the flight-path angle), bank from how fast
the heading is changing (a coordinated turn: tan(bank) = v * turn_rate / g).
"""
import math
from collections import deque

import feed as feedmod
from geo import FPM, KT, Projection, wrap_deg

MAX_EXTRAPOLATE_SEC = 12.0  # dead-reckon past the newest report at most this long
STALE_SEC = 20.0  # no position for this long: the track can't be ridden
FORGET_SEC = 90.0  # ...and this long: dropped entirely
HISTORY_SEC = 60.0
G = 9.81
MAX_BANK_DEG = 30.0
CAMERA_BLEND = 0.25  # per-frame smoothing of heading/pitch (at 10fps)
TURN_BASELINE_SEC = 3.0  # turn rate is measured across reports at least this far apart --
# readsb's track is quantized (~0.35 deg), so over a 1s gap the noise alone reads as ~1 deg/s
TURN_TAU_SEC = 2.0  # smoothing time constant for the turn rate the bank comes from
BANK_TAU_SEC = 1.0  # ...and for the bank itself


class Track:
    def __init__(self, hex_):
        self.hex = hex_
        self.reports = deque()  # (t, x, z, msl_m, vx, vz, vy, track_deg, turn_dps) -- velocities in m/s
        self.info = None  # latest feed.Report, for callsign/category/etc.

    def add(self, r, proj, msl_m, prev_msl):
        x, z = proj.to_xz(r.lat, r.lon)
        if r.gs_kt is not None and r.track_deg is not None:
            v = r.gs_kt * KT
            vx, vz = v * math.sin(math.radians(r.track_deg)), v * math.cos(math.radians(r.track_deg))
        elif self.reports:
            t0, x0, z0 = self.reports[-1][:3]
            dt = max(r.t - t0, 0.1)
            vx, vz = (x - x0) / dt, (z - z0) / dt
        else:
            vx = vz = 0.0
        vy = r.rate_fpm * FPM if r.rate_fpm is not None else 0.0
        if msl_m is None:
            msl_m = prev_msl
        if msl_m is None:
            return
        track = r.track_deg if r.track_deg is not None else math.degrees(math.atan2(vx, vz)) % 360
        # Turn rate: change in reported track since a report at least TURN_BASELINE_SEC old.
        turn = 0.0
        for old in reversed(self.reports):
            if r.t - old[0] >= TURN_BASELINE_SEC:
                if r.t - old[0] < 4 * TURN_BASELINE_SEC:
                    turn = wrap_deg(track - old[7]) / (r.t - old[0])
                break
        self.reports.append((r.t, x, z, msl_m, vx, vz, vy, track, turn))
        # Not every message carries callsign/category/squawk -- keep the last ones heard.
        if self.info is not None:
            r.callsign = r.callsign or self.info.callsign
            r.category = r.category or self.info.category
            r.squawk = r.squawk or self.info.squawk
            if r.rssi is None:
                r.rssi = self.info.rssi
        self.info = r
        while self.reports and self.reports[0][0] < r.t - HISTORY_SEC:
            self.reports.popleft()

    @property
    def last_t(self):
        return self.reports[-1][0] if self.reports else 0.0

    def state_at(self, t):
        """-> (x, z, msl_m, vx, vz, vy, track_deg, turn_dps) at time t, or
        None if t is too far past the newest report."""
        rs = self.reports
        if not rs:
            return None
        if t <= rs[0][0]:
            return rs[0][1:]
        if t >= rs[-1][0]:
            t0, x, z, y, vx, vz, vy, track, turn = rs[-1]
            dt = t - t0
            if dt > MAX_EXTRAPOLATE_SEC:
                return None
            return x + vx * dt, z + vz * dt, y + vy * dt, vx, vz, vy, track, turn
        # Bracketing pair (reports are few, a linear scan from the end is fine).
        i = len(rs) - 1
        while rs[i - 1][0] > t:
            i -= 1
        a, b = rs[i - 1], rs[i]
        h = b[0] - a[0]
        s = (t - a[0]) / h
        # Cubic Hermite on x and z, tangents = velocity * interval.
        h00 = 2 * s**3 - 3 * s**2 + 1
        h10 = s**3 - 2 * s**2 + s
        h01 = -2 * s**3 + 3 * s**2
        h11 = s**3 - s**2
        d00 = 6 * s**2 - 6 * s
        d10 = 3 * s**2 - 4 * s + 1
        d01 = -6 * s**2 + 6 * s
        d11 = 3 * s**2 - 2 * s
        x = h00 * a[1] + h10 * h * a[4] + h01 * b[1] + h11 * h * b[4]
        z = h00 * a[2] + h10 * h * a[5] + h01 * b[2] + h11 * h * b[5]
        vx = (d00 * a[1] + d10 * h * a[4] + d01 * b[1] + d11 * h * b[4]) / h
        vz = (d00 * a[2] + d10 * h * a[5] + d01 * b[2] + d11 * h * b[5]) / h
        y = a[3] + (b[3] - a[3]) * s
        vy = a[6] + (b[6] - a[6]) * s
        track = (a[7] + wrap_deg(b[7] - a[7]) * s) % 360
        turn = a[8] + (b[8] - a[8]) * s
        return x, z, y, vx, vz, vy, track, turn


class Traffic:
    """Every aircraft heard recently, plus the day's barometric correction."""

    def __init__(self, lat0, lon0):
        self.proj = Projection(lat0, lon0)
        self.tracks = {}
        self.baro_offset_ft = 0.0  # geom - baro, smoothed over many aircraft

    def update(self, reports, now):
        diffs = [r.alt_geom_ft - r.alt_baro_ft for r in reports
                 if r.alt_geom_ft is not None and r.alt_baro_ft is not None
                 and not r.on_ground and r.alt_baro_ft < 20000]
        if diffs:
            diffs.sort()
            self.baro_offset_ft += (diffs[len(diffs) // 2] - self.baro_offset_ft) * 0.2
        for r in reports:
            tr = self.tracks.setdefault(r.hex, Track(r.hex))
            prev = tr.reports[-1][3] if tr.reports else None
            tr.add(r, self.proj, feedmod.msl_m(r, self.baro_offset_ft), prev)
        for hex_ in [h for h, tr in self.tracks.items() if tr.last_t < now - FORGET_SEC]:
            del self.tracks[hex_]

    def rideable(self, now):
        """Tracks with a fresh position, airborne."""
        return [tr for tr in self.tracks.values()
                if tr.reports and tr.last_t > now - STALE_SEC and tr.info and not tr.info.on_ground]


class Camera:
    """The smoothed pose the renderer draws from: position (m, y above the
    ground plane), heading/pitch/bank in degrees, plus readouts for the panel."""

    def __init__(self, ground_msl_m):
        self.ground_msl_m = ground_msl_m
        self.x = self.z = 0.0
        self.y = 500.0
        self.heading = self.pitch = self.bank = 0.0
        self.turn_rate = 0.0  # deg/s, smoothed
        self.gs_kt = self.vs_fpm = 0.0
        self.msl_ft = 0.0
        self._fresh = True

    def reset(self):
        self._fresh = True

    def follow(self, state, dt):
        """Heading follows the reported track (interpolated between
        reports), not the curve's own direction -- that kinks slightly at
        every report and read as the wing rocking once a second."""
        x, z, msl, vx, vz, vy, heading, turn = state
        speed = math.hypot(vx, vz)
        pitch = math.degrees(math.atan2(vy, max(speed, 20.0)))
        fresh = self._fresh
        self._fresh = False
        k = 1.0 if fresh else CAMERA_BLEND
        k_turn = 1.0 if fresh else 1 - math.exp(-dt / TURN_TAU_SEC)
        k_bank = 1.0 if fresh else 1 - math.exp(-dt / BANK_TAU_SEC)
        self.turn_rate += (turn - self.turn_rate) * k_turn
        bank = math.degrees(math.atan(speed * math.radians(self.turn_rate) / G))
        bank = max(-MAX_BANK_DEG, min(MAX_BANK_DEG, bank))
        self.x, self.z = x, z
        self.y = max(msl - self.ground_msl_m, 8.0)
        self.heading = (self.heading + wrap_deg(heading - self.heading) * k) % 360
        self.pitch += (pitch - self.pitch) * k
        self.bank += (bank - self.bank) * k_bank
        self.gs_kt = speed / KT
        self.vs_fpm = vy / FPM
        self.msl_ft = msl / 0.3048
