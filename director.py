"""Chooses which aircraft AIR METAL rides.

Only aircraft within RANGE_NM of the Metal Shop are candidates -- farther
out, JOAN JET's overlay covers them, and the windshield shows mostly
distant haze. They're ranked by kind first, then distance. Lower score is
better:

    TIER_NM * tier                     -- 0 helicopters, 1 small or low, 2 the rest
  + distance from the shop (NM)
  + AWAY_PENALTY if heading away from the shop
  + RECENT_PENALTY if ridden lately (about a tier's worth)
  + STALE_NM_PER_SEC per second of its longest gap between position
    reports lately, past FRESH_SEC (spotty reception makes for short rides)

A ride lasts until the aircraft is out of range and heading away, its
signal is gone (flight.MAX_EXTRAPOLATE_SEC of dead reckoning), or
MAX_RIDE_SEC is up. After MIN_RIDE_SEC, a candidate of a better tier cuts
in. A ride picked from the remote isn't cut in on.

Remote Left/Right cycle the NEARBY_COUNT nearest aircraft -- the same list
as JOAN JET's NEARBY line (the nearest, then the four after it).
"""
import math

from geo import KT, M_PER_NM

RANGE_NM = 9.0
NEARBY_COUNT = 5
MAX_PICK_ALT_FT = 25000  # higher than this, there's little to see but haze
TIER_NM = 10.0  # one tier outranks anything a tier below, anywhere in range
HELICOPTER = {"A7"}
SMALL = {"A1", "B1", "B4", "B6"}  # light (< 15,500 lb), glider, ultralight, drone
LOW_ALT_FT = 3000  # below this (MSL), any aircraft counts as low
AWAY_PENALTY = 3.0
RECENT_PENALTY = 10.0
RECENT_SEC = 600.0
MIN_RIDE_SEC = 45.0
MAX_RIDE_SEC = 300.0
SLOW_KT = 30.0  # hovering/slow: neither inbound nor outbound
STALE_NM_PER_SEC = 1.0
FRESH_SEC = 3.0  # position reports younger than this cost nothing


def approach(state):
    """-> (cpa_nm, t_cpa_sec, dist_nm) for a flight.state_at() tuple: where
    its ground track, held straight, passes nearest the shop, and when."""
    x, z, _, vx, vz = state[:5]
    dist = math.hypot(x, z)
    v2 = vx * vx + vz * vz
    if v2 < (SLOW_KT * KT) ** 2:
        return dist / M_PER_NM, 0.0, dist / M_PER_NM
    t = -(x * vx + z * vz) / v2
    cx, cz = x + vx * max(t, 0.0), z + vz * max(t, 0.0)
    return math.hypot(cx, cz) / M_PER_NM, t, dist / M_PER_NM


def tier(info):
    if info.category in HELICOPTER:
        return 0
    if info.category in SMALL or (info.alt_baro_ft is not None and info.alt_baro_ft < LOW_ALT_FT):
        return 1
    return 2


class Director:
    def __init__(self):
        self.riding = None
        self.ride_start = 0.0
        self.manual = False  # picked from the remote
        self.recent = {}  # hex -> when we stopped riding it

    def score(self, tr, state, now):
        _, t_cpa, dist_nm = approach(state)
        s = TIER_NM * tier(tr.info) + dist_nm
        if t_cpa < 0:
            s += AWAY_PENALTY
        if now - self.recent.get(tr.hex, -1e9) < RECENT_SEC:
            s += RECENT_PENALTY
        s += max(tr.max_gap(now) - FRESH_SEC, 0.0) * STALE_NM_PER_SEC
        return s

    def ranked(self, traffic, now, t_view):
        """[(score, tier, hex)] best first, for every aircraft in range."""
        out = []
        for tr in traffic.rideable(now):
            st = tr.state_at(t_view)
            if st is None or (tr.info.alt_baro_ft or 0) > MAX_PICK_ALT_FT:
                continue
            if math.hypot(st[0], st[1]) / M_PER_NM > RANGE_NM:
                continue
            out.append((self.score(tr, st, now), tier(tr.info), tr.hex))
        return sorted(out)

    def nearby(self, traffic, now, t_view):
        """The NEARBY_COUNT nearest airborne aircraft, nearest first."""
        out = []
        for tr in traffic.rideable(now):
            st = tr.state_at(t_view)
            if st is not None:
                out.append((math.hypot(st[0], st[1]), tr.hex))
        return [h for _, h in sorted(out)[:NEARBY_COUNT]]

    def update(self, traffic, now, t_view):
        """Decides whether to keep riding or switch. -> the hex to ride, or None."""
        tr = traffic.tracks.get(self.riding) if self.riding else None
        st = tr.state_at(t_view) if tr else None
        if st is None:
            # Signal gone (or riding nothing): don't count it as ridden, so
            # it can be picked again if it comes back.
            best = self.ranked(traffic, now, t_view)
            self._ride(best[0][2] if best else None, now, ridden=False)
            return self.riding
        ridden = now - self.ride_start
        if ridden < MIN_RIDE_SEC:
            return self.riding
        _, t_cpa, dist_nm = approach(st)
        best = self.ranked(traffic, now, t_view)
        if (dist_nm > RANGE_NM and t_cpa < 0) or ridden > MAX_RIDE_SEC:
            others = [h for _, _, h in best if h != self.riding] or [h for _, _, h in best]
            self._ride(others[0] if others else None, now)
        elif not self.manual and best and best[0][1] < tier(tr.info):
            self._ride(best[0][2], now)
        return self.riding

    def step(self, traffic, now, t_view, direction):
        """Remote Left/Right: the next/previous of the nearest aircraft."""
        hexes = self.nearby(traffic, now, t_view)
        if not hexes:
            return self.riding
        if self.riding in hexes:
            i = hexes.index(self.riding) + direction
        else:
            i = 0 if direction > 0 else -1
        self._ride(hexes[i % len(hexes)], now)
        self.manual = True
        return self.riding

    def _ride(self, hex_, now, ridden=True):
        if self.riding and ridden:
            self.recent[self.riding] = now
        for h in [h for h, t in self.recent.items() if now - t > RECENT_SEC]:
            del self.recent[h]
        self.riding = hex_
        self.ride_start = now
        self.manual = False
