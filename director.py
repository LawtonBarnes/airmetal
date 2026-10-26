"""Chooses which aircraft AIR METAL rides.

Each candidate is scored by its closest point of approach (CPA) to the
Metal Shop -- where its current ground track, held straight, passes nearest
the origin, and how soon. Lower score is better:

    CPA distance (NM)
  + altitude / ALT_FT_PER_NM          -- lower aircraft preferred
  + time to CPA beyond SOON_SEC       -- prefer ones arriving soon
  + AWAY_PENALTY if already past CPA  -- heading away from the shop
  + up to IMMINENT_PENALTY if within IMMINENT_SEC of it -- it'd be a short ride
  + RECENT_PENALTY if ridden lately
  + STALE_NM_PER_SEC per second of its longest gap between position
    reports lately, past FRESH_SEC (spotty reception at the edge of the
    antenna's range makes for short rides that drop out)

A ride lasts until the aircraft is past its closest approach and opening
(distance at least PASSED_MARGIN_NM beyond its CPA) or MAX_RIDE_SEC, or its
data goes stale. An inbound ride always runs until it passes; an outbound
one (a fallback pick, nothing inbound around) gives way to a clearly better
candidate once MIN_RIDE_SEC is up.
"""
import math

from geo import KT, M_PER_NM

MAX_PICK_ALT_FT = 25000  # higher than this, there's little to see but haze
ALT_FT_PER_NM = 2000.0  # 2,000 ft of altitude costs as much as passing 1 NM farther away
SOON_SEC = 300.0
LATE_NM_PER_MIN = 0.5  # each minute past SOON_SEC costs this much
AWAY_PENALTY = 50.0
RECENT_PENALTY = 30.0
RECENT_SEC = 600.0
MIN_RIDE_SEC = 45.0
MAX_RIDE_SEC = 300.0
PASSED_MARGIN_NM = 2.0
UPGRADE_MARGIN = 10.0  # a candidate must beat the current ride's score by this much to cut in
SLOW_KT = 30.0  # hovering/slow: treat as passing where it is now
STALE_NM_PER_SEC = 1.0
FRESH_SEC = 3.0  # position reports younger than this cost nothing
IMMINENT_SEC = 30.0  # inbound but this close to its CPA: a short ride, so it costs up to...
IMMINENT_PENALTY = 10.0  # ...this much (pick the next ride early in its approach)


def approach(state):
    """-> (cpa_nm, t_cpa_sec, dist_nm) for a flight.state_at() tuple."""
    x, z, _, vx, vz = state[:5]
    dist = math.hypot(x, z)
    v2 = vx * vx + vz * vz
    if v2 < (SLOW_KT * KT) ** 2:
        return dist / M_PER_NM, 0.0, dist / M_PER_NM
    t = -(x * vx + z * vz) / v2
    cx, cz = x + vx * max(t, 0.0), z + vz * max(t, 0.0)
    return math.hypot(cx, cz) / M_PER_NM, t, dist / M_PER_NM


class Director:
    def __init__(self):
        self.riding = None
        self.ride_start = 0.0
        self.ride_score = 0.0
        self.ride_cpa_nm = None
        self.recent = {}  # hex -> when we stopped riding it

    def score(self, tr, state, now):
        cpa_nm, t_cpa, _ = approach(state)
        alt = tr.info.alt_baro_ft or 0.0
        s = cpa_nm + alt / ALT_FT_PER_NM + max(t_cpa - SOON_SEC, 0.0) / 60.0 * LATE_NM_PER_MIN
        if t_cpa < 0:
            s += AWAY_PENALTY
        elif t_cpa < IMMINENT_SEC:
            s += (IMMINENT_SEC - t_cpa) / IMMINENT_SEC * IMMINENT_PENALTY
        if now - self.recent.get(tr.hex, -1e9) < RECENT_SEC:
            s += RECENT_PENALTY
        s += max(tr.max_gap(now) - FRESH_SEC, 0.0) * STALE_NM_PER_SEC
        return s

    def ranked(self, traffic, now, t_view):
        """[(score, hex)] best first, for every rideable aircraft."""
        out = []
        for tr in traffic.rideable(now):
            st = tr.state_at(t_view)
            if st is None or (tr.info.alt_baro_ft or 0) > MAX_PICK_ALT_FT:
                continue
            out.append((self.score(tr, st, now), tr.hex))
        return sorted(out)

    def update(self, traffic, now, t_view):
        """Decides whether to keep riding or switch. -> the hex to ride, or None."""
        tr = traffic.tracks.get(self.riding) if self.riding else None
        st = tr.state_at(t_view) if tr else None
        if st is not None:
            ridden = now - self.ride_start
            cpa_nm, t_cpa, dist_nm = approach(st)
            passed = t_cpa < 0 and dist_nm > (self.ride_cpa_nm or cpa_nm) + PASSED_MARGIN_NM
            self.ride_cpa_nm = min(self.ride_cpa_nm or dist_nm, dist_nm)
            if ridden < MIN_RIDE_SEC:
                return self.riding
            if not passed and ridden < MAX_RIDE_SEC:
                if t_cpa >= 0:
                    return self.riding  # inbound: ride it until it passes
                # Outbound (a fallback pick): keep it unless something clearly better is around.
                best = self.ranked(traffic, now, t_view)
                if not best or best[0][1] == self.riding or best[0][0] > self.score(tr, st, now) - UPGRADE_MARGIN:
                    return self.riding
        best = [h for _, h in self.ranked(traffic, now, t_view) if h != self.riding] or \
               [h for _, h in self.ranked(traffic, now, t_view)]
        self._ride(best[0] if best else None, traffic, now, t_view)
        return self.riding

    def step(self, traffic, now, t_view, direction):
        """Remote Left/Right: the previous/next aircraft in score order."""
        hexes = [h for _, h in self.ranked(traffic, now, t_view)]
        if not hexes:
            return self.riding
        i = hexes.index(self.riding) + direction if self.riding in hexes else 0
        self._ride(hexes[i % len(hexes)], traffic, now, t_view)
        return self.riding

    def _ride(self, hex_, traffic, now, t_view):
        if hex_ == self.riding:
            return
        if self.riding:
            self.recent[self.riding] = now
        for h in [h for h, t in self.recent.items() if now - t > RECENT_SEC]:
            del self.recent[h]
        self.riding = hex_
        self.ride_start = now
        self.ride_cpa_nm = None
        if hex_:
            tr = traffic.tracks[hex_]
            st = tr.state_at(t_view)
            self.ride_score = self.score(tr, st, now) if st else 0.0
