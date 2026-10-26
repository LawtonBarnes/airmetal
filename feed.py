"""ADS-B traffic: P4's readsb aircraft.json, live over HTTP or replayed from a
recording (tools/record_feed.py's gzipped JSON lines).

Either way the app sees the same thing: Feed.poll() returns a list of
Report -- one per aircraft whose *position* is new since the last call,
time-stamped with when that position was actually received (readsb's
`now - seen_pos`), not when we happened to fetch the file.
"""
import gzip
import json
import threading
import time
import urllib.request
from dataclasses import dataclass

from geo import FT


@dataclass
class Report:
    hex: str
    t: float  # unix time the position was received
    lat: float
    lon: float
    alt_baro_ft: float | None  # None if unknown; 0 with on_ground
    alt_geom_ft: float | None  # WGS84 ellipsoid height
    on_ground: bool
    gs_kt: float | None
    track_deg: float | None
    rate_fpm: float | None  # baro_rate, else geom_rate
    callsign: str
    category: str
    squawk: str
    rssi: float | None  # dBFS, the receiver's signal strength


def _report(a, now):
    if "lat" not in a or "lon" not in a or "seen_pos" not in a:
        return None
    alt = a.get("alt_baro")
    on_ground = alt == "ground"
    return Report(
        hex=a["hex"],
        t=now - float(a["seen_pos"]),
        lat=float(a["lat"]),
        lon=float(a["lon"]),
        alt_baro_ft=0.0 if on_ground else (float(alt) if alt is not None else None),
        alt_geom_ft=float(a["alt_geom"]) if "alt_geom" in a else None,
        on_ground=on_ground,
        gs_kt=float(a["gs"]) if "gs" in a else None,
        track_deg=float(a["track"]) if "track" in a else (float(a["calc_track"]) if "calc_track" in a else None),
        rate_fpm=float(a["baro_rate"]) if "baro_rate" in a else (float(a["geom_rate"]) if "geom_rate" in a else None),
        callsign=(a.get("flight") or "").strip(),
        category=a.get("category", ""),
        squawk=a.get("squawk", ""),
        rssi=float(a["rssi"]) if "rssi" in a else None,
    )


class _Base:
    def __init__(self):
        self._last_t = {}  # hex -> time of the last position handed out

    def _new_reports(self, snapshot):
        now = float(snapshot.get("now") or snapshot.get("t") or 0)
        out = []
        for a in snapshot.get("aircraft", []):
            r = _report(a, now)
            if r is None or r.t <= self._last_t.get(r.hex, 0) + 0.05:
                continue
            self._last_t[r.hex] = r.t
            out.append(r)
        return out

    def clock(self):
        """The feed's notion of "now" (replays run on recorded time)."""
        return time.time()


class LiveFeed(_Base):
    """Polls the URL on a background thread so the render loop never waits on the network."""

    def __init__(self, url, poll_sec=1.0):
        super().__init__()
        self.url, self.poll_sec = url, poll_sec
        self._snap = None
        self._lock = threading.Lock()
        self.error = None
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while True:
            try:
                with urllib.request.urlopen(self.url, timeout=3) as resp:
                    snap = json.load(resp)
                with self._lock:
                    self._snap = snap
                self.error = None
            except (OSError, ValueError) as exc:
                self.error = str(exc)
            time.sleep(self.poll_sec)

    def poll(self):
        with self._lock:
            snap, self._snap = self._snap, None
        return self._new_reports(snap) if snap else []


class ReplayFeed(_Base):
    """Plays a recording back on its own clock, `speed`x real time, looping."""

    def __init__(self, path, speed=1.0, start_offset=0.0):
        super().__init__()
        self.frames = []
        with gzip.open(path, "rt") as f:
            try:
                for line in f:
                    self.frames.append(json.loads(line))
            except (EOFError, ValueError):
                pass  # a recording still being written ends mid-line
        if not self.frames:
            raise ValueError(f"no frames in {path}")
        self.t0 = self.frames[0]["t"]
        self.t_end = self.frames[-1]["t"]
        self.speed = speed
        self.start_offset = start_offset
        self._restart()

    def _restart(self):
        self.wall0 = time.time()
        self.i = 0
        self._last_t = {}
        self.loop_count = getattr(self, "loop_count", -1) + 1

    def clock(self):
        t = self.t0 + self.start_offset + (time.time() - self.wall0) * self.speed
        if t > self.t_end:
            self._restart()
            return self.clock()
        return t

    def poll(self):
        now = self.clock()
        out = []
        while self.i < len(self.frames) and self.frames[self.i]["t"] <= now:
            out += self._new_reports(self.frames[self.i])
            self.i += 1
        return out


def msl_m(report, baro_offset_ft):
    """Best-guess altitude above mean sea level, in meters.

    Geometric altitude is GPS height above the WGS84 ellipsoid; around
    Dallas the geoid sits ~26m below the ellipsoid, so MSL = geom + 26m.
    Barometric altitude is pressure altitude (always referenced to
    29.92inHg), off from true by the day's pressure -- `baro_offset_ft`
    is the fleet-wide geom-minus-baro difference measured from aircraft
    that report both (see Traffic.baro_offset_ft)."""
    if report.alt_geom_ft is not None:
        return report.alt_geom_ft * FT + 26.0
    if report.alt_baro_ft is not None:
        return (report.alt_baro_ft + baro_offset_ft) * FT + 26.0
    return None
