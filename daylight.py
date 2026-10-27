"""Time of day: where the sun is, and what the scene's light looks like.

sun_position() is the usual low-precision solar ephemeris (good to a
fraction of a degree, plenty for lighting). lighting_at() turns the sun's
elevation into a Lighting by blending between hand-picked keyframes --
day, golden hour, sunset, civil and nautical dusk, night:

  light      multiplies every lit color (ground, roads, buildings, water)
  lights_on  0..1, how far the city's own lights are up (land-cover glow,
             road and runway lights, lit windows)
  stars      0..1, how far the stars are out
  sky        a sky color for any elevation above the horizon and any
             direction relative to the sun (the sunset glow sits on the
             sun's side) -- see Lighting.sky_lut

Daytime is "day" and leaves the drone-matched colors exactly as they are.
"""
import math
from dataclasses import dataclass

import numpy as np

J2000_UNIX = 946728000.0  # 2000-01-01 12:00 UTC


def sun_position(t_unix, lat, lon):
    """-> (elevation, azimuth) in degrees; azimuth clockwise from north."""
    d = (t_unix - J2000_UNIX) / 86400.0
    g = math.radians((357.529 + 0.98560028 * d) % 360)
    q = (280.459 + 0.98564736 * d) % 360
    ecl_lon = math.radians((q + 1.915 * math.sin(g) + 0.020 * math.sin(2 * g)) % 360)
    tilt = math.radians(23.439 - 0.00000036 * d)
    ra = math.atan2(math.cos(tilt) * math.sin(ecl_lon), math.cos(ecl_lon))
    dec = math.asin(math.sin(tilt) * math.sin(ecl_lon))
    gmst_h = (18.697374558 + 24.06570982441908 * d) % 24
    ha = math.radians(gmst_h * 15 + lon) - ra
    la = math.radians(lat)
    elev = math.asin(math.sin(la) * math.sin(dec) + math.cos(la) * math.cos(dec) * math.cos(ha))
    az = math.atan2(-math.sin(ha), math.tan(dec) * math.cos(la) - math.sin(la) * math.cos(ha))
    return math.degrees(elev), math.degrees(az) % 360


# Keyframes by sun elevation (deg), high to low: horizon color toward the
# sun, horizon color away from it, the sky overhead, how high (deg) the
# horizon color reaches, light (rgb multiplier), lights_on.
KEYS = [
    (4.0, (236, 198, 152), (170, 186, 208), (78, 128, 198), 30.0, (1.0, 0.9, 0.76), 0.0),  # golden hour
    (0.0, (246, 150, 84), (150, 140, 162), (58, 90, 160), 25.0, (0.72, 0.54, 0.44), 0.25),  # sunset
    (-4.0, (196, 98, 72), (66, 64, 104), (28, 40, 92), 22.0, (0.34, 0.29, 0.38), 0.75),  # civil dusk
    (-9.0, (72, 46, 68), (30, 32, 60), (12, 16, 44), 20.0, (0.15, 0.15, 0.21), 1.0),  # nautical dusk
    (-13.0, (38, 30, 44), (34, 30, 46), (5, 7, 18), 20.0, (0.08, 0.09, 0.13), 1.0),  # night
    (-18.0, (26, 18, 18), (24, 17, 18), (0, 0, 3), 8.0, (0.05, 0.05, 0.08), 1.0),  # full dark: black sky, city glow on the horizon
]
STARS_FROM, STARS_FULL = -12.0, -17.0  # sun elevations where stars start to show, and are all out
DAY_ELEV = 10.0  # at and above this: plain day; between it and KEYS[0], blended


@dataclass
class Lighting:
    day: float  # 1 = plain day, 0 = KEYS[...] only
    key_a: int  # the two keys blended (indices into KEYS), and how far toward b
    key_b: int
    mix: float
    light: np.ndarray
    lights_on: float
    sun_az: float
    stars: float = 0.0

    def sky_lut(self, day_sky, azimuth_steps, elev_steps, sin_top):
        """(azimuth_steps, elev_steps, 3) sky colors: [a] runs from facing
        away from the sun (0) to facing it; [e] is linear in sin(elevation)
        from the horizon to sin_top. `day_sky` (elev_steps, 3) is the plain
        day gradient."""
        if self.day >= 1.0:
            return np.broadcast_to(day_sky, (azimuth_steps, elev_steps, 3)).copy()
        elev = np.degrees(np.arcsin(np.arange(elev_steps) / (elev_steps - 1) * sin_top))
        toward = np.linspace(0.0, 1.0, azimuth_steps)[:, None, None] ** 2.5

        def key_sky(k):
            _, sun_hz, anti_hz, zen, reach, _, _ = KEYS[k]
            hz = np.array(anti_hz, np.float32) + (np.array(sun_hz, np.float32) - anti_hz) * toward
            up = (np.clip(elev / reach, 0, 1) ** 0.6)[None, :, None]
            return hz * (1 - up) + np.array(zen, np.float32) * up

        sky = key_sky(self.key_a) * (1 - self.mix) + key_sky(self.key_b) * self.mix
        return day_sky[None] * self.day + sky * (1 - self.day)


def lighting_at(elev, az):
    lt = _lighting_at(elev, az)
    lt.stars = min(max((STARS_FROM - elev) / (STARS_FROM - STARS_FULL), 0.0), 1.0)
    return lt


def _lighting_at(elev, az):
    if elev >= DAY_ELEV:
        return Lighting(1.0, 0, 0, 0.0, np.ones(3, np.float32), 0.0, az)
    if elev >= KEYS[0][0]:
        day = (elev - KEYS[0][0]) / (DAY_ELEV - KEYS[0][0])
        light = np.ones(3, np.float32) * day + np.array(KEYS[0][5], np.float32) * (1 - day)
        return Lighting(day, 0, 0, 0.0, light, 0.0, az)
    for i in range(len(KEYS) - 1):
        hi, lo = KEYS[i], KEYS[i + 1]
        if elev >= lo[0]:
            m = (hi[0] - elev) / (hi[0] - lo[0])
            light = np.array(hi[5], np.float32) * (1 - m) + np.array(lo[5], np.float32) * m
            return Lighting(0.0, i, i + 1, m, light, hi[6] * (1 - m) + lo[6] * m, az)
    last = len(KEYS) - 1
    return Lighting(0.0, last, last, 0.0, np.array(KEYS[last][5], np.float32), KEYS[last][6], az)
