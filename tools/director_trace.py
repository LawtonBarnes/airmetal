"""Runs the director over a recording offline (as fast as it can) and lists
every ride it would have taken: when, how long, and how close/low it came.

Usage: python tools/director_trace.py RECORDING.jsonl.gz
"""
import gzip
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402
import director  # noqa: E402
import feed  # noqa: E402
import flight  # noqa: E402
from geo import M_PER_NM  # noqa: E402

s = config.load()
frames = [json.loads(line) for line in gzip.open(sys.argv[1], "rt")]
src = feed._Base()
traffic = flight.Traffic(s["lat"], s["lon"])
d = director.Director()
t0 = frames[0]["t"]
i, t = 0, t0 + 15
rides = []  # [hex, start, end, min_dist_nm, min_alt_ft, callsign, category]
while t < frames[-1]["t"]:
    while i < len(frames) and frames[i]["t"] <= t:
        traffic.update(src._new_reports(frames[i]), frames[i]["t"])
        i += 1
    h = d.update(traffic, t, t - s["delay_sec"])
    if h and (not rides or rides[-1][0] != h):
        rides.append([h, t, t, 99.0, 99999.0, "", ""])
    if h:
        tr = traffic.tracks[h]
        st = tr.state_at(t - s["delay_sec"])
        r = rides[-1]
        r[2] = t
        if st:
            r[3] = min(r[3], math.hypot(st[0], st[1]) / M_PER_NM)
        r[4] = min(r[4], tr.info.alt_baro_ft or 0)
        r[5], r[6] = tr.info.callsign or h, tr.info.category
    t += 1.0

print(f"{'start':>6s} {'secs':>5s} {'aircraft':10s} {'cat':4s} {'closest NM':>10s} {'lowest ft':>9s}")
for h, a, b, dist, alt, cs, cat in rides:
    print(f"{(a - t0) / 60:5.1f}m {b - a:5.0f} {cs:10s} {cat:4s} {dist:10.1f} {alt:9.0f}")
print(f"{len(rides)} rides in {(frames[-1]['t'] - t0) / 60:.0f} min; "
      f"median closest {sorted(r[3] for r in rides)[len(rides) // 2]:.1f} NM")
