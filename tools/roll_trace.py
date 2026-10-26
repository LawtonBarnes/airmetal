"""Flies the camera through a recording offline (no display, as fast as it
can) and reports how jittery the estimated bank is -- the mean absolute
frame-to-frame bank change at 10fps, per aircraft, for the ones that turn.

Usage: python tools/roll_trace.py RECORDING.jsonl.gz
"""
import gzip
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402
import feed  # noqa: E402
import flight  # noqa: E402

s = config.load()
frames = []
with gzip.open(sys.argv[1], "rt") as f:
    for line in f:
        frames.append(json.loads(line))
src = feed._Base()
traffic = flight.Traffic(s["lat"], s["lon"])
cams = {}
history = {}
i = 0
t = frames[0]["t"] + 10
DT = 0.1
while t < frames[-1]["t"]:
    while i < len(frames) and frames[i]["t"] <= t:
        traffic.update(src._new_reports(frames[i]), frames[i]["t"])
        i += 1
    for h, tr in traffic.tracks.items():
        st = tr.state_at(t - s["delay_sec"])
        if st is None or tr.info.on_ground:
            continue
        cam = cams.setdefault(h, flight.Camera(s["ground_msl_m"]))
        cam.follow(st, DT)
        history.setdefault(h, []).append(cam.bank)
    t += DT

rows = []
for h, banks in history.items():
    if len(banks) < 300:
        continue
    jitter = sum(abs(b - a) for a, b in zip(banks, banks[1:])) / (len(banks) - 1)
    rows.append((max(abs(b) for b in banks), jitter, h, len(banks)))
rows.sort(reverse=True)
print(f"{'hex':8s} {'secs':>5s} {'max bank':>9s} {'jitter deg/frame':>17s}")
for mx, jit, h, n in rows[:15]:
    print(f"{h:8s} {n / 10:5.0f} {mx:9.1f} {jit:17.3f}")
print(f"mean jitter over {len(rows)} aircraft: {sum(r[1] for r in rows) / len(rows):.3f} deg/frame")
