# AIR METAL

A retro flight sim for a 2x2 wall of CRTs that rides real aircraft, live,
as they fly over Dallas. Part of the METAL SHOP McBrain fleet (launched by
STRINGS, assigned from SCRUTE), and a companion to JOAN JET: it reads the
same ADS-B receiver.

The wall is one 1024x768 frame from a Raspberry Pi 3B+, split into four
512x384 tubes by an OREI 2x2 video-wall controller:

- **Top two tubes: the windshield.** A flat-shaded, 1990-style 3D view from
  the cockpit of the aircraft being ridden: sky with haze, textured ground,
  and OpenStreetMap roads, rail, rivers and lakes out to 60 NM. The ground is
  real land cover (USGS NLCD 2021 -- tree-canopy neighborhoods, commercial
  roofs, forest, pasture, water), looked up per pixel where each view ray
  meets the ground, SNES "Mode 7" style, blended and mipmapped so it doesn't
  shimmer in motion.
- **Bottom two: the instrument panel.** An analog six-pack (attitude,
  ground speed, turn coordinator, heading, altimeter, vertical speed),
  plus distance-from-the-shop and signal-strength gauges, and a data box
  with the callsign, transponder code and aircraft category.

## Where the traffic comes from

P4's readsb (the SDR that JOAN JET shows), polled over HTTP once a second:
`http://<P4>/tar1090/data/aircraft.json`. The camera flies `delay_sec`
(2.5s) behind live, so it can interpolate between position reports
instead of guessing ahead of them; pitch, bank and turn rate come from the
track itself.

**The director** picks what to ride: aircraft inbound to the Metal Shop,
scored on how close they'll pass, how low, how soon, and how steadily
they're being received. A ride runs until the aircraft is 2 NM past its
closest point, then hands off (a callsign card shows for a few seconds).
With nothing worth riding, standby hovers 400 ft over the shop, panning
across White Rock Lake toward downtown.

## Controls

| Key | Action |
|---|---|
| Left / Right | Previous / next aircraft in the director's ranking |
| Back | ABOUT screen (feed, aircraft tracked, ride, settings) |
| Power | Shutdown dialog |
| Home | Quit to SCRUTE |
| Q / Esc / Menu | Quit |

ABOUT and the shutdown dialog each sit inside one tube, so no text
crosses a bezel; the panel keeps running live under ABOUT.

## Install (PRODUCTION)

```sh
sudo git clone https://github.com/LawtonBarnes/airmetal /opt/airmetal
sudo chown -R metalshop:metalshop /opt/airmetal
sudo install -m 755 /opt/airmetal/airmetal-launcher /usr/local/bin/airmetal
```

Needs a 1024x768 16bpp framebuffer (PRODUCTION's `config.txt` already
boots at XGA for the wall), python3-pygame, python3-numpy and
python3-evdev. `settings.ini` is optional -- copy `settings.example.ini`.
After that, deploys go through `sync-fleet.sh` like every other app.

`splash.png` (optional, 711x450 -- the same share of the wall as the
fleet's 500x281 splash on one CRT) shows at launch; without it a
placeholder box stands in.

## Development (on the PC)

```sh
python main.py --window                      # live feed, in a window
python main.py --window --replay recordings/feed_20261026.jsonl.gz --speed 2
```

In a window: arrows, Enter, Backspace = BACK, P = Power, H = Home,
Esc/Q = quit.

- `tools/build_scenery.py` -- builds `data/scenery_*.npz` from JOAN JET's
  basemap (PC only).
- `tools/build_landcover.py` -- builds `data/landcover_*.npz` (32 m land
  classes, +-112 km) from NLCD via MRLC's map server (PC only).
- `tools/bench_mode7.py` -- the textured-ground cost spike.
- `tools/director_trace.py` -- replays a recording through the director
  and lists every ride it would take.
- `tools/screens_test.py` -- renders splash/ABOUT/dialog PNGs with the
  tube seams marked.
- `tools/snapshot.py`, `time_render.py`, `profile_render.py`,
  `bench_options.py`, `roll_trace.py` -- render checks and benchmarks.
- `tools/wall_test.sh` -- runs a dev copy on PRODUCTION's wall outside
  STRINGS.

Recordings (`recordings/`, gitignored) come from `record_feed.py` on P4,
one gzipped JSON line per aircraft.json poll.
