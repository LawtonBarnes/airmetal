"""AIR METAL settings: compiled-in constants plus settings.ini overrides.

settings.ini (gitignored, per machine -- copy settings.example.ini) holds
what differs between hosts or gets tuned live; everything else lives here.
"""
import configparser
import os

VERSION = "0.1"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SETTINGS_PATH = os.path.join(BASE_DIR, "settings.ini")
FONT_PATH = os.path.join(BASE_DIR, "VCR_OSD_MONO_1.001.ttf")
SPLASH_FONT_PATH = os.path.join(BASE_DIR, "FjallaOne-Regular.ttf")
DATA_DIR = os.path.join(BASE_DIR, "data")
SPLASH_PATH = os.path.join(BASE_DIR, "splash.png")  # optional -- see screens.build_splash
LOGO_PATH = os.path.join(BASE_DIR, "metalshop-logo.png")  # ABOUT screen

# The wall: one XGA frame, split 2x2 by the OREI into 512x384 quadrants.
FRAME_W, FRAME_H = 1024, 768
VIEW_H = 384  # the top two tubes: the windshield
VIEW_LOW_W, VIEW_LOW_H = 512, 192  # 3D render size, pixel-doubled to 1024x384
TARGET_FPS = 10  # EXITROW (Pi 3B+); AIR METAL's own is [view] fps

# EXITROW (exitrow.py): one puppet tube, 720x480 shown 4:3 (pixels 8/9 as
# wide as tall). The 3D view renders at half size and is pixel-doubled.
EXIT_FRAME_W, EXIT_FRAME_H = 720, 480
EXIT_VIEW_W, EXIT_VIEW_H = 360, 240
EXIT_PIXEL_ASPECT = (4 / 3) / (720 / 480)
EXIT_ART_DIR = os.path.join(BASE_DIR, "exitrow_art")

DEFAULTS = {
    "location": {
        "name": "METAL SHOP",
        "lat": "32.83475356507972",
        "lon": "-96.68858948607502",
        # Ground height (m MSL) the flat world sits at -- Dallas is ~130-180m.
        "ground_msl_m": "150",
    },
    "feed": {
        # P4's readsb, served by lighttpd. P4 has a DHCP reservation.
        "url": "http://192.168.68.70/tar1090/data/aircraft.json",
        "poll_sec": "1.0",
    },
    "link": {
        # GET /riding on this port tells JOAN JET which aircraft to circle.
        "port": "8430",
    },
    "view": {
        "fov_deg": "70",
        # Textured ground resolution: the land-cover pass runs at 1/this of
        # the 512x192 view (2 = 256x96, free on a Pi 3B+; 1 = full, ~30ms more).
        "ground_scale": "2",
        # "smooth": blended texels + mipmaps, no shimmer in motion.
        # "nearest": chunky one-texel samples (cheaper, "digital camo").
        "ground_filter": "smooth",
        # Time of day for the lighting: "auto" (the real clock and sun), or
        # a fixed local time like "19:05" (testing; or a favorite hour).
        "clock": "auto",
        # Color saturation of the whole scene, x the drone-matched colors
        # (1 = as photographed, which reads a little muted on the CRTs).
        "saturation": "1.7",
        # Pyramid trees on wooded land near the camera, below ~2,100 ft on the
        # altimeter. Off: tried 2026-10-27, too pointy and sparse to read as
        # trees, and denser costs too much on a Pi 3B+.
        "trees": "off",
        # The windshield's last step: one fixed 256-color palette through a
        # 4x4 ordered dither, so ground, roads and sky look like one engine
        # (on/off; ~17ms a frame on a Pi 3B+). Picked on the wall 2026-10-29.
        "dither": "on",
        # Roads etc. are drawn out to this share of each style's distance
        # (scenery.LINE_STYLES). Full range sometimes drew roads past the
        # horizon haze (user, 2026-11-25); EXITROW's half range looked right.
        "line_range": "0.5",
        # Softening while the windshield is scaled up to the wall (blur.py):
        # off, soft (a smooth 2x upscale -- free) or more (plus a sideways
        # smear, like composite video). For the 21" wall to sit closer to
        # EXITROW's 14" tubes: soft on the wall 2026-11-25, then "more".
        "blur": "more",
        # Windshield pixel size in wall pixels: 1 renders the full 1024x384
        # (needs QUAD's Pi 5), 2 renders 512x192 and pixel-doubles (the
        # Pi 3B+ days), 4 renders 256x96. The OREI stretches each 512x384
        # quadrant ~1.4x to NTSC, so a rendered pixel comes out unevenly
        # sized and shimmers as it moves; smaller ones hide that. The ground
        # keeps the same number of samples at every size.
        "pixel_size": "2",
        # Frames a second the wall aims for. 10 on the Pi 3B+; QUAD's Pi 5
        # runs 20 at pixel_size 1 (2026-11-25) -- smoother motion, less jitter.
        "fps": "10",
        # How far behind live the camera flies, so it can interpolate
        # between position reports instead of guessing ahead of them.
        "delay_sec": "2.5",
    },
    "windsock": {
        # The shop's windsock (windsock.py), so it can be picked out from
        # the air. Height in meters -- a 4-story building is ~14, about 7 px
        # tall from a mile out. 40 was still "pretty tiny even at low
        # altitudes" from a passing chopper (user, 2026-11-25): 4x that.
        "height_m": "160",
        # Compass direction the wind blows FROM (the sock points the other
        # way). height_m = 0: no windsock.
        "wind_from": "180",
    },
    "exitrow": {
        # Which side of the cabin this tube is: port (P1) or starboard (P3).
        "side": "port",
        # AIR METAL's ride link on QUAD: which aircraft to look out of.
        "airmetal_url": "http://192.168.68.71:8430/riding",
        "fov_deg": "60",
        # Degrees below level the window looks (plus the aircraft's bank).
        "look_down": "15",
        # Standby: parked at a gate (Love Field, nose toward the concourse).
        "gate_lat": "32.84649",
        "gate_lon": "-96.85010",
        "gate_heading": "135",
        # Roads drawn out to this share of AIR METAL's distances -- the
        # side window looks down, and a Pi 3B+ has ~100 ms a frame.
        "line_range": "0.5",
        # The textured ground's sample grid, 1/this of the 360x240 view
        # (AIR METAL's [view] ground_scale is the windshield's). 3 = 120x80:
        # a little softer than 2, about half the work.
        "ground_scale": "3",
    },
}


def load():
    cp = configparser.ConfigParser()
    cp.read_dict(DEFAULTS)
    cp.read(SETTINGS_PATH)
    return {
        "location_name": cp.get("location", "name"),
        "lat": cp.getfloat("location", "lat"),
        "lon": cp.getfloat("location", "lon"),
        "ground_msl_m": cp.getfloat("location", "ground_msl_m"),
        "feed_url": cp.get("feed", "url"),
        "poll_sec": cp.getfloat("feed", "poll_sec"),
        "fov_deg": cp.getfloat("view", "fov_deg"),
        "ground_scale": cp.getfloat("view", "ground_scale"),
        "ground_filter": cp.get("view", "ground_filter"),
        "trees": cp.getboolean("view", "trees"),
        "dither": cp.getboolean("view", "dither"),
        "saturation": cp.getfloat("view", "saturation"),
        "line_range": cp.getfloat("view", "line_range"),
        "blur": cp.get("view", "blur").strip().lower(),
        "pixel_size": cp.getint("view", "pixel_size"),
        "fps": cp.getfloat("view", "fps"),
        "clock": cp.get("view", "clock"),
        "delay_sec": cp.getfloat("view", "delay_sec"),
        "link_port": cp.getint("link", "port"),
        "windsock_height_m": cp.getfloat("windsock", "height_m"),
        "windsock_wind_from": cp.getfloat("windsock", "wind_from"),
        "exit_side": cp.get("exitrow", "side").strip().lower(),
        "exit_airmetal_url": cp.get("exitrow", "airmetal_url"),
        "exit_fov_deg": cp.getfloat("exitrow", "fov_deg"),
        "exit_look_down": cp.getfloat("exitrow", "look_down"),
        "exit_gate_lat": cp.getfloat("exitrow", "gate_lat"),
        "exit_gate_lon": cp.getfloat("exitrow", "gate_lon"),
        "exit_gate_heading": cp.getfloat("exitrow", "gate_heading"),
        "exit_line_range": cp.getfloat("exitrow", "line_range"),
        "exit_ground_scale": cp.getint("exitrow", "ground_scale"),
    }
