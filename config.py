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
TARGET_FPS = 10

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
        # How far behind live the camera flies, so it can interpolate
        # between position reports instead of guessing ahead of them.
        "delay_sec": "2.5",
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
        "ground_scale": cp.getint("view", "ground_scale"),
        "ground_filter": cp.get("view", "ground_filter"),
        "trees": cp.getboolean("view", "trees"),
        "saturation": cp.getfloat("view", "saturation"),
        "clock": cp.get("view", "clock"),
        "delay_sec": cp.getfloat("view", "delay_sec"),
        "link_port": cp.getint("link", "port"),
    }
