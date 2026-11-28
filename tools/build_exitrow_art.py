"""EXITROW's window overlays, from the user's Photoshop file -> exitrow_art/.

    python tools/build_exitrow_art.py "C:/Users/HP/Desktop/AIR METAL/WINGS-8-BIT-RETRO-OVERLAY.psd" \
        ["C:/Users/HP/Desktop/AIR METAL/SPLASH-EXIT-ROW.psd"]

The PSD is 720x480, drawn as the PORT window (the starboard window is the
same art flipped left-right at run time). Layers used:
  Jet Wing, Cessna Wing   wing + the rounded black window frame, together
  Jet Wing Red Lights     the jet wing's lit red lights (sunset/night, untinted)
  BLADES1..3(.png)        rotor blades along the top
  Chopper Overlay         the helicopter window matte (over the blades)
Written: wing_jet.png, wing_jet_lights.png, wing_cessna.png, blades1-3.png,
heli_window.png (RGBA, full frame, alpha cut at 50%), plus frame.png -- the window frame alone, for helicopters (blades go
under it). The frame comes from the jet layer's left edge, where nothing
but frame is drawn, mirrored to the right. (Helicopters show the blades
with no frame since 2026-11-25; frame.png is kept for trying it again.)

With the second argument, splash.png: the user's full-screen splash art
(2026-11-28), the PSD's "Port" layer (4:3, logo top right) stretched to
the 720x480 frame like AIR METAL's. Starboard flips it at run time, and
the title + version is drawn by exitrow_screens.build_splash.

Runs on the PC (needs psd-tools); the Pis only read the PNGs.
"""
import os
import sys

import numpy as np
from PIL import Image
from psd_tools import PSDImage

W, H = 720, 480
LAYERS = {"Jet Wing": "wing_jet", "Jet Wing Red Lights": "wing_jet_lights", "Cessna Wing": "wing_cessna",
          "BLADES1": "blades1", "BLADES2": "blades2", "BLADES3": "blades3",
          "Chopper Overlay": "heli_window"}
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "exitrow_art")


def full_frame(layer):
    """A layer on a transparent 720x480 canvas (layers may hang off the edges)."""
    canvas = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    layer.visible = True  # hidden layers (often the Cessna) composite empty otherwise
    img = layer.composite(force=True).convert("RGBA")
    x0, y0 = layer.bbox[:2]
    canvas.alpha_composite(img, (max(0, x0), max(0, y0)), (max(0, -x0), max(0, -y0)))
    return canvas


def frame_from(wing):
    """The window frame alone: per row, the opaque run in from the left
    edge, mirrored to the right edge. Rows that are opaque all the way
    across (none in the current art) stay fully framed."""
    a = np.asarray(wing)[..., 3] > 0
    clear = ~a
    first_clear = np.where(clear.any(1), clear.argmax(1), W)
    out = np.zeros((H, W, 4), np.uint8)
    out[..., 3] = 0
    cols = np.arange(W)
    mask = (cols[None, :] < first_clear[:, None]) | (cols[None, :] >= W - first_clear[:, None])
    out[mask] = (0, 0, 0, 255)
    return Image.fromarray(out, "RGBA")


def main():
    psd = PSDImage.open(sys.argv[1])
    if (psd.width, psd.height) != (W, H):
        sys.exit(f"expected a {W}x{H} PSD, got {psd.width}x{psd.height}")
    os.makedirs(OUT, exist_ok=True)
    found = {}
    for layer in psd:
        name = LAYERS.get(layer.name.removesuffix(".png"))
        if name:
            img = full_frame(layer)
            # Hard edges only: the Pi blits these colorkeyed.
            a = np.asarray(img).copy()
            a[..., 3] = np.where(a[..., 3] >= 128, 255, 0)
            img = Image.fromarray(a, "RGBA")
            img.save(os.path.join(OUT, name + ".png"))
            found[name] = img
            print(name)
    missing = set(LAYERS.values()) - set(found)
    if missing:
        sys.exit(f"missing layers: {sorted(missing)}")
    frame_from(found["wing_jet"]).save(os.path.join(OUT, "frame.png"))
    print("frame")
    if len(sys.argv) > 2:
        port = next(l for l in PSDImage.open(sys.argv[2]).descendants() if l.name == "Port")
        port.topil().convert("RGB").resize((W, H), Image.LANCZOS).save(os.path.join(OUT, "splash.png"))
        print("splash")


if __name__ == "__main__":
    main()
