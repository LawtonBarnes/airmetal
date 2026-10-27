"""Where frames go, and where keys come from: a desktop window for
development, or the Pi's /dev/fb0 plus the remote (evdev), the way every
METAL SHOP app runs on the fleet.

Both turn raw keys into the same small set of actions (see ACTIONS), so
main.py decides what a key means once, for both.

The framebuffer path writes the two halves separately -- the windshield
(top 1024x384) and the panel (bottom 1024x384) -- each converted to RGB565
by an SDL blit into a 16-bit surface (~4ms for a whole XGA frame on a Pi 3B,
per LOUDNESS).
"""
import mmap
import os
import struct
import sys
from pathlib import Path

import pygame

import config

KDSETMODE, KD_TEXT, KD_GRAPHICS = 0x4B3A, 0x00, 0x01
FBIO_WAITFORVSYNC = 0x40044620
MOUSE_MOVE_THRESHOLD = 12  # cumulative REL_X/REL_Y units before it counts as one direction press

# home: quit to SCRUTE's app menu. quit: plain quit. back: ABOUT.
# power: the shutdown dialog. enter/left/right/up/down: the D-pad.
ACTIONS = ("home", "quit", "back", "power", "enter", "left", "right", "up", "down")


def open_output(window):
    return WindowOutput() if window else FramebufferOutput()


class WindowOutput:
    """PC stand-ins for the remote: H = Home, Backspace = BACK, P = Power."""

    KEYS = {
        pygame.K_h: "home",
        pygame.K_ESCAPE: "quit", pygame.K_q: "quit",
        pygame.K_BACKSPACE: "back",
        pygame.K_p: "power",
        pygame.K_RETURN: "enter", pygame.K_KP_ENTER: "enter",
        pygame.K_LEFT: "left", pygame.K_RIGHT: "right", pygame.K_UP: "up", pygame.K_DOWN: "down",
    }

    def __init__(self):
        pygame.display.set_caption("AIR METAL")
        self.screen = pygame.display.set_mode((config.FRAME_W, config.FRAME_H))

    def actions(self):
        out = []
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                out.append("quit")
            elif ev.type == pygame.KEYDOWN and ev.key in self.KEYS:
                out.append(self.KEYS[ev.key])
        return out

    def write(self, top, bottom):
        self.screen.blit(top, (0, 0))
        self.screen.blit(bottom, (0, config.VIEW_H))
        pygame.display.flip()

    def close(self):
        pass


class FramebufferOutput:
    def __init__(self, dev="/dev/fb0"):
        global fcntl
        import fcntl  # noqa: F401 -- Linux only, like evdev
        import evdev
        from evdev import ecodes as e
        self.ecodes = e
        # Same key map as the rest of the fleet.
        self.keys = {
            e.KEY_HOMEPAGE: "home", e.KEY_HOME: "home",
            e.KEY_Q: "quit", e.KEY_ESC: "quit", e.KEY_COMPOSE: "quit",
            e.KEY_BACK: "back",
            e.KEY_POWER: "power",
            e.KEY_ENTER: "enter", e.KEY_KPENTER: "enter", e.BTN_LEFT: "enter", e.BTN_MOUSE: "enter",
            e.KEY_LEFT: "left", e.KEY_RIGHT: "right", e.KEY_UP: "up", e.KEY_DOWN: "down",
        }
        self._rel_accum = {e.REL_X: 0, e.REL_Y: 0}
        sys_dir = Path("/sys/class/graphics") / Path(dev).name
        self.width, self.height = (int(x) for x in (sys_dir / "virtual_size").read_text().split(","))
        bpp = int((sys_dir / "bits_per_pixel").read_text())
        self.stride = int((sys_dir / "stride").read_text())
        if (self.width, self.height, bpp) != (config.FRAME_W, config.FRAME_H, 16):
            raise RuntimeError(f"AIR METAL needs a 1024x768 16bpp framebuffer, got "
                               f"{self.width}x{self.height} {bpp}bpp -- see config.txt")
        self.fd = os.open(dev, os.O_RDWR)
        self.mm = mmap.mmap(self.fd, self.stride * self.height, mmap.MAP_SHARED,
                            mmap.PROT_WRITE | mmap.PROT_READ)
        self.half = pygame.Surface((config.FRAME_W, config.VIEW_H), 0, 16)
        self.vsync = True
        # Console into graphics mode so the text console can't draw over us.
        self.tty_fd = None
        try:
            self.tty_fd = os.open("/dev/tty", os.O_RDWR)
            fcntl.ioctl(self.tty_fd, KDSETMODE, KD_GRAPHICS)
        except OSError as exc:
            print(f"Console graphics mode not available: {exc}", file=sys.stderr)
        # Every key-capable input device, grabbed so nothing else sees our keys.
        self.devices = []
        for path in evdev.list_devices():
            d = evdev.InputDevice(path)
            if d.capabilities().get(e.EV_KEY):
                try:
                    d.grab()
                except OSError:
                    pass
                self.devices.append(d)

    def actions(self):
        e = self.ecodes
        out = []
        for d in self.devices:
            try:
                events = list(d.read())
            except (BlockingIOError, OSError):
                continue
            for ev in events:
                if ev.type == e.EV_KEY and ev.value == 1 and ev.code in self.keys:
                    out.append(self.keys[ev.code])
                elif ev.type == e.EV_REL and ev.code in self._rel_accum:
                    # Air-mouse movement -> discrete D-pad presses, as in loudness.py.
                    self._rel_accum[ev.code] += ev.value
                    accum = self._rel_accum[ev.code]
                    if abs(accum) >= MOUSE_MOVE_THRESHOLD:
                        self._rel_accum[ev.code] = 0
                        if ev.code == e.REL_X:
                            out.append("right" if accum > 0 else "left")
                        else:
                            out.append("down" if accum > 0 else "up")
        return out

    def _write_rows(self, surf, y0):
        self.half.blit(surf, (0, 0))
        raw = self.half.get_buffer().raw
        row = config.FRAME_W * 2
        if self.stride == row:
            self.mm.seek(y0 * self.stride)
            self.mm.write(raw)
        else:
            for y in range(config.VIEW_H):
                self.mm.seek((y0 + y) * self.stride)
                self.mm.write(raw[y * row:(y + 1) * row])

    def write(self, top, bottom):
        if self.vsync:
            try:
                fcntl.ioctl(self.fd, FBIO_WAITFORVSYNC, struct.pack("I", 0))
            except OSError:
                self.vsync = False
        self._write_rows(top, 0)
        self._write_rows(bottom, config.VIEW_H)

    def close(self):
        for d in self.devices:
            try:
                d.ungrab()
            except OSError:
                pass
        if self.tty_fd is not None:
            try:
                fcntl.ioctl(self.tty_fd, KDSETMODE, KD_TEXT)
                os.write(self.tty_fd, b"\033[2J\033[H")
            except OSError:
                pass
            os.close(self.tty_fd)
        self.mm.close()
        os.close(self.fd)
