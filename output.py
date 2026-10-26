"""Where frames go, and where keys come from: a desktop window for
development, or the Pi's /dev/fb0 plus the remote (evdev), the way every
METAL SHOP app runs on the fleet.

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
EXIT_GOTO_HOME = 42  # see STRINGS/SCRUTE: Home hands off to the app menu


def open_output(window):
    return WindowOutput() if window else FramebufferOutput()


class WindowOutput:
    exit_code = 0

    def __init__(self):
        pygame.display.set_caption("AIR METAL")
        self.screen = pygame.display.set_mode((config.FRAME_W, config.FRAME_H))

    def handle_input(self, app):
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                return False
            if ev.type == pygame.KEYDOWN:
                if ev.key in (pygame.K_ESCAPE, pygame.K_q):
                    return False
                if ev.key == pygame.K_RIGHT:
                    app.pick(app.feed.clock(), +1)
                elif ev.key == pygame.K_LEFT:
                    app.pick(app.feed.clock(), -1)
        return True

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
        from evdev import ecodes
        self.ecodes = ecodes
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
        self.exit_code = 0
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
            if d.capabilities().get(ecodes.EV_KEY):
                try:
                    d.grab()
                except OSError:
                    pass
                self.devices.append(d)

    def handle_input(self, app):
        e = self.ecodes
        for d in self.devices:
            try:
                events = list(d.read())
            except (BlockingIOError, OSError):
                continue
            for ev in events:
                if ev.type != e.EV_KEY or ev.value != 1:
                    continue
                if ev.code in (e.KEY_HOMEPAGE, e.KEY_HOME):
                    self.exit_code = EXIT_GOTO_HOME
                    return False
                if ev.code in (e.KEY_Q, e.KEY_ESC, e.KEY_COMPOSE):
                    return False
                if ev.code == e.KEY_RIGHT:
                    app.pick(app.feed.clock(), +1)
                elif ev.code == e.KEY_LEFT:
                    app.pick(app.feed.clock(), -1)
        return True

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
