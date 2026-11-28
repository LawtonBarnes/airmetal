"""AIR METAL's flyover sound: a loop for the aircraft being ridden, louder
the closer it is to the Metal Shop and silent past [audio] range_nm.

  jet    -- a steady roar (most categories)
  heli   -- thump-thump blade slap (A7)
  prop   -- a higher, buzzy propeller (A1 LIGHT, B4 ultralight)
  (none) -- gliders, balloons, drones, ground vehicles

Everything is synthesized at startup with numpy, no sound files. The
loops are built to repeat seamlessly (noise is made in the frequency
domain, so it's periodic; tones and blade thumps divide the loop
exactly). The sound is cheated toward the low end -- a sub-bass layer
under each -- so it rumbles the subwoofers and is felt more than heard.

Distance is slant range from the shop (ground distance and altitude),
so a jet at 3,000 ft straight overhead is ~0.5 NM away. Volume is
(1 - d / range)^2: max right at the shop, gone at range_nm. Changes are
eased over EASE_SEC so a switch of aircraft fades rather than jumps.

On QUAD's Pi 5 the HDMI PCM device is exclusive, and pygame already
opens it at pygame.init(); this just uses that mixer. If there's no
audio device, it quietly does nothing.
"""
import math

import numpy as np
import pygame

RATE = 44100
EASE_SEC = 0.6
SILENT = {"B1", "B2", "B6", "C1", "C2", "C3"}
PROP = {"A1", "B4"}
HELI = {"A7"}


def kind_for(category):
    if category in HELI:
        return "heli"
    if category in PROP:
        return "prop"
    if category in SILENT:
        return None
    return "jet"


# -- synthesis -------------------------------------------------------------

def _noise(n, shape, rng):
    """Periodic noise of n samples whose amplitude spectrum is shape(f)."""
    f = np.fft.rfftfreq(n, 1 / RATE)
    amp = shape(np.maximum(f, 1.0))
    amp[0] = 0.0
    spec = amp * np.exp(2j * np.pi * rng.random(len(f)))
    x = np.fft.irfft(spec, n)
    return x / np.abs(x).max()


def _band(lo, hi, f):
    """A soft band-pass gain curve."""
    return 1 / np.sqrt(1 + (lo / f) ** 4) / np.sqrt(1 + (f / hi) ** 4)


def _normalize(x, peak=0.9):
    return x * (peak / np.abs(x).max())


def jet(rng, seconds=4.0):
    n = int(RATE * seconds)
    t = np.arange(n) / RATE
    roar = _noise(n, lambda f: f ** -0.5 / (1 + (f / 700) ** 2), rng)  # pink-ish, dull top
    rumble = _noise(n, lambda f: _band(25, 70, f), rng)  # the sub layer
    swell = 1 + 0.12 * np.sin(2 * np.pi * 0.5 * t) + 0.06 * np.sin(2 * np.pi * 1.25 * t + 1)
    return _normalize((0.55 * roar + 0.9 * rumble) * swell)


def heli(rng, blade_hz=12.5, seconds=4.0):
    period = int(round(RATE / blade_hz))  # 3528 samples at 12.5 Hz: exact
    k = np.arange(period) / RATE
    thump = (np.sin(2 * np.pi * 55 * k) * np.exp(-k / 0.030)  # body
             + 0.9 * np.sin(2 * np.pi * 30 * k) * np.exp(-k / 0.045))  # sub punch
    slap = _noise(period, lambda f: _band(250, 1400, f), rng) * np.exp(-k / 0.010)
    pulse = thump + 0.6 * slap  # the slap keeps it a "thump" on small speakers too
    count = int(round(seconds * blade_hz))
    # Two blades, slightly unequal: a heavier "thump" then a lighter one.
    x = np.concatenate([pulse * (1.0 if i % 2 == 0 else 0.8) for i in range(count)])
    turbine = _noise(len(x), lambda f: f ** -0.5 / (1 + (f / 2500) ** 2), rng)
    return _normalize(x + 0.12 * turbine)


def prop(rng, buzz_hz=80.0, seconds=4.0):
    period = int(round(RATE / buzz_hz))  # 551 samples: 80.04 Hz
    n = period * int(round(seconds * RATE / period))
    t = np.arange(n) / RATE
    f0 = RATE / period
    buzz = sum(np.sin(2 * np.pi * f0 * h * t + h) / h ** 1.1 for h in range(1, 28))
    sub = np.sin(2 * np.pi * f0 / 2 * t)  # engine firing, an octave down
    hiss = _noise(n, lambda f: _band(300, 3000, f), rng)
    wobble = 1 + 0.08 * np.sin(2 * np.pi * 3 * np.arange(n) / n)  # 3 slow surges a loop
    return _normalize((0.6 * buzz / np.abs(buzz).max() + 0.7 * sub + 0.1 * hiss) * wobble)


# -- playback --------------------------------------------------------------

class FlyoverSound:
    def __init__(self, range_nm=1.5, volume=0.8, enabled=True):
        self.range_nm = max(0.1, range_nm)
        self.volume = max(0.0, min(1.0, volume))
        self.channels = {}
        self.level = {}
        if not enabled:
            return
        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init(RATE, -16, 2, 2048)
            rate, _, chans = pygame.mixer.get_init()
        except pygame.error:
            return
        if rate != RATE:
            print(f"audio: mixer runs at {rate} Hz, loops are built for {RATE}; pitch will be off")
        rng = np.random.default_rng(7)
        for name, build in (("jet", jet), ("heli", heli), ("prop", prop)):
            pcm = (build(rng) * 32767).astype(np.int16)
            pcm = np.repeat(pcm[:, None], chans, axis=1) if chans > 1 else pcm
            sound = pygame.sndarray.make_sound(np.ascontiguousarray(pcm))
            ch = sound.play(loops=-1)
            if ch is None:
                continue
            ch.set_volume(0.0)
            self.channels[name] = ch
            self.level[name] = 0.0

    def gain_at(self, distance_nm):
        g = max(0.0, 1.0 - distance_nm / self.range_nm)
        return g * g

    def update(self, kind, distance_nm, dt):
        """kind: jet/heli/prop or None (silence); distance: slant range, NM."""
        if not self.channels:
            return
        target = self.gain_at(distance_nm) if kind else 0.0
        step = min(1.0, dt / EASE_SEC)
        for name, ch in self.channels.items():
            want = target if name == kind else 0.0
            cur = self.level[name] + (want - self.level[name]) * step
            if abs(cur - want) < 1e-3:
                cur = want
            self.level[name] = cur
            ch.set_volume(cur * self.volume)

    def close(self):
        for ch in self.channels.values():
            ch.stop()
