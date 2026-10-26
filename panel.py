"""The analog six-pack on the bottom two tubes (1024x384), after the user's
"Computer Sim" mockup: EGA-palette pixel art, black dials on slate plates.

Only what ADS-B actually carries is shown, labeled for what it is --
ground speed, not airspeed; attitude and turn rate are estimates from the
flight path (see flight.py). The mockup's two gauges with no ADS-B source
became DISTANCE (to the Metal Shop) and SIGNAL (receiver RSSI), and the
data box's radio rows show the squawk, category and ICAO hex.

Everything static (plates, screws, faces, ticks, labels) is drawn once
into a background; each frame only blits it and draws the moving parts.
"""
import math

import pygame

import config

W, H = 1024, 384

PLATE = (52, 64, 86)
PLATE_HI = (88, 104, 128)
PLATE_LO = (26, 32, 44)
SCREW = (127, 159, 170)
SCREW_DARK = (42, 52, 64)
FACE = (8, 8, 10)
BEZEL = (30, 34, 42)
BEZEL_HI = (110, 118, 132)
WHITE = (235, 235, 235)
LABEL = (170, 178, 210)
YELLOW = (255, 214, 0)
GREEN = (60, 230, 90)
RED = (230, 40, 30)
ORANGE = (255, 150, 0)
SKY = (42, 127, 255)
GROUND = (140, 74, 10)
BLACK = (0, 0, 0)
KEY = (255, 0, 255)  # colorkey for round cut-outs

R = 70  # gauge face radius
VSI_SWEEP = 155  # degrees from 0 (9 o'clock) to +-2000 fpm
LEFT = {"att": (215, 112), "gs": (390, 112), "tc": (215, 276), "hdg": (390, 276)}
RIGHT = {"alt": (614, 112), "vsi": (786, 112), "dist": (614, 276), "sig": (786, 276)}
DATA_X, DATA_W = 872, 104

CATEGORY_NAMES = {
    "A1": "LIGHT", "A2": "SMALL", "A3": "LARGE", "A4": "B757", "A5": "HEAVY",
    "A6": "HI PERF", "A7": "ROTOR", "B1": "GLIDER", "B2": "BALLOON", "B4": "ULTRALT",
    "B6": "DRONE", "C1": "VEHICLE", "C2": "VEHICLE",
}


def _font(size):
    return pygame.font.Font(config.FONT_PATH, size)


def _text(surf, font, s, color, center=None, topleft=None, midleft=None, midright=None):
    img = font.render(s, False, color)
    rect = img.get_rect()
    if center is not None:
        rect.center = center
    elif topleft is not None:
        rect.topleft = topleft
    elif midleft is not None:
        rect.midleft = midleft
    elif midright is not None:
        rect.midright = midright
    surf.blit(img, rect)


def _polar(c, r, deg):
    """Point r from center c at deg clockwise from 12 o'clock."""
    a = math.radians(deg)
    return c[0] + r * math.sin(a), c[1] - r * math.cos(a)


def _needle(surf, c, deg, length, width, color=WHITE, tail=10):
    tip = _polar(c, length, deg)
    back = _polar(c, -tail, deg)
    side_l = _polar(c, width, deg - 90)
    side_r = _polar(c, width, deg + 90)
    pygame.draw.polygon(surf, color, [tip, side_r, back, side_l])


class Panel:
    def __init__(self):
        self.f_small = _font(13)
        self.f_tiny = _font(11)
        self.f_num = _font(16)
        self.f_label = _font(14)
        self.f_data = _font(17)
        self.f_call = _font(22)
        self.bg = self._build_background()
        self.card = self._build_compass_card()
        self.surface = pygame.Surface((W, H))

    # -- static background -------------------------------------------------

    def _plate(self, s, rect):
        pygame.draw.rect(s, PLATE, rect)
        pygame.draw.line(s, PLATE_HI, rect.topleft, rect.topright, 2)
        pygame.draw.line(s, PLATE_HI, rect.topleft, rect.bottomleft, 2)
        pygame.draw.line(s, PLATE_LO, rect.bottomleft, rect.bottomright, 2)
        pygame.draw.line(s, PLATE_LO, rect.topright, rect.bottomright, 2)

    def _screw(self, s, c):
        pygame.draw.circle(s, SCREW_DARK, c, 6)
        pygame.draw.circle(s, SCREW, c, 5)
        pygame.draw.line(s, SCREW_DARK, (c[0] - 3, c[1]), (c[0] + 3, c[1]), 2)

    def _face(self, s, c):
        pygame.draw.circle(s, BLACK, c, R + 9)
        pygame.draw.circle(s, BEZEL_HI, c, R + 7)
        pygame.draw.circle(s, BEZEL, c, R + 5)
        pygame.draw.circle(s, FACE, c, R)
        for dx, dy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
            self._screw(s, (int(c[0] + dx * (R + 12)), int(c[1] + dy * (R + 12))))

    def _ticks(self, s, c, start, sweep, n_major, n_minor, labels=None, label_r=R - 24):
        for i in range(n_major * n_minor + 1):
            deg = start + sweep * i / (n_major * n_minor)
            major = i % n_minor == 0
            pygame.draw.line(s, WHITE, _polar(c, R - 3, deg), _polar(c, R - (13 if major else 7), deg),
                             3 if major else 1)
            if major and labels:
                _text(s, self.f_num, labels[i // n_minor], WHITE, center=_polar(c, label_r, deg))

    def _build_background(self):
        s = pygame.Surface((W, H))
        s.fill(BLACK)
        self._plate(s, pygame.Rect(16, 16, 490, 360))
        self._plate(s, pygame.Rect(518, 16, 490, 360))
        for x in (30, 492, 532, 994):
            for y in (30, 362):
                self._screw(s, (x, y))
        # Switch column (decoration, after the mockup -- MASTER lit, it's live).
        col = pygame.Rect(40, 40, 90, 270)
        pygame.draw.rect(s, PLATE_LO, col, 2)
        for i, name in enumerate(("MASTER", "AVIONICS", "FUEL PUMP")):
            y = 52 + i * 88
            _text(s, self.f_small, name, LABEL, center=(col.centerx, y))
            sw = pygame.Rect(54, y + 16, 22, 46)
            pygame.draw.rect(s, BLACK, sw)
            pygame.draw.rect(s, (90, 96, 104), sw.inflate(-6, -6))
            knob = pygame.Rect(sw.x + 6, sw.y + 6, 10, 16)
            pygame.draw.rect(s, RED if i == 0 else (205, 205, 205), knob)
            _text(s, self.f_small, "ON", WHITE, midleft=(84, y + 28))
            _text(s, self.f_small, "OFF", WHITE, midleft=(84, y + 52))
        for c in (*LEFT.values(), *RIGHT.values()):
            self._face(s, c)
        # Dials with a scale use a 270-degree sweep; captions sit in the gap at the bottom.
        # Ground speed: 0-500 kt.
        c = LEFT["gs"]
        self._ticks(s, c, -135, 270, 5, 5, ["0", "100", "200", "300", "400", "500"], label_r=R - 25)
        _text(s, self.f_tiny, "GND SPEED", WHITE, center=(c[0], c[1] - 22))
        _text(s, self.f_tiny, "KTS", WHITE, center=(c[0], c[1] + 24))
        # Turn coordinator.
        c = LEFT["tc"]
        for deg in (-110, -70, 70, 110):  # level and standard-rate marks, each side
            pygame.draw.line(s, WHITE, _polar(c, R - 4, deg), _polar(c, R - 16, deg), 3)
        _text(s, self.f_num, "L", WHITE, center=_polar(c, R - 26, -96))
        _text(s, self.f_num, "R", WHITE, center=_polar(c, R - 26, 96))
        _text(s, self.f_small, "TURN COORD", WHITE, center=(c[0], c[1] + 46))
        _text(s, self.f_small, "2 MIN", WHITE, center=(c[0], c[1] + 58))
        # Altimeter: 0-9 around the dial.
        c = RIGHT["alt"]
        self._ticks(s, c, 0, 360, 10, 5, [str(i) for i in range(10)] + [""])
        _text(s, self.f_label, "ALT", WHITE, center=(c[0], c[1] - 24))
        # Vertical speed: +-2000 fpm, 0 at 9 o'clock.
        c = RIGHT["vsi"]
        for i in range(-20, 21):
            deg = -90 + i / 20 * VSI_SWEEP
            major = i % 5 == 0
            pygame.draw.line(s, WHITE, _polar(c, R - 3, deg), _polar(c, R - (13 if major else 7), deg),
                             3 if major else 1)
            if major and i:
                _text(s, self.f_num, str(abs(i) // 1), WHITE, center=_polar(c, R - 24, deg))
        _text(s, self.f_num, "0", WHITE, center=_polar(c, R - 24, -90))
        _text(s, self.f_tiny, "UP", WHITE, center=(c[0] - 22, c[1] - 13))
        _text(s, self.f_tiny, "DN", WHITE, center=(c[0] - 22, c[1] + 13))
        _text(s, self.f_tiny, "x100", WHITE, center=(c[0] + 20, c[1] - 9))
        _text(s, self.f_tiny, "FPM", WHITE, center=(c[0] + 20, c[1] + 9))
        # Distance to the Metal Shop: 0-20 NM.
        c = RIGHT["dist"]
        self._ticks(s, c, -135, 270, 4, 5, ["0", "5", "10", "15", "20"], label_r=R - 25)
        _text(s, self.f_tiny, "TO SHOP", WHITE, center=(c[0], c[1] - 22))
        _text(s, self.f_tiny, "NM", WHITE, center=(c[0], c[1] + 24))
        # Signal strength (readsb RSSI): -40..0 dBFS.
        c = RIGHT["sig"]
        self._ticks(s, c, -135, 270, 4, 5, ["-40", "-30", "-20", "-10", "0"], label_r=R - 27)
        _text(s, self.f_tiny, "SIGNAL", WHITE, center=(c[0], c[1] - 22))
        _text(s, self.f_tiny, "DBFS", WHITE, center=(c[0], c[1] + 24))
        return s

    def _build_compass_card(self):
        d = 2 * R
        s = pygame.Surface((d, d))
        s.fill(KEY)
        pygame.draw.circle(s, FACE, (R, R), R)
        s.set_colorkey(KEY)
        c = (R, R)
        for i in range(72):
            deg = i * 5
            major = i % 2 == 0
            pygame.draw.line(s, WHITE, _polar(c, R - 3, deg), _polar(c, R - (11 if major else 6), deg),
                             2 if major else 1)
        for deg, lab in ((0, "N"), (90, "E"), (180, "S"), (270, "W"), (30, "3"), (60, "6"), (120, "12"),
                         (150, "15"), (210, "21"), (240, "24"), (300, "30"), (330, "33")):
            font = self.f_label if lab.isalpha() else self.f_small
            img = font.render(lab, False, WHITE)
            img = pygame.transform.rotate(img, -deg)
            s.blit(img, img.get_rect(center=_polar(c, R - 22, deg)))
        return s

    # -- per frame -----------------------------------------------------------

    def _attitude(self, s, c, pitch, bank):
        d = 2 * R
        ball = pygame.Surface((d, d))
        ball.fill(SKY)
        k = 2.2  # px per degree of pitch
        cx, cy = R, R + pitch * k
        a = math.radians(-bank)
        dx, dy = math.cos(a), -math.sin(a)  # horizon direction, rotated by bank
        big = 3 * d
        nx, ny = -dy, dx  # points toward the ground side
        p1 = (cx - dx * big, cy - dy * big)
        p2 = (cx + dx * big, cy + dy * big)
        pygame.draw.polygon(ball, GROUND, [p1, p2, (p2[0] + nx * big, p2[1] + ny * big),
                                           (p1[0] + nx * big, p1[1] + ny * big)])
        pygame.draw.line(ball, WHITE, p1, p2, 2)
        for deg in (-20, -10, 10, 20):
            off = -deg * k
            half = 18 if abs(deg) == 10 else 26
            mx, my = cx + nx * off, cy + ny * off
            pygame.draw.line(ball, WHITE, (mx - dx * half, my - dy * half), (mx + dx * half, my + dy * half), 2)
        # Mask to the round face.
        mask = pygame.Surface((d, d))
        mask.fill((255, 0, 255))
        pygame.draw.circle(mask, (0, 0, 0), (R, R), R)
        mask.set_colorkey((0, 0, 0))
        ball.blit(mask, (0, 0))
        ball.set_colorkey((255, 0, 255))
        s.blit(ball, (c[0] - R, c[1] - R))
        # Bank pointer + fixed airplane symbol.
        tip = _polar(c, R - 4, -bank)
        pygame.draw.polygon(s, WHITE, [tip, _polar(c, R - 14, -bank - 5), _polar(c, R - 14, -bank + 5)])
        pygame.draw.polygon(s, YELLOW, [(c[0] - 5, c[1] - 9), (c[0] + 5, c[1] - 9), (c[0], c[1] - 1)])
        pygame.draw.line(s, YELLOW, (c[0] - 44, c[1]), (c[0] - 14, c[1]), 4)
        pygame.draw.line(s, YELLOW, (c[0] + 14, c[1]), (c[0] + 44, c[1]), 4)
        pygame.draw.circle(s, YELLOW, c, 4)
        pygame.draw.circle(s, (60, 60, 60), (c[0], c[1] + R - 10), 6)

    def _turn_coordinator(self, s, c, turn_rate):
        tilt = max(-30.0, min(30.0, turn_rate / 3.0 * 20.0))  # standard rate (3 deg/s) = the 20 deg mark
        a = math.radians(tilt)
        def rot(px, py):
            return c[0] + px * math.cos(a) - py * math.sin(a), c[1] + px * math.sin(a) + py * math.cos(a)
        pygame.draw.polygon(s, WHITE, [rot(-46, -2), rot(46, -2), rot(46, 3), rot(-46, 3)])
        pygame.draw.polygon(s, WHITE, [rot(-6, -14), rot(6, -14), rot(6, 6), rot(-6, 6)])
        pygame.draw.polygon(s, WHITE, [rot(-14, 10), rot(14, 10), rot(14, 13), rot(-14, 13)])
        # Inclinometer: coordinated, so the ball stays centered.
        tube = pygame.Rect(c[0] - 34, c[1] + 22, 68, 14)
        pygame.draw.rect(s, (200, 200, 205), tube, border_radius=7)
        pygame.draw.circle(s, BLACK, (c[0], c[1] + 29), 6)

    def draw(self, cam, info, distance_nm, status_ok):
        s = self.surface
        s.blit(self.bg, (0, 0))
        self._attitude(s, LEFT["att"], cam.pitch, cam.bank)
        self._turn_coordinator(s, LEFT["tc"], cam.turn_rate)
        # Heading: card rotates under a fixed airplane and lubber line.
        c = LEFT["hdg"]
        card = pygame.transform.rotate(self.card, cam.heading)
        s.blit(card, card.get_rect(center=c))
        pygame.draw.polygon(s, ORANGE, [(c[0], c[1] - R + 2), (c[0] - 7, c[1] - R - 10), (c[0] + 7, c[1] - R - 10)])
        pygame.draw.polygon(s, WHITE, [(c[0], c[1] - 22), (c[0] - 4, c[1] - 8), (c[0] - 22, c[1] + 2),
                                       (c[0] - 22, c[1] + 6), (c[0] - 4, c[1] + 2), (c[0] - 3, c[1] + 16),
                                       (c[0] - 10, c[1] + 22), (c[0] + 10, c[1] + 22), (c[0] + 3, c[1] + 16),
                                       (c[0] + 4, c[1] + 2), (c[0] + 22, c[1] + 6), (c[0] + 22, c[1] + 2),
                                       (c[0] + 4, c[1] - 8)])
        _needle(s, LEFT["gs"], -135 + 270 * min(cam.gs_kt, 500) / 500, R - 12, 4)
        # Altimeter: 100s hand (long, thin) and 1000s hand (short, wide), plus a readout box.
        c = RIGHT["alt"]
        alt = max(cam.msl_ft, 0)
        box = pygame.Rect(c[0] - 6, c[1] + 18, 56, 20)
        pygame.draw.rect(s, BLACK, box)
        pygame.draw.rect(s, WHITE, box, 1)
        _text(s, self.f_small, f"{alt:5.0f}", WHITE, center=box.center)
        _needle(s, c, (alt % 10000) / 10000 * 360, R - 34, 6)
        _needle(s, c, (alt % 1000) / 1000 * 360, R - 12, 3)
        vs = max(-2000, min(2000, cam.vs_fpm))
        _needle(s, RIGHT["vsi"], -90 + vs / 2000 * VSI_SWEEP, R - 12, 4)
        _needle(s, RIGHT["dist"], -135 + 270 * min(distance_nm, 20) / 20, R - 12, 4)
        rssi = info.rssi if info and info.rssi is not None else -40
        _needle(s, RIGHT["sig"], -135 + 270 * (max(-40, min(0, rssi)) + 40) / 40, R - 12, 4)
        for c in (*LEFT.values(), *RIGHT.values()):
            if c is not LEFT["att"]:
                pygame.draw.circle(s, (70, 70, 74), c, 5)
        self._data_box(s, cam, info, status_ok)
        return s

    def _box(self, s, y, h):
        rect = pygame.Rect(DATA_X, y, DATA_W, h)
        pygame.draw.rect(s, BLACK, rect)
        pygame.draw.rect(s, PLATE_HI, rect, 1)
        return rect

    def _data_box(self, s, cam, info, status_ok):
        callsign = (info.callsign or info.hex.upper()) if info else "------"
        r = self._box(s, 30, 34)
        _text(s, self.f_call, callsign[:8], YELLOW, midleft=(r.x + 6, r.centery))
        r = self._box(s, 70, 78)
        for i, (lab, val) in enumerate((("ALT", f"{cam.msl_ft:5.0f}"), ("SPD", f"{cam.gs_kt:3.0f}"),
                                        ("HDG", f"{cam.heading:03.0f}"))):
            y = r.y + 14 + i * 24
            _text(s, self.f_data, lab, WHITE, midleft=(r.x + 6, y))
            _text(s, self.f_data, val, YELLOW, midright=(r.right - 6, y))
        r = self._box(s, 154, 48)
        if status_ok:
            _text(s, self.f_data, "ADS-B", GREEN, midleft=(r.x + 6, r.y + 13))
            _text(s, self.f_data, "ONLINE", GREEN, midleft=(r.x + 6, r.y + 34))
        else:
            _text(s, self.f_data, "ADS-B", RED, midleft=(r.x + 6, r.y + 13))
            _text(s, self.f_data, "NO DATA", RED, midleft=(r.x + 6, r.y + 34))
        rows = (("XPDR", info.squawk if info and info.squawk else "----"),
                ("CAT", CATEGORY_NAMES.get(info.category, info.category or "--") if info else "--"),
                ("HEX", info.hex.upper() if info else "------"))
        for i, (lab, val) in enumerate(rows):
            r = self._box(s, 208 + i * 32, 28)
            _text(s, self.f_small, lab, WHITE, midleft=(r.x + 5, r.centery))
            _text(s, self.f_small, val, YELLOW, midright=(r.right - 5, r.centery))
