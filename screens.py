"""The fleet's standard screens -- splash, ABOUT, the shutdown dialog --
adapted to the 2x2 wall.

Every other app draws these centered on one 720x480 CRT. Here the frame is
split across four tubes with bezels between them, so anything centered on
the frame would be cut down the middle by a seam. The splash is the
exception (it's art, and spans the wall on purpose); ABOUT and the power
dialog each live inside one tube, sized so they read the same as on a
single CRT: a tube shows 384 of the frame's lines where a lone CRT shows
480, so fonts are 0.8x the fleet's (title 36 -> 30, body 26 -> 22).

Each tube is 512x384 shown 4:3, i.e. square pixels -- so the METAL SHOP
logo needs no 720/640 stretch here, just the same 0.8x.
"""
import sys

import pygame

import config

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
ORANGE = (0xFF, 0xA5, 0x00)
GRAY = (90, 90, 90)

TUBE_W, TUBE_H = config.FRAME_W // 2, config.FRAME_H // 2
TOP_LEFT, TOP_RIGHT, BOTTOM_LEFT, BOTTOM_RIGHT = (0, 0), (1, 0), (0, 1), (1, 1)
TUBE_SCALE = 384 / 480

# Splash: unlike the rest of the fleet, AIR METAL's art fills the whole
# wall (splash.png is 1024x768). VERSION sits centered under the logo in
# the top-right tube, in the art's cream (f3e0cd) Fjalla One, no outline --
# placed from the user's SPLASH-AIR-METAL-SCREEN-LAYOUT.png (2026-10-29).
# Until splash.png exists, a placeholder box stands in for it.
SPLASH_SECONDS = 5.0
SPLASH_VERSION_FONT_SIZE = 26
SPLASH_VERSION_COLOR = (0xF3, 0xE0, 0xCD)
SPLASH_VERSION_CENTER = (768, 310)  # where the letters' ink is centered

ABOUT_MARGIN_Y = 38  # the fleet's 48px, per tube
ABOUT_LINE_GAP = 6
ABOUT_COLUMN_GAP = 13
ABOUT_TITLE_FONT_SIZE = 30
ABOUT_BODY_FONT_SIZE = 22

POWER_OPTIONS = ["NO", "YES", "RESTART"]
POWER_TUBE = TOP_LEFT


def tube_rect(tube):
    col, row = tube
    return pygame.Rect(col * TUBE_W, row * TUBE_H, TUBE_W, TUBE_H)


def outlined_text(font, text, color, outline):
    """text in color over a black copy stamped around it, outline px thick."""
    fill = font.render(text, True, color)
    edge = font.render(text, True, BLACK)
    surf = pygame.Surface((fill.get_width() + 2 * outline, fill.get_height() + 2 * outline), pygame.SRCALPHA)
    for dx in range(-outline, outline + 1):
        for dy in range(-outline, outline + 1):
            if dx * dx + dy * dy <= outline * outline:
                surf.blit(edge, (outline + dx, outline + dy))
    surf.blit(fill, (outline, outline))
    return surf


def build_splash(path, version):
    """Full-frame splash art, VERSION centered under the logo.
    No convert(): in framebuffer mode there's no SDL display to convert to."""
    canvas = pygame.Surface((config.FRAME_W, config.FRAME_H))
    canvas.fill(BLACK)
    try:
        img = pygame.image.load(str(path))
        if img.get_size() != canvas.get_size():
            img = pygame.transform.smoothscale(img, canvas.get_size())
        canvas.blit(img, (0, 0))
    except (pygame.error, OSError, FileNotFoundError):
        pygame.draw.rect(canvas, GRAY, canvas.get_rect(), 3)
        font = pygame.font.Font(config.FONT_PATH, 60)
        label = font.render("AIR METAL", True, WHITE)
        canvas.blit(label, label.get_rect(center=canvas.get_rect().center))
    font = pygame.font.Font(config.SPLASH_FONT_PATH, SPLASH_VERSION_FONT_SIZE)
    surf = font.render(f"VERSION {version}", True, SPLASH_VERSION_COLOR)
    ink = surf.get_bounding_rect()
    canvas.blit(surf, (SPLASH_VERSION_CENTER[0] - ink.centerx, SPLASH_VERSION_CENTER[1] - ink.centery))
    return canvas


def load_about_logo(path):
    try:
        img = pygame.image.load(str(path))
    except (pygame.error, OSError, FileNotFoundError) as exc:
        print(f"Logo load failed: {exc}", file=sys.stderr)
        return None
    w, h = img.get_size()
    return pygame.transform.smoothscale(img, (round(w * TUBE_SCALE), round(h * TUBE_SCALE)))


class Screens:
    def __init__(self, logo_path):
        self.title_font = pygame.font.Font(config.FONT_PATH, ABOUT_TITLE_FONT_SIZE)
        self.body_font = pygame.font.Font(config.FONT_PATH, ABOUT_BODY_FONT_SIZE)
        self.logo = load_about_logo(logo_path)

    def build_about(self, title, rows):
        """The windshield half of the ABOUT screen: title and (label, value)
        rows in the top-left tube -- labels orange and right-aligned, values
        white, the fleet layout -- and the METAL SHOP logo centered in the
        top-right tube. The panel keeps running live underneath."""
        canvas = pygame.Surface((config.FRAME_W, config.VIEW_H))
        canvas.fill(BLACK)
        left = tube_rect(TOP_LEFT)
        title_surf = self.title_font.render(title, True, ORANGE)
        canvas.blit(title_surf, (left.centerx - title_surf.get_width() // 2, ABOUT_MARGIN_Y))
        if self.logo is not None:
            canvas.blit(self.logo, self.logo.get_rect(center=tube_rect(TOP_RIGHT).center))
        if not rows:
            return canvas
        f = self.body_font
        label_surfs = [f.render(label, True, ORANGE) if label else None for label, _ in rows]
        value_surfs = [f.render(str(value), True, WHITE) for _, value in rows]
        label_w = max((s.get_width() for s in label_surfs if s), default=0)
        value_w = max(s.get_width() for s in value_surfs)
        line_h = f.get_linesize() + ABOUT_LINE_GAP
        top = ABOUT_MARGIN_Y + title_surf.get_height()
        bottom = TUBE_H - ABOUT_MARGIN_Y
        x0 = left.centerx - (label_w + ABOUT_COLUMN_GAP + value_w) // 2
        y = top + (bottom - top - line_h * len(rows)) // 2
        for label_surf, value_surf in zip(label_surfs, value_surfs):
            if label_surf:
                canvas.blit(label_surf, (x0 + label_w - label_surf.get_width(), y))
            canvas.blit(value_surf, (x0 + label_w + ABOUT_COLUMN_GAP, y))
            y += line_h
        return canvas

    def draw_power_dialog(self, surf, selection):
        """The fleet's ARE YOU SURE box, centered in POWER_TUBE. `surf` is the
        windshield (POWER_TUBE is a top tube)."""
        lines = [self.title_font.render(t, True, WHITE) for t in ("ARE YOU SURE YOU WANT", "TO SHUT DOWN?")]
        opts = [self.body_font.render(o, True, BLACK if i == selection else ORANGE)
                for i, o in enumerate(POWER_OPTIONS)]
        pad_x, pad_y, gap = 26, 18, 32
        opts_w = sum(s.get_width() for s in opts) + gap * (len(opts) - 1)
        content_w = max(max(s.get_width() for s in lines), opts_w)
        content_h = sum(s.get_height() for s in lines) + 8 * (len(lines) - 1) + 24 + opts[0].get_height()
        box = pygame.Surface((content_w + pad_x * 2, content_h + pad_y * 2))
        box.fill(BLACK)
        pygame.draw.rect(box, ORANGE, box.get_rect(), 3)
        y = pad_y
        for s in lines:
            box.blit(s, ((box.get_width() - s.get_width()) // 2, y))
            y += s.get_height() + 8
        y += 16
        x = (box.get_width() - opts_w) // 2
        for i, s in enumerate(opts):
            if i == selection:
                pygame.draw.rect(box, ORANGE, (x - 8, y - 5, s.get_width() + 16, s.get_height() + 10))
            box.blit(s, (x, y))
            x += s.get_width() + gap
        surf.blit(box, box.get_rect(center=tube_rect(POWER_TUBE).center))
