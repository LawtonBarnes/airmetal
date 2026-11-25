"""EXITROW's fleet screens -- splash, ABOUT, the shutdown dialog -- on one
720x480 tube, laid out like every other single-CRT app (copied from
THUNDERSTRUCK / SPACE TRUCKIN per the fleet's no-shared-library
convention; AIR METAL's own screens.py is the 2x2-wall version).

Splash: the user's 737 exit-window safety decal (exitrow_art/splash.png,
pre-stretched for the 720-wide NTSC frame by tools/build_exitrow_art.py),
raised from center, orange VERSION underneath.
"""
import sys

import pygame

import config

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
ORANGE = (0xFF, 0xA5, 0x00)

SPLASH_SECONDS = 5.0
SPLASH_Y_OFFSET = 40
SPLASH_VERSION_FONT_SIZE = 22
SPLASH_VERSION_GAP = 20

ABOUT_MARGIN_Y = 48
ABOUT_LINE_GAP = 8
ABOUT_COLUMN_GAP = 16
TITLE_FONT_SIZE = 36
BODY_FONT_SIZE = 26
OPTION_FONT_SIZE = 32

POWER_OPTIONS = ["NO", "YES", "RESTART"]


class Screens:
    def __init__(self, size, logo_path, splash_path):
        self.size = size
        self.title_font = pygame.font.Font(config.FONT_PATH, TITLE_FONT_SIZE)
        self.body_font = pygame.font.Font(config.FONT_PATH, BODY_FONT_SIZE)
        self.option_font = pygame.font.Font(config.FONT_PATH, OPTION_FONT_SIZE)
        self.logo = self._load_logo(logo_path)
        self.splash_path = splash_path

    def _load_logo(self, path):
        try:
            img = pygame.image.load(path)
        except (pygame.error, OSError) as exc:
            print(f"Logo load failed: {exc}", file=sys.stderr)
            return None
        w, h = img.get_size()
        # Square-pixel art on the 720-wide NTSC frame: stretch 720/640.
        return pygame.transform.smoothscale(img, (round(w * 720 / 640), h))

    def build_splash(self, version):
        w, h = self.size
        canvas = pygame.Surface(self.size)
        canvas.fill(BLACK)
        vfont = pygame.font.Font(config.FONT_PATH, SPLASH_VERSION_FONT_SIZE)
        v = vfont.render(f"VERSION {version}", True, ORANGE)
        try:
            img = pygame.image.load(self.splash_path)
        except (pygame.error, OSError) as exc:
            print(f"Splash load failed: {exc}", file=sys.stderr)
            img = self.title_font.render("EXITROW", True, ORANGE)
        img_y = (h - img.get_height()) // 2 - SPLASH_Y_OFFSET
        canvas.blit(img, ((w - img.get_width()) // 2, img_y))
        canvas.blit(v, ((w - v.get_width()) // 2, img_y + img.get_height() + SPLASH_VERSION_GAP))
        return canvas

    def build_about(self, title, rows):
        canvas = pygame.Surface(self.size)
        frame_w, frame_h = self.size
        canvas.fill(BLACK)
        title_surf = self.title_font.render(title, True, ORANGE)
        canvas.blit(title_surf, ((frame_w - title_surf.get_width()) // 2, ABOUT_MARGIN_Y))
        top = ABOUT_MARGIN_Y + title_surf.get_height()
        bottom = frame_h - ABOUT_MARGIN_Y
        if self.logo is not None:
            bottom -= self.logo.get_height()
            canvas.blit(self.logo, ((frame_w - self.logo.get_width()) // 2, bottom))
        if not rows:
            return canvas
        label_surfs = [self.body_font.render(label, True, ORANGE) if label else None for label, _ in rows]
        value_surfs = [self.body_font.render(str(value), True, WHITE) for _, value in rows]
        label_w = max((s.get_width() for s in label_surfs if s), default=0)
        value_w = max(s.get_width() for s in value_surfs)
        line_h = self.body_font.get_linesize() + ABOUT_LINE_GAP
        x0 = (frame_w - (label_w + ABOUT_COLUMN_GAP + value_w)) // 2
        y = top + (bottom - top - line_h * len(rows)) // 2
        for label_surf, value_surf in zip(label_surfs, value_surfs):
            if label_surf:
                canvas.blit(label_surf, (x0 + label_w - label_surf.get_width(), y))
            canvas.blit(value_surf, (x0 + label_w + ABOUT_COLUMN_GAP, y))
            y += line_h
        return canvas

    def draw_power_dialog(self, canvas, selection):
        lines = ["ARE YOU SURE YOU WANT", "TO SHUT DOWN?"]
        line_surfs = [self.title_font.render(line, True, WHITE) for line in lines]
        option_surfs = [self.option_font.render(opt, True, BLACK if i == selection else ORANGE)
                        for i, opt in enumerate(POWER_OPTIONS)]
        pad_x, pad_y, gap = 40, 24, 40
        options_w = sum(s.get_width() for s in option_surfs) + gap * (len(option_surfs) - 1)
        content_w = max(max(s.get_width() for s in line_surfs), options_w)
        content_h = (sum(s.get_height() for s in line_surfs) + 10 * (len(line_surfs) - 1)
                     + 30 + option_surfs[0].get_height())
        box = pygame.Surface((content_w + pad_x * 2, content_h + pad_y * 2))
        box.fill(BLACK)
        pygame.draw.rect(box, ORANGE, box.get_rect(), 3)
        y = pad_y
        for surf in line_surfs:
            box.blit(surf, ((box.get_width() - surf.get_width()) // 2, y))
            y += surf.get_height() + 10
        y += 20
        x = (box.get_width() - options_w) // 2
        for i, surf in enumerate(option_surfs):
            if i == selection:
                pygame.draw.rect(box, ORANGE, pygame.Rect(x - 10, y - 6, surf.get_width() + 20, surf.get_height() + 12))
            box.blit(surf, (x, y))
            x += surf.get_width() + gap
        frame_w, frame_h = canvas.get_size()
        canvas.blit(box, ((frame_w - box.get_width()) // 2, (frame_h - box.get_height()) // 2))
