"""Clean, everyday-app look for the translator: rounded cards, soft colours, plain words.

All shapes and text are drawn with Pillow in one pass per frame; the camera picture is
placed with NumPy first. Colours live in the Theme object T (light by default).
"""
import os
import time
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

from common import ROOT, _is_rtl, _has_raqm, shape_rtl


def _rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


THEMES = {
    "light": {
        "bg": "#F3F5F9", "card": "#FFFFFF", "border": "#E3E8EF", "text": "#111827",
        "muted": "#6B7280", "soft": "#F1F4F8", "accent": "#2563EB", "accent_soft": "#E6EEFD",
        "on_accent": "#FFFFFF", "green": "#16A34A", "green_soft": "#E7F6EC", "amber": "#D97706",
        "video_bg": "#0B0D10",
    },
    "dark": {
        "bg": "#0F1115", "card": "#1A1D23", "border": "#2A2F37", "text": "#F3F4F6",
        "muted": "#9CA3AF", "soft": "#23272E", "accent": "#3B82F6", "accent_soft": "#1C2A45",
        "on_accent": "#FFFFFF", "green": "#22C55E", "green_soft": "#15301F", "amber": "#F59E0B",
        "video_bg": "#000000",
    },
}


class Theme:
    def __init__(self, name="light"):
        self.set(name)

    def set(self, name):
        self.name = name
        for k, v in THEMES[name].items():
            setattr(self, k, _rgb(v))

    def toggle(self):
        self.set("dark" if self.name == "light" else "light")

    @property
    def dark(self):
        return self.name == "dark"

    def bgr(self, key):
        r, g, b = getattr(self, key)
        return (b, g, r)


T = Theme("light")

# ---------------------------------------------------------------- fonts
_REG = [os.path.join(ROOT, "fonts", "font.ttf"), "C:/Windows/Fonts/segoeui.ttf",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf", "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansArabic-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]
_BOLD = [os.path.join(ROOT, "fonts", "font-bold.ttf"), "C:/Windows/Fonts/segoeuib.ttf",
         "C:/Windows/Fonts/arialbd.ttf", "/usr/share/fonts/truetype/noto/NotoSansArabic-Bold.ttf",
         "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]
_fonts = {}


def font(size, bold=False):
    key = (size, bold)
    if key not in _fonts:
        f = None
        for p in (_BOLD if bold else []) + _REG:
            if os.path.exists(p):
                f = ImageFont.truetype(p, size)
                break
        _fonts[key] = f or ImageFont.load_default()
    return _fonts[key]


def text_width(s, size, bold=False):
    f = font(size, bold)
    if _is_rtl(s):
        return f.getlength(s, direction="rtl") if _has_raqm() else f.getlength(shape_rtl(s))
    return f.getlength(s)


# ---------------------------------------------------------------- canvas
class Canvas:
    """Draw on a BGR frame with Pillow: opaque shapes, see-through shapes, then text on top."""

    def __init__(self, img_bgr):
        self.out = img_bgr
        self.im = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)).convert("RGBA")
        self.ov = Image.new("RGBA", self.im.size, (0, 0, 0, 0))
        self.d = ImageDraw.Draw(self.im)
        self.o = ImageDraw.Draw(self.ov)
        self.texts = []

    def box(self, x, y, w, h, fill, radius=16, outline=None, width=1, alpha=None):
        xy = (int(x), int(y), int(x + w), int(y + h))
        if alpha is not None:
            self.o.rounded_rectangle(xy, radius, fill=tuple(fill) + (int(alpha * 255),))
        else:
            self.d.rounded_rectangle(xy, radius, fill=fill, outline=outline, width=width)

    def circle(self, cx, cy, r, fill, outline=None, width=1):
        self.d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=fill, outline=outline, width=width)

    def text(self, s, xy, size=16, color=None, anchor="la", bold=False):
        if s:
            self.texts.append((str(s), xy, size, color or T.text, anchor, bold))

    def finish(self):
        img = Image.alpha_composite(self.im, self.ov)
        d = ImageDraw.Draw(img)
        raqm = _has_raqm()
        for s, xy, size, color, anchor, bold in self.texts:
            f = font(size, bold)
            if _is_rtl(s):
                if raqm:
                    d.text(xy, s, font=f, fill=color, anchor=anchor, direction="rtl")
                else:
                    d.text(xy, shape_rtl(s), font=f, fill=color, anchor=anchor)
            else:
                d.text(xy, s, font=f, fill=color, anchor=anchor)
        self.out[:] = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)
        return self.out


def inside(pt, r):
    return pt is not None and r[0] <= pt[0] <= r[0] + r[2] and r[1] <= pt[1] <= r[1] + r[3]


def button(c, x, y, w, h, label, style="secondary", hover=False, size=17):
    """Rounded button. style: primary | secondary | ghost. Returns its rect."""
    if style == "primary":
        fill = T.accent if not hover else tuple(min(255, int(v * 1.12)) for v in T.accent)
        c.box(x, y, w, h, fill, radius=h // 2)
        fg = T.on_accent
    elif style == "disabled":
        c.box(x, y, w, h, T.border, radius=h // 2)
        fg = T.muted
    elif style == "ghost":
        c.box(x, y, w, h, T.soft if hover else T.card, radius=h // 2, outline=T.border, width=1)
        fg = T.text
    else:
        c.box(x, y, w, h, T.accent_soft if hover else T.soft, radius=h // 2)
        fg = T.text
    c.text(label, (x + w / 2, y + h / 2), size, fg, "mm", bold=True)
    return (x, y, w, h)


# ---------------------------------------------------------------- camera
HAND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10),
              (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18),
              (18, 19), (19, 20), (0, 17)]
POSE_EDGES = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24)]
_masks = {}


def _round_mask(w, h, r):
    key = (w, h, r)
    if key not in _masks:
        m = Image.new("L", (w * 2, h * 2), 0)
        ImageDraw.Draw(m).rounded_rectangle((0, 0, w * 2 - 1, h * 2 - 1), r * 2, fill=255)
        _masks[key] = (np.array(m.resize((w, h), Image.LANCZOS)) / 255.0)[..., None]
    return _masks[key]


def camera(img, frame, results, x, y, w, h, show_tracking=False, radius=22):
    """Places the whole camera picture (never cropped) in a rounded box.
    With show_tracking the body + hand points are drawn on top."""
    view = np.empty((h, w, 3), np.uint8)
    view[:] = T.bgr("video_bg")
    if frame is not None:
        fh, fw = frame.shape[:2]
        s = min(w / fw, h / fh)
        rw, rh = int(round(fw * s)), int(round(fh * s))
        ox, oy = (w - rw) // 2, (h - rh) // 2
        view[oy:oy + rh, ox:ox + rw] = cv2.resize(frame, (rw, rh), interpolation=cv2.INTER_AREA)
        if show_tracking:
            P = lambda q: (int(q.x * rw + ox), int(q.y * rh + oy))
            acc = T.bgr("accent")
            pose = getattr(results, "pose_landmarks", None)
            if pose is not None:
                lm = pose.landmark
                vis = [float(getattr(q, "visibility", 1.0) or 0.0) for q in lm]
                for a, b in POSE_EDGES:
                    if a < len(lm) and b < len(lm) and vis[a] > 0.5 and vis[b] > 0.5:
                        cv2.line(view, P(lm[a]), P(lm[b]), (255, 255, 255), 4, cv2.LINE_AA)
                        cv2.line(view, P(lm[a]), P(lm[b]), acc, 2, cv2.LINE_AA)
            for lms in (getattr(results, "left_hand_landmarks", None),
                        getattr(results, "right_hand_landmarks", None)):
                if lms is None:
                    continue
                pts = [P(q) for q in lms.landmark]
                for a, b in HAND_EDGES:
                    cv2.line(view, pts[a], pts[b], (255, 255, 255), 3, cv2.LINE_AA)
                for q in pts:
                    cv2.circle(view, q, 4, acc, -1, cv2.LINE_AA)
    m = _round_mask(w, h, radius)
    bg = np.empty_like(view)
    bg[:] = T.bgr("bg")
    img[y:y + h, x:x + w] = (view * m + bg * (1 - m)).astype(np.uint8)
