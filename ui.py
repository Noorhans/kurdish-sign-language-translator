"""Drawing kit shared by app.py and collect.py: themes (dark / light), panels, buttons,
and the camera viewport with full-body + hand skeleton overlay.

Shapes are drawn with OpenCV; all text is queued and drawn in ONE Pillow pass per frame
(fast, and the only way to get properly joined right-to-left Kurdish script).
"""
import os
import time
import numpy as np
import cv2

from common import ROOT, _FONT_CANDIDATES, _is_rtl, _has_raqm, shape_rtl


def _bgr(hex_):
    hex_ = hex_.lstrip("#")
    r, g, b = (int(hex_[i:i + 2], 16) for i in (0, 2, 4))
    return (b, g, r)


# ---------------------------------------------------------------- themes
THEMES = {
    "dark": {
        "bg": "#070B10", "panel": "#0D141C", "panel2": "#132030", "line": "#1C2A39",
        "grid": "#0F1922", "text": "#E6EDF3", "muted": "#8193A6", "dim": "#3A4A5C",
        "cyan": "#00E5FF", "green": "#2BFF9E", "amber": "#FFB020", "red": "#FF3B5C",
        "on_accent": "#04121A",
    },
    "light": {
        "bg": "#EEF3F7", "panel": "#FFFFFF", "panel2": "#F2F6FA", "line": "#CBD6E1",
        "grid": "#DCE4EC", "text": "#0B1420", "muted": "#4A5868", "dim": "#93A2B1",
        "cyan": "#007E9C", "green": "#08875A", "amber": "#B45F00", "red": "#D7263D",
        "on_accent": "#FFFFFF",
    },
}


class Theme:
    """Current colours. Always read them as T.text, T.cyan ... at draw time."""

    def __init__(self, name="dark"):
        self.set(name)

    def set(self, name):
        self.name = name
        for k, v in THEMES[name].items():
            setattr(self, k, _bgr(v))

    def toggle(self):
        self.set("light" if self.name == "dark" else "dark")

    @property
    def dark(self):
        return self.name == "dark"


T = Theme("dark")

# ---------------------------------------------------------------- fonts
_MONO_CANDIDATES = [
    os.path.join(ROOT, "fonts", "mono.ttf"),
    "C:/Windows/Fonts/consola.ttf",
    "C:/Windows/Fonts/cour.ttf",
    "/System/Library/Fonts/Menlo.ttc",
    "/System/Library/Fonts/Monaco.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
]
_MONO_BOLD = [
    os.path.join(ROOT, "fonts", "mono-bold.ttf"),
    "C:/Windows/Fonts/consolab.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
]
_fonts = {}


def _load(cands, size):
    from PIL import ImageFont
    for p in cands:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return None


def font(size, mono=False, bold=False):
    key = (size, mono, bold)
    if key not in _fonts:
        f = None
        if mono and bold:
            f = _load(_MONO_BOLD, size)
        if f is None and mono:
            f = _load(_MONO_CANDIDATES, size)
        if f is None:
            f = _load(_FONT_CANDIDATES, size)
        if f is None:
            from PIL import ImageFont
            f = ImageFont.load_default()
        _fonts[key] = f
    return _fonts[key]


class Painter:
    """Collects text for one frame, then draws it all at once with flush()."""

    def __init__(self, img):
        self.img = img
        self.items = []

    def text(self, s, xy, size=20, color=None, anchor="la", mono=False, bold=False):
        if s:
            self.items.append((str(s), xy, size, color or T.text, anchor, mono, bold))

    def width(self, s, size=20, mono=False, bold=False):
        f = font(size, mono, bold)
        if _is_rtl(s):
            if _has_raqm():
                return f.getlength(s, direction="rtl")
            return f.getlength(shape_rtl(s))
        return f.getlength(s)

    def flush(self):
        if not self.items:
            return self.img
        from PIL import Image, ImageDraw
        pil = Image.fromarray(cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB))
        d = ImageDraw.Draw(pil)
        raqm = _has_raqm()
        for s, xy, size, color, anchor, mono, bold in self.items:
            f = font(size, mono, bold)
            fill = color[::-1]
            if _is_rtl(s):
                if raqm:
                    d.text(xy, s, font=f, fill=fill, anchor=anchor, direction="rtl")
                else:
                    d.text(xy, shape_rtl(s), font=f, fill=fill, anchor=anchor)
            else:
                d.text(xy, s, font=f, fill=fill, anchor=anchor)
        self.img[:] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        self.items.clear()
        return self.img


# ---------------------------------------------------------------- shapes
def rect(img, x, y, w, h, color, thick=-1):
    cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), color, thick, cv2.LINE_AA)


def blend_rect(img, x, y, w, h, color, alpha):
    x, y, w, h = int(x), int(y), int(w), int(h)
    x0, y0 = max(x, 0), max(y, 0)
    roi = img[y0:y + h, x0:x + w]
    if roi.size:
        roi[:] = (roi * (1 - alpha) + np.array(color) * alpha).astype(np.uint8)


def brackets(img, x, y, w, h, color, l=18, t=2):
    """Corner brackets around a box (HUD style)."""
    x, y, w, h = int(x), int(y), int(w), int(h)
    for cx, cy, dx, dy in ((x, y, 1, 1), (x + w, y, -1, 1), (x, y + h, 1, -1), (x + w, y + h, -1, -1)):
        cv2.line(img, (cx, cy), (cx + dx * l, cy), color, t, cv2.LINE_AA)
        cv2.line(img, (cx, cy), (cx, cy + dy * l), color, t, cv2.LINE_AA)


def panel(img, p, x, y, w, h, title=None, accent=None, right_note=None):
    rect(img, x, y, w, h, T.panel)
    rect(img, x, y, w, h, T.line, 1)
    cv2.line(img, (int(x), int(y)), (int(x + 40), int(y)), accent or T.cyan, 3)
    if title:
        p.text(title, (x + 14, y + 11), 13, T.muted, mono=True, bold=True)
    if right_note:
        p.text(right_note, (x + w - 14, y + 11), 13, T.muted, "ra", mono=True)


def bar(img, x, y, w, h, frac, color, marker=None, bg=None):
    frac = float(np.clip(frac, 0, 1))
    rect(img, x, y, w, h, bg or T.line)
    if frac > 0:
        rect(img, x, y, max(1, w * frac), h, color)
    if marker is not None:
        mx = int(x + w * marker)
        cv2.line(img, (mx, int(y - 4)), (mx, int(y + h + 4)), T.text, 1, cv2.LINE_AA)


def dot(img, x, y, r, color, ring=False):
    cv2.circle(img, (int(x), int(y)), r, color, 1 if ring else -1, cv2.LINE_AA)


def pulse(speed=3.0):
    """0..1 value that pulses over time (for blinking indicators)."""
    return 0.5 + 0.5 * np.sin(time.time() * speed)


def pill(img, p, x, y, label, color, anchor_right=False, size=14):
    """Status pill with a glowing dot. Returns its width."""
    tw = p.width(label, size, mono=True, bold=True)
    w, h = tw + 40, 30
    if anchor_right:
        x = x - w
    blend_rect(img, x, y, w, h, color, 0.14)
    rect(img, x, y, w, h, color, 1)
    dot(img, x + 15, y + h / 2, 5, color)
    p.text(label, (x + 28, y + h / 2), size, color, "lm", mono=True, bold=True)
    return w


def keycap(img, p, x, y, key, label):
    """Draws  [KEY] label  and returns the x after it."""
    kw = max(28, p.width(key, 13, mono=True, bold=True) + 16)
    rect(img, x, y, kw, 24, T.panel2)
    rect(img, x, y, kw, 24, T.dim, 1)
    p.text(key, (x + kw / 2, y + 12), 13, T.text, "mm", mono=True, bold=True)
    p.text(label, (x + kw + 8, y + 12), 13, T.muted, "lm", mono=True)
    return x + kw + 8 + p.width(label, 13, mono=True) + 22


def button(img, p, x, y, w, h, label, key, hover=False, primary=False, active=False):
    """Clickable-looking button with its keyboard shortcut. Returns its rect."""
    if primary:
        rect(img, x, y, w, h, T.cyan)
        fg, kc = T.on_accent, T.on_accent
    else:
        rect(img, x, y, w, h, T.panel2)
        if hover or active:
            blend_rect(img, x, y, w, h, T.cyan, 0.16)
        rect(img, x, y, w, h, T.cyan if (hover or active) else T.line, 1)
        fg, kc = T.text, T.muted
    if primary and hover:
        rect(img, x - 2, y - 2, w + 4, h + 4, T.text, 1)
    kw = p.width(key, 11, mono=True, bold=True) + 12
    rect(img, x + 10, y + h / 2 - 10, kw, 20, kc, 1)
    p.text(key, (x + 10 + kw / 2, y + h / 2), 11, kc, "mm", mono=True, bold=True)
    p.text(label, (x + 10 + kw + (w - 20 - kw) / 2, y + h / 2), 15, fg, "mm", bold=True)
    return (x, y, w, h)


def inside(pt, r):
    return pt is not None and r[0] <= pt[0] <= r[0] + r[2] and r[1] <= pt[1] <= r[1] + r[3]


# ---------------------------------------------------------------- camera viewport
HAND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10),
              (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18),
              (18, 19), (19, 20), (0, 17)]
TIPS = (4, 8, 12, 16, 20)
POSE_EDGES = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24),
              (23, 25), (25, 27), (24, 26), (26, 28), (27, 29), (29, 31), (27, 31), (28, 30),
              (30, 32), (28, 32), (15, 17), (15, 19), (17, 19), (16, 18), (16, 20), (18, 20)]
POSE_JOINTS = (11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28)


def _cover(fw, fh, w, h):
    s = max(w / fw, h / fh)
    return s, (w - fw * s) / 2.0, (h - fh * s) / 2.0


def viewport(img, p, results, frame, x, y, w, h, mode="camera", frame_color=None, label=None):
    """The real camera picture, shown whole (nothing cropped).

    mode "camera":   the person exactly as the camera sees them; tracking runs invisibly
    mode "tracking": the same picture with the body + hand tracking drawn on top
    Returns (number of hands found, body found)."""
    x, y, w, h = int(x), int(y), int(w), int(h)
    fh, fw = frame.shape[:2] if frame is not None else (720, 1280)
    s = min(w / fw, h / fh)                       # fit: the whole picture is visible
    ox, oy = (w - fw * s) / 2.0, (h - fh * s) / 2.0
    view = np.zeros((h, w, 3), np.uint8)          # bars around a 4:3 webcam stay black
    if frame is not None:
        rw, rh = int(round(fw * s)), int(round(fh * s))
        x0, y0 = int(round(ox)), int(round(oy))
        small = cv2.resize(frame, (rw, rh), interpolation=cv2.INTER_AREA)
        view[y0:y0 + rh, x0:x0 + rw] = small[:h - y0, :w - x0]
    draw = mode == "tracking"

    def P(q):
        return (int(q.x * fw * s + ox), int(q.y * fh * s + oy))

    def seg(a, b, col, thick):
        cv2.line(view, a, b, (10, 10, 10), thick + 3, cv2.LINE_AA)
        cv2.line(view, a, b, col, thick, cv2.LINE_AA)

    # ---- body
    body = False
    pose = getattr(results, "pose_landmarks", None)
    if pose is not None:
        lm = pose.landmark
        n = len(lm)
        vis = [float(getattr(q, "visibility", 1.0) or 0.0) if hasattr(q, "visibility") else 1.0 for q in lm]
        pts = [P(q) for q in lm]
        body_col = T.cyan
        for a, b in (POSE_EDGES if draw else []):
            if a < n and b < n and vis[a] > 0.5 and vis[b] > 0.5:
                seg(pts[a], pts[b], body_col, 2)
        for i in (POSE_JOINTS if draw else []):
            if i < n and vis[i] > 0.5:
                cv2.circle(view, pts[i], 5, T.bg, -1, cv2.LINE_AA)
                cv2.circle(view, pts[i], 5, body_col, 2, cv2.LINE_AA)
        body = n > 12 and vis[11] > 0.5 and vis[12] > 0.5
        if draw and n > 12 and body:   # little "TARGET" tag over the head
            hx, hy = pts[0]
            sw = abs(pts[11][0] - pts[12][0])
            r = max(24, int(sw * 0.45))
            brackets(view, hx - r, hy - r, 2 * r, 2 * r, body_col, 10, 1)

    # ---- hands
    n_hands, tags = 0, []
    for lms, col, tag in ((getattr(results, "left_hand_landmarks", None), T.cyan, "L"),
                          (getattr(results, "right_hand_landmarks", None), T.green, "R")):
        if lms is None:
            continue
        n_hands += 1
        if not draw:
            continue
        pts = [P(q) for q in lms.landmark]
        for a, b in HAND_EDGES:
            seg(pts[a], pts[b], col, 2)
        for i, q in enumerate(pts):
            r = 6 if i in TIPS else 4
            cv2.circle(view, q, r, (20, 20, 20), -1, cv2.LINE_AA)
            cv2.circle(view, q, r, col, 2 if i in TIPS else 1, cv2.LINE_AA)
        xs, ys = [q[0] for q in pts], [q[1] for q in pts]
        bx, by = min(xs) - 14, min(ys) - 14
        bw, bh = max(xs) - min(xs) + 28, max(ys) - min(ys) + 28
        brackets(view, bx, by, bw, bh, col, 10, 2)
        tags.append((f"HAND_{tag}", x + bx, y + by - 18, col))

    img[y:y + h, x:x + w] = view
    for t, tx, ty, col in tags:
        if y <= ty <= y + h - 16 and x <= tx <= x + w - 70:
            blend_rect(img, tx - 4, ty - 2, p.width(t, 12, True, True) + 8, 17, T.bg, 0.75)
            p.text(t, (tx, ty), 12, col, mono=True, bold=True)

    rect(img, x, y, w, h, T.line, 1)
    brackets(img, x - 6, y - 6, w + 12, h + 12, frame_color or T.cyan, 24, 3)
    if label:
        tw = p.width(label, 13, mono=True, bold=True)
        blend_rect(img, x + 8, y + 8, tw + 20, 26, T.bg, 0.7)
        p.text(label, (x + 18, y + 21), 13, T.text, "lm", mono=True, bold=True)
    return n_hands, body
