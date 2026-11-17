"""Shared helpers: phrase list, landmark extraction, normalisation, drawing, Kurdish text."""
import os
import glob
import time
import json
import numpy as np
import cv2

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
MODEL_DIR = os.path.join(ROOT, "model")
PHRASES_FILE = os.path.join(ROOT, "phrases.txt")

SEQ_LEN = 30          # frames per sample (~1 s at 30 fps)
IDLE_ID = "idle"      # "not signing" class, always included

# Upper-body pose landmarks only (0-24: face points, shoulders, elbows, wrists, hips).
POSE_IDX = list(range(25))
N_POSE, N_HAND = len(POSE_IDX), 21
FEAT_DIM = (N_POSE + 2 * N_HAND) * 3   # 201 values per frame


# --------------------------------------------------------------------------- phrases
def load_phrases(path=PHRASES_FILE):
    """Returns list of dicts {id, ku, en}. 'idle' is always index 0."""
    phrases = [{"id": IDLE_ID, "ku": "", "en": "(idle)"}]
    seen = {IDLE_ID}
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split("|")]
            pid = parts[0]
            if not pid.replace("_", "").isalnum() or not pid.isascii():
                raise ValueError(f"phrases.txt line {n}: id '{pid}' must be English letters/numbers/_ only")
            if pid in seen:
                raise ValueError(f"phrases.txt line {n}: duplicate id '{pid}'")
            seen.add(pid)
            ku = parts[1] if len(parts) > 1 and parts[1] else pid
            en = parts[2] if len(parts) > 2 else ""
            phrases.append({"id": pid, "ku": ku, "en": en})
    return phrases


def count_samples(pid):
    return len(glob.glob(os.path.join(DATA_DIR, pid, "*.npy")))


# --------------------------------------------------------------------------- mediapipe
# Uses the MediaPipe Tasks API (Hand + Pose landmarkers), which works on all current
# mediapipe versions. Model files (~10 MB) download once into mp_models/ on first run.
MP_MODEL_DIR = os.path.join(ROOT, "mp_models")
_MP_URLS = {
    "hand_landmarker.task": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
                            "hand_landmarker/float16/latest/hand_landmarker.task",
    "pose_landmarker_lite.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
                                 "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
}


def _mp_model(name):
    path = os.path.join(MP_MODEL_DIR, name)
    if not os.path.exists(path):
        import urllib.request
        os.makedirs(MP_MODEL_DIR, exist_ok=True)
        print(f"Downloading {name} (one time only)...")
        urllib.request.urlretrieve(_MP_URLS[name], path + ".part")
        os.replace(path + ".part", path)
    return path


class _LMList:
    def __init__(self, pts):
        self.landmark = pts


class Results:
    def __init__(self, pose=None, left=None, right=None):
        self.pose_landmarks = _LMList(pose) if pose else None
        self.left_hand_landmarks = _LMList(left) if left else None
        self.right_hand_landmarks = _LMList(right) if right else None


class Tracker:
    """Finds both hands + upper body in each frame. Frames must already be mirrored."""

    def __init__(self):
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python import vision
        mode = vision.RunningMode.VIDEO
        self.hands = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=_mp_model("hand_landmarker.task")),
            running_mode=mode, num_hands=2, min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5, min_tracking_confidence=0.5))
        self.pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=_mp_model("pose_landmarker_lite.task")),
            running_mode=mode, num_poses=1))
        self.ts = 0

    def process(self, frame_bgr):
        import mediapipe as mp
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self.ts = max(self.ts + 1, int(time.monotonic() * 1000))   # must keep increasing
        hr = self.hands.detect_for_video(img, self.ts)
        pr = self.pose.detect_for_video(img, self.ts)

        left = right = None
        found = []
        for lms, hd in zip(hr.hand_landmarks, hr.handedness):
            found.append((hd[0].category_name if hd else "", lms))
        # if both hands got the same label, decide by position (person's right hand is on
        # the right side of a mirrored image)
        if len(found) == 2 and found[0][0] == found[1][0]:
            found.sort(key=lambda t: t[1][0].x)
            found = [("Left", found[0][1]), ("Right", found[1][1])]
        for label, lms in found:
            if label == "Left" and left is None:
                left = lms
            elif right is None:
                right = lms
            else:
                left = lms
        pose = pr.pose_landmarks[0] if pr.pose_landmarks else None
        return Results(pose, left, right)

    def close(self):
        self.hands.close(); self.pose.close()


def make_holistic():
    return Tracker()


def run_holistic(tracker, frame_bgr):
    return tracker.process(frame_bgr)


def _lm_array(lms, idx=None, n=None):
    if lms is None:
        return np.zeros(((len(idx) if idx else n), 3), dtype=np.float32), False
    pts = lms.landmark
    if idx:
        pts = [pts[i] for i in idx]
    return np.array([[p.x, p.y, p.z] for p in pts], dtype=np.float32), True


def extract_raw(results):
    """Raw (un-normalised) landmarks for one frame + which parts were found."""
    pose, has_pose = _lm_array(results.pose_landmarks, POSE_IDX)
    lh, has_lh = _lm_array(results.left_hand_landmarks, n=N_HAND)
    rh, has_rh = _lm_array(results.right_hand_landmarks, n=N_HAND)
    return pose, lh, rh, has_pose, has_lh, has_rh


def normalise(pose, lh, rh, has_pose, has_lh, has_rh):
    """Make features independent of where the person stands and how far from the camera.
    Origin = middle of shoulders, unit = shoulder width. Missing parts stay 0."""
    if has_pose:
        ls, rs = pose[11], pose[12]
        centre = (ls + rs) / 2.0
        scale = float(np.linalg.norm(ls[:2] - rs[:2])) or 1.0
    else:
        hands = [h for h, ok in ((lh, has_lh), (rh, has_rh)) if ok]
        centre = np.mean(np.concatenate(hands), axis=0) if hands else np.zeros(3, np.float32)
        scale = 0.3
    out = []
    for arr, ok in ((pose, has_pose), (lh, has_lh), (rh, has_rh)):
        out.append(((arr - centre) / scale) if ok else np.zeros_like(arr))
    return np.concatenate(out).reshape(-1).astype(np.float32)


def frame_features(results):
    raw = extract_raw(results)
    hands_visible = raw[4] or raw[5]
    return normalise(*raw), hands_visible


# --------------------------------------------------------------------------- drawing
HAND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10),
              (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18),
              (18, 19), (19, 20), (0, 17)]


def draw_hands_only(results, w, h, canvas=None):
    """Draws ONLY the hand skeletons on a dark canvas (face and body are never shown)."""
    if canvas is None:
        canvas = np.full((h, w, 3), (24, 20, 18), dtype=np.uint8)
    for lms, color in ((results.left_hand_landmarks, (255, 170, 60)),
                       (results.right_hand_landmarks, (90, 220, 120))):
        if lms is None:
            continue
        pts = [(int(p.x * w), int(p.y * h)) for p in lms.landmark]
        for a, b in HAND_EDGES:
            cv2.line(canvas, pts[a], pts[b], color, 3, cv2.LINE_AA)
        for p in pts:
            cv2.circle(canvas, p, 5, (255, 255, 255), -1, cv2.LINE_AA)
    return canvas


# --------------------------------------------------------------------------- Kurdish text
_FONT_CANDIDATES = [
    os.path.join(ROOT, "fonts", "NotoSansArabic-Regular.ttf"),
    os.path.join(ROOT, "fonts", "font.ttf"),
    "C:/Windows/Fonts/segoeui.ttf",
    "C:/Windows/Fonts/tahoma.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "/System/Library/Fonts/Supplemental/GeezaPro.ttc",
    "/Library/Fonts/Arial Unicode.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansArabic-Regular.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]
_font_cache = {}


def _font(size):
    from PIL import ImageFont
    if size not in _font_cache:
        for path in _FONT_CANDIDATES:
            if os.path.exists(path):
                _font_cache[size] = ImageFont.truetype(path, size)
                break
        else:
            _font_cache[size] = ImageFont.load_default()
    return _font_cache[size]


def shape_rtl(text):
    """Joins Arabic-script letters and orders them right-to-left so they display correctly."""
    import arabic_reshaper
    try:
        from bidi import get_display
    except ImportError:
        from bidi.algorithm import get_display
    return get_display(arabic_reshaper.reshape(text))


_RAQM = None


def _has_raqm():
    global _RAQM
    if _RAQM is None:
        from PIL import features
        _RAQM = bool(features.check("raqm"))
    return _RAQM


def _is_rtl(text):
    return any("؀" <= c <= "ۿ" or "ﭐ" <= c <= "﻿" for c in text)


def draw_text(img, text, xy, size=36, color=(255, 255, 255), anchor="la"):
    """Draws Kurdish/English text on an OpenCV image (cv2.putText cannot do Kurdish)."""
    from PIL import Image, ImageDraw
    if not text:
        return img
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(pil)
    if _has_raqm():      # Pillow shapes Arabic script itself
        kw = {"direction": "rtl"} if _is_rtl(text) else {}
        d.text(xy, text, font=_font(size), fill=color[::-1], anchor=anchor, **kw)
    else:                # basic Pillow: join letters + reverse ourselves
        d.text(xy, shape_rtl(text) if _is_rtl(text) else text, font=_font(size),
               fill=color[::-1], anchor=anchor)
    img[:] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    return img


def save_labels(phrases):
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(os.path.join(MODEL_DIR, "labels.json"), "w", encoding="utf-8") as f:
        json.dump(phrases, f, ensure_ascii=False, indent=2)


def load_labels():
    with open(os.path.join(MODEL_DIR, "labels.json"), encoding="utf-8") as f:
        return json.load(f)
