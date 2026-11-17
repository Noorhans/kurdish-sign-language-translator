"""Live Kurdish Sign Language -> Kurdish text translator.

Usage:
    python app.py                    # light theme
    python app.py --theme dark
    python app.py --camera 1

Everything works with the mouse. Keyboard shortcuts too:
    ENTER      make a sentence from your words
    BACKSPACE  undo the last word
    C          clear all
    V          show / hide the tracking points on the video
    T          light / dark mode
    F          full screen
    + / -      stricter / more relaxed recognition
    Q / ESC    quit

The camera picture is shown whole and as-is, and never saved.
Sentence building uses Claude if ANTHROPIC_API_KEY is set; otherwise words are just joined.
"""
import argparse
import os
import threading
import time
from collections import deque
import numpy as np
import cv2

from common import (SEQ_LEN, IDLE_ID, MODEL_DIR, load_labels, make_holistic, run_holistic,
                    frame_features)
import friendly as fx
from friendly import T

W, H = 1280, 720
CX, CY, CW, CH = 24, 80, 832, 468          # camera
WX, WY, WW, WH = 24, 564, 832, 140         # your words
RX, RW = 880, 376                          # right column

SENTENCE_MODEL = "claude-haiku-5-5"


class Recogniser:
    """Sliding-window recogniser with smoothing so one sign = one word."""

    def __init__(self, model, labels, threshold=0.8, stable=3, every=2):
        self.model, self.labels = model, labels
        self.idle = next((i for i, l in enumerate(labels) if l["id"] == IDLE_ID), -1)
        self.threshold, self.stable, self.every = threshold, stable, every
        self.window = deque(maxlen=SEQ_LEN)
        self.hands = deque(maxlen=SEQ_LEN)
        self.history = deque(maxlen=5)
        self.frame_i = 0
        self.candidate, self.streak = None, 0
        self.last_word, self.last_time = None, 0.0
        self.cooldown_until = 0.0
        self.released = True
        self.probs = np.zeros(len(labels))

    def reset_window(self):
        self.window.clear(); self.hands.clear(); self.history.clear()
        self.candidate, self.streak = None, 0

    def push(self, feat, hands_visible):
        """Returns an accepted label index or None."""
        self.window.append(feat); self.hands.append(hands_visible)
        self.frame_i += 1
        now = time.time()
        if len(self.window) < SEQ_LEN or self.frame_i % self.every:
            return None
        if sum(self.hands) < SEQ_LEN * 0.4:          # no hands -> nothing to read
            self.history.clear(); self.probs[:] = 0
            self.candidate, self.streak, self.released = None, 0, True
            return None
        x = np.expand_dims(np.stack(self.window), 0)
        p = self.model(x, training=False).numpy()[0]
        self.history.append(p)
        self.probs = np.mean(self.history, axis=0)
        top = int(self.probs.argmax())
        conf = float(self.probs[top])

        if top == self.idle or conf < self.threshold or now < self.cooldown_until:
            self.candidate, self.streak = None, 0
            self.released = True
            return None
        if top != self.last_word:
            self.released = True
        if top == self.candidate:
            self.streak += 1
        else:
            self.candidate, self.streak = top, 1
        if self.streak < self.stable:
            return None
        # the same word twice in a row only after the hands relax in between
        if top == self.last_word and not self.released:
            return None
        self.last_word, self.last_time, self.released = top, now, False
        self.cooldown_until = now + 0.6
        self.reset_window()
        return top


def build_sentence(words, out):
    """Turns a list of signed glosses into one natural Kurdish (Sorani) sentence."""
    fallback = " ".join(words)
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        out["text"], out["busy"] = fallback, False
        return
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=key)
        msg = client.messages.create(
            model=SENTENCE_MODEL, max_tokens=120,
            messages=[{"role": "user", "content":
                       "These words were recognised in order from Kurdish Sign Language. Sign "
                       "language has its own grammar, so rewrite them as ONE short, natural "
                       "Sorani Kurdish sentence with the same meaning. Do not add new ideas. "
                       "Reply with the sentence only, in Kurdish script.\n\nWords: " + " | ".join(words)}])
        out["text"] = msg.content[0].text.strip() or fallback
    except Exception as e:                    # no internet etc. -> still show something
        print("sentence building failed:", e)
        out["text"] = fallback
    out["busy"] = False




# ------------------------------------------------------------------------------ screen
def draw_ui(img, frame, res, hands_now, rec, labels, words, sentence, s, last_hit):
    """Draws one frame. Returns {action: rect} for every clickable button."""
    now = time.time()
    flash = now - last_hit < 1.8
    rects = {}
    hov = lambda r: fx.inside(s["mouse"], r)

    img[:] = T.bgr("bg")
    fx.camera(img, frame, res, CX, CY, CW, CH, show_tracking=s["tracking"])
    c = fx.Canvas(img)

    # ---- top bar
    c.circle(44, 40, 20, T.accent)
    c.text("KS", (44, 40), 15, T.on_accent, "mm", bold=True)
    c.text("Sign Translator", (76, 30), 22, T.text, "lm", bold=True)
    c.text("وەرگێڕی زمانی ئاماژەی کوردی", (76, 56), 14, T.muted, "lm")
    x = W - 24
    for action, label in (("full", "Exit full screen" if s["full"] else "Full screen"),
                          ("theme", "Dark mode" if not T.dark else "Light mode"),
                          ("tracking", "Hide tracking" if s["tracking"] else "Show tracking")):
        bw = fx.text_width(label, 15, True) + 36
        x -= bw
        r = (x, 18, bw, 40)
        rects[action] = fx.button(c, *r, label, "ghost", hov(r), size=15)
        x -= 10

    # ---- status chip on the video
    if sentence["busy"]:
        st, col, ku = "Making your sentence…", T.accent, ""
    elif flash:
        st, col, ku = "Got it!", T.green, ""
    elif hands_now:
        st, col, ku = "Reading your sign…", T.amber, ""
    else:
        st, col, ku = "Raise your hands to start", T.muted, "دەستەکانت بەرز بکەرەوە"
    tw = fx.text_width(st, 16, True) + (fx.text_width(ku, 16) + 18 if ku else 0)
    c.box(CX + 16, CY + 16, tw + 52, 42, T.card, radius=21)
    c.circle(CX + 38, CY + 37, 6, col)
    c.text(st, (CX + 54, CY + 37), 16, T.text, "lm", bold=True)
    if ku:
        c.text(ku, (CX + 54 + fx.text_width(st, 16, True) + 18, CY + 37), 16, T.muted, "lm")

    # ---- subtitle on the video
    if flash and rec.last_word is not None:
        wd = labels[rec.last_word]
        bw = max(fx.text_width(wd["ku"], 40, True), fx.text_width(wd["en"], 16)) + 72
        bh = 96 if (s["en"] and wd["en"]) else 72
        bx, by = CX + CW / 2 - bw / 2, CY + CH - bh - 20
        c.box(bx, by, bw, bh, (0, 0, 0), radius=20, alpha=0.62)
        c.text(wd["ku"], (CX + CW / 2, by + 36), 40, (255, 255, 255), "mm", bold=True)
        if s["en"] and wd["en"]:
            c.text(wd["en"], (CX + CW / 2, by + 74), 16, (209, 213, 219), "mm")

    # ---- toast
    if s["toast"] and now - s["toast"][1] < 1.6:
        t = s["toast"][0]
        bw = fx.text_width(t, 15, True) + 40
        c.box(CX + CW / 2 - bw / 2, CY + 16, bw, 40, (17, 24, 39), radius=20, alpha=0.8)
        c.text(t, (CX + CW / 2, CY + 36), 15, (255, 255, 255), "mm", bold=True)

    # ---- detected sign card
    y = CY
    c.box(RX, y, RW, 280, T.card, radius=20, outline=T.border)
    c.text("Detected sign", (RX + 24, y + 22), 15, T.muted, bold=True)
    order = [int(i) for i in np.argsort(rec.probs)[::-1]
             if rec.probs[i] > 0.01 and labels[i]["id"] != IDLE_ID]
    if flash and rec.last_word is not None:
        idx, p, col, note = rec.last_word, 1.0, T.green, "Got it"
    elif order and hands_now:
        idx = order[0]
        p = float(rec.probs[idx])
        ok = p >= rec.threshold
        col, note = (T.green if ok else T.accent), f"{p * 100:.0f}% sure"
    else:
        idx, p, col, note = None, 0.0, T.accent, ""
    if idx is not None:
        ku, size = labels[idx]["ku"], 50
        while size > 24 and fx.text_width(ku, size, True) > RW - 48:
            size -= 2
        c.text(ku, (RX + RW / 2, y + 116), size, T.text, "mm", bold=True)
        if s["en"] and labels[idx]["en"]:
            c.text(labels[idx]["en"], (RX + RW / 2, y + 172), 18, T.muted, "mm")
    else:
        c.text("Waiting for a sign", (RX + RW / 2, y + 128), 20, T.muted, "mm")
    c.box(RX + 24, y + 222, RW - 48, 10, T.soft, radius=5)
    if p > 0:
        c.box(RX + 24, y + 222, max(10, (RW - 48) * min(p, 1.0)), 10, col, radius=5)
    c.text(note, (RX + 24, y + 252), 15, col if note == "Got it" else T.muted, "lm", bold=note == "Got it")

    # ---- how it works card
    y = CY + 296
    c.box(RX, y, RW, 172, T.card, radius=20, outline=T.border)
    c.text("How it works", (RX + 24, y + 22), 15, T.muted, bold=True)
    for k, t in enumerate(("Raise your hands into view",
                           "Make one sign, then rest",
                           "Press Make sentence")):
        sy = y + 70 + k * 36
        c.circle(RX + 38, sy, 13, T.accent_soft)
        c.text(str(k + 1), (RX + 38, sy), 14, T.accent, "mm", bold=True)
        c.text(t, (RX + 62, sy), 17, T.text, "lm")

    # ---- buttons
    r = (RX, WY, RW, 60)
    rects["sentence"] = fx.button(c, *r, "Make sentence", "primary" if words else "disabled",
                                  hov(r) and bool(words), size=18)
    half = (RW - 12) / 2
    r = (RX, WY + 76, half, 64)
    rects["undo"] = fx.button(c, *r, "Undo", "ghost", hov(r))
    r = (RX + half + 12, WY + 76, half, 64)
    rects["clear"] = fx.button(c, *r, "Clear all", "ghost", hov(r))

    # ---- your words card
    c.box(WX, WY, WW, WH, T.card, radius=20, outline=T.border)
    c.text("Your words", (WX + 24, WY + 20), 15, T.muted, bold=True)
    if words:
        c.text(f"{len(words)} word{'s' if len(words) != 1 else ''}", (WX + WW - 24, WY + 20), 15, T.muted, "ra")
    if not words:
        c.text("The signs you make will appear here", (WX + 24, WY + 74), 18, T.muted, "lm")
    else:
        avail, used, keep = WW - 48, 0, []
        for j in range(len(words) - 1, -1, -1):        # Kurdish reads right-to-left
            t = labels[words[j]]["ku"]
            bw = fx.text_width(t, 20, True) + 32
            if used + bw > avail:
                break
            keep.insert(0, (j, t, bw))
            used += bw + 8
        xr = WX + WW - 24
        for j, t, bw in keep:
            newest = j == len(words) - 1
            fill = T.green_soft if (newest and flash) else T.soft
            c.box(xr - bw, WY + 48, bw, 40, fill, radius=20)
            c.text(t, (xr - bw / 2, WY + 68), 20, T.green if (newest and flash) else T.text, "mm", bold=True)
            xr -= bw + 8
        if sentence["busy"]:
            c.text("Making your sentence…", (WX + WW - 24, WY + 114), 17, T.muted, "rm")
        elif sentence["text"]:
            c.text(sentence["text"], (WX + WW - 24, WY + 112), 24, T.accent, "rm", bold=True)
        else:
            c.text("Press Make sentence to turn your words into a sentence", (WX + 24, WY + 114), 15, T.muted, "lm")
    c.finish()
    return rects


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--threshold", type=float, default=0.7)
    ap.add_argument("--theme", choices=["light", "dark"], default="light")
    args = ap.parse_args()
    T.set(args.theme)

    import tensorflow as tf
    path = os.path.join(MODEL_DIR, "ksl_model.keras")
    if not os.path.exists(path):
        raise SystemExit("No trained model yet. Run:  python train.py")
    model = tf.keras.models.load_model(path)
    labels = load_labels()
    rec = Recogniser(model, labels, threshold=args.threshold)
    rec.model(np.zeros((1, SEQ_LEN, model.input_shape[-1]), np.float32))   # warm-up

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit("Could not open the webcam. Try:  python app.py --camera 1")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    tracker = make_holistic()
    win = "Sign Translator"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, W, H)

    s = {"tracking": False, "en": True, "full": False, "mouse": None, "click": None, "toast": None}

    def on_mouse(event, x, y, flags, param):
        s["mouse"] = (x, y)
        if event == cv2.EVENT_LBUTTONUP:
            s["click"] = (x, y)
    cv2.setMouseCallback(win, on_mouse)

    words, sentence = [], {"text": "", "busy": False}
    last_hit = 0.0

    def toast(t):
        s["toast"] = (t, time.time())

    def do(action):
        if action == "sentence" and words and not sentence["busy"]:
            sentence.update(text="", busy=True)
            threading.Thread(target=build_sentence,
                             args=([labels[i]["ku"] for i in words], sentence), daemon=True).start()
        elif action == "undo" and words:
            words.pop(); sentence["text"] = ""
        elif action == "clear":
            words.clear(); sentence["text"] = ""; rec.reset_window()
        elif action == "tracking":
            s["tracking"] = not s["tracking"]
            toast("Tracking shown" if s["tracking"] else "Tracking hidden")
        elif action == "theme":
            T.toggle()
        elif action == "full":
            s["full"] = not s["full"]
            cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN if s["full"] else cv2.WINDOW_NORMAL)
        elif action == "quit":
            return False
        return True

    keymap = {13: "sentence", 10: "sentence", 8: "undo", 127: "undo", 27: "quit",
              ord("c"): "clear", ord("C"): "clear", ord("v"): "tracking", ord("V"): "tracking",
              ord("t"): "theme", ord("T"): "theme", ord("f"): "full", ord("F"): "full",
              ord("q"): "quit", ord("Q"): "quit"}

    canvas = np.zeros((H, W, 3), np.uint8)
    running = True
    while running:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        res = run_holistic(tracker, frame)
        feat, hands = frame_features(res)
        hit = rec.push(feat, hands)
        if hit is not None:
            words.append(hit)
            sentence["text"] = ""
            last_hit = time.time()
            print("+", labels[hit]["id"], labels[hit]["ku"])

        rects = draw_ui(canvas, frame, res, hands, rec, labels, words, sentence, s, last_hit)
        cv2.imshow(win, canvas)

        if s["click"] is not None:
            for action, r in rects.items():
                if fx.inside(s["click"], r):
                    running = do(action)
                    break
            s["click"] = None

        k = cv2.waitKey(1) & 0xFF
        if k in keymap:
            running = do(keymap[k]) and running
        elif k in (ord("+"), ord("=")):
            rec.threshold = min(0.99, rec.threshold + 0.05)
            toast(f"Stricter recognition ({rec.threshold:.2f})")
        elif k in (ord("-"), ord("_")):
            rec.threshold = max(0.3, rec.threshold - 0.05)
            toast(f"More relaxed recognition ({rec.threshold:.2f})")
        elif k in (ord("e"), ord("E")):
            s["en"] = not s["en"]
        if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
