"""Live Kurdish Sign Language -> Kurdish text translator.

Usage:
    python app.py                          # webcam 0, dark theme, camera view
    python app.py --theme light            # start in light mode
    python app.py --view tracking          # start with the tracking lines drawn on the video
    python app.py --camera 1 --threshold 0.8

Everything can be done with the mouse (buttons at the bottom) or the keyboard:
    ENTER      turn the recognised signs into a full Kurdish sentence
    BACKSPACE  remove the last word
    C          clear the transcript
    V          show / hide the tracking lines on the video (hidden by default)
    T          switch theme: dark <-> light
    F          fullscreen on/off
    + / -      raise / lower the confidence threshold
    E          show / hide English
    Q / ESC    quit

The camera picture is shown as-is and never saved; tracking runs in the background.
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
import ui
from ui import T

W, H = 1280, 720
VX, VY, VW, VH = 24, 78, 800, 450          # camera viewport (16:9)
RX, RW = 848, 408                          # right column
TY, SY, BY = 544, 614, 672                 # transcript, sentence, buttons rows

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
BUTTONS = [  # action, key label
    ("sentence", "ENTER"), ("undo", "BKSP"), ("clear", "C"), ("view", "V"),
    ("theme", "T"), ("full", "F"), ("quit", "Q"),
]


def button_label(action, s):
    return {
        "sentence": "Sentence",
        "undo": "Undo word",
        "clear": "Clear all",
        "view": "Show tracking" if s["view"] == "camera" else "Hide tracking",
        "theme": "Light mode" if T.dark else "Dark mode",
        "full": "Exit fullscreen" if s["full"] else "Fullscreen",
        "quit": "Quit",
    }[action]


def draw_ui(img, frame, res, hands_now, rec, labels, words, sentence, fps, s, last_hit):
    """Draws one full frame of the translator. `s` is the UI state dict."""
    p = ui.Painter(img)
    now = time.time()
    flash = now - last_hit < 1.6

    # ---- top bar
    p.text("KSL", (24, 17), 24, T.cyan, mono=True, bold=True)
    p.text("// TRANSLATOR", (76, 22), 17, T.text, mono=True, bold=True)
    p.text("AI · COMPUTER VISION", (248, 25), 12, T.muted, mono=True)
    p.text("وەرگێڕی زمانی ئاماژەی کوردی", (W - 24, 14), 24, T.text, "ra")
    cv2.line(img, (0, 62), (W, 62), T.line, 1)
    if sentence["busy"]:
        state, col = "GENERATING SENTENCE", T.cyan
    elif flash:
        state, col = "SIGN RECOGNIZED", T.green
    elif hands_now:
        state, col = "READING YOUR SIGN", T.amber
    else:
        state, col = "RAISE YOUR HANDS", T.muted
    ui.pill(img, p, 640 - p.width(state, 14, True, True) / 2 - 20, 16, state, col)

    # ---- camera viewport
    frame_col = T.green if flash else (T.amber if hands_now else T.cyan)
    label = "LIVE CAMERA" if s["view"] == "camera" else "LIVE CAMERA + TRACKING"
    n, body = ui.viewport(img, p, res, frame, VX, VY, VW, VH, s["view"], frame_col, label)
    stat = f"HANDS {n}/2   BODY {'OK' if body else '--'}"
    sw = p.width(stat, 13, True, True)
    ui.blend_rect(img, VX + 8, VY + VH - 34, sw + 20, 26, T.bg, 0.7)
    p.text(stat, (VX + 18, VY + VH - 21), 13, T.green if n else T.muted, "lm", mono=True, bold=True)
    fps_t = f"{fps:4.1f} FPS"
    fw_ = p.width(fps_t, 13, True)
    ui.blend_rect(img, VX + VW - fw_ - 28, VY + 8, fw_ + 20, 26, T.bg, 0.7)
    p.text(fps_t, (VX + VW - 18, VY + 21), 13, T.muted, "rm", mono=True)

    if not hands_now and not flash:                    # small hint at the top, never over the person
        ui.blend_rect(img, VX + VW / 2 - 200, VY + 44, 400, 64, T.bg, 0.6)
        p.text("دەستەکانت بەرز بکەرەوە", (VX + VW / 2, VY + 64), 24, T.text, "mm")
        p.text("Raise your hands to start signing", (VX + VW / 2, VY + 94), 13,
               T.muted, "mm", mono=True)
    if flash and rec.last_word is not None:            # TV-style subtitle
        word = labels[rec.last_word]
        tw = max(p.width(word["ku"], 40), 220) + 60
        sx = VX + VW / 2 - tw / 2
        ui.blend_rect(img, sx, VY + VH - 96, tw, 82, T.bg, 0.6)
        cv2.line(img, (int(sx), VY + VH - 96), (int(sx + tw), VY + VH - 96), T.green, 2)
        p.text(word["ku"], (VX + VW / 2, VY + VH - 64), 40, T.green, "mm")
        if s["en"] and word["en"]:
            p.text(word["en"].upper(), (VX + VW / 2, VY + VH - 28), 13, T.text, "mm", mono=True)

    # ---- detected sign
    ui.panel(img, p, RX, VY, RW, 200, "DETECTED SIGN", right_note=f"SENSITIVITY {rec.threshold:.2f}")
    order = [int(i) for i in np.argsort(rec.probs)[::-1]
             if rec.probs[i] > 0.01 and labels[i]["id"] != IDLE_ID]
    top_p = float(rec.probs[order[0]]) if order else 0.0
    if flash and rec.last_word is not None:
        big, big_en, big_p, big_col, conf_label = (labels[rec.last_word]["ku"], labels[rec.last_word]["en"],
                                                   1.0, T.green, "ACCEPTED")
    elif order and hands_now:
        i = order[0]
        ok = top_p >= rec.threshold
        big, big_en, big_p, big_col = labels[i]["ku"], labels[i]["en"], top_p, (T.green if ok else T.text)
        conf_label = f"CONFIDENCE {top_p * 100:.0f}%"
    else:
        big, big_en, big_p, big_col, conf_label = "", "", 0.0, T.text, "WAITING FOR A SIGN"
    if big:
        p.text(big, (RX + RW - 22, VY + 42), 48, big_col, "ra")
        if s["en"] and big_en:
            p.text(big_en.upper(), (RX + RW - 22, VY + 112), 14, T.muted, "ra", mono=True)
    else:
        p.text("— — —", (RX + RW / 2, VY + 84), 30, T.dim, "mm", mono=True)
    p.text(conf_label, (RX + 22, VY + 150), 13, big_col if big else T.muted, mono=True, bold=True)
    ui.bar(img, RX + 22, VY + 174, RW - 44, 8, big_p, big_col if big else T.dim, marker=rec.threshold)

    # ---- other guesses
    gy = VY + 212
    ui.panel(img, p, RX, gy, RW, 110, "OTHER GUESSES")
    alts = order[1:4] if (order and hands_now and not flash) else []
    for k in range(3):
        y = gy + 42 + k * 24
        if k < len(alts):
            i = alts[k]
            pr = float(rec.probs[i])
            ui.bar(img, RX + 22, y - 2, 110, 4, pr, T.cyan)
            p.text(f"{pr * 100:3.0f}%", (RX + 144, y), 12, T.muted, "lm", mono=True)
            p.text(labels[i]["ku"], (RX + RW - 22, y), 19, T.text, "rm")
        else:
            cv2.line(img, (RX + 22, y), (RX + RW - 22, y), T.line, 1)

    # ---- how to use
    hy = gy + 122
    ui.panel(img, p, RX, hy, RW, VY + VH - hy, "HOW TO USE", accent=T.amber)
    steps = ["Raise your hands into the camera",
             "Make one sign, then rest your hands",
             "The word appears as a subtitle",
             "Press  Sentence  to finish"]
    for k, t in enumerate(steps):
        y = hy + 40 + k * 21
        ui.dot(img, RX + 30, y, 9, T.amber, ring=True)
        p.text(str(k + 1), (RX + 30, y), 11, T.amber, "mm", mono=True, bold=True)
        p.text(t, (RX + 48, y), 14, T.text, "lm")

    # ---- transcript
    ui.panel(img, p, VX, TY, W - 48, 62, "TRANSCRIPT", accent=T.green,
             right_note=f"{len(words)} WORD{'S' if len(words) != 1 else ''}")
    avail, used, keep = (W - 40) - (VX + 130), 0, []
    for j in range(len(words) - 1, -1, -1):          # Kurdish reads right-to-left
        txt = labels[words[j]]["ku"]
        w = p.width(txt, 22) + 28
        if used + w > avail:
            break
        keep.insert(0, (j, txt, w))
        used += w + 10
    xr = W - 40
    for j, txt, w in keep:
        x0 = xr - w
        newest = j == len(words) - 1
        col = T.green if (newest and flash) else (T.cyan if newest else T.dim)
        if newest and flash:
            ui.blend_rect(img, x0, TY + 26, w, 30, T.green, 0.15)
        else:
            ui.rect(img, x0, TY + 26, w, 30, T.panel2)
        ui.rect(img, x0, TY + 26, w, 30, col, 1)
        p.text(txt, (x0 + w / 2, TY + 40), 22, T.text, "mm")
        xr -= w + 10
    if not words:
        p.text("your signs will appear here", (W - 40, TY + 41), 14, T.dim, "rm", mono=True)
    if len(keep) < len(words):
        p.text(f"+{len(words) - len(keep)}", (VX + 20, TY + 41), 14, T.muted, "lm", mono=True)

    # ---- sentence
    ui.panel(img, p, VX, SY, W - 48, 50, "SENTENCE", accent=T.cyan)
    if sentence["busy"]:
        a = ui.pulse(6)
        p.text("GENERATING" + "." * int(now * 3 % 4), (W - 40, SY + 27), 15,
               tuple(int(c * (0.5 + 0.5 * a)) for c in T.cyan), "rm", mono=True, bold=True)
    elif sentence["text"]:
        p.text(sentence["text"], (W - 40, SY + 26), 26, T.green, "rm")
    else:
        p.text("press  Sentence  to join the words into a Kurdish sentence",
               (W - 40, SY + 28), 14, T.dim, "rm", mono=True)

    # ---- buttons (clickable)
    rects = {}
    bw = (W - 48 - 10 * (len(BUTTONS) - 1)) / len(BUTTONS)
    for k, (action, key) in enumerate(BUTTONS):
        bx = VX + k * (bw + 10)
        r = (bx, BY, bw, 40)
        rects[action] = ui.button(img, p, bx, BY, bw, 40, button_label(action, s), key,
                                  hover=ui.inside(s["mouse"], r), primary=(action == "sentence" and bool(words)))
    p.flush()
    return rects


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--threshold", type=float, default=0.7)
    ap.add_argument("--theme", choices=["dark", "light"], default="dark")
    ap.add_argument("--view", choices=["camera", "tracking"], default="camera")
    args = ap.parse_args()
    T.set(args.theme)

    import tensorflow as tf
    model = tf.keras.models.load_model(os.path.join(MODEL_DIR, "ksl_model.keras"))
    labels = load_labels()
    rec = Recogniser(model, labels, threshold=args.threshold)
    rec.model(np.zeros((1, SEQ_LEN, model.input_shape[-1]), np.float32))   # warm-up

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit("Could not open the webcam. Try --camera 1")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    tracker = make_holistic()
    win = "KSL Translator"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, W, H)

    s = {"view": args.view, "en": True, "full": False, "mouse": None, "click": None}

    def on_mouse(event, x, y, flags, param):
        s["mouse"] = (x, y)
        if event == cv2.EVENT_LBUTTONUP:
            s["click"] = (x, y)
    cv2.setMouseCallback(win, on_mouse)

    words, sentence = [], {"text": "", "busy": False}
    last_hit = 0.0
    t_prev, fps = time.time(), 0.0

    def do(action):
        nonlocal words
        if action == "sentence" and words and not sentence["busy"]:
            sentence.update(text="", busy=True)
            threading.Thread(target=build_sentence,
                             args=([labels[i]["ku"] for i in words], sentence), daemon=True).start()
        elif action == "undo" and words:
            words.pop(); sentence["text"] = ""
        elif action == "clear":
            words.clear(); sentence["text"] = ""; rec.reset_window()
        elif action == "view":
            s["view"] = "tracking" if s["view"] == "camera" else "camera"
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
              ord("c"): "clear", ord("C"): "clear", ord("v"): "view", ord("V"): "view",
              ord("t"): "theme", ord("T"): "theme", ord("f"): "full", ord("F"): "full",
              ord("q"): "quit", ord("Q"): "quit"}

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

        canvas = np.full((H, W, 3), T.bg, np.uint8)
        now = time.time()
        fps = 0.9 * fps + 0.1 / max(now - t_prev, 1e-3); t_prev = now
        rects = draw_ui(canvas, frame, res, hands, rec, labels, words, sentence, fps, s, last_hit)
        cv2.imshow(win, canvas)

        if s["click"] is not None:
            for action, r in rects.items():
                if ui.inside(s["click"], r):
                    running = do(action)
                    break
            s["click"] = None

        k = cv2.waitKey(1) & 0xFF
        if k in keymap:
            running = do(keymap[k]) and running
        elif k in (ord("+"), ord("=")):
            rec.threshold = min(0.99, rec.threshold + 0.05)
        elif k in (ord("-"), ord("_")):
            rec.threshold = max(0.3, rec.threshold - 0.05)
        elif k in (ord("e"), ord("E")):
            s["en"] = not s["en"]
        # closing the window with its X button also quits
        if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
