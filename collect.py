"""Record training samples from the webcam.

Usage:
    python collect.py --signer noorhan                 # goes through every phrase that needs samples
    python collect.py --signer ahmad --phrase stop     # record one phrase only
    python collect.py --signer noorhan --samples 40    # samples per phrase for this signer
    python collect.py --status                         # show how many samples each phrase has

Keys:  SPACE record the batch for this phrase    P pause the batch
       R delete the last sample (bad take)        N next phrase   B previous phrase
       Q quit (everything recorded is already saved)

Keys V (show / hide tracking lines) and T (dark / light theme) work here too.
Only landmark numbers are saved, never camera images.
"""
import argparse
import glob
import os
import time
import numpy as np
import cv2

from common import (DATA_DIR, SEQ_LEN, IDLE_ID, load_phrases, make_holistic, run_holistic,
                    frame_features)
import ui
from ui import T

W, H = 1280, 720
VX, VY, VW, VH = 24, 76, 784, 441
RX, RW = 832, 424
READY_TIME = 1.2      # pause between samples (s)


def signer_count(pid, signer):
    return len(glob.glob(os.path.join(DATA_DIR, pid, f"{signer}_*.npy")))


def show_status():
    phrases = load_phrases()
    print(f"\n{'phrase':<22}{'samples':>8}   signers")
    print("-" * 52)
    total = 0
    for p in phrases:
        files = glob.glob(os.path.join(DATA_DIR, p["id"], "*.npy"))
        total += len(files)
        signers = {}
        for f in files:
            s = os.path.basename(f).split("_")[0]
            signers[s] = signers.get(s, 0) + 1
        flag = "  <-- needs more" if len(files) < 30 else ""
        print(f"{p['id']:<22}{len(files):>8}   " +
              ", ".join(f"{k}:{v}" for k, v in sorted(signers.items())) + flag)
    print("-" * 52)
    print(f"{len(phrases)} classes (incl. idle), {total} samples total\n")


class Screen:
    def __init__(self, signer, phrases, targets):
        self.signer, self.phrases, self.targets = signer, phrases, targets
        self.msg, self.msg_col, self.msg_t = "", None, 0.0
        self.view = "camera"

    def note(self, text, col):
        self.msg, self.msg_col, self.msg_t = text, col, time.time()
        print(text)

    def draw(self, frame, res, phrase, idx, state, frac=0.0, countdown=0.0):
        img = np.full((H, W, 3), T.bg, np.uint8)
        p = ui.Painter(img)
        pid = phrase["id"]
        done, target = signer_count(pid, self.signer), self.targets[pid]

        # ---- top bar
        p.text("KSL", (24, 16), 22, T.cyan, mono=True, bold=True)
        p.text("// DATA RECORDER", (70, 20), 16, T.text, mono=True, bold=True)
        p.text(f"SIGNER: {self.signer.upper()}", (W - 24, 20), 14, T.muted, "ra", mono=True, bold=True)
        cv2.line(img, (0, 56), (W, 56), T.line, 1)
        if state == "rec":
            label, col = "RECORDING", T.red
        elif state == "ready":
            label, col = "GET READY", T.amber
        else:
            label, col = "STANDBY", T.cyan
        ui.pill(img, p, 640 - p.width(label, 14, True, True) / 2 - 20, 14, label, col)

        # ---- viewport
        frame_col = T.red if state == "rec" else (T.amber if state == "ready" else T.cyan)
        n, _ = ui.viewport(img, p, res, frame, VX, VY, VW, VH, self.view, frame_col,
                           "LIVE CAMERA" if self.view == "camera" else "LIVE CAMERA + TRACKING")
        ui.blend_rect(img, VX + 8, VY + VH - 34, 104, 26, T.bg, 0.7)
        p.text(f"HANDS {n}/2", (VX + 18, VY + VH - 28), 13, T.green if n else T.muted, mono=True, bold=True)
        if state == "ready":
            p.text(f"{countdown:.1f}", (VX + VW / 2, VY + VH - 70), 64, T.amber, "mm", mono=True, bold=True)
        if state == "rec":
            a = ui.pulse(8)
            ui.dot(img, VX + VW - 30, VY + 26, 8, tuple(int(c * (0.4 + 0.6 * a)) for c in T.red))
            p.text("REC", (VX + VW - 46, VY + 26), 14, T.red, "rm", mono=True, bold=True)
            ui.bar(img, VX, VY + VH + 4, VW, 4, frac, T.red, bg=T.bg)

        # ---- current phrase
        ui.panel(img, p, RX, VY, RW, 178, "CURRENT PHRASE",
                 right_note=f"{idx + 1:02d} / {len(self.phrases):02d}")
        if pid == IDLE_ID:
            p.text("بێ ئاماژە", (RX + RW - 22, VY + 46), 46, T.text, "ra")
            p.text("IDLE  ·  NOT SIGNING", (RX + RW - 22, VY + 112), 14, T.amber, "ra", mono=True, bold=True)
            p.text("rest · fidget · raise & drop hands · half-start a sign",
                   (RX + 22, VY + 148), 12, T.muted, mono=True)
        else:
            p.text(phrase["ku"], (RX + RW - 22, VY + 46), 46, T.text, "ra")
            if phrase["en"]:
                p.text(phrase["en"].upper(), (RX + RW - 22, VY + 112), 14, T.muted, "ra", mono=True)
            p.text(f"id: {pid}", (RX + 22, VY + 148), 12, T.dim, mono=True)

        # ---- samples for this phrase
        sy = VY + 190
        ui.panel(img, p, RX, sy, RW, 128, "SAMPLES", accent=T.green,
                 right_note=f"{done:02d} / {target:02d}")
        cols = 15
        rows = int(np.ceil(target / cols))
        step = (RW - 44) / cols
        for k in range(min(target, cols * 4)):
            cx = RX + 22 + step * (k % cols) + step / 2
            cy = sy + 46 + (k // cols) * (60 / max(rows, 1) if rows > 2 else 24)
            if k < done:
                ui.dot(img, cx, cy, 6, T.green)
            elif k == done and state == "rec":
                ui.dot(img, cx, cy, 6, T.red)
            else:
                ui.dot(img, cx, cy, 6, T.dim, ring=True)
        ui.bar(img, RX + 22, sy + 108, RW - 44, 4, done / max(target, 1), T.green)

        # ---- overall progress
        oy = sy + 140
        total_t = sum(self.targets[q["id"]] for q in self.phrases)
        total_d = sum(min(signer_count(q["id"], self.signer), self.targets[q["id"]]) for q in self.phrases)
        complete = sum(signer_count(q["id"], self.signer) >= self.targets[q["id"]] for q in self.phrases)
        ui.panel(img, p, RX, oy, RW, VY + VH - oy, "OVERALL PROGRESS",
                 right_note=f"{complete}/{len(self.phrases)} PHRASES DONE")
        pct = total_d / max(total_t, 1)
        p.text(f"{pct * 100:.0f}%", (RX + 22, oy + 52), 30, T.cyan, "lm", mono=True, bold=True)
        ui.bar(img, RX + 110, oy + 50, RW - 132, 6, pct, T.cyan)

        # ---- instruction / message bar
        my = 537
        ui.panel(img, p, VX, my, W - 48, 80, "INSTRUCTION", accent=frame_col)
        if state == "rec":
            instr, icol = "SIGN NOW — finish the sign before the red bar ends", T.red
        elif state == "ready":
            instr, icol = "Put your hands in the start position…", T.amber
        else:
            instr, icol = "Press SPACE to record this phrase", T.cyan
        p.text(instr, (VX + 22, my + 50), 20, icol, "lm", mono=True, bold=True)

        ly = 627
        ui.panel(img, p, VX, ly, W - 48, 56, "LOG")
        if self.msg and time.time() - self.msg_t < 6:
            p.text(self.msg, (VX + 80, ly + 30), 15, self.msg_col or T.muted, "lm", mono=True)
        else:
            p.text(f"{READY_TIME:.1f}s get-ready  ·  {SEQ_LEN} frames per sample  ·  "
                   f"only landmarks are saved, never images", (VX + 80, ly + 30), 13, T.dim, "lm", mono=True)

        x = VX
        for key, lab in (("SPACE", "record"), ("P", "pause"), ("R", "delete last"),
                         ("N", "next"), ("B", "back"), ("V", "tracking"), ("T", "theme"), ("Q", "quit")):
            x = ui.keycap(img, p, x, 691, key, lab)
        p.flush()
        return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--signer", help="name of the person signing, e.g. noorhan")
    ap.add_argument("--phrase", help="record only this phrase id")
    ap.add_argument("--samples", type=int, default=30, help="samples per phrase for this signer")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--status", action="store_true", help="print sample counts and exit")
    args = ap.parse_args()
    if args.status:
        show_status(); return
    if not args.signer:
        raise SystemExit("Add your name:  python collect.py --signer yourname")
    signer = "".join(c for c in args.signer.lower() if c.isalnum()) or "signer"

    phrases = load_phrases()
    if args.phrase:
        phrases = [p for p in phrases if p["id"] == args.phrase]
        if not phrases:
            raise SystemExit(f"'{args.phrase}' is not in phrases.txt")
    # idle needs more examples (many different resting positions)
    targets = {p["id"]: (args.samples * 2 if p["id"] == IDLE_ID else args.samples) for p in phrases}
    # start at the first phrase that still needs samples
    pi = next((i for i, p in enumerate(phrases) if signer_count(p["id"], signer) < targets[p["id"]]), None)
    if pi is None:
        print("All phrases already have enough samples for", signer)
        return

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit("Could not open the webcam. Try --camera 1")
    tracker = make_holistic()
    screen = Screen(signer, phrases, targets)
    win = "KSL Recorder"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, W, H)

    def read():
        ok, frame = cap.read()
        if not ok:
            raise SystemExit("Webcam stopped sending frames.")
        frame = cv2.flip(frame, 1)                              # mirrored, feels natural
        return frame, run_holistic(tracker, frame)

    last_saved = None
    while 0 <= pi < len(phrases):
        phrase = phrases[pi]
        pid, target = phrase["id"], targets[phrase["id"]]
        os.makedirs(os.path.join(DATA_DIR, pid), exist_ok=True)

        # ---- standby: wait for a key
        frame, res = read()
        cv2.imshow(win, screen.draw(frame, res, phrase, pi, "idle"))
        k = cv2.waitKey(1) & 0xFF
        if k in (ord("q"), ord("Q"), 27):
            break
        if k in (ord("n"), ord("N")):
            pi = min(pi + 1, len(phrases) - 1) if pi < len(phrases) - 1 else pi
            continue
        if k in (ord("b"), ord("B")):
            pi = max(0, pi - 1)
            continue
        if k in (ord("r"), ord("R")):
            if last_saved and os.path.exists(last_saved):
                os.remove(last_saved)
                screen.note(f"✗ deleted last sample ({os.path.basename(last_saved)})", T.amber)
                last_saved = None
            continue
        if k in (ord("v"), ord("V")):
            screen.view = "tracking" if screen.view == "camera" else "camera"
            continue
        if k in (ord("t"), ord("T")):
            ui.T.toggle()
            continue
        if k != ord(" "):
            continue
        if signer_count(pid, signer) >= target:
            screen.note(f"✓ '{pid}' is already complete — press N for the next phrase", T.green)
            continue

        # ---- record the batch
        paused = False
        while signer_count(pid, signer) < target and not paused:
            t0 = time.time()
            while time.time() - t0 < READY_TIME:
                frame, res = read()
                left = READY_TIME - (time.time() - t0)
                cv2.imshow(win, screen.draw(frame, res, phrase, pi, "ready", countdown=left))
                k = cv2.waitKey(1) & 0xFF
                if k in (ord("p"), ord("P"), ord("q"), ord("Q"), 27):
                    paused = True
                    screen.note("‖ paused — press SPACE to continue", T.amber)
                    break
            if paused:
                break

            seq, hand_frames = [], 0
            while len(seq) < SEQ_LEN:
                frame, res = read()
                feat, hands = frame_features(res)
                seq.append(feat)
                hand_frames += hands
                cv2.imshow(win, screen.draw(frame, res, phrase, pi, "rec", frac=len(seq) / SEQ_LEN))
                cv2.waitKey(1)

            if pid != IDLE_ID and hand_frames < SEQ_LEN * 0.4:
                screen.note(f"✗ not saved: hands visible in only {hand_frames}/{SEQ_LEN} frames", T.red)
                continue
            path = os.path.join(DATA_DIR, pid, f"{signer}_{int(time.time() * 1000)}.npy")
            np.save(path, np.stack(seq))
            last_saved = path
            screen.note(f"✓ saved {pid}  {signer_count(pid, signer)}/{target}", T.green)

        if not paused:
            screen.note(f"✓ '{pid}' complete", T.green)
            nxt = next((i for i in range(pi + 1, len(phrases))
                        if signer_count(phrases[i]["id"], signer) < targets[phrases[i]["id"]]), None)
            if nxt is None:
                break
            pi = nxt

    cap.release()
    cv2.destroyAllWindows()
    print("Saved. Check progress with:  python collect.py --status")
    print("When every phrase has enough samples, run:  python train.py")


if __name__ == "__main__":
    main()
