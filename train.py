"""Train the sign classifier on everything in data/.

Usage:
    python train.py                 # default settings
    python train.py --epochs 300    # train longer

Writes model/ksl_model.keras, model/labels.json and model/report.txt
(per-phrase accuracy + which phrases get confused with each other).
"""
import argparse
import glob
import os
import numpy as np

from common import (DATA_DIR, MODEL_DIR, SEQ_LEN, FEAT_DIM, N_POSE, N_HAND,
                    load_phrases, save_labels, count_samples)

MIN_SAMPLES = 5

N_LM = N_POSE + 2 * N_HAND
GROUPS = [(0, N_POSE), (N_POSE, N_POSE + N_HAND), (N_POSE + N_HAND, N_LM)]


# --------------------------------------------------------------------------- data
def load_dataset(phrases):
    X, y, signers = [], [], []
    for ci, p in enumerate(phrases):
        files = sorted(glob.glob(os.path.join(DATA_DIR, p["id"], "*.npy")))
        for f in files:
            a = np.load(f).astype(np.float32)
            if a.shape != (SEQ_LEN, FEAT_DIM):
                print("  skipping (wrong shape)", f, a.shape)
                continue
            X.append(a); y.append(ci)
            signers.append(os.path.basename(f).split("_")[0])
    return np.array(X), np.array(y), np.array(signers)


# --------------------------------------------------------------------------- augmentation
def augment(seq, rng):
    s = seq.reshape(SEQ_LEN, N_LM, 3).copy()
    present = np.abs(s).sum(axis=2) > 0                    # missing landmarks stay zero

    # speed change (people sign faster/slower)
    speed = rng.uniform(0.8, 1.25)
    # + start offset: live windows never line up exactly with the recorded clip
    src = np.clip(np.arange(SEQ_LEN) * speed + rng.uniform(-6, 6), 0, SEQ_LEN - 1)
    lo = np.floor(src).astype(int); hi = np.minimum(lo + 1, SEQ_LEN - 1); w = (src - lo)[:, None, None]
    s = s[lo] * (1 - w) + s[hi] * w
    present = present[lo] & present[hi]

    # rotate / scale / shift (camera angle, distance, position)
    ang = np.deg2rad(rng.uniform(-12, 12))
    rot = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]], np.float32)
    s[..., :2] = s[..., :2] @ rot.T
    s *= rng.uniform(0.85, 1.15)
    s[..., :2] += rng.uniform(-0.15, 0.15, size=2)
    s += rng.normal(0, 0.015, size=s.shape)               # tracking jitter

    # occasionally lose a hand for a few frames (tracking drop-outs)
    if rng.random() < 0.3:
        g = GROUPS[rng.integers(1, 3)]
        t0 = rng.integers(0, SEQ_LEN - 3)
        present[t0:t0 + rng.integers(1, 4), g[0]:g[1]] = False

    s[~present] = 0
    return s.reshape(SEQ_LEN, FEAT_DIM).astype(np.float32)


def make_sequence_cls():
    import tensorflow as tf

    class AugSeq(tf.keras.utils.Sequence):
        def __init__(self, X, y, batch, seed=0, **kw):
            super().__init__(**kw)
            self.X, self.y, self.batch = X, y, batch
            self.rng = np.random.default_rng(seed)
            self.idx = np.arange(len(X))
            self.on_epoch_end()

        def __len__(self):
            return int(np.ceil(len(self.X) / self.batch))

        def __getitem__(self, i):
            b = self.idx[i * self.batch:(i + 1) * self.batch]
            return np.stack([augment(self.X[j], self.rng) for j in b]), self.y[b]

        def on_epoch_end(self):
            self.rng.shuffle(self.idx)

    return AugSeq


# --------------------------------------------------------------------------- model
def build_model(n_classes):
    import tensorflow as tf
    L = tf.keras.layers
    inp = L.Input((SEQ_LEN, FEAT_DIM))
    x = L.Conv1D(128, 5, padding="same", activation="relu")(inp)
    x = L.BatchNormalization()(x)
    x = L.Dropout(0.2)(x)
    x = L.Bidirectional(L.GRU(128, return_sequences=True))(x)
    x = L.Dropout(0.3)(x)
    x = L.Bidirectional(L.GRU(64))(x)
    x = L.Dense(128, activation="relu")(x)
    x = L.Dropout(0.4)(x)
    out = L.Dense(n_classes, activation="softmax")(x)
    m = tf.keras.Model(inp, out)
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return m


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--val", type=float, default=0.2, help="fraction kept aside for testing")
    args = ap.parse_args()

    import tensorflow as tf
    phrases = load_phrases()
    # Train only on phrases that actually have recordings, so a partly recorded list still works.
    ready = [p for p in phrases if count_samples(p["id"]) >= MIN_SAMPLES]
    skipped = [p["id"] for p in phrases if p not in ready]
    if skipped:
        print(f"\nSkipping {len(skipped)} phrase(s) with fewer than {MIN_SAMPLES} samples: {skipped}")
    if not ready or ready[0]["id"] != "idle":
        raise SystemExit("Record the 'idle' class first:  python collect.py --signer yourname --phrase idle")
    if len(ready) < 3:
        raise SystemExit("Need at least 2 recorded phrases (plus idle) to train.")
    phrases = ready
    X, y, signers = load_dataset(phrases)
    if len(X) == 0:
        raise SystemExit("No data yet. Run collect.py first.")

    counts = np.bincount(y, minlength=len(phrases))
    print(f"\n{len(X)} samples, {len(phrases)} classes, signers: {sorted(set(signers))}")
    print("Training on:", ", ".join(p["id"] for p in phrases if p["id"] != "idle"))

    # stratified split
    rng = np.random.default_rng(42)
    tr, va = [], []
    for c in range(len(phrases)):
        ids = rng.permutation(np.where(y == c)[0])
        k = max(1, int(round(len(ids) * args.val)))
        va += list(ids[:k]); tr += list(ids[k:])
    tr, va = np.array(tr), np.array(va)

    weights = {i: len(y) / (len(phrases) * c) for i, c in enumerate(counts)}
    AugSeq = make_sequence_cls()
    train_seq = AugSeq(X[tr], y[tr], args.batch)

    model = build_model(len(phrases))
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = os.path.join(MODEL_DIR, "ksl_model.keras")
    cbs = [
        tf.keras.callbacks.ModelCheckpoint(path, monitor="val_accuracy", save_best_only=True, verbose=0),
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=30, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, min_lr=1e-5),
    ]
    model.fit(train_seq, validation_data=(X[va], y[va]), epochs=args.epochs,
              class_weight=weights, callbacks=cbs, verbose=2)
    model.save(path)
    save_labels(phrases)

    # ---- report
    pred = model.predict(X[va], verbose=0).argmax(1)
    acc = (pred == y[va]).mean()
    lines = [f"Validation accuracy: {acc * 100:.1f}%  ({len(va)} test samples)\n",
             "Per phrase (worst first):"]
    per = []
    for c, p in enumerate(phrases):
        m = y[va] == c
        per.append(((pred[m] == c).mean() if m.any() else 1.0, p, counts[c]))
    for a, p, n in sorted(per, key=lambda t: t[0]):
        lines.append(f"  {a * 100:5.1f}%  {p['id']:<20} ({n} samples)")
    conf = {}
    for t, pr in zip(y[va], pred):
        if t != pr:
            conf[(t, pr)] = conf.get((t, pr), 0) + 1
    if conf:
        lines.append("\nMost confused (true -> predicted):")
        for (t, pr), n in sorted(conf.items(), key=lambda kv: -kv[1])[:10]:
            lines.append(f"  {phrases[t]['id']} -> {phrases[pr]['id']}  x{n}")
    report = "\n".join(lines)
    with open(os.path.join(MODEL_DIR, "report.txt"), "w", encoding="utf-8") as f:
        f.write(report)
    print("\n" + report)
    print(f"\nSaved {path}.  Now run:  python app.py")


if __name__ == "__main__":
    main()
