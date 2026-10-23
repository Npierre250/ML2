"""
model.py - NumPy autoencoder (no deep-learning framework needed) + the scorer used at inference time.

Training: Adam, mini-batches, optional input noise (denoising autoencoder), early stopping on the
reconstruction loss of HEALTHY validation data. Only healthy data is ever used to fit the network.
"""
import json

import numpy as np

from anomaly.config import SENSORS


class Autoencoder:
    def __init__(self, sizes, seed=42):
        """sizes e.g. [4, 32, 2, 32, 4]: symmetric, the narrowest layer in the middle is the bottleneck."""
        rng = np.random.default_rng(seed)
        self.sizes = list(sizes)
        self.W = [rng.normal(0, np.sqrt(2.0 / a), (a, b)) for a, b in zip(sizes[:-1], sizes[1:])]  # He init
        self.b = [np.zeros(b) for b in sizes[1:]]

    def forward(self, x, keep=False, upto=None):
        acts, n = [x], len(self.W) if upto is None else upto
        for i in range(n):
            z = acts[-1] @ self.W[i] + self.b[i]
            acts.append(z if i == len(self.W) - 1 else np.maximum(z, 0.0))       # ReLU; linear output layer
        return acts if keep else acts[-1]

    def encode(self, X):
        """Latent coordinates (output of the bottleneck layer) - used for the latent-space figure."""
        return self.forward(X, upto=len(self.W) // 2)

    def errors(self, X):
        """Per-row reconstruction error (mean squared error) = the anomaly score."""
        return ((self.forward(X) - X) ** 2).mean(axis=1)

    def fit(self, X, Xval=None, epochs=80, batch=256, lr=1e-3, patience=8, noise=0.0, seed=0, verbose=False):
        """Returns {'train': [...], 'val': [...]} (per-epoch MSE). With Xval, stops early and keeps the best weights."""
        rng = np.random.default_rng(seed)
        params = self.W + self.b
        m = [np.zeros_like(p) for p in params]
        v = [np.zeros_like(p) for p in params]
        t, n, L = 0, len(X), len(self.W)
        hist = {"train": [], "val": []}
        best, best_w, bad = np.inf, None, 0
        for ep in range(epochs):
            idx, total = rng.permutation(n), 0.0
            for s in range(0, n, batch):
                xb = X[idx[s:s + batch]]
                xin = xb + noise * rng.standard_normal(xb.shape) if noise else xb
                acts = self.forward(xin, keep=True)
                err = acts[-1] - xb
                total += float((err ** 2).sum())
                delta = 2.0 * err / len(xb)
                gW, gb = [None] * L, [None] * L
                for i in reversed(range(L)):
                    gW[i], gb[i] = acts[i].T @ delta, delta.sum(0)
                    if i > 0:
                        delta = (delta @ self.W[i].T) * (acts[i] > 0)
                t += 1
                for k, (p, g) in enumerate(zip(params, gW + gb)):                     # Adam
                    m[k] = 0.9 * m[k] + 0.1 * g
                    v[k] = 0.999 * v[k] + 0.001 * g * g
                    p -= lr * (m[k] / (1 - 0.9 ** t)) / (np.sqrt(v[k] / (1 - 0.999 ** t)) + 1e-8)
            hist["train"].append(total / X.size)
            if Xval is not None:
                vl = float(self.errors(Xval).mean())
                hist["val"].append(vl)
                if vl < best - 1e-6:
                    best, bad, best_w = vl, 0, ([w.copy() for w in self.W], [b.copy() for b in self.b])
                else:
                    bad += 1
                    if bad >= patience:
                        break
            if verbose and ep % 10 == 0:
                print(f"  epoch {ep + 1:3d}  train {hist['train'][-1]:.5f}  val {hist['val'][-1] if hist['val'] else float('nan'):.5f}")
        if best_w is not None:
            self.W, self.b = best_w
        return hist


def make_sizes(d, bottleneck, hidden=32, depth=1):
    """depth=1: d-h-b-h-d ; depth=2: d-h-h/2-b-h/2-h-d"""
    if depth == 1:
        return [d, hidden, bottleneck, hidden, d]
    return [d, hidden, hidden // 2, bottleneck, hidden // 2, hidden, d]


def attribute(res, resid_std):
    """
    Which sensor deviates most? `res` = (input - reconstruction) in normalised units, shape (n, d).
    Residuals are divided by each sensor's HEALTHY residual spread so sensors are comparable, and only
    UPWARD deviations count (fault direction; reconstruction otherwise smears error onto healthy sensors).
    A hint, not proof of cause: several sensors can be off at once.
    """
    up = np.clip(res, 0, None) / resid_std
    both = np.abs(res) / resid_std
    return np.where(up.max(axis=1) > 0, up.argmax(axis=1), both.argmax(axis=1))


# ── persisted bundle: weights + per-machine-type normalisation + threshold + metadata ─────────────────
def save_bundle(path, model, norm, threshold, extra=None, resid_std=None):
    arrays = {f"W{i}": w for i, w in enumerate(model.W)}
    arrays.update({f"b{i}": b for i, b in enumerate(model.b)})
    meta = {"sizes": model.sizes, "norm": norm, "threshold": float(threshold), "features": SENSORS,
            "extra": extra or {}, "resid_std": [float(v) for v in resid_std] if resid_std is not None else None}
    np.savez(path, meta=json.dumps(meta), **arrays)


def read_meta(path):
    try:
        return json.loads(str(np.load(path, allow_pickle=False)["meta"]))
    except Exception:
        return None


class AnomalyScorer:
    """Load a bundle once and score readings one at a time (stream processor, Flink map function, ...)."""

    def __init__(self, path):
        z = np.load(path, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        self.model = Autoencoder(meta["sizes"])
        n = len(meta["sizes"]) - 1
        self.model.W = [z[f"W{i}"] for i in range(n)]
        self.model.b = [z[f"b{i}"] for i in range(n)]
        self.norm, self.threshold = meta["norm"], meta["threshold"]
        self.resid_std = np.array(meta.get("resid_std") or [1.0] * len(SENSORS))

    def normalise(self, machine_type, values):
        s = self.norm.get(machine_type) or self.norm["__global__"]
        return (np.asarray(values, float) - np.array(s["mean"])) / np.array(s["std"])

    def score_detail(self, machine_type, values):
        x = self.normalise(machine_type, values)[None, :]
        res = x - self.model.forward(x)
        err = float((res ** 2).mean())
        pick = int(attribute(res, self.resid_std)[0])
        return {"err": err, "flag": err > self.threshold, "ratio": err / self.threshold, "root_cause": SENSORS[pick]}

    def score(self, machine_type, values):
        d = self.score_detail(machine_type, values)
        return d["err"], d["flag"]
