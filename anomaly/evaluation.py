"""evaluation.py - splitting, normalisation and metrics shared by training, benchmark and tests."""
import numpy as np

from anomaly.config import SENSORS
from anomaly.model import Autoencoder, make_sizes


def split(df):
    """Time-ordered 70/15/15 split (no leakage from the future)."""
    n = len(df)
    df = df.reset_index(drop=True)
    return df[: int(.70 * n)], df[int(.70 * n): int(.85 * n)], df[int(.85 * n):]


def fit_norm(normal_df):
    """Per-machine-type mean/std fitted on HEALTHY training rows only."""
    norm = {mt: {"mean": g[SENSORS].mean().tolist(), "std": (g[SENSORS].std() + 1e-6).tolist()}
            for mt, g in normal_df.groupby("machine_type")}
    norm["__global__"] = {"mean": normal_df[SENSORS].mean().tolist(), "std": (normal_df[SENSORS].std() + 1e-6).tolist()}
    return norm


def normalise(df, norm):
    out = np.empty((len(df), len(SENSORS)))
    for mt, idx in df.groupby("machine_type").groups.items():
        pos = df.index.get_indexer(idx)
        s = norm[mt]
        out[pos] = (df.loc[idx, SENSORS].to_numpy() - s["mean"]) / s["std"]
    return out


def roc_auc(scores, labels):
    scores, labels = np.asarray(scores, float), np.asarray(labels, bool)
    order = scores.argsort(kind="mergesort")
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    sorted_s = scores[order]
    start = 0
    for i in range(1, len(scores) + 1):                          # average ranks over ties
        if i == len(scores) or sorted_s[i] != sorted_s[start]:
            ranks[order[start:i]] = (start + 1 + i) / 2
            start = i
    n_pos, n_neg = labels.sum(), (~labels).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def average_precision(scores, labels):
    """Area under the precision-recall curve (step-wise). The more honest metric when faults are rare."""
    scores, labels = np.asarray(scores, float), np.asarray(labels, bool)
    y = labels[np.argsort(-scores, kind="mergesort")]
    tp = np.cumsum(y)
    return float((tp / np.arange(1, len(y) + 1))[y].mean()) if y.any() else float("nan")


def prf(pred, y):
    pred, y = np.asarray(pred, bool), np.asarray(y, bool)
    tp, fp = int((pred & y).sum()), int((pred & ~y).sum())
    fn, tn = int((~pred & y).sum()), int((~pred & ~y).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0,
            "fpr": fp / (fp + tn) if fp + tn else 0.0, "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def roc_curve(scores, labels, points=100):
    scores, labels = np.asarray(scores, float), np.asarray(labels, bool)
    ts = np.quantile(scores, np.linspace(0, 1, points))
    return [(scores[~labels] >= t).mean() for t in ts], [(scores[labels] >= t).mean() for t in ts]


def detection_by_type(pred, fault_types):
    """Share of readings flagged, per fault type ('none' = false-alarm rate on healthy readings)."""
    pred, ft = np.asarray(pred, bool), np.asarray(fault_types)
    return {t: float(pred[ft == t].mean()) for t in np.unique(ft)}


def fit_autoencoders(Xtr, Xval_healthy, Xval_all, yval, bottlenecks=(1, 2, 3), hidden=32, depth=1,
                     epochs=80, noise=0.0, seed=42):
    """
    One autoencoder per candidate bottleneck width, ranked by validation ROC-AUC (a few labelled faults
    in the validation split are enough to choose the width). Early stopping uses HEALTHY validation loss.
    Returns ({b: (model, history, val_auc)}, best_b).
    """
    out = {}
    for b in bottlenecks:
        m = Autoencoder(make_sizes(Xtr.shape[1], b, hidden, depth), seed=seed)
        hist = m.fit(Xtr, Xval_healthy, epochs=epochs, noise=noise)
        out[b] = (m, hist, roc_auc(m.errors(Xval_all), yval))
    return out, max(out, key=lambda b: out[b][2])
