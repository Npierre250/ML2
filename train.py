#!/usr/bin/env python3
"""
train.py - train the autoencoder on HEALTHY readings only, choose the alert threshold, evaluate, save.

  python train.py                          # simulate 40k readings, train, save models/ae_bundle.npz
  python train.py --csv data/sensors.csv   # use an existing labelled CSV instead
  python train.py --bottleneck 2 --percentile 97
"""
import argparse
import os
from datetime import datetime

import numpy as np

from anomaly import config, evaluation as E, simulator
from anomaly.model import attribute, save_bundle
from anomaly.storage import get_storage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None, help="labelled CSV (is_fault, fault_type); default: simulate")
    ap.add_argument("--records", type=int, default=40000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=config.MODEL_PATH)
    ap.add_argument("--bottleneck", default="auto", help="width of the bottleneck, or 'auto' (1-4, best validation AUC)")
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--percentile", type=float, default=99.0, help="threshold = this percentile of healthy validation error")
    a = ap.parse_args()

    if a.csv:
        df = simulator.read_csv(a.csv)
    else:
        df = simulator.generate_dataframe(a.records, 0.15, a.seed)
        simulator.write_csv(df, config.TRAIN_CSV, labels=True)
        print(f"Simulated {len(df):,} readings -> {config.TRAIN_CSV}")
    train, val, test = E.split(df)
    normal = train[train.is_fault == 0]
    print(f"Data: {len(df):,} readings, {df.is_fault.mean():.1%} faults | training on {len(normal):,} HEALTHY readings only")

    norm = E.fit_norm(normal)
    Xtr, Xvh = E.normalise(normal, norm), E.normalise(val[val.is_fault == 0], norm)
    Xva, Xte = E.normalise(val, norm), E.normalise(test, norm)
    yva, y, ft = val.is_fault.to_numpy().astype(bool), test.is_fault.to_numpy().astype(bool), test.fault_type.to_numpy()

    cands = (1, 2, 3, 4) if a.bottleneck == "auto" else (int(a.bottleneck),)
    models, b = E.fit_autoencoders(Xtr, Xvh, Xva, yva, cands, a.hidden, epochs=a.epochs)
    for w, (_, h, v) in models.items():
        print(f"  bottleneck {w}: validation ROC-AUC {v:.3f}  ({len(h['train'])} epochs){'  <- selected' if w == b else ''}")
    model = models[b][0]

    thr = float(np.percentile(model.errors(Xvh), a.percentile))
    resid_std = (model.forward(Xvh) - Xvh).std(axis=0)
    err = model.errors(Xte)
    m = E.prf(err > thr, y)
    by_type = E.detection_by_type(err > thr, ft)
    auc, ap_ = E.roc_auc(err, y), E.average_precision(err, y)
    print(f"\nTest set ({len(y):,} readings), threshold = p{a.percentile:g} of healthy validation error = {thr:.4f}")
    print(f"  ROC-AUC {auc:.3f} | avg precision {ap_:.3f} | precision {m['precision']:.3f} | recall {m['recall']:.3f} | "
          f"F1 {m['f1']:.3f} | false-alarm rate {m['fpr']:.1%}")
    print("  detected by type:", {k: f"{v:.0%}" for k, v in sorted(by_type.items())}, "(none = false alarms)")

    sel = (err > thr) & (ft == "threshold")                        # obvious faults: the perturbed sensors are known
    if sel.any():
        pick = attribute(Xte[sel] - model.forward(Xte[sel]), resid_std)
        print(f"  sensor hint check (obvious faults, which never touch power): {sel.sum()} detected; hint named "
              f"temperature/vibration/sound in {np.isin(pick, [0, 1, 2]).mean():.0%}")

    extra = {"auc": round(auc, 4), "ap": round(ap_, 4), "precision": round(m["precision"], 4), "recall": round(m["recall"], 4),
             "f1": round(m["f1"], 4), "fpr": round(m["fpr"], 4), "bottleneck": b, "train_rows": int(len(Xtr)),
             "percentile": a.percentile, "by_type": {k: round(v, 4) for k, v in by_type.items()},
             "trained_at": datetime.now().isoformat(timespec="seconds")}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    save_bundle(a.out, model, norm, thr, extra, resid_std)
    get_storage().upload(a.out, config.D_MODELS)
    print(f"\nSaved model -> {a.out} (and to the {config.STORAGE} store)")


if __name__ == "__main__":
    main()
