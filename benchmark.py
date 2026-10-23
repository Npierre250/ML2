#!/usr/bin/env python3
"""
benchmark.py - the evidence: autoencoder vs rules vs classical baselines.

  python benchmark.py                 # 5 seeds, about 5 minutes -> docs/RESULTS.md, docs/fig_*.png, README results block
  python benchmark.py --seeds 3       # faster

Protocol (every seed gets a fresh simulated dataset):
  * time-ordered 70/15/15 train / validation / test split; the test split is touched once, at the very end
  * every learned model is fitted on HEALTHY training rows only; faults are never used for fitting
  * threshold A ("label-free"): 99th percentile of healthy validation error, the same ~1 % false-alarm budget for all
  * threshold B ("tuned"): percentile in {97 ... 99.9} that maximises F1 on the VALIDATION split, applied to every
    learned detector alike (uses the few labelled faults of the validation split, never the test split)
  * the autoencoder: bottleneck width 1-4 chosen on validation ROC-AUC; 3 random restarts per width, lowest healthy
    validation loss kept; early stopping on healthy validation loss
  * extra evidence: fault-size sweep, sensor-noise robustness, paired per-seed comparison, training-failure rate
"""
import argparse
import json
import re

import numpy as np

from anomaly import baselines as B
from anomaly import evaluation as E
from anomaly import simulator as S
from anomaly.config import SENSORS
from anomaly.model import Autoencoder, make_sizes

LEARNED = ["Mahalanobis", "PCA (1 comp.)", "PCA (2 comp.)", "Isolation Forest", "Autoencoder"]
ALL = ["Rules (original)"] + LEARNED
TYPES = ["threshold", "decoupling", "mild_offset", "none"]
SIZES = (0.05, 0.10, 0.15, 0.20, 0.30, 0.40)
NOISE = (0.0, 0.01, 0.02, 0.04)


def metrics(scores, thr, y, ft):
    pred = np.asarray(scores) > thr
    m = E.prf(pred, y)
    return {"auc": E.roc_auc(scores, y), "ap": E.average_precision(scores, y), "precision": m["precision"],
            "recall": m["recall"], "f1": m["f1"], "fpr": m["fpr"], "by_type": E.detection_by_type(pred, ft)}


def prepare(df, norm):
    return E.normalise(df, norm)


def noisy(df, sigma, rng):
    """Add Gaussian sensor noise of sigma x (normal range width of that sensor and machine type)."""
    out = df.copy()
    for s in SENSORS:
        width = np.array([S.MACHINE_TYPES[t][s][1] - S.MACHINE_TYPES[t][s][0] for t in out.machine_type])
        out[s] = out[s].to_numpy() + rng.normal(0, 1, len(out)) * sigma * width
    return out


def run_seed(seed, n, epochs, keep=False, collapse_runs=0):
    df = S.generate_dataframe(n, 0.15, seed)
    tr, va, te = E.split(df)
    normal = tr[tr.is_fault == 0]
    norm = E.fit_norm(normal)
    Xtr, Xvh = prepare(normal, norm), prepare(va[va.is_fault == 0], norm)
    Xva, Xte = prepare(va, norm), prepare(te, norm)
    yva, y, ft = va.is_fault.to_numpy().astype(bool), te.is_fault.to_numpy().astype(bool), te.fault_type.to_numpy()

    dets = {"Mahalanobis": B.Mahalanobis().fit(Xtr), "PCA (1 comp.)": B.PCADetector(1).fit(Xtr),
            "PCA (2 comp.)": B.PCADetector(2).fit(Xtr), "Isolation Forest": B.isolation_forest(Xtr)}
    cands, best_b = E.fit_autoencoders(Xtr, Xvh, Xva, yva, epochs=epochs, seed=seed)
    ae, hist, _ = cands[best_b]
    dets["Autoencoder"] = ae
    dets = {k: v for k, v in dets.items() if v is not None}

    res = {"p99": {}, "tuned": {}, "pct": {}}
    res["p99"]["Rules (original)"] = res["tuned"]["Rules (original)"] = metrics(B.rule_severity(te).astype(float), 0.5, y, ft)
    for name, d in dets.items():
        evh, eva, ete = d.errors(Xvh), d.errors(Xva), d.errors(Xte)
        pct, _ = E.choose_percentile(evh, eva, yva)
        res["p99"][name] = metrics(ete, np.percentile(evh, 99), y, ft)
        res["tuned"][name] = metrics(ete, np.percentile(evh, pct), y, ft)
        res["pct"][name] = pct
    res["ae_info"] = {"bottleneck": best_b, "epochs_run": len(hist["train"]), "restart_losses": ae.restart_losses}

    ab = {}                                                              # ablations (label-free p99 threshold)
    for b, (m, _, _) in cands.items():
        ab[f"bottleneck {b}"] = metrics(m.errors(Xte), np.percentile(m.errors(Xvh), 99), y, ft)
    for label, kw in (("2 hidden layers", dict(depth=2)), ("tanh activation", dict(act="tanh")),
                      ("denoising (noise 0.1)", dict(noise=0.1)), ("ReLU on the code layer (old design)", dict(linear_code=False))):
        m = Autoencoder(make_sizes(4, best_b, 32, kw.get("depth", 1)), seed=seed, activation=kw.get("act", "relu"),
                        linear_code=kw.get("linear_code", True))
        m.fit(Xtr, Xvh, epochs=epochs, noise=kw.get("noise", 0.0))
        ab[label] = metrics(m.errors(Xte), np.percentile(m.errors(Xvh), 99), y, ft)
    first = Autoencoder(make_sizes(4, best_b, 32), seed=seed * 1000 + best_b * 10)   # = the first restart only
    first.fit(Xtr, Xvh, epochs=epochs)
    ab["single run (no restarts)"] = metrics(first.errors(Xte), np.percentile(first.errors(Xvh), 99), y, ft)
    res["_ablation"] = ab

    res["_sweep"] = {str(p): {k: E.prf(ae.errors(Xte) > np.percentile(ae.errors(Xvh), p), y)[k] for k in ("precision", "recall", "fpr")}
                     for p in (90, 95, 97, 99, 99.5, 99.9)}

    sev = {name: {} for name in ALL}                                      # fault-size sweep: detection of mild offsets
    for frac in SIZES:
        ds = S.generate_dataframe(3000, 1.0, seed + 500, fault_mix=(0, 0, 1), offset_range=(frac, frac))
        Xs = prepare(ds, norm)
        sev["Rules (original)"][str(frac)] = float((B.rule_severity(ds) >= 1).mean())
        for name, d in dets.items():
            sev[name][str(frac)] = float((d.errors(Xs) > np.percentile(d.errors(Xvh), 99)).mean())
    res["_severity"] = sev

    rng = np.random.default_rng(seed)       # sensor-noise robustness: models trained on clean data, noisy test readings
    nz = {name: {} for name in ALL}
    for sigma in NOISE:
        tn = noisy(te, sigma, rng)
        Xn = prepare(tn, norm)
        Xvn = prepare(noisy(va[va.is_fault == 0], sigma, rng), norm)      # healthy validation data at the SAME noise level
        sc = B.rule_severity(tn).astype(float)
        r0 = E.prf(sc >= 1, y)
        nz["Rules (original)"][str(sigma)] = {"auc": E.roc_auc(sc, y), "f1": r0["f1"], "fpr": r0["fpr"], "f1_recal": r0["f1"]}
        for name, d in dets.items():
            e = d.errors(Xn)
            clean, recal = E.prf(e > np.percentile(d.errors(Xvh), 99), y), E.prf(e > np.percentile(d.errors(Xvn), 99), y)
            nz[name][str(sigma)] = {"auc": E.roc_auc(e, y), "f1": clean["f1"], "fpr": clean["fpr"],
                                    "f1_recal": recal["f1"], "fpr_recal": recal["fpr"]}
    res["_noise"] = nz

    if collapse_runs:                          # how often does ONE training run fail? same initialisation seeds for both designs
        col = {}
        for label, lin in (("relu_code", False), ("linear_code", True)):
            losses = []
            for j in range(collapse_runs):
                m = Autoencoder(make_sizes(4, best_b, 32), seed=9000 + j, linear_code=lin)
                m.fit(Xtr, Xvh, epochs=epochs)
                losses.append(float(m.errors(Xvh).mean()))
            col[label] = {"runs": collapse_runs, "failed": int(sum(l > 0.3 for l in losses))}
        res["_collapse"] = col

    art = {}
    if keep:
        art = {k: d.errors(Xte) for k, d in dets.items() if k != "Mahalanobis" or True}
        art.update(y=y, ft=ft, hist=hist, thr=float(np.percentile(ae.errors(Xvh), 99)), sev=B.rule_severity(te).astype(float),
                   latent_model=cands[2][0], Xte=Xte)
    return res, art


# ── aggregation ───────────────────────────────────────────────────────────────
def agg(runs, getter):
    v = np.array([getter(r) for r in runs], float)
    return v.mean(), v.std()


def fmt(mean, std, pct=False):
    return f"{mean * 100:.1f} ± {std * 100:.1f} %" if pct else f"{mean:.3f} ± {std:.3f}"


def bold(name, text):
    return f"**{text}**" if name == "Autoencoder" else text


def main_table(runs, mode, names):
    L = ["| Detector | ROC-AUC | Avg. precision | Precision | Recall | F1 | False-alarm rate |", "|---|---|---|---|---|---|---|"]
    for d in names:
        row = [fmt(*agg(runs, lambda r: r[mode][d][k])) for k in ("auc", "ap", "precision", "recall", "f1")]
        row.append(fmt(*agg(runs, lambda r: r[mode][d]["fpr"]), pct=True))
        L.append(f"| {bold(d, d)} | " + " | ".join(row) + " |")
    return "\n".join(L)


def type_table(runs, mode, names):
    L = ["| Detector | Obvious: over threshold | Subtle: decoupling | Subtle: mild offset | False alarms on healthy |", "|---|---|---|---|---|"]
    for d in names:
        cells = [fmt(*agg(runs, lambda r: r[mode][d]["by_type"][t]), pct=True) for t in ("threshold", "decoupling", "mild_offset", "none")]
        L.append(f"| {bold(d, d)} | " + " | ".join(cells) + " |")
    return "\n".join(L)


def tables(runs):
    names = [d for d in ALL if d in runs[0]["p99"]]
    out = {"main": main_table(runs, "p99", names), "types": type_table(runs, "p99", names),
           "tuned": main_table(runs, "tuned", names), "tuned_types": type_table(runs, "tuned", names)}
    A = ["| Variant | ROC-AUC | F1 | Detects mild offset | Detects decoupling |", "|---|---|---|---|---|"]
    for k in runs[0]["_ablation"]:
        g = lambda f: agg(runs, lambda r: f(r["_ablation"][k]))
        A.append(f"| {k} | {fmt(*g(lambda m: m['auc']))} | {fmt(*g(lambda m: m['f1']))} | "
                 f"{fmt(*g(lambda m: m['by_type']['mild_offset']), pct=True)} | {fmt(*g(lambda m: m['by_type']['decoupling']), pct=True)} |")
    out["ablation"] = "\n".join(A)
    P = ["| Threshold percentile | Precision | Recall | False-alarm rate |", "|---|---|---|---|"]
    for pct in runs[0]["_sweep"]:
        g = lambda k: agg(runs, lambda r: r["_sweep"][pct][k])
        P.append(f"| {pct} | {fmt(*g('precision'), pct=True)} | {fmt(*g('recall'), pct=True)} | {fmt(*g('fpr'), pct=True)} |")
    out["sweep"] = "\n".join(P)
    Z = ["| Detector | " + " | ".join(f"{int(f * 100)} % of range" for f in SIZES) + " |", "|---|" + "---|" * len(SIZES)]
    for d in [x for x in ALL if x in runs[0]["_severity"]]:
        Z.append(f"| {bold(d, d)} | " + " | ".join(f"{agg(runs, lambda r: r['_severity'][d][str(f)])[0] * 100:.0f} %" for f in SIZES) + " |")
    out["severity"] = "\n".join(Z)
    for key, field, title in (("noise", "f1", "F1, threshold calibrated on CLEAN data"),
                              ("noise_recal", "f1_recal", "F1, threshold re-calibrated on healthy data at the same noise level")):
        N = [f"| Detector ({title}) | " + " | ".join(f"noise {s * 100:.0f} %" for s in NOISE) + " |", "|---|" + "---|" * len(NOISE)]
        for d in [x for x in ALL if x in runs[0]["_noise"]]:
            N.append(f"| {bold(d, d)} | " + " | ".join(f"{agg(runs, lambda r: r['_noise'][d][str(s)][field])[0]:.3f}" for s in NOISE) + " |")
        out[key] = "\n".join(N)
    out["noise_fpr"] = " / ".join(f"{agg(runs, lambda r: r['_noise']['Autoencoder'][str(s)]['fpr'])[0] * 100:.1f} %" for s in NOISE)
    return out


def paired(runs):
    """Per-seed paired comparison of F1 (label-free threshold): autoencoder minus each baseline."""
    L = ["| Autoencoder minus ... | Mean F1 difference | Autoencoder better in |", "|---|---|---|"]
    for d in [x for x in ALL if x != "Autoencoder" and x in runs[0]["p99"]]:
        diff = np.array([r["p99"]["Autoencoder"]["f1"] - r["p99"][d]["f1"] for r in runs])
        L.append(f"| {d} | {diff.mean():+.3f} ± {diff.std():.3f} | {int((diff > 0).sum())} of {len(diff)} seeds |")
    return "\n".join(L)


def figures(art, runs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = {"Rules (original)": "#EF4444", "Mahalanobis": "#F59E0B", "PCA (1 comp.)": "#8B5CF6", "PCA (2 comp.)": "#A78BFA",
           "Isolation Forest": "#64748B", "Autoencoder": "#0D9488"}
    y, ft = art["y"], art["ft"]
    scores = {"Rules (original)": art["sev"], **{k: art[k] for k in LEARNED if k in art}}

    h = art["hist"]
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.plot(h["train"], label="training loss"); ax.plot(h["val"], label="validation loss (healthy)")
    ax.set(xlabel="epoch", ylabel="reconstruction MSE", title="Learning curve (early stopping keeps the best epoch)", yscale="log")
    ax.legend(); ax.grid(alpha=.3); fig.tight_layout(); fig.savefig("docs/fig_learning_curve.png", dpi=140); plt.close(fig)

    e = art["Autoencoder"]; fig, ax = plt.subplots(figsize=(7.5, 4))
    bins = np.logspace(np.log10(max(e.min(), 1e-5)), np.log10(e.max()), 60)
    for t, c in (("none", "#22C55E"), ("mild_offset", "#F59E0B"), ("decoupling", "#3B82F6"), ("threshold", "#EF4444")):
        x = e[ft == t]
        ax.hist(x, bins=bins, weights=np.ones(len(x)) / len(x), alpha=.5, color=c, label=t if t != "none" else "healthy")
    ax.axvline(art["thr"], color="k", ls="--", label="alert threshold (p99 healthy)")
    ax.set(xscale="log", xlabel="reconstruction error (log scale)", ylabel="fraction of that reading type per bin", title="Autoencoder error by reading type")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig("docs/fig_error_distribution.png", dpi=140); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for name, s in scores.items():
        fpr, tpr = E.roc_curve(s, y)
        lw = 2.4 if name == "Autoencoder" else 1.2
        axes[0].plot(fpr, tpr, color=col[name], lw=lw, label=name)
        o = np.argsort(-s, kind="mergesort"); yy = y[o]; tp = np.cumsum(yy)
        axes[1].plot(tp / yy.sum(), tp / np.arange(1, len(yy) + 1), color=col[name], lw=lw, label=name)
    axes[0].plot([0, 1], [0, 1], "k--", lw=.7)
    axes[0].set(xlabel="false positive rate", ylabel="true positive rate", title="ROC curve")
    axes[1].set(xlabel="recall", ylabel="precision", title="Precision-recall curve", ylim=(0, 1.02))
    for a in axes:
        a.grid(alpha=.3); a.legend(fontsize=7)
    fig.tight_layout(); fig.savefig("docs/fig_roc_pr.png", dpi=140); plt.close(fig)

    names = [d for d in ALL if d in runs[0]["p99"]]
    fig, ax = plt.subplots(figsize=(9, 4.2)); w = 0.8 / len(names)
    labels = ["Obvious\n(over threshold)", "Subtle\n(decoupling)", "Subtle\n(mild offset)", "False alarms\n(healthy)"]
    for i, d in enumerate(names):
        vals = [agg(runs, lambda r: r["p99"][d]["by_type"][t]) for t in TYPES]
        ax.bar(np.arange(4) + i * w, [v[0] * 100 for v in vals], w, yerr=[v[1] * 100 for v in vals], color=col[d], label=d, capsize=2)
    ax.set_xticks(np.arange(4) + 0.4 - w / 2); ax.set_xticklabels(labels)
    ax.set(ylabel="% of readings flagged", title="What each detector catches (mean ± std over seeds)"); ax.legend(fontsize=7, ncol=3)
    ax.grid(axis="y", alpha=.3); fig.tight_layout(); fig.savefig("docs/fig_detection_by_type.png", dpi=140); plt.close(fig)

    Z = art["latent_model"].encode(art["Xte"]); fig, ax = plt.subplots(figsize=(6.2, 5))
    for t, c in (("none", "#22C55E"), ("mild_offset", "#F59E0B"), ("decoupling", "#3B82F6"), ("threshold", "#EF4444")):
        m = ft == t
        ax.scatter(Z[m, 0], Z[m, 1], s=5, alpha=.45, color=c, label=t if t != "none" else "healthy")
    ax.set(xlabel="latent z1", ylabel="latent z2", title="2-D latent space (bottleneck = 2)")
    ax.legend(markerscale=3, fontsize=8); fig.tight_layout(); fig.savefig("docs/fig_latent_space.png", dpi=140); plt.close(fig)

    ps = list(runs[0]["_sweep"]); fig, ax = plt.subplots(figsize=(6, 3.8))
    for k, c in (("precision", "#0D9488"), ("recall", "#3B82F6"), ("fpr", "#EF4444")):
        ax.plot(ps, [agg(runs, lambda r: r["_sweep"][p][k])[0] for p in ps], "o-", color=c, label={"fpr": "false-alarm rate"}.get(k, k))
    ax.set(xlabel="threshold = percentile of healthy validation error", title="Choosing the alert threshold"); ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig("docs/fig_threshold_tradeoff.png", dpi=140); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))                                      # NEW: fault-size sweep + noise
    for d in names:
        lw = 2.6 if d == "Autoencoder" else 1.3
        axes[0].plot([f * 100 for f in SIZES], [agg(runs, lambda r: r["_severity"][d][str(f)])[0] * 100 for f in SIZES], "o-", color=col[d], lw=lw, label=d)
        axes[1].plot([s * 100 for s in NOISE], [agg(runs, lambda r: r["_noise"][d][str(s)]["f1_recal"])[0] for s in NOISE], "o-", color=col[d], lw=lw, label=d)
        axes[1].plot([s * 100 for s in NOISE], [agg(runs, lambda r: r["_noise"][d][str(s)]["f1"])[0] for s in NOISE], "--", color=col[d], lw=0.9)
    axes[0].set(xlabel="size of the sensor offset (% of its normal range)", ylabel="% of faulty readings detected",
                title="How small a fault can each detector see?")
    axes[1].set(xlabel="added sensor noise (% of normal range)", ylabel="F1 score", title="Noisier sensors (solid: re-calibrated threshold, dashed: clean threshold)", ylim=(0, 1.02))
    for a in axes:
        a.grid(alpha=.3); a.legend(fontsize=7)
    fig.tight_layout(); fig.savefig("docs/fig_sensitivity_noise.png", dpi=140); plt.close(fig)


def headline(runs, t):
    a, m = (lambda d, k: agg(runs, lambda r: r["p99"][d][k])[0]), None
    best_base = max([d for d in LEARNED if d != "Autoencoder" and d in runs[0]["p99"]], key=lambda d: a(d, "f1"))
    ae_mild = agg(runs, lambda r: r["p99"]["Autoencoder"]["by_type"]["mild_offset"])[0]
    bb_mild = agg(runs, lambda r: r["p99"][best_base]["by_type"]["mild_offset"])[0]
    tuned = agg(runs, lambda r: r["tuned"]["Autoencoder"]["f1"])
    return (f"With the label-free threshold the autoencoder reaches F1 {a('Autoencoder', 'f1'):.3f} and ROC-AUC {a('Autoencoder', 'auc'):.3f}, "
            f"against F1 {a('Rules (original)', 'f1'):.3f} for the original rules and {a(best_base, 'f1'):.3f} for the best baseline ({best_base}). "
            f"It detects {ae_mild * 100:.0f} % of mild offsets (best baseline: {bb_mild * 100:.0f} %) at a measured false-alarm rate of "
            f"{a('Autoencoder', 'fpr') * 100:.1f} % (target 1 %). With the validation-tuned threshold F1 rises to {tuned[0]:.3f} ± {tuned[1]:.3f}.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--records", type=int, default=40000)
    ap.add_argument("--epochs", type=int, default=100)
    a = ap.parse_args()
    runs, art = [], None
    for s in range(a.seeds):
        print(f"seed {s + 1}/{a.seeds} ...", flush=True)
        r, ar = run_seed(s, a.records, a.epochs, keep=(s == 0), collapse_runs=40 if s == 0 else 0)
        runs.append(r)
        art = art or ar
    T = tables(runs)
    col = runs[0]["_collapse"]
    n_lin = col["linear_code"]["runs"]
    zero_note = f" (0 failures in {n_lin} runs: with 95 % confidence the failure rate is below about {3 / n_lin:.0%})" if col["linear_code"]["failed"] == 0 else ""
    chosen_b = [r["ae_info"]["bottleneck"] for r in runs]
    chosen_p = [r["pct"]["Autoencoder"] for r in runs]
    head = headline(runs, T)
    md = f"""# Results

Simulated data ({a.records:,} readings per seed, 15 % faults: 40 % obvious, 40 % decoupling, 20 % mild offset),
**{a.seeds} independent seeds**, mean ± standard deviation, **test split only** (the last 15 % in time).
Learned models are fitted on healthy readings only. Autoencoder bottleneck width chosen per seed: {chosen_b};
tuned threshold percentile chosen per seed: {chosen_p}.

{head}

## 1. Overall detection quality (label-free threshold: 99th percentile of healthy validation error)

{T['main']}

## 2. What each detector catches, by fault type

{T['types']}

![Detection by type](fig_detection_by_type.png)

## 3. Is the network better than the baselines, seed by seed?

{paired(runs)}

## 4. Same comparison with a validation-tuned threshold

The percentile (97 ... 99.9) that maximises F1 on the **validation** split is applied to each learned detector alike.

{T['tuned']}

{T['tuned_types']}

## 5. How small a fault can each detector see?

Share of mild-offset faults detected (label-free threshold) when every faulty reading has an offset of exactly the stated size.
Offsets are kept inside the normal range, so threshold rules can never see them.

{T['severity']}

## 6. Robustness to noisier sensors

Models trained on clean data; Gaussian noise (a percentage of each sensor's normal range) is added to every sensor of the test readings.
With a threshold calibrated on clean data, noise raises the healthy error of every learned detector and the autoencoder, which
reconstructs the clean pattern very precisely, is affected most (its false-alarm rate at the four noise levels: {T['noise_fpr']}).
Re-calibrating the threshold on healthy readings at the current noise level needs no fault labels and removes the problem.

{T['noise']}

{T['noise_recal']}

![Fault size and noise](fig_sensitivity_noise.png)

## 7. Curves

![ROC and precision-recall](fig_roc_pr.png)

## 8. Autoencoder ablations (same protocol)

{T['ablation']}

## 9. Choosing the alert threshold (autoencoder)

{T['sweep']}

![Threshold trade-off](fig_threshold_tradeoff.png)

## 10. Training reliability

Single training runs (bottleneck {chosen_b[0]}, same 40 random initialisation seeds for both designs, seed-0 data):
**ReLU on the code layer (old design): {col['relu_code']['failed']} of {col['relu_code']['runs']} runs collapsed** to a trivial solution
(healthy validation error above 0.3; a working run reaches about 0.017). **Linear code layer (shipped): {col['linear_code']['failed']} of {col['linear_code']['runs']}**{zero_note}.
A 1-unit ReLU code layer can die (negative for every input), after which the network outputs a constant; a linear code layer cannot.
On top of that, training keeps the best of 3 restarts (lowest healthy validation error, no labels needed) and automatically
adds up to 5 more if all of them collapsed. Healthy-validation losses of the restarts kept for seed 0: {[round(x, 4) for x in runs[0]['ae_info']['restart_losses']]}.

## 11. What the network learned

![Error distribution](fig_error_distribution.png)
![Learning curve](fig_learning_curve.png)
![Latent space](fig_latent_space.png)
"""
    open("docs/RESULTS.md", "w").write(md)
    json.dump(runs, open("docs/results.json", "w"), indent=1, default=float)
    figures(art, runs)

    block = (f"{head}\n\n**Label-free threshold** (mean ± std over {a.seeds} seeds):\n\n{T['main']}\n\n"
             f"**By fault type** (share of readings flagged):\n\n{T['types']}\n")
    rd = open("README.md").read()
    if "<!-- RESULTS:START -->" in rd:
        rd = re.sub(r"<!-- RESULTS:START -->.*<!-- RESULTS:END -->", lambda _: f"<!-- RESULTS:START -->\n{block}\n<!-- RESULTS:END -->", rd, flags=re.S)
        open("README.md", "w").write(rd)
    print("\n" + md.split("## 7.")[0])
