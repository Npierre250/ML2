#!/usr/bin/env python3
"""
stream_processor.py - the real-time pipeline.

  TCP stream -> parse -> rule check (original thresholds)
                      -> autoencoder reconstruction error + sensor hint
                      -> alert engine (severity, per-machine cooldown, optional webhook)
                      -> batches written to storage (local folder or HDFS)
                      -> live metrics, incl. detection rate per fault type when the stream carries labels

Outputs under <store>/flink/output/streaming/:
  all/ every enriched reading   critical/ rule severity >= 2   ae_anomalies/ autoencoder flags
  alerts/ de-duplicated alerts  metrics/metrics.json live counters (read by the dashboard)
"""
import json
import logging
import os
import socket
import sys
import time
from datetime import datetime, timezone

from anomaly import config
from anomaly.alerting import AlertEngine
from anomaly.model import AnomalyScorer
from anomaly.simulator import CSV_COLUMNS, LABEL_COLUMNS
from anomaly.storage import get_storage

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("processor")
HISTORY_POINTS = 90
NUMERIC = ("temperature", "vibration", "sound_level", "power_consumption", "operating_hours")


def now_utc():
    return datetime.now(timezone.utc).replace(tzinfo=None)


def parse(line):
    """CSV line -> dict, or None. Accepts 8 fields (no labels) or 10 fields (+ is_fault, fault_type)."""
    parts = line.strip().split(",")
    if len(parts) not in (len(CSV_COLUMNS), len(CSV_COLUMNS) + 2):
        return None
    try:
        r = dict(zip(CSV_COLUMNS + LABEL_COLUMNS, parts))
        for k in NUMERIC:
            r[k] = float(r[k])
        r["error_count"] = int(r["error_count"])
        r["is_fault"] = int(r["is_fault"]) if "is_fault" in r else None
        r.setdefault("fault_type", None)
        return r
    except ValueError:
        return None


def apply_rules(r):
    a = []
    if r["temperature"] > config.TEMP_THRESHOLD:
        a.append(f"HIGH_TEMP({r['temperature']:.1f}C)")
    if r["vibration"] > config.VIBRATION_THRESHOLD:
        a.append(f"HIGH_VIB({r['vibration']:.1f}mm/s)")
    if r["sound_level"] > config.SOUND_THRESHOLD:
        a.append(f"HIGH_SOUND({r['sound_level']:.1f}dB)")
    r["anomalies"], r["severity"], r["is_critical"] = a, len(a), len(a) >= 2


def apply_autoencoder(r, scorer):
    d = scorer.score_detail(r["machine_type"], [r["temperature"], r["vibration"], r["sound_level"], r["power_consumption"]])
    r["ae_score"], r["ae_anomaly"] = round(d["err"], 5), bool(d["flag"])
    r["ae_ratio"], r["root_cause"] = round(d["ratio"], 3), d["root_cause"]


# ── metrics ───────────────────────────────────────────────────────────────────
def cm():
    return {"tp": 0, "fp": 0, "fn": 0, "tn": 0}


def blank_metrics():
    return {"total": 0, "rule_flags": 0, "ae_anomalies": 0, "true_faults": 0, "batches": 0, "labelled": 0,
            "rules": cm(), "ae": cm(), "by_fault_type": {}, "by_machine": {}, "root_causes": {},
            "alerts": {"WARNING": 0, "CRITICAL": 0, "suppressed": 0}, "history": [],
            "started_at": datetime.now().isoformat(timespec="seconds"), "updated_at": None}


def tally(c, predicted, truth):
    c[("t" if predicted == truth else "f") + ("p" if predicted else "n")] += 1


def update_metrics(m, r):
    rule_flag = r["severity"] >= 1
    m["total"] += 1
    m["rule_flags"] += rule_flag
    m["ae_anomalies"] += r["ae_anomaly"]
    if r["is_fault"] is not None:                                   # ground truth available (demo / evaluation stream)
        truth = bool(r["is_fault"])
        m["labelled"] += 1
        m["true_faults"] += truth
        tally(m["rules"], rule_flag, truth)
        tally(m["ae"], r["ae_anomaly"], truth)
        t = m["by_fault_type"].setdefault(r["fault_type"], {"n": 0, "rules": 0, "ae": 0})
        t["n"] += 1
        t["rules"] += rule_flag
        t["ae"] += r["ae_anomaly"]
    mm = m["by_machine"].setdefault(r["machine_id"], {"n": 0, "ae": 0, "type": r["machine_type"]})
    mm["n"] += 1
    mm["ae"] += r["ae_anomaly"]
    if r["ae_anomaly"]:
        m["root_causes"][r["root_cause"]] = m["root_causes"].get(r["root_cause"], 0) + 1


def history_point(batch):
    n = max(len(batch), 1)
    lab = [r for r in batch if r["is_fault"] is not None]
    return {"t": now_utc().strftime("%H:%M:%S"), "n": len(batch),
            "ae": round(sum(r["ae_anomaly"] for r in batch) / n, 3),
            "rules": round(sum(r["severity"] >= 1 for r in batch) / n, 3),
            "truth": round(sum(r["is_fault"] for r in lab) / max(len(lab), 1), 3)}


# ── main loop ─────────────────────────────────────────────────────────────────
def load_scorer(store):
    if not os.path.exists(config.MODEL_PATH):
        log.info("Model not found locally, trying the %s store...", store.name)
        store.download(config.D_MODELS, os.path.basename(config.MODEL_PATH), config.MODEL_PATH)
    if not os.path.exists(config.MODEL_PATH):
        sys.exit("No trained model found. Run:  python train.py")
    return AnomalyScorer(config.MODEL_PATH)


def flush(store, m, batch, alerts, engine):
    name = f"{now_utc().strftime('%Y%m%d_%H%M%S')}_batch_{m['batches'] + 1:05d}.json"   # timestamp first -> sorts in time order
    dump = lambda rows: "".join(json.dumps(r) + "\n" for r in rows)
    crit = [r for r in batch if r["is_critical"]]
    ae = [r for r in batch if r["ae_anomaly"]]
    store.write_text(config.D_ALL, name, dump(batch))
    for rows, d in ((crit, config.D_CRIT), (ae, config.D_AE), (alerts, config.D_ALERTS)):
        if rows:
            store.write_text(d, name, dump(rows))
    for a in alerts:
        m["alerts"][a["level"]] += 1
    m["alerts"]["suppressed"] = engine.suppressed
    m["batches"] += 1
    m["history"] = (m["history"] + [history_point(batch)])[-HISTORY_POINTS:]
    m["updated_at"] = datetime.now().isoformat(timespec="seconds")
    store.write_text(config.D_METRICS, "metrics.json", json.dumps(m))
    log.info("batch %05d | %d readings | %d rule flags | %d AE anomalies | %d alerts | total %d",
             m["batches"], len(batch), sum(r["severity"] >= 1 for r in batch), len(ae), len(alerts), m["total"])


def run():
    store = get_storage()
    scorer, engine = load_scorer(store), AlertEngine()
    try:
        m = json.loads(store.read_text(config.D_METRICS, "metrics.json")) or blank_metrics()
        for k, v in blank_metrics().items():
            m.setdefault(k, v)
        log.info("Resuming counters from previous run (%d readings)", m["total"])
    except ValueError:
        m = blank_metrics()
    log.info("Storage=%s | threshold=%.4f | stream %s:%d | alert cooldown %.0fs%s", store.name, scorer.threshold,
             config.SOCKET_HOST, config.SOCKET_PORT, engine.cooldown, " | webhook ON" if engine.webhook else "")
    while True:
        try:
            sock = socket.create_connection((config.SOCKET_HOST, config.SOCKET_PORT), timeout=5)
            log.info("Connected to sensor stream")
            sock.settimeout(2)
            buf, batch, alerts, last = b"", [], [], time.time()
            while True:
                try:
                    chunk = sock.recv(65536)
                    if not chunk:
                        break
                    buf += chunk
                except socket.timeout:
                    pass
                *lines, buf = buf.split(b"\n")
                for raw in lines:
                    r = parse(raw.decode(errors="ignore"))
                    if not r:
                        continue
                    apply_rules(r)
                    apply_autoencoder(r, scorer)
                    r["processed_at"] = now_utc().isoformat()
                    update_metrics(m, r)
                    a = engine.consider(r)
                    if a:
                        alerts.append(a)
                    batch.append(r)
                if batch and (len(batch) >= config.BATCH_SIZE or time.time() - last >= config.BATCH_SECONDS):
                    flush(store, m, batch, alerts, engine)
                    batch, alerts, last = [], [], time.time()
        except (ConnectionRefusedError, socket.timeout, OSError) as e:
            log.warning("Stream unavailable (%s) - retrying in 3s", e)
            time.sleep(3)
        except KeyboardInterrupt:
            log.info("Stopped")
            return


if __name__ == "__main__":
    run()
