"""
simulator.py - factory sensor simulator with ground-truth labels and three kinds of fault.

Healthy machines: one hidden "load" drives temperature, vibration, sound and power through curved
(physically plausible) responses, so the four signals are strongly related. That relationship is
exactly what an autoencoder can learn.

Faults (the label is always known, which makes honest evaluation possible):
  threshold    OBVIOUS   a sensor is pushed past its safe limit (what the original project injected)
  decoupling   SUBTLE    every sensor stays inside its normal range, but they no longer agree with each
                         other, e.g. hot motor + low power draw + low vibration (cooling/bearing problem)
  mild_offset  SUBTLE    one or two sensors are shifted by 20-35 % of their range, in a direction that keeps
                         them inside the normal range (a labelled fault is always a real change)

Threshold rules can only ever see 'threshold' faults - by construction. How well a model detects the
subtle types is MEASURED (see benchmark.py), not assumed.
"""
import math
import random
import socket
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from anomaly.config import SENSORS

# Normal operating ranges per machine type (same as the original project; all strictly below the rule thresholds)
MACHINE_TYPES = {
    "CNC_Machine":     {"temperature": (30, 68), "vibration": (1, 16), "sound_level": (55, 75), "power_consumption": (5, 35)},
    "Conveyor_Belt":   {"temperature": (20, 60), "vibration": (2, 17), "sound_level": (50, 70), "power_consumption": (1, 15)},
    "Hydraulic_Press": {"temperature": (35, 72), "vibration": (3, 19), "sound_level": (60, 78), "power_consumption": (10, 50)},
    "Robotic_Arm":     {"temperature": (25, 65), "vibration": (1, 15), "sound_level": (45, 72), "power_consumption": (3, 25)},
    "Pump_Station":    {"temperature": (30, 70), "vibration": (2, 18), "sound_level": (55, 76), "power_consumption": (8, 40)},
}
MACHINES = [{"id": f"{t[:3].upper()}-{i:03d}", "type": t} for t in MACHINE_TYPES for i in range(1, 4)]

# How each sensor responds to machine load u in [0,1] (curved, not straight lines)
RESPONSE = {
    "temperature":       lambda u: 1 - (1 - u) ** 2,                 # saturates
    "vibration":         lambda u: u ** 2,                           # grows faster than load
    "sound_level":       lambda u: math.log1p(9 * u) / math.log(10),  # log-like
    "power_consumption": lambda u: u ** 1.5,
}

FAULT_TYPES = ["threshold", "decoupling", "mild_offset"]
FAULT_MIX = [0.4, 0.4, 0.2]

CSV_COLUMNS = ["machine_id", "machine_type", "temperature", "vibration", "sound_level",
               "power_consumption", "operating_hours", "error_count"]
LABEL_COLUMNS = ["is_fault", "fault_type"]


def _clip(x, lo=0.0, hi=1.0):
    return min(max(x, lo), hi)


def _healthy(ranges, rng, loads=None):
    """Sensor values for one healthy reading. `loads` lets a fault give sensors different hidden loads."""
    u = rng.random()
    out = {}
    for s in SENSORS:
        load = _clip((loads[s] if loads else u) + rng.gauss(0, 0.04))
        pos = _clip(RESPONSE[s](load) + rng.gauss(0, 0.02))
        lo, hi = ranges[s]
        out[s] = lo + (hi - lo) * pos
    return out


def make_reading(machine, fault_prob=0.15, rng=random, fault_mix=FAULT_MIX):
    """One reading as a dict, including ground truth (is_fault, fault_type)."""
    ranges = MACHINE_TYPES[machine["type"]]
    ftype = "none"
    if rng.random() < fault_prob:
        ftype = rng.choices(FAULT_TYPES, weights=fault_mix)[0]

    if ftype == "decoupling":
        ua = rng.random()
        ub = rng.uniform(ua + 0.5, 1.0) if ua < 0.5 else rng.uniform(0.0, ua - 0.5)
        group = set(rng.sample(SENSORS, rng.randint(1, 3)))
        v = _healthy(ranges, rng, {s: (ua if s in group else ub) for s in SENSORS})
    else:
        v = _healthy(ranges, rng)

    if ftype == "mild_offset":
        for s in rng.sample(SENSORS, rng.randint(1, 2)):
            lo, hi = ranges[s]
            shift = rng.uniform(0.20, 0.35) * (hi - lo)
            room = [d for d in (1, -1) if lo <= v[s] + d * shift <= hi]      # only directions that stay in range,
            d = rng.choice(room) if room else (1 if hi - v[s] > v[s] - lo else -1)   # so the fault is never clipped away
            v[s] = _clip(v[s] + d * shift, lo, hi)
    elif ftype == "threshold":
        kind = rng.choice(["temperature", "vibration", "sound_level", "multi"])
        if kind in ("temperature", "multi"):
            v["temperature"] = rng.uniform(76, 105)
        if kind in ("vibration", "multi"):
            v["vibration"] = rng.uniform(21, 40)
        if kind in ("sound_level", "multi"):
            v["sound_level"] = rng.uniform(82, 110)

    return {"machine_id": machine["id"], "machine_type": machine["type"],
            **{s: round(v[s], 2) for s in SENSORS},
            "operating_hours": round(rng.uniform(0, 8760), 2),
            "error_count": rng.randint(0, 3),            # independent of faults (the original leaked the label here)
            "is_fault": int(ftype != "none"), "fault_type": ftype}


def to_csv_line(r, labels=False):
    cols = CSV_COLUMNS + (LABEL_COLUMNS if labels else [])
    return ",".join(str(r[c]) for c in cols)


def generate_dataframe(n, fault_prob=0.15, seed=0):
    """n readings in stream order (round-robin over machines) as a DataFrame with labels."""
    rng = random.Random(seed)
    return pd.DataFrame([make_reading(MACHINES[i % len(MACHINES)], fault_prob, rng) for i in range(n)])


def write_csv(df, path, labels=True):
    import os
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    cols = CSV_COLUMNS + (LABEL_COLUMNS if labels else [])
    with open(path, "w") as f:
        f.write("# " + ",".join(cols) + "\n")                # '#' header: skipped by csv.allow-comments / comment='#'
        for row in df[cols].itertuples(index=False):
            f.write(",".join(map(str, row)) + "\n")


def read_csv(path):
    return pd.read_csv(path, comment="#", header=None,
                       names=CSV_COLUMNS + LABEL_COLUMNS)


def _bind(srv, host, port):
    """Bind to `host`; if it is not an address of this machine, listen on all interfaces."""
    try:
        srv.bind((host, port))
        return host
    except (OSError, socket.gaierror):
        print(f"[Stream] '{host}' is not an address of this container -> listening on 0.0.0.0", flush=True)
        srv.bind(("0.0.0.0", port))
        return "0.0.0.0"


def serve(host="0.0.0.0", port=9999, interval=0.5, fault_prob=0.15, labels=False, seed=None, ready=None):
    """TCP server: streams one CSV line per reading to whoever connects; accepts reconnects forever."""
    rng = random.Random(seed)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        bound = _bind(srv, host, port)
        srv.listen(1)
        print(f"[Stream] Listening on {bound}:{port} | fault prob {fault_prob:.0%} | interval {interval}s | "
              f"labels {'on' if labels else 'off'}", flush=True)
        if ready:
            ready.set()
        while True:
            conn, addr = srv.accept()
            print(f"[Stream] client connected: {addr}", flush=True)
            try:
                while True:
                    for m in MACHINES:                       # one reading per machine per round
                        conn.sendall((to_csv_line(make_reading(m, fault_prob, rng), labels) + "\n").encode())
                    time.sleep(interval)
            except (BrokenPipeError, ConnectionResetError):
                print("[Stream] client disconnected - waiting for a new connection", flush=True)
            finally:
                conn.close()
