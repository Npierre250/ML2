#!/usr/bin/env python3
"""
run_demo.py - the whole system on your laptop with one command.

  python run_demo.py                    # train (if needed) -> stream -> real-time scoring -> dashboard
  python run_demo.py --retrain          # force a new model
  python run_demo.py --duration 60      # stop automatically after 60 s

Open http://localhost:5000   (Ctrl+C stops everything)
"""
import argparse
import os
import shutil
import subprocess
import sys
import time
import webbrowser

os.environ["STORAGE"] = "local"
from anomaly import config  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--records", type=int, default=30000, help="simulated readings used for training")
ap.add_argument("--epochs", type=int, default=60)
ap.add_argument("--interval", type=float, default=0.1, help="seconds between rounds of 15 readings")
ap.add_argument("--retrain", action="store_true")
ap.add_argument("--keep", action="store_true", help="keep results of earlier runs")
ap.add_argument("--no-browser", action="store_true")
ap.add_argument("--duration", type=float, default=None)
a = ap.parse_args()

py, env, procs = sys.executable, dict(os.environ), []
step = lambda msg: print(f"\n\033[1;36m== {msg}\033[0m", flush=True)

try:
    if a.retrain or not os.path.exists(config.MODEL_PATH):
        step("1/3 Training the autoencoder on healthy readings")
        subprocess.run([py, "train.py", "--records", str(a.records), "--epochs", str(a.epochs)], check=True, env=env)
    else:
        step("1/3 Using the existing model (run with --retrain to train again)")
    if not a.keep:
        shutil.rmtree(config.LOCAL_STORE / "flink" / "output", ignore_errors=True)      # counters start at zero

    step("2/3 Starting the sensor stream and the real-time processor")
    procs.append(subprocess.Popen([py, "simulate_factory_data.py", "--mode", "stream", "--interval", str(a.interval),
                                   "--emit-label"], env=env))
    time.sleep(1.5)
    procs.append(subprocess.Popen([py, "stream_processor.py"], env=env))

    step("3/3 Starting the dashboard")
    procs.append(subprocess.Popen([py, "dashboard.py"], env=env))
    url = f"http://localhost:{config.DASHBOARD_PORT}"
    time.sleep(2)
    print(f"\n  Dashboard ready -> {url}\n  (Ctrl+C to stop)\n", flush=True)
    if not a.no_browser:
        webbrowser.open(url)
    t0 = time.time()
    while a.duration is None or time.time() - t0 < a.duration:
        time.sleep(1)
        if any(p.poll() is not None for p in procs):
            print("A component stopped unexpectedly - shutting down.")
            break
except KeyboardInterrupt:
    pass
finally:
    for p in procs:
        p.terminate()
    print("\nStopped.")
