#!/usr/bin/env python3
"""
Factory sensor data simulator - command line (same name and options as the original project).

  python simulate_factory_data.py --mode stream --port 9999 --interval 0.5 --emit-label
  python simulate_factory_data.py --mode batch  --n-rows 50000 --output data/sensors.csv --emit-label
"""
import argparse
import os

from anomaly import config, simulator

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", default=os.environ.get("MODE", "stream"), choices=["stream", "batch"])
    ap.add_argument("--host", default="0.0.0.0", help="address to listen on (stream mode)")
    ap.add_argument("--port", type=int, default=config.SOCKET_PORT)
    ap.add_argument("--interval", type=float, default=0.5, help="seconds between rounds of 15 readings")
    ap.add_argument("--fault-prob", type=float, default=0.15)
    ap.add_argument("--n-rows", type=int, default=50000)
    ap.add_argument("--output", default=config.TRAIN_CSV)
    ap.add_argument("--emit-label", action="store_true", help="append is_fault and fault_type (ground truth)")
    ap.add_argument("--seed", type=int, default=None)
    a = ap.parse_args()
    if a.mode == "stream":
        simulator.serve(a.host, a.port, a.interval, a.fault_prob, a.emit_label, a.seed)
    else:
        df = simulator.generate_dataframe(a.n_rows, a.fault_prob, a.seed or 0)
        simulator.write_csv(df, a.output, a.emit_label)
        print(f"[Batch] {len(df):,} rows -> {a.output}  (faults: {df.is_fault.mean():.1%}, "
              f"by type: {df[df.is_fault == 1].fault_type.value_counts().to_dict()})")
