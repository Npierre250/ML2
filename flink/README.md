# Optional: running the scorer inside Apache Flink (+ PostgreSQL)

> **Status: not tested.** Flink and PostgreSQL were not available while this project was built. Everything else in the
> repository (training, benchmark, real-time processor, dashboard, alerts, HDFS storage against a mock) was run and tested.
> The plain-Python `stream_processor.py` is the tested real-time path.

Your original `Machine_learning_two` stack (Flink 1.18, 3 TaskManagers, PostgreSQL, `init_db.sql`) can stay as it is.
What changes is the scoring step: replace the hand-written inference code by `anomaly.model.AnomalyScorer`, which
loads the bundle produced by `train.py` (`models/ae_bundle.npz`, needs only `numpy`).

```python
# in your Flink job (e.g. inside ParseAndScoreFunction)
from pyflink.datastream.functions import MapFunction
from anomaly.model import AnomalyScorer

class AEScore(MapFunction):
    def open(self, runtime_context):
        self.scorer = AnomalyScorer("/app/models/ae_bundle.npz")     # loaded once per task
    def map(self, r):                                                # r: dict with the sensor fields
        d = self.scorer.score_detail(r["machine_type"],
              [r["temperature"], r["vibration"], r["sound_level"], r["power_consumption"]])
        return {**r, "recon_error": d["err"], "ae_anomaly": int(d["flag"]), "root_cause": d["root_cause"]}
```

Needed in the Flink image: `numpy`, and the `anomaly/` package plus the model file mounted/copied
(e.g. `-pyfs anomaly,models`). Connect the job to **`generator:9999`** (the generator is the TCP server; the JobManager
listens on nothing at port 9000). Add `is_fault` / `fault_type` columns to `init_db.sql` if you want ground-truth
evaluation in SQL.
