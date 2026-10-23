# Comparing the two projects and how they were merged

| | `Machine_learning_two` | `FinalProject-of-Iot2` |
|---|---|---|
| **Course goal** | Machine Learning II / Neural Networks: autoencoder anomaly detection (the approved proposal) | IoT / Big Data: Apache Flink running inside a Hadoop cluster |
| **ML component** | NumPy autoencoder (6→4→2→4→6), threshold μ+3σ | none - fixed thresholds only |
| **Infrastructure** | Docker: Flink 1.18 + 3 TaskManagers + PostgreSQL | Hadoop 3.3 (HDFS + YARN) + Flink on YARN, auto-start scripts, live dashboard |
| **Resources** | about 8 GB RAM | about 14 GB RAM |
| **Strengths** | the ML idea, SQL schema and views, Docker packaging | Flink proven on a real cluster, HDFS storage, one-command start, lessons on ports and checkpoints |
| **Weaknesses found** | generator could not start (syntax error); generator bound to the wrong host; rules scored 100 % against the ground truth so "AE vs rules" proved nothing; `error_count` leaked the label; 6 input features of which 2 were pure noise | no neural network at all; data had independent sensors (nothing to learn) |

## Decision

The approved proposal is a **Machine Learning II / Neural Networks** project, so it is graded on the network, the
training method and the evaluation. The ML repository is therefore the base, and the IoT repository contributes what
makes the ML result a *system* rather than a notebook.

## What was taken from each

**From `Machine_learning_two`:** the autoencoder idea and the unsupervised protocol (train on normal data, flag high
reconstruction error); the sensor schema and machine types; the Docker-based packaging; the PostgreSQL schema idea
(kept as an optional add-on in `flink/README.md`).

**From `FinalProject-of-Iot2`:** HDFS as a storage backend (`STORAGE=hdfs`, `scripts/start_cluster.sh`); the continuous
stream-to-storage processor design (`write_hdfs.py` became `stream_processor.py`); the live dashboard look; one-command
start-up; the port lesson (the sensor stream uses 9999 because 9000 is the HDFS NameNode port).

## What was fixed or added (why it matters for an ML course)

| Problem in the originals | Change |
|---|---|
| Faults were created by crossing the same thresholds the rules check, so rules were perfect and the AE could only tie | Three fault types, two of them *subtle* (every sensor in its normal range, but inconsistent with each other); the AE is compared against rules per fault type |
| No ground-truth label, so no honest metrics | The simulator emits `is_fault` and `fault_type` (used only for evaluation, never for training) |
| `error_count` was 0-3 when healthy and 0-10 when faulty: a label leak | `error_count` is independent of faults; the model uses only the 4 sensor signals named in the proposal |
| Healthy sensors were independent uniform noise | One hidden machine load drives all four sensors through curved responses, so there is a relationship to learn |
| One threshold rule (μ+3σ) and no comparison | Threshold chosen from healthy validation error; compared with Mahalanobis, PCA (the linear autoencoder), Isolation Forest and the original rules, over several seeds, with ablations |
| No tests | 20 unit tests and a CI workflow |
| Generator crashed on import | Fixed; also reconnects after a client disconnects |
