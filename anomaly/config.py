"""Central configuration. Override anything with environment variables."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# The four sensor signals from the approved proposal (model input)
SENSORS = ["temperature", "vibration", "sound_level", "power_consumption"]

# Fixed-threshold rules of the original project (README "Fault Threshold" column)
TEMP_THRESHOLD, VIBRATION_THRESHOLD, SOUND_THRESHOLD = 75.0, 20.0, 80.0

# STORAGE=local -> files under ./data_store ; STORAGE=hdfs -> real HDFS via the `hdfs dfs` CLI
STORAGE = os.getenv("STORAGE", "local").lower()
LOCAL_STORE = Path(os.getenv("LOCAL_STORE", ROOT / "data_store"))
HDFS_ROOT = os.getenv("HDFS_ROOT", "/flink")

# Port 9999, not 9000: 9000 collides with the HDFS NameNode RPC port (lesson from the IoT/Hadoop project)
SOCKET_HOST = os.getenv("SOCKET_HOST", "localhost")
SOCKET_PORT = int(os.getenv("SOCKET_PORT", "9999"))
DASHBOARD_PORT = int(os.getenv("DASHBOARD_PORT", "5000"))

BATCH_SIZE = int(os.getenv("BATCH_SIZE", "200"))
BATCH_SECONDS = int(os.getenv("BATCH_SECONDS", "10"))

MODEL_PATH = os.getenv("MODEL_PATH", str(ROOT / "models" / "ae_bundle.npz"))
TRAIN_CSV = os.getenv("TRAIN_CSV", str(ROOT / "data" / "sensors.csv"))

# Directories inside the store (HDFS: /flink/<dir>)
D_INPUT, D_MODELS = "input", "models"
D_ALL, D_CRIT = "output/streaming/all", "output/streaming/critical"
D_AE, D_ALERTS = "output/streaming/ae_anomalies", "output/streaming/alerts"
D_METRICS, D_CKPT = "output/streaming/metrics", "checkpoints"

# Alerting
ALERT_COOLDOWN_S = float(os.getenv("ALERT_COOLDOWN_S", "15"))
ALERT_WEBHOOK = os.getenv("ALERT_WEBHOOK", "")
CRITICAL_RATIO = float(os.getenv("CRITICAL_RATIO", "3"))
