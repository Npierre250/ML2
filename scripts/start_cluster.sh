#!/bin/bash
# Hadoop / Google-Cloud VM edition (from the IoT project): results and the model live in HDFS.
# Usage:  PUBLIC_IP=<vm ip> bash scripts/start_cluster.sh
set -u
cd "$(dirname "$0")/.."
export STORAGE=hdfs
PUBLIC_IP="${PUBLIC_IP:-<your-vm-external-ip>}"
pkill -f simulate_factory_data.py 2>/dev/null; pkill -f stream_processor.py 2>/dev/null; pkill -f dashboard.py 2>/dev/null
sleep 1
echo "[1/4] Training (healthy data only) and uploading the model to HDFS /flink/models"
python3 train.py --records 40000
echo "[2/4] Sensor stream on :9999"
nohup python3 simulate_factory_data.py --mode stream --interval 0.5 --emit-label > logs_generator.txt 2>&1 &
sleep 3
echo "[3/4] Real-time processor (reads the stream, writes to HDFS)"
nohup python3 stream_processor.py > logs_processor.txt 2>&1 &
echo "[4/4] Dashboard"
nohup python3 dashboard.py > logs_dashboard.txt 2>&1 &
sleep 3
echo "Dashboard: http://$PUBLIC_IP:5000   |   HDFS UI: http://$PUBLIC_IP:9870   |   YARN UI: http://$PUBLIC_IP:8088"
echo "Stop with: bash scripts/stop.sh"
