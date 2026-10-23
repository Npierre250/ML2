#!/bin/bash
pkill -f simulate_factory_data.py 2>/dev/null
pkill -f stream_processor.py 2>/dev/null
pkill -f dashboard.py 2>/dev/null
echo "stopped"
