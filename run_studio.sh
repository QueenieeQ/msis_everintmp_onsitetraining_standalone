#!/usr/bin/env bash
# Start the Onsite Training Studio.  Usage: ./run_studio.sh [--workspace DIR] [--port 8766]
cd "$(dirname "$0")" && exec python3 -m onsite_training_studio "$@"
