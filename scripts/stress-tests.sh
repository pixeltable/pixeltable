#!/bin/bash

# Runs a standardized configuration of Pixeltable stress tests for a specified duration.

if [ -z "$2" ]; then
    echo "Usage: stress-tests.sh <num-workers> <duration-in-seconds>"
    exit 1
fi

SCRIPT_DIR=$(dirname "$0")
WORKERS="$1"
DURATION="$2"

# For now, we disable certain operations and data types that have known concurrency bugs.
echo "Running random-ops (12 workers for $DURATION seconds) ..."
python tool/random_ops.py "$WORKERS" "$DURATION" --exclude rename_view -Drandom_img_freq=0 -Drandom_json_freq=0 -Drandom_array_freq=0
echo ""
python tool/print_random_ops_stats.py
