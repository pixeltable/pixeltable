#!/bin/bash

# Runs a standardized configuration of Pixeltable stress tests for a specified duration.

if [ -z "$1" ]; then
    echo "Usage: stress-tests.sh <duration-in-seconds>"
    exit 1
fi

SCRIPT_DIR=$(dirname "$0")
DURATION="$1"

# For now, we disable certain operations and data types that have known concurrency bugs.
echo "Running random-ops (12 workers for $DURATION seconds) ..."
python tool/random_ops.py 12 "$DURATION" --exclude rename_view -Drandom_img_freq=0 -Drandom_json_freq=0 -Drandom_array_freq=0
echo ""
python tool/print_random_ops_stats.py
