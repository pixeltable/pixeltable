#!/bin/bash

SCRIPT_DIR=$(dirname "$0")
PXT_DIR=$(realpath "$SCRIPT_DIR/..")
mkdir -p "$PXT_DIR"/target

echo "Running random-ops (12 workers for 120 seconds) ..."
LOG_FILE="$PXT_DIR"/target/random-ops.log
python tool/random_ops.py 12 120 --exclude rename_view -Drandom_img_freq=0 -Drandom_json_freq=0 -Drandom_array_freq=0
python tool/print_random_ops_stats.py

IGNORE_ERRORS='That Pixeltable operation could not be completed|Table was dropped|Path.*does not exist'
if [ -n "$(grep ERROR "$LOG_FILE" | grep -vE "$IGNORE_ERRORS")" ]; then
    echo "Errors occurred during the stress test, such as:"
    echo "$(grep ERROR "$LOG_FILE" | grep -vE "$IGNORE_ERRORS" | head -5)"
    echo "See the logfile for more details: $LOG_FILE"
    exit 1
fi
