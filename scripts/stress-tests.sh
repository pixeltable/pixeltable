#!/bin/bash

SCRIPT_DIR=$(dirname "$0")
PXT_DIR=$(realpath "$SCRIPT_DIR/..")
mkdir -p "$PXT_DIR"/target

echo "Running random-tbl-ops (12 workers for 120 seconds) ..."
LOG_FILE="$PXT_DIR"/target/random-tbl-ops.log
python "$PXT_DIR"/tool/worker_harness.py 12 120 "$PXT_DIR"/tool/random_tbl_ops.py 2>&1 > "$LOG_FILE"

IGNORE_ERRORS='That Pixeltable operation could not be completed|Table was dropped|Path.*does not exist'
if [ -n "$(grep ERROR "$LOG_FILE" | grep -vE "$IGNORE_ERRORS")" ]; then
    echo "Errors occurred during the stress test, such as:"
    echo "$(grep ERROR "$LOG_FILE" | grep -vE "$IGNORE_ERRORS" | head -5)"
    echo "See the logfile for more details: $LOG_FILE"
    exit 1
fi
