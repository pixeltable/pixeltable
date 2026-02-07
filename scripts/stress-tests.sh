#!/bin/bash -e

# Runs a standardized configuration of Pixeltable stress tests for a specified duration.

if [ -z "$2" ]; then
    echo "Usage: stress-tests.sh [--read-only] <num-workers> <duration-in-seconds>"
    echo "--read-only: if specified, only Worker 0 will perform write operations."
    exit 1
fi

SCRIPT_DIR=$(dirname "$0")
if [ "$1" == "--read-only" ]; then
    export PXT_STRESS_TESTS_READ_ONLY=1
    shift
fi
WORKERS="$1"
DURATION="$2"

PIXELTABLE_HOME=${PIXELTABLE_HOME:-~/.pixeltable}

# Remove the log of the previous run
rm -f "$PIXELTABLE_HOME"/random-ops.log

# Drop the previous database if exists
export PIXELTABLE_DB="random_ops"
if [ -d "$PIXELTABLE_HOME" ]; then
    echo "Cleaning $PIXELTABLE_DB postgres DB ..."
    POSTGRES_BIN_PATH=$(python -c 'import pixeltable_pgserver; import sys; sys.stdout.write(str(pixeltable_pgserver._commands.POSTGRES_BIN_PATH))')
    PIXELTABLE_URL="postgresql://postgres:@/postgres?host=$PIXELTABLE_HOME/pgdata"
    "$POSTGRES_BIN_PATH/psql" "$PIXELTABLE_URL" -U postgres -c "DROP DATABASE IF EXISTS $PIXELTABLE_DB;"
fi

echo "Running random-ops ($WORKERS workers for $DURATION seconds) ..."
if [ -n "$PXT_STRESS_TESTS_READ_ONLY" ]; then
    echo "Read-only mode enabled: only Worker 0 will perform write operations."
    # In read-only mode, all operations are enabled.
    python tool/random_ops.py "$WORKERS" "$DURATION" --read-only-workers $(( WORKERS - 1 ))
else
    # In read/write mode, we disable certain operations and data types that have known concurrency bugs.
    python tool/random_ops.py "$WORKERS" "$DURATION" --exclude rename_view -Drandom_img_freq=0 -Drandom_json_freq=0 -Drandom_array_freq=0
fi
echo ""
python tool/print_random_ops_stats.py
