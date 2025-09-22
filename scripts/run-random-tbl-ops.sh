#!/bin/bash -e

SCRIPT_DIR=$(dirname "$0")

export PIXELTABLE_HOME=~/.pixeltable

# Remove old log
rm random-tbl-ops.log || true

# Drop random_tbl_ops DB
POSTGRES_BIN_PATH=$(python -c 'import pixeltable_pgserver; import sys; sys.stdout.write(str(pixeltable_pgserver._commands.POSTGRES_BIN_PATH))')
PIXELTABLE_URL="postgresql://postgres:@/postgres?host=$PIXELTABLE_HOME/pgdata"
"$POSTGRES_BIN_PATH/psql" "$PIXELTABLE_URL" -U postgres -c "DROP DATABASE IF EXISTS random_tbl_ops;"

# Run worker harness script
python "$SCRIPT_DIR/../tool/worker_harness.py" $1 $2 "$SCRIPT_DIR/../tool/random_tbl_ops_2.py"
echo

# Print stats
python "$SCRIPT_DIR/../tool/print_random_ops_stats.py"
