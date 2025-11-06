#!/bin/bash -e

# Initializes a clean DB, runs random_ops.py, and prints statistics at the end.

SCRIPT_DIR=$(dirname "$0")

export PIXELTABLE_HOME=~/.pixeltable

# Remove old log
rm -f "$PIXELTABLE_HOME"/random-ops.log

# Drop random_ops DB
POSTGRES_BIN_PATH=$(python -c 'import pixeltable_pgserver; import sys; sys.stdout.write(str(pixeltable_pgserver._commands.POSTGRES_BIN_PATH))')
PIXELTABLE_URL="postgresql://postgres:@/postgres?host=$PIXELTABLE_HOME/pgdata"
"$POSTGRES_BIN_PATH/psql" "$PIXELTABLE_URL" -U postgres -c "DROP DATABASE IF EXISTS random_ops;"

# Run script
python "$SCRIPT_DIR/../tool/random_ops.py" "$@"
echo

# Print stats
python "$SCRIPT_DIR/../tool/print_random_ops_stats.py"
