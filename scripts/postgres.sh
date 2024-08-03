#!/bin/bash -e
# Run this script inside your conda environment to open a postgres connection to your Pixeltable DB.

POSTGRES_BIN_PATH=$(python -c 'import pgserver; import sys; sys.stdout.write(str(pgserver._commands.POSTGRES_BIN_PATH))')
if [ -z "$PIXELTABLE_HOME" ]; then
    PIXELTABLE_HOME=~/.pixeltable
fi
PIXELTABLE_URL="postgresql://postgres:@/postgres?host=$PIXELTABLE_HOME/pgdata"
"$POSTGRES_BIN_PATH/psql" "$PIXELTABLE_URL" -U postgres
