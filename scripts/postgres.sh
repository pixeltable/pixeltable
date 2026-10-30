#!/bin/bash -e
# Run this script inside your conda environment to open a postgres connection to your Pixeltable DB.

POSTGRES_BIN_PATH=$(python -c 'import sys; from pixeltable_pgserver.utils import POSTGRES_VERSIONS, TARGET_POSTGRES_VERSION; sys.stdout.write(str(POSTGRES_VERSIONS[TARGET_POSTGRES_VERSION]))')
if [ -z "$PIXELTABLE_HOME" ]; then
    PIXELTABLE_HOME=~/.pixeltable
fi
PIXELTABLE_URL="postgresql://postgres:@/postgres?host=$PIXELTABLE_HOME/pgdata"
"$POSTGRES_BIN_PATH/psql" "$PIXELTABLE_URL" -U postgres "$@"
