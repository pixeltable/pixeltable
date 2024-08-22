#!/bin/bash -e

POSTGRES_BIN_PATH=$(python -c 'import pixeltable_pgserver; import sys; sys.stdout.write(str(pixeltable_pgserver._commands.POSTGRES_BIN_PATH))')
if [ -z "$PIXELTABLE_HOME" ]; then
    PIXELTABLE_HOME=~/.pixeltable
fi
PIXELTABLE_URL="postgresql://postgres:@/postgres?host=$PIXELTABLE_HOME/pgdata"


echo "THIS COMMAND WILL DELETE EVERYTHING IN YOUR PIXELTABLE DATABASE AT: $PIXELTABLE_HOME/pgdata"
read -p "Type \"delete\" if you're sure: "

if [[ "$REPLY" != "delete" ]]; then
    exit 0
fi

echo "Deleting!"

PG_DISCONNECT="SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE pid <> pg_backend_pid() AND datname = 'pixeltable';"
PG_DROP="DROP DATABASE pixeltable;"

"$POSTGRES_BIN_PATH/psql" "$PIXELTABLE_URL" -U postgres -c "$PG_DISCONNECT"
"$POSTGRES_BIN_PATH/psql" "$PIXELTABLE_URL" -U postgres -c "$PG_DROP"
