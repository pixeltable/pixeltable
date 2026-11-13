#!/bin/bash
# Prints the state of the Pixeltable Postgres server, to show why a statement is not returning: every backend's state,
# wait event and blocking pids; the locks held or awaited in each database with a backend outside the idle state; and
# each server process's run state and CPU time. CI runs this shortly before the unit-test step's timeout.
# Run it inside your conda environment. It reads only Postgres's statistics and lock views.

pgdata="${PIXELTABLE_PGDATA:-${PIXELTABLE_HOME:-$HOME/.pixeltable}/pgdata}"
if [[ ! -f "$pgdata/postmaster.pid" ]]; then
    echo "No Postgres server running in $pgdata"
    exit 0
fi
# postmaster.pid: line 1 is the postmaster's pid, line 4 its port, line 5 its socket directory
postmaster_pid=$(sed -n 1p "$pgdata/postmaster.pid")
port=$(sed -n 4p "$pgdata/postmaster.pid")
socket_dir=$(sed -n 5p "$pgdata/postmaster.pid")
bin_path=$(python -c 'import sys; from pixeltable_pgserver.utils import POSTGRES_VERSIONS, TARGET_POSTGRES_VERSION; sys.stdout.write(str(POSTGRES_VERSIONS[TARGET_POSTGRES_VERSION]))')

# Time limits, so that a stuck server cannot hold up the caller
export PGCONNECT_TIMEOUT=10
export PGOPTIONS='-c statement_timeout=10s'

psql_db() {
    "$bin_path/psql" -X -P pager=off -h "$socket_dir" -p "$port" -U postgres -d "$1" -c "$2"
}

echo "=== Postgres backends (postmaster pid $postmaster_pid)"
psql_db postgres "
    SELECT pid, datname, backend_type, state, wait_event_type, wait_event, pg_blocking_pids(pid) AS blocked_by,
        date_trunc('second', now() - xact_start) AS xact_age, date_trunc('second', now() - state_change) AS state_age,
        backend_xid AS xid, left(regexp_replace(query, '\s+', ' ', 'g'), 150) AS query
    FROM pg_stat_activity
    WHERE pid <> pg_backend_pid()
    ORDER BY datname NULLS FIRST, pid"

dbs=$(
    "$bin_path/psql" -X -A -t -h "$socket_dir" -p "$port" -U postgres -d postgres -c "
        SELECT DISTINCT datname FROM pg_stat_activity
        WHERE datname IS NOT NULL AND state <> 'idle' AND pid <> pg_backend_pid()"
)
for db in $dbs; do
    echo "=== Locks in $db"
    psql_db "$db" "
        SELECT l.pid, l.locktype, l.relation::regclass AS relation, l.page, l.tuple, l.transactionid AS xid,
            l.mode, l.granted, date_trunc('second', now() - l.waitstart) AS waiting
        FROM pg_locks l JOIN pg_stat_activity a USING (pid)
        WHERE a.datname = current_database() AND l.pid <> pg_backend_pid()
        ORDER BY l.pid, l.granted, l.locktype, relation"
done

echo "=== Postgres processes"
children=$(pgrep -d, -P "$postmaster_pid")
ps -o pid,ppid,stat,etime,time,command -p "$postmaster_pid${children:+,$children}"
