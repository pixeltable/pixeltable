#!/usr/bin/env bash
# Start a single-node CockroachDB container on localhost:26257 for CI tests
# and apply the cluster settings Pixeltable expects.
set -euo pipefail

ROACH_VERSION='v26.1.3'
ROACH_NAME='roach-ci'

docker run -d --rm \
    --name "$ROACH_NAME" \
    -p 26257:26257 \
    -p 8080:8080 \
    "cockroachdb/cockroach:$ROACH_VERSION" \
    start-single-node --insecure \
    --cache=1GiB \
    --max-sql-memory=1GiB

for _ in $(seq 1 60); do
    if docker exec "$ROACH_NAME" /cockroach/cockroach sql --insecure -e 'SELECT 1' >/dev/null 2>&1; then
        break
    fi
    sleep 2
done

docker exec "$ROACH_NAME" /cockroach/cockroach sql --insecure -e "
    CREATE DATABASE IF NOT EXISTS pixeltable;
    SET CLUSTER SETTING feature.vector_index.enabled = true;
    SET CLUSTER SETTING sql.defaults.experimental_temporary_tables.enabled = true;
    ALTER RANGE default CONFIGURE ZONE USING gc.ttlseconds = 300;
    ALTER DATABASE pixeltable CONFIGURE ZONE USING gc.ttlseconds = 300;
    SET CLUSTER SETTING sql.txn.read_committed_isolation.enabled = true;
"
