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
    --listen-addr=0.0.0.0:26257 \
    --http-addr=0.0.0.0:8080 \
    --cache=256MiB \
    --max-sql-memory=256MiB

ready=0
for _ in $(seq 1 60); do
    if ! docker ps --format '{{.Names}}' | grep -q "^${ROACH_NAME}$"; then
        echo "Container ${ROACH_NAME} is not running. Last logs:" >&2
        docker logs "$ROACH_NAME" 2>&1 || true
        exit 1
    fi
    if docker exec "$ROACH_NAME" /cockroach/cockroach sql --insecure -e 'SELECT 1' >/dev/null 2>&1; then
        ready=1
        break
    fi
    sleep 2
done

if [ "$ready" -ne 1 ]; then
    echo "CockroachDB did not become ready in time. Logs:" >&2
    docker logs "$ROACH_NAME" 2>&1 || true
    exit 1
fi

docker exec "$ROACH_NAME" /cockroach/cockroach sql --insecure -e "
    CREATE DATABASE IF NOT EXISTS pixeltable;
    SET CLUSTER SETTING feature.vector_index.enabled = true;
    SET CLUSTER SETTING sql.defaults.experimental_temporary_tables.enabled = true;
    ALTER RANGE default CONFIGURE ZONE USING gc.ttlseconds = 60;
    ALTER DATABASE pixeltable CONFIGURE ZONE USING gc.ttlseconds = 60;
"
