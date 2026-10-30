#!/bin/bash

# Prints the machine's memory use every few seconds, until the shell that started it exits.
# Usage: log-memory.sh <interval-seconds> &

interval="${1:-15}"
parent="$PPID"

while kill -0 "$parent" 2>/dev/null; do
    mem=$(free -m | awk '/^Mem:/ {printf "used=%dMB available=%dMB", $3, $7}')
    top=$(ps -eo pid=,rss=,args= --sort=-rss | awk 'NR <= 5 {
        pid = $1; rss_mb = int($2 / 1024); $1 = $2 = ""; sub(/^ +/, "")
        printf "%spid=%s rss=%dMB %s", (NR > 1 ? "; " : ""), pid, rss_mb, substr($0, 1, 80)
    }')
    echo "[memory $(date -u +%H:%M:%S)] $mem largest=[${top}]"
    sleep "$interval"
done
