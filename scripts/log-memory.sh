#!/bin/bash

# Prints the machine's memory use every few seconds, until the shell that started it exits.
# Usage: log-memory.sh <interval-seconds> &

interval="${1:-15}"
parent="$PPID"

while kill -0 "$parent" 2>/dev/null; do
    mem=$(free -m | awk '/^Mem:/ {printf "used=%dMB available=%dMB", $3, $7}')
    largest=$(ps -eo rss= --sort=-rss | head -3 | awk '{printf "%s%dMB", (NR > 1 ? "," : ""), $1 / 1024}')
    echo "[memory $(date -u +%H:%M:%S)] $mem largest_processes=$largest"
    sleep "$interval"
done
