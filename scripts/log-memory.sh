#!/bin/bash

# Prints the machine's memory use every few seconds, until the shell that started it exits.
# Usage: log-memory.sh <interval-seconds> &

interval="${1:-15}"
parent="$PPID"
cmd_width=80

# Label a pid so xdist workers (gwN), per-worker proxy daemons, postgres, and ollama are distinguishable.
# PYTEST_XDIST_WORKER and PIXELTABLE_PROXY_DAEMON live in the environment, not the command line.
proc_label() {
    local pid="$1"
    local args="$2"
    local envfile="/proc/${pid}/environ"
    local tag=""
    if [ -r "$envfile" ]; then
        local envvars worker daemon
        envvars=$(tr '\0' '\n' < "$envfile" 2>/dev/null)
        worker=$(printf '%s\n' "$envvars" | sed -n 's/^PYTEST_XDIST_WORKER=//p' | head -n1)
        daemon=$(printf '%s\n' "$envvars" | sed -n 's/^PIXELTABLE_PROXY_DAEMON=//p' | head -n1)
        if [ -n "$worker" ]; then
            tag="[${worker}] "
        fi
        if [ -n "$daemon" ] && [ "$daemon" != "0" ]; then
            tag="${tag}[proxy] "
        fi
    fi
    local short
    short=$(printf '%s' "$args" | tr -s '[:space:]' ' ' | cut -c1-"$cmd_width")
    printf '%s%s' "$tag" "$short"
}

while kill -0 "$parent" 2>/dev/null; do
    mem=$(free -m | awk '/^Mem:/ {printf "used=%dMB available=%dMB", $3, $7}')
    top=""
    while IFS= read -r line; do
        # awk already folded pid and rss; the rest of the line is args (may contain spaces)
        pid=${line%% *}
        rest=${line#* }
        rss_mb=${rest%% *}
        args=${rest#* }
        label=$(proc_label "$pid" "$args")
        if [ -n "$top" ]; then
            top="${top}; "
        fi
        top="${top}pid=${pid} rss=${rss_mb}MB ${label}"
    done < <(ps -eo pid=,rss=,args= --sort=-rss | awk 'NR <= 5 {
        pid = $1
        rss_mb = int($2 / 1024)
        $1 = ""
        $2 = ""
        sub(/^ +/, "")
        printf "%s %d %s\n", pid, rss_mb, $0
    }')
    echo "[memory $(date -u +%H:%M:%S)] $mem largest=[${top}]"
    sleep "$interval"
done
