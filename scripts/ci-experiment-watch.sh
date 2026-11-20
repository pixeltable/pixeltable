#!/bin/bash
# TEMPORARY (experiment branch only): evidence for the macOS psycopg hang and the Ubuntu runner loss.

pgdata="${PIXELTABLE_PGDATA:-${PIXELTABLE_HOME:-$HOME/.pixeltable}/pgdata}"
handled=' '
iteration=0
prev_pswpout=0
prev_oom_count=0

pg_query() {
    "$bin_path/psql" -X -A -t -F ' ' -h "$socket_dir" -p "$port" -U postgres -d postgres -c "$1" 2> /dev/null
}

check_hangs() {
    [[ -f "$pgdata/postmaster.pid" ]] || return
    port=$(sed -n 4p "$pgdata/postmaster.pid")
    socket_dir=$(sed -n 5p "$pgdata/postmaster.pid")
    if [[ -z "$bin_path" ]]; then
        bin_path=$(python -c 'import sys; from pixeltable_pgserver.utils import POSTGRES_VERSIONS, TARGET_POSTGRES_VERSION; sys.stdout.write(str(POSTGRES_VERSIONS[TARGET_POSTGRES_VERSION]))' 2> /dev/null)
    fi
    [[ -n "$bin_path" ]] || return
    local rows
    rows=$(pg_query "
        SELECT pid, application_name, extract(epoch FROM now() - state_change)::int
        FROM pg_stat_activity
        WHERE backend_type = 'client backend' AND state = 'idle in transaction'
            AND application_name LIKE 'pixeltable-%' AND now() - state_change > interval '${EXP_HANG_THRESHOLD:-8 minutes}'")
    [[ -n "$rows" ]] || return
    while read -r backend_pid app_name idle_secs; do
        [[ "$handled" == *" $backend_pid "* ]] && continue
        handled="$handled$backend_pid "
        local client_pid=${app_name#pixeltable-}
        echo "EXP-HANG ===== backend $backend_pid idle in transaction for ${idle_secs}s; client pid $client_pid: $(ps -o command= -p "$client_pid" | cut -c1-120)"
        echo "EXP-HANG wait function in this job: PSYCOPG_WAIT_FUNC=${PSYCOPG_WAIT_FUNC:-<default>}"
        ./scripts/dump-pg-state.sh 2>&1 | sed 's/^/EXP-HANG pg: /'
        echo "EXP-HANG lsof of client pid $client_pid:"
        lsof -n -P -p "$client_pid" 2>&1 | sed 's/^/EXP-HANG lsof: /'
        echo "EXP-HANG sending SIGUSR2 to $client_pid for the psycopg generator dump and flush probe"
        kill -USR2 "$client_pid" 2> /dev/null
        sleep 20 > /dev/null 2>&1
        local marker
        marker=$(cat "$RUNNER_TEMP/exp-probe-$client_pid" 2> /dev/null)
        echo "EXP-HANG probe result for $client_pid: ${marker:-<none>}"
        echo "EXP-HANG send() failures logged by pid $client_pid so far:"
        grep " pid $client_pid " "$EXP_SEND_LOG_PATH" 2> /dev/null | sed 's/^/EXP-HANG   /'
        echo "EXP-HANG backend $backend_pid 20 s after the probe: $(pg_query "SELECT state, extract(epoch FROM now() - state_change)::int, left(regexp_replace(query, '\s+', ' ', 'g'), 150) FROM pg_stat_activity WHERE pid = $backend_pid")"
        if [[ "$marker" == 'psycopg busy=1'* ]]; then
            echo "EXP-HANG CONFIRMED HANG: worker $client_pid is blocked in psycopg with a busy connection; killing it so that xdist replaces it"
            kill -9 "$client_pid" 2> /dev/null
        else
            echo "EXP-HANG not blocked in psycopg (long transaction, not a hang); leaving worker $client_pid running"
        fi
    done <<< "$rows"
}

sample_memory() {
    [[ "$(uname)" == 'Linux' ]] || return
    local avail swap_total swap_free pswpout
    avail=$(awk '/^MemAvailable:/ {print int($2/1024)}' /proc/meminfo)
    swap_total=$(awk '/^SwapTotal:/ {print int($2/1024)}' /proc/meminfo)
    swap_free=$(awk '/^SwapFree:/ {print int($2/1024)}' /proc/meminfo)
    pswpout=$(awk '/^pswpout/ {print $2}' /proc/vmstat)
    echo "EXP-MEM $(date -u +%H:%M:%S) avail=${avail}MB swap_used=$((swap_total - swap_free))MB pswpout_delta=$((pswpout - prev_pswpout)) load=$(cut -d' ' -f1-3 /proc/loadavg)"
    prev_pswpout=$pswpout
    if ((iteration % 4 == 0)); then
        ps -eo pid=,rss=,args= --sort=-rss | head -8 | awk '{printf "EXP-MEM   top pid=%s rss=%dMB ", $1, $2/1024; $1=""; $2=""; print substr($0, 3, 110)}'
    fi
    local oom
    oom=$(sudo dmesg 2> /dev/null | grep -iE 'out of memory|oom-kill|killed process|oom_reaper')
    local oom_count
    oom_count=$(printf '%s' "$oom" | grep -c . )
    if ((oom_count > prev_oom_count)); then
        printf '%s\n' "$oom" | tail -n $((oom_count - prev_oom_count)) | sed 's/^/EXP-MEM OOM: /'
        prev_oom_count=$oom_count
    fi
}

tail_sendlog() {
    [[ -n "$EXP_SEND_LOG_PATH" && -f "$EXP_SEND_LOG_PATH" ]] || return
    local lines
    lines=$(wc -l < "$EXP_SEND_LOG_PATH")
    if ((lines > sendlog_seen)); then
        tail -n $((lines - sendlog_seen)) "$EXP_SEND_LOG_PATH"
        sendlog_seen=$lines
    fi
}

sendlog_seen=0
while true; do
    sample_memory
    tail_sendlog
    if ((iteration % 2 == 0)); then
        check_hangs
    fi
    iteration=$((iteration + 1))
    sleep 15 > /dev/null 2>&1
done
