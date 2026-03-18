#!/bin/bash
# Monitor SLURM jobs until all finish, then tail the logs.
JOBS=(24192954 24192955 24192956)
INTERVAL=60

echo "Monitoring jobs: ${JOBS[*]}"
echo "Press Ctrl+C to stop."
echo

while true; do
    clear
    echo "=== $(date '+%H:%M:%S') ==="
    squeue -u haroldh -o "%-10i %-10j %-8T %-10M %R" 2>/dev/null

    still_running=0
    for jid in "${JOBS[@]}"; do
        if squeue -j "$jid" -h &>/dev/null; then
            still_running=1
        fi
    done

    if [ "$still_running" -eq 0 ]; then
        echo
        echo "=== All jobs finished ==="
        for jid in "${JOBS[@]}"; do
            logfile=$(ls results/logs/*_${jid}.out 2>/dev/null | head -1)
            errfile=$(ls results/logs/*_${jid}.err 2>/dev/null | head -1)
            echo
            echo "--- Job $jid: $logfile ---"
            [ -f "$logfile" ] && tail -30 "$logfile"
            if [ -s "$errfile" ]; then
                echo "--- STDERR ---"
                tail -10 "$errfile"
            fi
        done
        break
    fi

    sleep "$INTERVAL"
done
