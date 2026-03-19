#!/bin/bash
# Monitor gdm SLURM jobs and results until all finish.
# Usage: bash monitor_jobs.sh [interval_seconds]

INTERVAL=${1:-30}
RESULTS_DIR=/cluster/home/haroldh/spgdmm/examples/results
LOG_DIR=${RESULTS_DIR}/logs

echo "Waiting for gdm jobs to appear..."
while ! squeue -u haroldh | grep -q gdm; do sleep 5; done

while squeue -u haroldh | grep -q gdm; do
    clear
    echo "=== $(date)  [refresh every ${INTERVAL}s — Ctrl-C to exit] ==="
    echo ""

    echo "--- Running / Pending ---"
    squeue -u haroldh --format="%-18i %-22j %-8T %M" | grep -E "JOBID|gdm"
    echo ""

    n_done=$(cat ${RESULTS_DIR}/*/cv_metrics.csv 2>/dev/null | grep -vc "^dataset" || echo 0)
    echo "--- Completed metrics (${n_done} rows) ---"
    cat ${RESULTS_DIR}/*/cv_metrics.csv 2>/dev/null \
        | awk -F, 'NR==1 || !seen[$1$2$3]++ {printf "%-10s %-35s %-6s %-8s %-8s %-8s\n", $1,$2,$3,$6,$7,$8}' \
        | head -30
    echo ""

    echo "--- Recent errors (non-empty .err files) ---"
    for f in $(ls -t ${LOG_DIR}/*.err 2>/dev/null | head -10); do
        if [ -s "$f" ]; then
            echo "  $(basename $f):"
            tail -3 "$f" | sed 's/^/    /'
        fi
    done

    sleep "${INTERVAL}"
done

clear
echo "=== $(date) — All gdm jobs finished ==="
echo ""
echo "--- Final metrics ---"
cat ${RESULTS_DIR}/*/cv_metrics.csv 2>/dev/null \
    | awk -F, 'NR==1 || !seen[$1$2$3]++ {printf "%-10s %-35s %-6s %-8s %-8s %-8s\n", $1,$2,$3,$6,$7,$8}'
