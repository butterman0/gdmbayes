#!/bin/bash
# Monitor the pilot job until it completes, then show results.
INTERVAL=30
RESULTS_DIR=/cluster/home/haroldh/spgdmm/examples/results/panama

while true; do
    clear
    echo "=== $(date) ==="
    echo ""
    echo "--- Queue ---"
    squeue -u haroldh | grep gdm || echo "  (no gdm jobs running)"
    echo ""
    echo "--- Output files ---"
    ls "$RESULTS_DIR" | grep -v freq || echo "  (none yet)"
    echo ""

    if ! squeue -u haroldh | grep -q gdm; then
        echo "--- Log tail ---"
        tail -20 "$RESULTS_DIR"/../logs/bayes_panama_*_s42.out 2>/dev/null || true
        echo ""
        echo "Pilot job finished. Exiting monitor."
        break
    fi

    sleep "$INTERVAL"
done
