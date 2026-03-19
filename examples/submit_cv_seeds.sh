#!/bin/bash
# Submit CV array jobs for seeds 42, 123, 456 across all three datasets.
# Usage: bash submit_cv_seeds.sh
#
# Each dataset × seed = one array job (9 tasks, one per model config).
# Full-data .nc models are reused if already present; only the CV fold is refit.

set -e
cd /cluster/home/haroldh/spgdmm/examples

for SEED in 42 123 456; do
    echo "--- Submitting seed=${SEED} ---"

    sbatch \
        --export=ALL,SEED=${SEED} \
        --output="results/logs/bayes_panama_%A_%a_s${SEED}.out" \
        --error="results/logs/bayes_panama_%A_%a_s${SEED}.err" \
        run_bayes_panama.sh

    sbatch \
        --export=ALL,SEED=${SEED} \
        --output="results/logs/bayes_sw_%A_%a_s${SEED}.out" \
        --error="results/logs/bayes_sw_%A_%a_s${SEED}.err" \
        run_bayes_sw.sh

    sbatch \
        --export=ALL,SEED=${SEED} \
        --output="results/logs/bayes_gcfr_%A_%a_s${SEED}.out" \
        --error="results/logs/bayes_gcfr_%A_%a_s${SEED}.err" \
        run_bayes_gcfr.sh
done

echo ""
echo "Submitted 9 array jobs (3 datasets × 3 seeds × 9 configs = 81 tasks total)."
echo "Monitor with: squeue -u haroldh | grep gdm"
