#!/bin/bash
#SBATCH --job-name=gdm-bayes-panama
#SBATCH --output=results/logs/bayes_panama_%A_%a.out
#SBATCH --error=results/logs/bayes_panama_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-8

# Array job: one task per model config (0-8).
# Each task runs 10-fold CV for one config.
# Non-spatial: ~5min/fit × 11 fits = ~1h.  Spatial: ~15min/fit × 11 fits = ~3h.
# Submit with: sbatch run_bayes_panama.sh

set -e
cd /cluster/home/haroldh/spgdmm/examples
mkdir -p results/logs results/panama ~/.cache/arviz

PYTHON=/cluster/home/haroldh/miniforge3/envs/spgdmm-test/bin/python
SEED=${SEED:-42}

echo "=== Bayesian spGDMM — Panama  config_idx=${SLURM_ARRAY_TASK_ID}  seed=${SEED} ==="
$PYTHON panama_example.py --mode bayes \
    --config_idx ${SLURM_ARRAY_TASK_ID} \
    --draws 1000 --tune 1000 --chains 4 --seed ${SEED} \
    --n_folds 10 \
    --output_dir results/panama

echo "Done (config_idx=${SLURM_ARRAY_TASK_ID}  seed=${SEED})."
