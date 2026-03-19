#!/bin/bash
#SBATCH --job-name=gdm-bayes-panama
#SBATCH --output=results/logs/bayes_panama_%A_%a.out
#SBATCH --error=results/logs/bayes_panama_%A_%a.err
#SBATCH --time=2:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --array=0-8

# Array job: one task per model config (0-8).
# Each task runs 10-fold CV for one config (~5min/fit × 10 folds = ~1h).
# Submit with: sbatch run_bayes_panama.sh

set -e
cd /cluster/home/haroldh/spgdmm/examples
mkdir -p results/logs results/panama ~/.cache/arviz

PYTHON=/cluster/home/haroldh/miniforge3/envs/spgdmm-test/bin/python

echo "=== Bayesian spGDMM — Panama  config_idx=${SLURM_ARRAY_TASK_ID} ==="
$PYTHON panama_example.py --mode bayes \
    --config_idx ${SLURM_ARRAY_TASK_ID} \
    --draws 1000 --tune 1000 --chains 4 --seed 42 \
    --n_folds 1 \
    --output_dir results/panama

echo "Done (config_idx=${SLURM_ARRAY_TASK_ID})."
