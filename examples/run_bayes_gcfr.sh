#!/bin/bash
#SBATCH --job-name=gdm-bayes-gcfr
#SBATCH --output=results/logs/bayes_gcfr_%A_%a.out
#SBATCH --error=results/logs/bayes_gcfr_%A_%a.err
#SBATCH --time=10:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --array=0-7

# Array job: one task per model configuration (0-7).
# Each task runs 5-fold CV for one config (~2h/fit × 5 folds = ~10h).
# Submit with: sbatch run_bayes_gcfr.sh

set -e
cd /cluster/home/haroldh/spgdmm/examples
mkdir -p results/logs results/gcfr ~/.cache/arviz

PYTHON=/cluster/home/haroldh/miniforge3/envs/spgdmm-test/bin/python

echo "=== Bayesian spGDMM — GCFR  config_idx=${SLURM_ARRAY_TASK_ID} ==="
$PYTHON gcfr_example.py --mode bayes \
    --config_idx ${SLURM_ARRAY_TASK_ID} \
    --draws 1000 --tune 1000 --chains 4 --seed 42 \
    --n_folds 1 \
    --output_dir results/gcfr

echo "Done (config_idx=${SLURM_ARRAY_TASK_ID})."
