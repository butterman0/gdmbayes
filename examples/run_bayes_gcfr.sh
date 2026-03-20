#!/bin/bash
#SBATCH --job-name=gdm-bayes-gcfr
#SBATCH --output=results/logs/bayes_gcfr_%A_%a.out
#SBATCH --error=results/logs/bayes_gcfr_%A_%a.err
#SBATCH --time=48:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --array=0-8

# Array job: one task per model configuration (0-8).
# Each task runs 10-fold CV for one config (~2h/fit × 10 folds = ~20h).
# Submit with: sbatch run_bayes_gcfr.sh

set -e
cd /cluster/home/haroldh/spgdmm/examples
mkdir -p results/logs results/gcfr ~/.cache/arviz

PYTHON=/cluster/home/haroldh/miniforge3/envs/spgdmm-test/bin/python
SEED=${SEED:-42}

echo "=== Bayesian spGDMM — GCFR  config_idx=${SLURM_ARRAY_TASK_ID}  seed=${SEED} ==="
$PYTHON gcfr_example.py --mode bayes \
    --config_idx ${SLURM_ARRAY_TASK_ID} \
    --draws 1000 --tune 1000 --chains 4 --seed ${SEED} \
    --n_folds 10 \
    --output_dir results/gcfr

echo "Done (config_idx=${SLURM_ARRAY_TASK_ID}  seed=${SEED})."
