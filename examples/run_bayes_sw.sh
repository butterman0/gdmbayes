#!/bin/bash
#SBATCH --job-name=gdm-bayes-sw
#SBATCH --output=results/logs/bayes_sw_%A_%a.out
#SBATCH --error=results/logs/bayes_sw_%A_%a.err
#SBATCH --time=4:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-8

# Array job: one task per model config (0-8).
# Each task runs 10-fold CV for one config (~20min/fit × 10 folds = ~3.5h).
# Submit with: sbatch run_bayes_sw.sh

set -e
cd /cluster/home/haroldh/spgdmm/examples
mkdir -p results/logs results/southwest ~/.cache/arviz

PYTHON=/cluster/home/haroldh/miniforge3/envs/spgdmm-test/bin/python
SEED=${SEED:-42}

echo "=== Bayesian spGDMM — SW Australia  config_idx=${SLURM_ARRAY_TASK_ID}  seed=${SEED} ==="
$PYTHON southwest_example.py --mode bayes \
    --config_idx ${SLURM_ARRAY_TASK_ID} \
    --draws 1000 --tune 1000 --chains 4 --seed ${SEED} \
    --n_folds 10 \
    --output_dir results/southwest

echo "Done (config_idx=${SLURM_ARRAY_TASK_ID}  seed=${SEED})."
