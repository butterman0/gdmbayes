#!/bin/bash
#SBATCH --job-name=panama-poly
#SBATCH --output=results/logs/panama_poly_%A_%a.out
#SBATCH --error=results/logs/panama_poly_%A_%a.err
#SBATCH --time=4:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --array=1,4,7

# Smoke test: covariate_dependent variance configs after R poly() refactor.
# 1-fold CV only to verify correctness.

set -e
cd /cluster/home/haroldh/spgdmm/examples
mkdir -p results/logs results/panama

PYTHON=/cluster/home/haroldh/miniforge3/envs/spgdmm-test/bin/python
SEED=42

echo "=== Panama poly test — config_idx=${SLURM_ARRAY_TASK_ID} ==="
echo "Commit: $(git -C /cluster/home/haroldh/spgdmm rev-parse --short HEAD)"
echo "Start: $(date)"

$PYTHON panama_example.py \
    --mode bayes \
    --config_idx ${SLURM_ARRAY_TASK_ID} \
    --draws 500 --tune 2000 --chains 4 --seed ${SEED} \
    --n_folds 1 \
    --skip_full_model \
    --output_dir results/panama

echo "End: $(date)"
echo "Done (config_idx=${SLURM_ARRAY_TASK_ID})."
