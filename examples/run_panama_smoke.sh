#!/bin/bash
#SBATCH --job-name=panama-smoke
#SBATCH --output=results/logs/panama_smoke_%A_%a.out
#SBATCH --error=results/logs/panama_smoke_%A_%a.err
#SBATCH --time=1:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-8

# Smoke test: all 9 configs, 3 folds of the 10-fold split (same seed/folds
# as the full run). ~6-15 min per config depending on spatial effect.

set -e
cd /cluster/home/haroldh/spgdmm/examples
mkdir -p results/logs results/panama

PYTHON=/cluster/home/haroldh/miniforge3/envs/spgdmm-test/bin/python
SEED=42

echo "=== Panama smoke — config_idx=${SLURM_ARRAY_TASK_ID}  3-fold  seed=${SEED} ==="
echo "Commit: $(git -C /cluster/home/haroldh/spgdmm rev-parse --short HEAD)"
echo "Start: $(date)"

$PYTHON panama_example.py \
    --mode bayes \
    --config_idx ${SLURM_ARRAY_TASK_ID} \
    --draws 1000 --tune 4000 --chains 4 --seed ${SEED} \
    --n_folds 3 \
    --skip_full_model \
    --output_dir results/panama

echo "End: $(date)"
echo "Done (config_idx=${SLURM_ARRAY_TASK_ID})."
