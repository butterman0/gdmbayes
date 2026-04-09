#!/bin/bash
#SBATCH --job-name=panama-gpfix
#SBATCH --output=results/logs/panama_gpfix_%A_%a.out
#SBATCH --error=results/logs/panama_gpfix_%A_%a.err
#SBATCH --time=2:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --array=0,1,3,4,6,7

# Test GP coordinate/kernel fix. All 6 configs White reports for Panama.
# 3-fold smoke test (same folds as eventual 10-fold).

set -e
cd /cluster/home/haroldh/spgdmm/examples
mkdir -p results/logs results/panama

PYTHON=/cluster/home/haroldh/miniforge3/envs/spgdmm-test/bin/python
SEED=42

echo "=== Panama GP-fix — config_idx=${SLURM_ARRAY_TASK_ID}  3-fold  seed=${SEED} ==="
echo "Commit: $(git -C /cluster/home/haroldh/spgdmm rev-parse --short HEAD)"
echo "Start: $(date)"

$PYTHON panama_example.py \
    --mode bayes \
    --config_idx ${SLURM_ARRAY_TASK_ID} \
    --draws 1000 --tune 4000 --chains 4 --seed ${SEED} \
    --n_folds 3 \
    --output_dir results/panama

echo "End: $(date)"
echo "Done (config_idx=${SLURM_ARRAY_TASK_ID})."
