#!/bin/bash
#SBATCH --job-name=pan1-white
#SBATCH --output=logs/bayes_white_%j.out
#SBATCH --error=logs/bayes_white_%j.err
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --account=share-ie-itk

set -e
cd /cluster/home/haroldh/spgdmm/experiments/panama_single_fold
mkdir -p logs results

source /cluster/home/haroldh/miniforge3/etc/profile.d/conda.sh
conda activate R

echo "=== Panama single-fold — White NIMBLE models {1,2,4,7} ==="
echo "Commit: $(git -C /cluster/home/haroldh/spgdmm rev-parse --short HEAD)"
echo "Start: $(date)"
Rscript run_bayes_white.R
echo "End: $(date)"
