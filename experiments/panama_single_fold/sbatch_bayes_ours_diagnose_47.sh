#!/bin/bash
#SBATCH --job-name=pan1-diag47
#SBATCH --output=logs/bayes_ours_diagnose_47_%j.out
#SBATCH --error=logs/bayes_ours_diagnose_47_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --account=share-ie-itk

set -e
cd /cluster/home/haroldh/spgdmm/experiments/panama_single_fold
mkdir -p logs results

source /cluster/home/haroldh/miniforge3/etc/profile.d/conda.sh
conda activate spgdmm-test

export DRAWS=${DRAWS:-1000}
export TUNE=${TUNE:-3000}
export CHAINS=${CHAINS:-4}
export TARGET_ACCEPT=${TARGET_ACCEPT:-0.97}

echo "=== Panama single-fold — diagnose models {4,7} R-hat (per-chain metrics) ==="
echo "Commit: $(git -C /cluster/home/haroldh/spgdmm rev-parse --short HEAD)"
echo "Draws=${DRAWS}  Tune=${TUNE}  Chains=${CHAINS}  target_accept=${TARGET_ACCEPT}"
echo "Start: $(date)"
python run_bayes_ours_diagnose_47.py
echo "End: $(date)"
