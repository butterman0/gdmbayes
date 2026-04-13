#!/bin/bash
#SBATCH --job-name=gdm-freq-sw
#SBATCH --output=results/logs/freq_sw_%j.out
#SBATCH --error=results/logs/freq_sw_%j.err
#SBATCH --time=1:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2

set -e
cd /cluster/home/haroldh/spgdmm/experiments/white2024_cv
mkdir -p results/logs results/southwest

PYTHON=/cluster/home/haroldh/miniforge3/envs/spgdmm-test/bin/python
SEED=${SEED:-42}

echo "=== Frequentist GDM 10-fold CV — SW Australia  seed=${SEED} ==="
$PYTHON southwest_example.py --mode freq --seed ${SEED} --output_dir results/southwest

echo "Done."
