#!/bin/bash
#SBATCH --job-name=gdm-freq-panama
#SBATCH --output=results/logs/freq_panama_%j.out
#SBATCH --error=results/logs/freq_panama_%j.err
#SBATCH --time=1:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2

set -e
cd /cluster/home/haroldh/spgdmm/experiments/white2024_cv
mkdir -p results/logs results/panama

PYTHON=/cluster/home/haroldh/miniforge3/envs/spgdmm-test/bin/python
SEED=${SEED:-42}

echo "=== Frequentist GDM 10-fold CV — Panama  seed=${SEED} ==="
$PYTHON panama_example.py --mode freq --seed ${SEED} --output_dir results/panama

echo "Done."
