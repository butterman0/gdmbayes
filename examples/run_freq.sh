#!/bin/bash
#SBATCH --job-name=gdm-freq
#SBATCH --output=results/logs/freq_%j.out
#SBATCH --error=results/logs/freq_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

set -e
cd /cluster/home/haroldh/spgdmm/examples
mkdir -p results/logs results/southwest results/panama results/gcfr

PYTHON=/cluster/home/haroldh/miniforge3/envs/spgdmm-test/bin/python

echo "=== Frequentist GDM: SW Australia ==="
$PYTHON southwest_example.py --mode freq --n_folds 1 --output_dir results/southwest

echo "=== Frequentist GDM: Panama ==="
$PYTHON panama_example.py --mode freq --n_folds 1 --output_dir results/panama

echo "=== Frequentist GDM: GCFR ==="
$PYTHON gcfr_example.py --mode freq --n_folds 1 --output_dir results/gcfr

echo "All done."
