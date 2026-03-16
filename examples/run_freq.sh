#!/bin/bash
#SBATCH --job-name=gdm-freq
#SBATCH --output=results/logs/freq_%j.out
#SBATCH --error=results/logs/freq_%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4

set -e
cd /cluster/home/haroldh/spgdmm/examples
mkdir -p results/logs

PYTHON=/cluster/home/haroldh/miniforge3/envs/spgdmm-test/bin/python

echo "=== Frequentist GDM: southwest ==="
$PYTHON southwest_example.py --mode freq --output_dir results/southwest

echo "=== Frequentist GDM: BCI ==="
$PYTHON bci_example.py --mode freq --output_dir results/bci

echo "All done."
