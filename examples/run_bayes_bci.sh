#!/bin/bash
#SBATCH --job-name=gdm-bayes-bci
#SBATCH --output=results/logs/bayes_bci_%j.out
#SBATCH --error=results/logs/bayes_bci_%j.err
#SBATCH --time=03:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8

set -e
cd /cluster/home/haroldh/spgdmm/examples
mkdir -p results/logs results/bci

PYTHON=/cluster/home/haroldh/miniforge3/envs/spgdmm-test/bin/python

echo "=== Bayesian spGDMM (no spatial): BCI ==="
$PYTHON bci_example.py --mode bayes --spatial none \
    --draws 1000 --tune 1000 --chains 4 --seed 42 \
    --output_dir results/bci

echo "=== Bayesian spGDMM (abs_diff): BCI ==="
$PYTHON bci_example.py --mode bayes --spatial abs_diff \
    --draws 1000 --tune 1000 --chains 4 --seed 42 \
    --output_dir results/bci

echo "All done."
