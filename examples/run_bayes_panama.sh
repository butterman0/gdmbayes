#!/bin/bash
#SBATCH --job-name=gdm-bayes-panama
#SBATCH --output=results/logs/bayes_panama_%j.out
#SBATCH --error=results/logs/bayes_panama_%j.err
#SBATCH --time=10:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8

set -e
cd /cluster/home/haroldh/spgdmm/examples
mkdir -p results/logs results/panama ~/.cache/arviz

PYTHON=/cluster/home/haroldh/miniforge3/envs/spgdmm-test/bin/python

echo "=== Bayesian spGDMM — Panama (all 8 configs × 10-fold CV) ==="
$PYTHON panama_example.py --mode bayes \
    --draws 1000 --tune 1000 --chains 4 --seed 42 \
    --output_dir results/panama

echo "All done."
