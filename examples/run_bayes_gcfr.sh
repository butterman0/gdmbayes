#!/bin/bash
#SBATCH --job-name=gdm-bayes-gcfr
#SBATCH --output=results/logs/bayes_gcfr_%j.out
#SBATCH --error=results/logs/bayes_gcfr_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

set -e
cd /cluster/home/haroldh/spgdmm/examples
mkdir -p results/logs results/gcfr ~/.cache/arviz

PYTHON=/cluster/home/haroldh/miniforge3/envs/spgdmm-test/bin/python

echo "=== Bayesian spGDMM (no spatial): GCFR ==="
$PYTHON gcfr_example.py --mode bayes --spatial none \
    --draws 1000 --tune 1000 --chains 4 --seed 42 \
    --output_dir results/gcfr

echo "=== Bayesian spGDMM (abs_diff): GCFR ==="
$PYTHON gcfr_example.py --mode bayes --spatial abs_diff \
    --draws 1000 --tune 1000 --chains 4 --seed 42 \
    --output_dir results/gcfr

echo "=== Bayesian spGDMM (squared_diff): GCFR ==="
$PYTHON gcfr_example.py --mode bayes --spatial squared_diff \
    --draws 1000 --tune 1000 --chains 4 --seed 42 \
    --output_dir results/gcfr

echo "All done."
