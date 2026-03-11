#!/bin/bash
# Quick verification script - run after installing spgdmm-test environment
# Usage: source verify_install.sh or bash verify_install.sh

echo "=========================================="
echo "spGDMM Package Verification"
echo "=========================================="

# Check if we're in the right directory
if [[ ! -f environment.yml ]]; then
    echo "Error: Please run this from the spgdmm-package directory"
    echo "  cd /cluster/home/haroldh/spGDMM/spgdmm-package"
    exit 1
fi

# Activate environment (try both mamba and conda)
echo ""
echo "Activating conda/mamba environment..."
if command -v mamba &> /dev/null; then
    source ~/mambaforge/etc/profile.d/conda.sh
    mamba activate spgdmm-test
elif command -v conda &> /dev/null; then
    source ~/miniforge3/etc/profile.d/conda.sh
    conda activate spgdmm-test
else
    echo "Error: Neither mamba nor conda found in common locations"
    echo "Try one of these to find conda:"
    echo "  which conda"
    echo "  ls ~/mambaforge/etc/profile.d/"
    echo "  ls ~/miniforge3/etc/profile.d/"
    exit 1
fi

echo "Active environment: $CONDA_DEFAULT_ENV"

# Run quick tests
echo ""
echo "Running quick verification..."

python -c "
try:
    from spgdmm import spGDMM, ModelVariant, ModelConfig, VarianceType, SpatialEffectType
    print('✓ Imports successful!')
    print(f'  spGDMM version: {spGDMM.version}')
    print(f'  Variants available: {len(ModelVariant)}')
except ImportError as e:
    print(f'✗ Import failed: {e}')
    exit(1)
"

echo ""
echo "Quick synthetic test..."
python -c "
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from spgdmm import spGDMM, ModelVariant

np.random.seed(42)
n = 20

X = pd.DataFrame({
    'xc': np.random.uniform(0, 100, n),
    'yc': np.random.uniform(0, 100, n),
    'time_idx': np.zeros(n, dtype=int),
    'temp': np.random.uniform(5, 20, n),
    'depth': np.random.uniform(0, 200, n),
})

biomass = np.random.exponential(1, (n, 10))
y = np.clip(pdist(biomass, 'braycurtis'), 1e-8, None)

model = spGDMM.from_variant(ModelVariant.MODEL1)
model._generate_and_preprocess_model_data(X, y)
model.build_model(model.X_transformed, model.y_transformed)

print('✓ Model built successfully!')
print(f'  Sites: {n}, Pairs: {len(y)}')
print(f'  Model has {len(model.model.free_RVs)} free variables')
"

echo ""
echo "=========================================="
echo "Verification Complete!"
echo "=========================================="
echo ""
echo "To run full tests:"
echo "  bash test_hpc.sh          # For interactive testing"
echo "  sbatch test_hpc.sbatch    # For SLURM submission"