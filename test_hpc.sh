#!/bin/bash
# Run spGDMM tests on HPC (non-SLURM version)

echo "=========================================="
echo "Running spGDMM Package Tests"
echo "=========================================="
echo "Date: $(date)"
echo "User: $USER"
echo "Host: $HOSTNAME"
echo ""

# Setup paths
export PACKAGE_DIR="/cluster/home/haroldh/spGDMM/spgdmm-package"
cd $PACKAGE_DIR

# Activate mamba environment
echo "Activating mamba environment..."
source ~/mambaforge/etc/profile.d/conda.sh
mamba activate spgdmm-test

# Test 1: Print environment info
echo "=========================================="
echo "Environment Info"
echo "=========================================="
python --version
which python
echo ""
echo "Installed packages:"
pip list | grep -E "(pymc|arviz|numpy|pandas|scipy|dms-variants|ISLP|json-tricks)"
echo ""

# Test 2: Quick import test
echo "=========================================="
echo "Quick Import Test"
echo "=========================================="
python -c "
from spgdmm import spGDMM, ModelVariant, ModelConfig, VarianceType, SpatialEffectType
print('✓ All imports successful!')
print(f'  spGDMM version: {spGDMM.version}')
print(f'  Model variants: {[v.value for v in ModelVariant]}')
"
echo ""

# Test 3: Run quick test script
echo "=========================================="
echo "Running Quick Test"
echo "=========================================="
python quick_test.py
echo ""

# Test 4: Run model variants example
echo "=========================================="
echo "Model Variants Demo"
echo "=========================================="
python examples/model_variants.py
echo ""

# Test 5: Run unit tests with pytest (if available)
echo "=========================================="
echo "Running Unit Tests"
echo "=========================================="
if python -c "import pytest" 2>/dev/null; then
    pytest tests/ -v --tb=short
else
    echo "pytest not installed, skipping unit tests"
    echo "Install with: pip install pytest"
fi
echo ""

# Test 6: Run basic usage example (with minimal sampling)
echo "=========================================="
echo "Basic Usage Example (Minimal Sampling)"
echo "=========================================="
python -c "
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from spgdmm import spGDMM, ModelVariant

print('Creating synthetic data...')
np.random.seed(42)
n_sites = 30

X = pd.DataFrame({
    'xc': np.random.uniform(0, 100, n_sites),
    'yc': np.random.uniform(0, 100, n_sites),
    'time_idx': np.zeros(n_sites, dtype=int),
    'temp': np.random.uniform(5, 20, n_sites),
    'salinity': np.random.uniform(30, 35, n_sites),
    'depth': np.random.uniform(0, 200, n_sites),
})

biomass = np.random.exponential(1, (n_sites, 15))
y = pdist(biomass, 'braycurtis')
y = np.clip(y, 1e-8, None)

print(f'Sites: {n_sites}, Pairs: {len(y)}')
print()
print('Creating model and fitting (minimal sampling)...')
model = spGDMM.from_variant(ModelVariant.MODEL1, deg=3, knots=2)

# Minimal sampling for quick test
idata = model.fit(
    X, y,
    random_seed=42,
    draws=50,
    tune=50,
    chains=2,
    progressbar=False,
)

print()
print('✓ Fit completed!')
print(f'  Posterior shape: {idata.posterior.dims}')
print(f'  Variables: {list(idata.posterior.data_vars.keys())}')
print()
print('Test sample size predictions...')
X_pred = X.iloc[:5]
pred = model.predict_posterior(X_pred, extend_idata=False)
print(f'  Predictions shape: {pred.shape}')
print(f'  Mean predictions: {pred.mean(dim=\"sample\").values}')
"
echo ""

echo "=========================================="
echo "All Tests Complete!"
echo "=========================================="