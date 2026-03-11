#!/bin/bash
# Quick setup script for spgdmm development environment

set -e

echo "Creating spGDMM mamba environment..."
mamba env create -f environment.yml || mamba env update -f environment.yml

echo ""
echo "Environment created successfully!"
echo ""
echo "To activate the environment, run:"
echo "  mamba activate spgdmm"
echo ""
echo "Then verify the installation:"
echo "  python -c 'from spgdmm import spGDMM, ModelVariant; print(\"spGDMM imported successfully!\")'"
echo ""
echo "To run tests:"
echo "  pytest tests/"
echo ""
echo "To run examples:"
echo "  python examples/basic_usage.py"
echo "  python examples/model_variants.py"