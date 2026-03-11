"""
GDM- Compatible Usage Example.

This example demonstrates how to use spGDMM with the same input/output
format as the R GDM package.

The workflow is:
1. Prepare biological and environmental data
2. Use format_site_pair() to create site-pair table
3. Fit GDM model with GDMModel or gdm() function
4. Access results in R GDM-compatible format
"""

import numpy as np
import pandas as pd
from spgdmm import (
    format_site_pair,
    BioFormat,
    GDMModel,
    gdm,
    ModelConfig,
    ModelVariant,
)


def example_basic_gdm_workflow():
    """Basic GDM workflow matching R GDM package."""
    print("=" * 60)
    print("Basic GDM-Compatible Workflow")
    print("=" * 60)

    # Step 1: Prepare biological data (FORMAT2: long format with site, species, coords)
    # This mimics typical ecological data from field surveys
    print("\n1. Preparing biological data...")

    bio_data = pd.DataFrame({
        "site": ["A", "A", "A", "B", "B", "B", "C", "C", "C"],
        "species": ["sp1", "sp2", "sp3", "sp1", "sp2", "sp3", "sp1", "sp2", "sp3"],
        "xCoord": [0.0, 0.0, 0.0, 10.0, 10.0, 10.0, 20.0, 20.0, 20.0],
        "yCoord": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "abundance": [10, 5, 8, 8, 3, 6, 5, 2, 4]
    })
    print(f"   Biological data: {len(bio_data)} records")

    # Step 2: Prepare environmental predictor data
    print("\n2. Preparing environmental data...")

    pred_data = pd.DataFrame({
        "temp": {"A": 10.0, "B": 15.0, "C": 20.0},
        "precip": {"A": 500.0, "B": 400.0, "C": 300.0}
    })
    print(f"   Predictors: {', '.join(pred_data.columns)}")

    # Step 3: Format data into site-pair table (matches R's formatsitepair)
    print("\n3. Creating site-pair table...")

    site_pair = format_site_pair(
        bio_data=bio_data,
        bio_format=BioFormat.FORMAT2,
        site_column="site",
        x_column="xCoord",
        y_column="yCoord",
        species_column="species",
        abund_column="abundance",
        pred_data=pred_data,
        dist="bray",
        verbose=True
    )
    print(f"   Site-pair shape: {site_pair.shape}")
    print(f"   Columns: {list(site_pair.columns)}")

    # Step 4: Fit GDM model (two equivalent ways)
    print("\n4a. Fitting GDM model using GDMModel class...")

    model_cls = GDMModel(geo=True, splines=3)
    result_cls = model_cls.fit(site_pair, dataname="my_study")

    print("\n4b. Alternative: Fit using gdm() function (matches R GDM)...")

    result_func = gdm(site_pair, geo=True, splines=3, dataname="my_study")

    # Step 5: Access results in R GDM-compatible format
    print("\n5. Results (R GDM-compatible format):")
    print(f"   - Data name: {result_cls.dataname}")
    print(f"   - Geographic distance included: {result_cls.geo}")
    print(f"   - Null deviance: {result_cls.nulldeviance:.3f}")
    print(f"   - GDM deviance: {result_cls.gdmdeviance:.3f}")
    print(f"   - Deviance explained: {result_cls.explained:.1f}%")
    print(f"   - Intercept: {result_cls.intercept:.3f}")
    print(f"   - Predictors (by importance): {', '.join(result_cls.predictors)}")
    print(f"   - Number of spline bases: {result_cls.splines}")
    print(f"   - Creation date: {result_cls.creationdate}")

    # Access coefficients
    print("\n6. Predictor coefficients:")
    for pred, coeff in result_cls.coefficients.items():
        print(f"   - {pred}: {coeff:.4f}")

    # Access observed vs predicted
    print("\n7. Model fit summary:")
    print(f"   - Observed distance range: [{result_cls.observed.min():.3f}, {result_cls.observed.max():.3f}]")
    print(f"   - Predicted distance range: [{result_cls.predicted.min():.3f}, {result_cls.predicted.max():.3f}]")
    print(f"   - Ecological distance range: [{result_cls.ecological.min():.3f}, {result_cls.ecological.max():.3f}]")

    # Step 8: Make predictions on new site pairs
    print("\n8. Making predictions on new data...")

    new_site_pair = site_pair.iloc[:2].copy()  # Use first 2 pairs as example
    predictions = model_cls.predict(new_site_pair)

    print(f"   Predicted distances: {predictions.predicted}")
    print(f"   Ecological distances: {predictions.ecological}")

    # Step 9: Access full inference data (spGDMM extension)
    print("\n9. Accessing full Bayesian inference data...")
    if result_cls.idata is not None:
        print(f"   - Posterior chains: {result_cls.idata.posterior.dims['chain']}")
        print(f"   - Posterior draws: {result_cls.idata.posterior.dims['draw']}")
        print(f"   - Posterior variables: {list(result_cls.idata.posterior.data_vars)}")

    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)

    return result_cls


def example_format1_usage():
    """Example using FORMAT1 (site x species matrix)."""
    print("\n" + "=" * 60)
    print("FORMAT1 Usage Example")
    print("=" * 60)

    # FORMAT1: Sites as rows, species as columns
    print("\n1. Using FORMAT1 (site x species abundance matrix)...")

    bio_data = pd.DataFrame({
        "sp1": {"A": 10, "B": 8, "C": 5, "D": 3},
        "sp2": {"A": 5, "B": 3, "C": 2, "D": 1},
        "sp3": {"A": 8, "B": 6, "C": 4, "D": 2},
    })
    print(f"   Shape: {bio_data.shape}")
    print(f"   Sites: {list(bio_data.index)}")
    print(f"   Species: {list(bio_data.columns)}")

    # Add coordinate information
    # For FORMAT1, coordinates need to be provided separately
    # Here we'll use site indices as coordinates for simplicity
    bio_data.index.name = "site"

    # With FORMAT1, we need to add coordinates to the data
    # This is a simplification - in practice, pass coordinates separately
    print("\n2. Creating site-pair table...")

    # For FORMAT1 with site x species, we need coordinates
    # Here we use site indices
    coords = np.array([[0, 0], [10, 0], [20, 0], [30, 0]])
    sites = list(bio_data.index)

    # predictor data
    pred_data = pd.DataFrame({
        "temp": {"A": 10.0, "B": 12.0, "C": 14.0, "D": 16.0},
        "elevation": {"A": 100, "B": 150, "C": 200, "D": 250}
    })
    pred_data.index.name = "site"

    # Note: format_site_pair with FORMAT1 needs coordinates
    # This is handled in the function by checking for coordinate columns
    print("\n   Note: FORMAT1 requires coordinate data")
    print("   In practice, use FORMAT2 or provide coordinate columns")

    return bio_data


def example_comparison_with_r_gdm():
    """Show side-by-side comparison with R GDM code."""
    print("\n" + "=" * 60)
    print("Comparison with R GDM Package")
    print("=" * 60)

    print("""
    # R GDM code:
    library(gdm)

    # Create site-pair table
    gdmTab <- formatsitepair(
        bioData = sppTab,
        bioFormat = 2,
        XColumn = "Long",
        YColumn = "Lat",
        sppColumn = "species",
        siteColumn = "site",
        predData = envTab
    )

    # Fit GDM model
    gdmMod <- gdm(data = gdmTab, geo = TRUE, splines = 3)

    # Access results
    gdmMod$explained        # Deviance explained
    gdmMod$(coefficients)   # Spline coefficients
    gdmMod$predictors       # Predictor names

    # Predictions
    pred <- gdm.predict(gdmMod, newSiteData)

    # Python spGDMM code (equivalent):
    from spgdmm import format_site_pair, BioFormat, gdm, GDMModel

    # Create site-pair table
    gdm_tab = format_site_pair(
        bio_data=spp_tab,
        bio_format=BioFormat.FORMAT2,
        x_column="Long",
        y_column="Lat",
        species_column="species",
        site_column="site",
        pred_data=env_tab
    )

    # Fit GDM model
    gdm_result = gdm(data=gdm_tab, geo=True, splines=3)
    # OR:
    gdm_model = GDMModel(geo=True, splines=3)
    gdm_result = gdm_model.fit(gdm_tab)

    # Access results (same attribute names as R!)
    gdm_result.explained        # Deviance explained
    gdm_result.coefficients     # Spline coefficients
    gdm_result.predictors       # Predictor names

    # Predictions
    predictions = gdm_model.predict(new_site_data)
    """)

    print("\nKey compatibility points:")
    print("  1. format_site_pair() matches formatsitepair() output format")
    print("  2. gdm() function matches R gdm() function signature")
    print("  3. Result object has same attributes as R GDM model object")
    print("  4. Additional Bayesian inference available via result.idata")


def main():
    """Run all examples."""
    # Basic workflow
    result = example_basic_gdm_workflow()

    # FORMAT1 example
    example_format1_usage()

    # R comparison
    example_comparison_with_r_gdm()

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()