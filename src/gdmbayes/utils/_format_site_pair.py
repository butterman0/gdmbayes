"""format_site_pair: Convert biological and environmental data to site-pair format.

This module provides functions to format site-level biological and environmental
data into the site-pair table format compatible with the R GDM package.

The site-pair table format matches the output of R's formatsitepair function:
- Column 1: Biological distance/dissimilarity (response)
- Column 2: Weight for model fitting
- Columns 3-4: s1.xCoord, s1.yCoord (first site coordinates)
- Columns 5-6: s2.xCoord, s2.yCoord (second site coordinates)
- Remaining columns: s1.{predictor} and s2.{predictor} for each predictor
"""

from enum import Enum
from typing import Literal, Optional, Union

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform


class BioFormat(str, Enum):
    """Biological data format codes matching R GDM package.

    Each format specifies a different way to organize the biological data:

    - FORMAT1: Site rows, columns are species with abundances
    - FORMAT2: Species-site matrix with columns for site, species, coordinate, abundance
    - FORMAT3: Dissimilarity matrix with site coordinates
    - FORMAT4: Pre-computed site-pair table format
    """

    FORMAT1 = 1  # Site x Species abundance matrix
    FORMAT2 = 2  # Separate site, species, coordinate columns
    FORMAT3 = 3  # Dissimilarity matrix format
    FORMAT4 = 4  # Site-pair table format


def format_site_pair(
    bio_data: pd.DataFrame,
    bio_format: Union[int, str, BioFormat] = BioFormat.FORMAT2,
    dist: str = "braycurtis",
    is_abundance: bool = True,
    site_column: str = "site",
    x_column: str = "xCoord",
    y_column: str = "yCoord",
    species_column: str = "species",
    abund_column: str = "abundance",
    spp_filter: int = 0,
    pred_data: Optional[pd.DataFrame] = None,
    dist_preds: Optional[dict] = None,
    weight_type: Literal["equal", "richness", "custom"] = "equal",
    custom_weights: Optional[pd.DataFrame] = None,
    sample_sites: Optional[float] = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Format biological and environmental data into a site-pair table.

    This function creates a site-pair dataframe compatible with the R GDM package's
    formatsitepair() function. The output format consists of site pairs with their
    biological dissimilarities, weights, coordinates, and paired environmental data.

    Parameters
    ----------
    bio_data : pd.DataFrame
        Biological data. Format depends on bio_format:
        - FORMAT1: Site rows, species columns (abundance data)
        - FORMAT2: Long format with site, species, coordinate, abundance columns
        - FORMAT3: Dissimilarity matrix with site coordinates
        - FORMAT4: Already in site-pair format (returns as-is)
    bio_format : int, str, or BioFormat, default=2
        Code specifying the format of bio_data:
        - 1: Site x Species abundance matrix
        - 2: Columns for site, species, coordinates, abundance
        - 3: Dissimilarity matrix format
        - 4: Site-pair table format (no transformation)
    dist : str, default="braycurtis"
        Dissimilarity/distance metric for biological distance calculation.
        Supported: "braycurtis", "euclidean", "manhattan", "cosine",
        "jaccard", "kulczynski", "chisq", etc. (scipy.spatial.distance metrics)
    is_abundance : bool, default=True
        True for abundance data, False for presence-absence data.
    site_column : str, default="site"
        Name of column containing unique site identifiers.
    x_column : str, default="xCoord"
        Name of column containing x-coordinates of sites.
    y_column : str, default="yCoord"
        Name of column containing y-coordinates of sites.
    species_column : str, default="species"
        Name of column containing unique species names.
    abund_column : str, default="abundance"
        Name of column containing abundance values.
    spp_filter : int, default=0
        Minimum species count threshold. Species occurring in fewer than this
        number of sites are removed.
    pred_data : pd.DataFrame, optional
        Environmental predictor data with site_id as index and predictor columns.
    dist_preds : dict, optional
        Dictionary of distance matrices as additional predictors. Keys are
        predictor names, values are site x site distance matrices.
    weight_type : {"equal", "richness", "custom"}, default="equal"
        How to weight site pairs:
        - "equal": All pairs have equal weight (1.0)
        - "richness": Weight proportional to site richness
        - "custom": Use custom weights from custom_weights
    custom_weights : pd.DataFrame, optional
        Two-column dataframe with site names and weights for weight_type="custom".
    sample_sites : float, optional
        Fraction of sites to use (0-1). If None, all sites used.
    verbose : bool, default=False
        Print summary information.

    Returns
    -------
    pd.DataFrame
        Site-pair dataframe in GDM-compatible format:
        - Columns 1-2:生物距离, 权重
        - Columns 3-4: s1.xCoord, s1.yCoord (site 1 coordinates)
        - Columns 5-6: s2.xCoord, s2.yCoord (site 2 coordinates)
        - Additional columns: s1.{predictor}, s2.{predictor} for each predictor

    Examples
    --------
    >>> import pandas as pd
    >>> from gdmbayes.utils import format_site_pair, BioFormat
    >>>
    >>> # FORMAT2: Long format with site, species, coordinates
    >>> bio_data = pd.DataFrame({
    ...     "site": ["A", "A", "B", "B"],
    ...     "species": ["sp1", "sp2", "sp1", "sp2"],
    ...     "xCoord": [0.0, 0.0, 1.0, 1.0],
    ...     "yCoord": [0.0, 0.0, 1.0, 1.0],
    ...     "abundance": [10, 5, 8, 3]
    ... })
    >>>
    >>> pred_data = pd.DataFrame({
    ...     "temp": {"A": 10.0, "B": 15.0},
    ...     "precip": {"A": 500.0, "B": 400.0}
    ... })
    >>>
    >>> site_pair = format_site_pair(
    ...     bio_data,
    ...     bio_format=BioFormat.FORMAT2,
    ...     pred_data=pred_data
    ... )

    Notes
    -----
    This function mirrors the behavior of the formatsitepair function in the
    R GDM package. The output format is designed to be directly compatible
    with both R GDM and Python spGDMM implementations.

    References
    ----------
    GDM package: https://cran.r-project.org/package=gdm
    """

    # Convert bio_format to enum if string or int
    if isinstance(bio_format, int):
        try:
            bio_format = BioFormat(str(bio_format))
        except (ValueError, TypeError):
            raise ValueError(
                f"bio_format must be 1, 2, 3, or 4, got {bio_format}"
            )
    elif isinstance(bio_format, str):
        try:
            bio_format = BioFormat(bio_format)
        except (ValueError, TypeError):
            raise ValueError(
                f"bio_format must be 1, 2, 3, or 4, got {bio_format}"
            )
    elif isinstance(bio_format, BioFormat):
        # Already a BioFormat enum
        pass
    else:
        raise ValueError(
            f"bio_format must be BioFormat enum, int, or str, got {type(bio_format)}"
        )

    # FORMAT4: Already in site-pair format
    if bio_format == BioFormat.FORMAT4:
        if verbose:
            print("Data already in site-pair format (FORMAT4), returning as-is.")
        return bio_data.copy()

    # Get site-level coordinates from bio_data
    site_coords, sites = _extract_site_coordinates(
        bio_data, bio_format, site_column, x_column, y_column, species_column
    )

    # Apply site sampling if specified
    if sample_sites is not None and 0 < sample_sites < 1:
        n_sample = max(2, int(len(sites) * sample_sites))
        sample_indices = np.random.choice(len(sites), n_sample, replace=False)
        sites = [sites[i] for i in sample_indices]
        site_coords = site_coords[sample_indices]
        bio_data = _filter_bio_data(bio_data, bio_format, sites, site_column, species_column)
        pred_data = _filter_pred_data(pred_data, sites)

    # Get site-level biological data
    biological_sites = _get_biological_sites(
        bio_data, bio_format, sites, site_column, species_column, abund_column
    )

    # Filter species by occurrence
    if spp_filter > 0:
        species_counts = (biological_sites > 0).sum()
        species_to_keep = species_counts[species_counts >= spp_filter].index
        biological_sites = biological_sites[species_to_keep]
        if verbose:
            print(f"Filtered to {len(species_to_keep)} species with >= {spp_filter} occurrences")

    # Convert to presence-absence if needed
    if not is_abundance:
        biological_sites = (biological_sites > 0).astype(int)

    # Calculate pairwise biological distances
    bio_distances = pdist(biological_sites.values, metric=dist)
    n_sites = len(sites)
    n_pairs = n_sites * (n_sites - 1) // 2

    if verbose:
        print(f"Created {n_pairs} site pairs from {n_sites} sites")
        print(f"Dissimilarity metric: {dist}")

    # Get all unique pairs of site indices
    pairs_list = [(i, j) for i in range(n_sites) for j in range(i + 1, n_sites)]

    # Build site-pair dataframe
    site_pair_df = pd.DataFrame()

    # Column 1: Biological distance
    site_pair_df["bio_distance"] = bio_distances

    # Column 2: Weights
    weights = _calculate_weights(
        biological_sites,
        weight_type,
        custom_weights,
        sites,
        n_pairs,
        pairs_list,
    )
    site_pair_df["weight"] = weights

    # Columns 3-4: Site 1 coordinates
    site_pair_df["s1.xCoord"] = [site_coords[i, 0] for i, _ in pairs_list]
    site_pair_df["s1.yCoord"] = [site_coords[i, 1] for i, _ in pairs_list]

    # Columns 5-6: Site 2 coordinates
    site_pair_df["s2.xCoord"] = [site_coords[j, 0] for _, j in pairs_list]
    site_pair_df["s2.yCoord"] = [site_coords[j, 1] for _, j in pairs_list]

    # Add environmental predictors
    if pred_data is not None:
        site_X = pred_data.loc[sites]
        for pred in site_X.columns:
            site_pair_df[f"s1.{pred}"] = [site_X.iloc[i][pred] for i, _ in pairs_list]
            site_pair_df[f"s2.{pred}"] = [site_X.iloc[j][pred] for _, j in pairs_list]

    # Add distance predictors
    if dist_preds is not None:
        for name, dist_matrix in dist_preds.items():
            site_pair_df[f"{name}"] = [dist_matrix[i, j] for i, j in pairs_list]

    return site_pair_df


def _extract_site_coordinates(
    bio_data: pd.DataFrame,
    bio_format: BioFormat,
    site_column: str,
    x_column: str,
    y_column: str,
    species_column: str,
) -> tuple[np.ndarray, list]:
    """Extract site coordinates from biological data."""
    if bio_format == BioFormat.FORMAT1:
        # FORMAT1: Site index with coordinates stored elsewhere
        # Try to get coordinates from index coordinates if available
        if hasattr(bio_data.index, 'levels'):
            # MultiIndex
            x_coords = bio_data.index.get_level_values(0).values if len(bio_data.index.names) > 0 else np.arange(len(bio_data))
            y_coords = bio_data.index.get_level_values(1).values if len(bio_data.index.names) > 1 else np.zeros(len(bio_data))
        else:
            # Simple index, assume sequential
            n = len(bio_data)
            x_coords = np.arange(n)
            y_coords = np.zeros(n)
        coords = np.column_stack([x_coords, y_coords])
        sites = [f"site_{i}" for i in range(len(bio_data))]

    elif bio_format == BioFormat.FORMAT2:
        # FORMAT2: Long format with coordinate columns
        # Get unique sites and their coordinates
        site_info = bio_data[[site_column, x_column, y_column]].drop_duplicates()
        site_info = site_info.sort_values(site_column)
        sites = site_info[site_column].tolist()
        coords = site_info[[x_column, y_column]].values

    elif bio_format == BioFormat.FORMAT3:
        # FORMAT3: Dissimilarity matrix format
        # Coordinates should be in separate data or indices
        if isinstance(bio_data.index, pd.MultiIndex):
            sites = bio_data.index.to_frame()[site_column].unique().tolist()
        else:
            sites = bio_data.index.tolist()

        # Extract coordinates if available in data attributes or separate column
        if hasattr(bio_data, 'coords'):
            coords = bio_data.coords[[x_column, y_column]].values
        else:
            # Default: use sequential coordinates
            coords = np.column_stack([
                np.arange(len(sites)),
                np.zeros(len(sites))
            ])

    else:
        raise ValueError(f"Unsupported bio_format: {bio_format}")

    return coords, sites


def _filter_bio_data(
    bio_data: pd.DataFrame,
    bio_format: BioFormat,
    sites: list,
    site_column: str,
    species_column: str,
) -> pd.DataFrame:
    """Filter biological data to keep only specified sites."""
    if bio_format == BioFormat.FORMAT1:
        return bio_data.loc[sites]
    elif bio_format == BioFormat.FORMAT2:
        return bio_data[bio_data[site_column].isin(sites)]
    elif bio_format == BioFormat.FORMAT3:
        return bio_data.loc[sites, sites]
    return bio_data


def _filter_pred_data(
    pred_data: Optional[pd.DataFrame],
    sites: list,
) -> Optional[pd.DataFrame]:
    """Filter predictor data to keep only specified sites."""
    if pred_data is None:
        return None
    return pred_data.loc[sites]


def _get_biological_sites(
    bio_data: pd.DataFrame,
    bio_format: BioFormat,
    sites: list,
    site_column: str,
    species_column: str,
    abund_column: str,
) -> pd.DataFrame:
    """Create site x species matrix from biological data."""
    if bio_format == BioFormat.FORMAT1:
        # Already in site x species format
        return bio_data.loc[sites]

    elif bio_format == BioFormat.FORMAT2:
        # Convert long format to site x species
        filtered = bio_data[bio_data[site_column].isin(sites)]
        pivot = filtered.pivot_table(
            index=site_column,
            columns=species_column,
            values=abund_column,
            fill_value=0,
            aggfunc='sum'
        )
        # Ensure all sites are present
        for site in sites:
            if site not in pivot.index:
                pivot.loc[site] = 0
        return pivot.loc[sites].sort_index()

    elif bio_format == BioFormat.FORMAT3:
        # Dissimilarity matrix - convert back to species data if possible
        # This is approximate using metric MDS
        from sklearn.manifold import MDS

        n_sites = len(sites)
        if n_sites < 4:
            raise ValueError("FORMAT3 requires at least 4 sites for reconstruction")

        mds = MDS(n_components=min(n_sites - 1, 10), dissimilarity='precomputed', random_state=42)
        coords = mds.fit_transform(bio_data.loc[sites, sites].values)

        # Use MDS coordinates as pseudo-species data
        return pd.DataFrame(coords, index=sites)

    raise ValueError(f"Unsupported bio_format: {bio_format}")


def _calculate_weights(
    biological_sites: pd.DataFrame,
    weight_type: str,
    custom_weights: Optional[pd.DataFrame],
    sites: list,
    n_pairs: int,
    pairs_list: list,
) -> np.ndarray:
    """Calculate weights for each site pair."""
    if weight_type == "equal":
        return np.ones(n_pairs, dtype=float)

    elif weight_type == "richness":
        # Richness = species count per site
        richness = (biological_sites > 0).sum(axis=1).values
        weights = []
        for i, j in pairs_list:
            w = np.mean([richness[i], richness[j]])
            weights.append(w)
        return np.array(weights)

    elif weight_type == "custom":
        if custom_weights is None:
            raise ValueError("custom_weights must be provided for weight_type='custom'")

        # Create site weight lookup
        site_weights = dict(zip(custom_weights.iloc[:, 0], custom_weights.iloc[:, 1]))

        weights = []
        for i, j in pairs_list:
            w1 = site_weights.get(sites[i], 1.0)
            w2 = site_weights.get(sites[j], 1.0)
            weights.append(np.mean([w1, w2]))
        return np.array(weights)

    raise ValueError(f"Unsupported weight_type: {weight_type}")


__all__ = ["format_site_pair", "BioFormat"]