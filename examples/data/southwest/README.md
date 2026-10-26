# Southwest Australia example data

Derived from the `southwest` dataset distributed with the R package **gdm**
(Fitzpatrick, Mokany, Manion, Nieto-Lugilde & Ferrier; CRAN, GPL-3): species
occurrences and environmental covariates at 94 sites in south-western Australia.

- `southwest_sites.csv` — one row per site: `xc` (longitude), `yc` (latitude),
  `time_idx` (all 0), `phTotal`, `bio5`, `bio19`, taken from the site-level
  environmental columns of `gdm::southwest`.
- `southwest_y.csv` — the condensed vector of pairwise Bray–Curtis dissimilarities
  between the 94 sites, computed from the species-by-site occurrence table with
  `vegan::vegdist`, in `scipy.spatial.distance.pdist` (row-major upper-triangle) order.

These files are redistributed here for the example notebook and the R `gdm` regression
test, with attribution to the `gdm` authors. Please cite the `gdm` package and
Ferrier et al. (2007) if you use them.
