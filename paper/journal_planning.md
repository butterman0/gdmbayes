# Journal Planning — gdmbayes Manuscript

## Paper Type

Software / application paper describing the **gdmbayes** Python package.
Contribution: first Python GDM implementation; full Bayesian backend (PyMC/NUTS);
replicates and extends White et al. (2024) spGDMM (R/NIMBLE).

Reference: White et al. (2024) *Methods in Ecology and Evolution* — spGDMM in R/NIMBLE.

---

## Target Journals (ranked)

| # | Journal | IF (2024) | CiteScore | Paper type | APC (USD) | Notes |
|---|---------|-----------|-----------|------------|-----------|-------|
| 1 | **Methods in Ecology & Evolution** | 6.64 | 13.6 | Application Papers | ~$2,600 (app discount) | **Primary target.** White et al. 2024 published here; dedicated Application Papers track; highest IF among ecology methods journals. |
| 2 | **Ecography** | 5.84 | 11.2 | Software Notes | ~$2,900 | Strong spatiotemporal ecology scope; must be open-source. |
| 3 | **Molecular Ecology Resources** | 5.78 | 11.7 | Core scope | ~$3,900 | Explicitly lists software packages as core content; fully OA from 2026. |
| 4 | **Ecological Informatics** | ~6.0 | 11.4 | Core scope | ~$3,200–3,500 | Computational ecology focus; software implementations in scope. |
| 5 | **Journal of Statistical Software** | 8.11 | 9.6 | Exclusively | $0 (diamond OA) | Highest IF; zero APC; more statistics-focused; slower review. |
| 6 | **JOSS** | N/A | N/A | Exclusively | $0 | Fast, lightweight; no WoS/Scopus indexing; good for open-source visibility. |
| 7 | **Ecology and Evolution** | 2.29 | 4.4 | Accepted | ~$2,700 | Lower IF; fallback. |
| 8 | **Frontiers in Ecology & Evolution** | 2.52 | 4.2 | Methods type | ~$2,800 | Lower IF; fallback. |

**Recommendation:** Submit to **MEE** as Application Paper first. If rejected, try **Ecography** Software Notes.
JOSS is a viable parallel submission (different scope — no overlap concern with MEE).

---

## Submission Checklist

- [ ] Panama CV results match White et al. (2024) Table 1 (CRPS M1 ≈ 0.0527, M8 ≈ 0.0450)
- [ ] GCFR dataset results
- [ ] Response curves / biological space figures (`plot_isplines`)
- [ ] README / package documentation complete
- [ ] PyPI package published
- [ ] Zenodo DOI for code release
- [ ] Cover letter drafted

---

## Key Selling Points

1. Only Python GDM implementation
2. Full Bayesian inference (posterior predictive intervals, CRPS-based CV)
3. sklearn-compatible API (`fit` / `predict` / `Pipeline`-ready preprocessor)
4. Modular variance and spatial-effect specifications (9 model configurations)
5. Replicates White et al. (2024) results with NUTS (faster mixing than AF_slice)
