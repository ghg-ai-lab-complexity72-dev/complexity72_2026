# `co2_gridcell_regression_soi_gosif.py`

Per-land-grid-cell linear regression of the annual atmospheric CO<sub>2</sub>
growth rate on two drivers, with a switch between the two predictor sets used 
in the paper. Produces the main regression figures, the coefficient maps, and
the per-location contribution figure, plus the beta coefficient fields as NetCDF files

---

## The `MODE` switch

Set the constant near the top of the file:

```python
MODE = 0   # 0 = SOI (DSs)   1 = GOSIF (DSb)
```

| | `MODE = 0` | `MODE = 1` |
|---|---|---|
| Model | `growth ~ β₀ + β_E·E_z + β_SOI·SOI_z` | `growth ~ β₀ + β_ΔE·ΔE_z + β_ΔSIF·ΔSIF_z` |
| Drivers | **absolute** emission + 7-month-lagged global SOI | **year-to-year differences** of emission and GOSIF |
| Needs `Input/SOI.txt` | yes | no |
| Output prefix | `soi_` | `delta_sif_` |

Following the paper, the GOSIF specification uses the *differenced* drivers,
so the switch changes both the second predictor and whether the drivers are
differenced. Each mode writes its own filenames, so the two never overwrite
each other — run the script twice (once per mode) to regenerate everything.

---

## Inputs and outputs

All paths are resolved **relative to the repository**, never to a particular
machine: the script walks up from its own location until it finds the repo
root, so it runs unchanged for anyone who clones `complexity72_2026`, and keeps
working if the file is moved to a different depth inside `Code/`.

**Reads**

* `Input/unified_annual_carbon_dataset_2015_2024.nc` — `growth_rate`,
  `growth_err`, `emission`, `gosif`, `delta_emission`, `delta_gosif`
* `Input/SOI.txt` — NOAA/CPC monthly SOI, standardized block (`MODE = 0` only)

**Writes**

| File | Mode | Manuscript position |
|---|---|---|
| `Figures_and_maps/soi_main_obs_pred_resid.pdf` | 0 | Fig. 3, Main |
| `Figures_and_maps/soi_variants.pdf` | 0 | Fig. 1, Supplementary |
| `Figures_and_maps/soi_beta_r2_2x2.pdf` | 0 | Fig. 2, Supplementary |
| `Figures_and_maps/soi_city_contribution_4x3.pdf` | 0 | Fig. 4, Main (updated) |
| `Output/soi_beta_maps.nc` | 0 | — (derived data) |
| `Figures_and_maps/delta_sif_main_obs_pred_resid.pdf` | 1 | Fig. 3, Supplementary |
| `Figures_and_maps/delta_sif_variants.pdf` | 1 | Fig. 4, Supplementary |
| `Figures_and_maps/delta_sif_beta_r2_2x2.pdf` | 1 | Fig. 5, Supplementary |
| `Figures_and_maps/delta_sif_city_contribution_4x3.pdf` | 1 | Fig. 6, Supplementary |
| `Output/delta_sif_beta_maps.nc` | 1 | — (derived data) |


### Coefficient NetCDF files

 Grid `lat` 180 × `lon` 360, cean cells `NaN`. Variables:

* `MODE = 0` → `beta0`, `bE`, `bSOI`, `r2`
* `MODE = 1` → `beta0`, `bdE`, `bdSIF`, `r2`

Each variable carries `long_name` and `units`; global attributes record the
model equation, mode, estimator, period and source dataset (`ncdump -h` to
inspect). Coefficients are in ppm yr⁻¹ per 1 standard deviation of the driver.

---

## Key parameters

| Parameter | Default | Meaning |
|---|---|---|
| `MODE` | `0` | SOI vs GOSIF specification |
| `YEARS` | `[2015, 2017, 2019, 2021, 2023]` | Years shown as rows in the map figures |
| `SOI_LAG_MONTHS` | `7` | Lag applied to monthly SOI before annual averaging |
| `MIN_OBS` | `6` | Minimum valid years for a cell to be fitted |
| `LAMBDAS` | `[0, 0.1, 0.3, 1, 3, 10]` | Ridge penalties scanned for the variant fits |
| `FIG_EXT` | `"pdf"` | Figure format |
| `SAVE_NETCDF` | `True` | Also write coefficient maps to `Output/` |
| `CITIES` | 12 entries | Locations in the contribution figure (4 rows × 3 columns) |


---

# `co2_cluster_regression_soi_gosif.py`

Cluster-level regression (Fig. 7, Main). For each persistent cluster C1–C5 and
year, growth rate, emission and GOSIF are averaged over the cluster's grid
cells, and two OLS models are fitted to the 10-year cluster-mean series:

| Model | Predictors |
|---|---|
| SOI | `y ~ β₀ + β_E·E_z + β_SOI·SOI_z` (7-month-lagged SOI) |
| GOSIF | `y ~ β₀ + β_E·E_z + β_SIF·GOSIF_z` (cluster-mean GOSIF) |


The figure uses the same style as Fig. 4: grey band = area-weighted global land
mean ± 1 SD; black = observed cluster mean with propagated error bars; red
dashed = prediction; bars = β_E·E_z and β₂·P_z (β₀ omitted).

**Error propagation.** Each grid cell carries a 1-σ uncertainty `growth_err`
(σᵢ). Because CAMS errors are strongly correlated between neighbouring cells,
the error of the cluster mean is propagated assuming fully correlated cell
errors:

σ(ȳ) = (1/N) Σᵢ σᵢ

Cells without `growth_err` are left out, and N is then the number of cells with
a finite error (this affects mainly C1 in 2015–2016, where about half of the
cells, mostly polar, have no error value).

**Reads** `Input/unified_annual_carbon_dataset_2015_2025.nc`  `Input/SOI.txt`,
`Output/clustering/co-occurrence-clusters.csv`. 
**Writes**

| File | Content |
|---|---|
| `Figures_and_maps/cluster_contribution_5x2.pdf` | Fig. 7, Main |
| `Output/cluster_regression_coefficients.csv` | β₀, β_E, β₂, R², LOO R² per cluster and model |
| `Output/cluster_mean_timeseries.csv` | cluster-year means, N cells, propagated error |



Dependencies: numpy, pandas, xarray, netCDF4, matplotlib, global-land-mask.
