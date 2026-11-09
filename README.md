
# complexity72_2026

Analysis code and derived data for our study of annual atmospheric CO₂ growth, 
its surface drivers, and how those relationships cluster spatially.

This repository reproduces the figures from the paper:

> **Investigation of regional variations in CO₂ growth rates: Integrating Emission 
> Inventories and Atmospheric Observations**  
> Yogesh Bali, Darja Cvetković, Juan Gancio, Adrián Gutiérrez-Arroyo, 
> Sofia Vazquez Alferez, Xuan Tung Vu, Jin Yan, Pietro Zgaga, 
> Fakhteh Ghanbarnejad, Nasrin Mostafavi Pak  
> arXiv:2606.28462 \[physics.ao-ph\] — [PDF](https://arxiv.org/pdf/2606.28462)

---

## Contents

- [Requirements](#requirements)
- [Repository layout](#repository-layout)
- [Pipeline A — clustering & stability](#pipeline-a--clustering--stability)
- [Pipeline B — grid-cell regression](#pipeline-b--grid-cell-regression)
- [Data](#data)
- [Quick start](#quick-start)

---

## Requirements

Python 3.10+.

```bash
pip install numpy pandas xarray netcdf4 matplotlib cartopy shapely
cartopy and shapely are only needed for the map figures. xarray +
netcdf4 are needed for loading the input dataset.

Repository layout
text
complexity72_2026/
├── Code/
│   ├── clustering/                          Pipeline A
│   │   ├── k_means.py
│   │   ├── cre_matrix.py
│   │   └── run_stability_pipeline.py
│   ├── linear_regression/                   Pipeline B
│   │   └── co2_gridcell_regression_soi_gosif.py
│   ├── cluster_plotting_maps_fig5.ipynb     Figures 5 and city overlays
│   └── build_interactive_dashboard.py       Interactive map
├── Input/                                   Source datasets
├── Output/                                  Derived CSVs, NetCDFs, binaries
│   └── clustering/
├── Figures_and_maps/                        All figures (PDF and PNG)
│   └── clustering/
├── old_backup_AUG25/                        Archived notebooks and CSVs
└── index.html                               Interactive dashboard landing page
All scripts locate the repository root automatically by walking up from their
own directory until they find Input/ and Output/, so they run unchanged
from any clone location.

Pipeline A — clustering & stability
Per-year KMeans clustering of land grid cells on emission and gosif,
followed by a connected-component analysis of which pixel groups stay
clustered together across years.

Run in order from Code/clustering/:

bash
python k_means.py                 # cluster each year, write per-pixel labels
python cre_matrix.py              # build same-cluster co-occurrence matrix
python run_stability_pipeline.py  # detect persistent components, plot them
Key outputs

Stage	Outputs
k_means.py	kmeans_classified_points_all_years.csv (input to stages 2–3), per-year cluster maps and mean-profile charts, cached k-selection scores
cre_matrix.py	pixel_cluster_stability_upper_uint8.bin — upper-triangle same-cluster counts, plus pixel_index.csv and metadata JSON
run_stability_pipeline.py	co-occurrence-clusters.csv (long format, one-hot component membership), component-size histogram and persistent-component maps
The number of clusters k per year is selected automatically (elbow by
default, silhouette optional), typically landing on k = 4 or k = 5.
All settings live at the top of each script.

Full details: Code/clustering/README.md.

Pipeline B — grid-cell regression
Per-land-grid-cell regression of the annual CO₂ growth rate on two drivers,
with a MODE switch between the two specifications in the paper:

MODE = 0 (SOI model, DSS): growth ~ β₀ + β_E·E_z + β_SOI·SOI_z

MODE = 1 (GOSIF model, DSb): growth ~ β₀ + β_ΔE·ΔE_z + β_ΔSIF·ΔSIF_z

The GOSIF specification uses year-to-year differences of the drivers;
the SOI specification uses the absolute emission field with a 7-month-lagged
global SOI index.

bash
cd Code/linear_regression
python co2_gridcell_regression_soi_gosif.py    # edit MODE at the top to switch
Each mode writes its own output prefix (soi_ or delta_sif_), so the two
never overwrite each other. Run the script twice — once per mode — to
regenerate all regression figures.

Reads

Input/unified_annual_carbon_dataset_2015_2024.nc

Input/SOI.txt (MODE 0 only)

Writes

Figures_and_maps/ — observed / predicted / residual maps, variants,
beta and R² maps, per-location contribution figures (PDF).

Output/ — soi_beta_maps.nc / delta_sif_beta_maps.nc containing
beta0, the two driver betas, and R² on the native grid
(ocean cells as NaN).

Full details: Code/linear_regression/README.md.

Data
File	Description
Input/unified_annual_carbon_dataset_2015_2025.nc	Main grid: annual growth_rate, growth_err, emission, gosif, delta_emission, delta_gosif, 2015–2025
Input/unified_annual_carbon_dataset_2015_2024.nc	Same fields, 2015–2024 — used by the regression for a common baseline
Input/SOI.txt	NOAA/CPC monthly Southern Oscillation Index (standardized block)
Input/tccon_all_stations_annual_growth_rates.nc	TCCON station observations (validation)
Datasets are read directly by both pipelines; they are not regenerated here.

Quick start
bash
git clone <repo-url>
cd complexity72_2026
pip install numpy pandas xarray netcdf4 matplotlib cartopy shapely

# Pipeline A: clustering and stability
cd Code/clustering
python k_means.py
python cre_matrix.py
python run_stability_pipeline.py

# Pipeline B: grid-cell regression (run once per mode)
cd ../linear_regression
python co2_gridcell_regression_soi_gosif.py    # MODE = 0
# edit MODE = 1 at the top of the file, then rerun
python co2_gridcell_regression_soi_gosif.py
All figures appear in Figures_and_maps/; all derived data in Output/.

Citation
If you use this code or data, please cite:

bibtex
@article{bali2026investigation,
  title   = {Investigation of regional variations in CO$_2$ growth rates: 
             Integrating Emission Inventories and Atmospheric Observations},
  author  = {Bali, Yogesh and Cvetkovi{\'c}, Darja and Gancio, Juan and 
             Guti{\'e}rrez-Arroyo, Adri{\'a}n and Vazquez Alferez, Sofia and 
             Vu, Xuan Tung and Yan, Jin and Zgaga, Pietro and 
             Ghanbarnejad, Fakhteh and Mostafavi Pak, Nasrin},
  journal = {arXiv preprint arXiv:2606.28462},
  year    = {2026},
  url     = {https://arxiv.org/abs/2606.28462}
}
License
Code is released for research use. See Input/README.md for dataset
provenance and licensing.
