**Note:** This README was generated with the assistance of AI. Please verify details before relying on it.

# Cluster Stability Pipeline

Three-stage pipeline that clusters annual carbon data with KMeans and analyses how stable those clusters are across years. Starting from a single NetCDF file, it produces per-year cluster maps, a pixel-level same-cluster co-occurrence matrix, and maps of persistent connected components — plus a long-format CSV ready for downstream modelling.

```
k_means.py  →  cre_matrix.py  →  run_stability_pipeline.py
```

---

## Contents

- [Requirements](#requirements)
- [Repository layout](#repository-layout)
- [Pipeline overview](#pipeline-overview)
  - [1. `k_means.py` — annual KMeans clustering](#1-k_meanspy--annual-kmeans-clustering)
  - [2. `cre_matrix.py` — same-cluster co-occurrence matrix](#2-cre_matrixpy--same-cluster-co-occurrence-matrix)
  - [3. `run_stability_pipeline.py` — components, maps, long CSV](#3-run_stability_pipelinepy--components-maps-long-csv)
- [Quick start](#quick-start)
- [Configuration reference](#configuration-reference)
- [Output files](#output-files)
- [How the stability analysis works](#how-the-stability-analysis-works)

---

## Requirements

Python 3.10+ (the scripts use `from __future__ import annotations` and modern typing).

```bash
pip install numpy pandas xarray netcdf4 matplotlib cartopy shapely
```

| Package | Needed by |
|---|---|
| `numpy`, `pandas` | all stages |
| `xarray`, `netcdf4` | `k_means.py`, long CSV in stage 3 |
| `matplotlib`, `cartopy`, `shapely` | maps and figures |

---

## Repository layout

All three scripts locate the repository root automatically by walking up from their own directory until they find a folder containing both `Input/` and `Output/` (or a `.git`). Output directories are created on demand.

```
complexity72_2026/
├── Input/
│   └── unified_annual_carbon_dataset_2015_2025.nc
├── Output/
│   └── clustering/                ← CSVs, JSON, BIN files
├── Figures_and_maps/
│   └── clustering/                ← PNGs
└── cre_matrix.py
    k_means.py
    run_stability_pipeline.py
```

---

## Pipeline overview

### 1. `k_means.py` — annual KMeans clustering

Clusters every valid land grid cell each year on the features `emission` and `gosif`, selects the number of clusters `k` automatically, and writes per-year maps and the per-pixel cluster labels the rest of the pipeline consumes.

**What it does**

- Loads `Input/unified_annual_carbon_dataset_2015_2025.nc`.
- For each year in `KMEANS_YEARS` (2015–2025):
  - Reads `emission` and `gosif`. Non-finite values are treated as `0`; sea points are dropped unless `INCLUDE_SEA_POINTS = True`.
  - Applies `log1p` to features in `LOG_TRANSFORM_FEATURES` (default: `emission`), then z-scores all features.
  - Selects `k ∈ [KMEANS_K_MIN, KMEANS_K_MAX]` via the **elbow** method (default) or **silhouette**.
  - Runs k-means++ with a fixed `RANDOM_SEED` (72), up to `MAX_ITERATIONS` (100), tolerance `1e-4`.
- Caches k-selection scores per year so re-runs are fast; set `REBUILD_K_SELECTION_CACHE = True` to force recompute.
- Writes a per-year cluster map with barycenters, a cluster mean-profile chart, and a barycenter CSV.

**Key settings**

| Setting | Default | Purpose |
|---|---|---|
| `KMEANS_YEARS` | 2015–2025 | Years to cluster |
| `KMEANS_K_MIN` / `KMEANS_K_MAX` | 2 / 12 | Search range for `k` |
| `KMEANS_K_SELECTION_METHOD` | `"elbow"` | `"elbow"` or `"silhouette"` |
| `KMEANS_FEATURES` | `("emission", "gosif")` | Clustering features |
| `LOG_TRANSFORM_FEATURES` | `{"emission"}` | Features to `log1p` |
| `INCLUDE_SEA_POINTS` | `False` | Include ocean cells |
| `KMEANS_N_JOBS` | `0` (auto) | Worker processes for k-search |
| `SILHOUETTE_SAMPLE_SIZE` | `None` (all) | Subsample for silhouette |

### 2. `cre_matrix.py` — same-cluster co-occurrence matrix

For every pair of pixels, counts **how many years they were in the same cluster**. This is the numerically heavy step, so it streams to disk in row blocks rather than materialising an `N × N` array in memory.

**What it does**

- Reads `Output/clustering/kmeans_classified_points_all_years.csv`.
- Writes `pixel_index.csv` mapping `pixel_id → (lon, lat)` (1-based IDs, matching binary offsets).
- Streams the **upper triangle** of the count matrix, row by row, in `ROW_BLOCK_SIZE` (1024) blocks:
  - `pixel_cluster_stability_upper_uint8.bin` — `uint8` same-cluster counts per pixel pair.
  - `pixel_cluster_stability_upper_denominator_uint8.bin` — only if some cluster values are missing; records how many years were comparable per pair.
- Writes `pixel_cluster_stability_metadata.json` describing format, shape, dtype, storage order (`for row i, columns j=i..N-1`) and how to convert counts to probabilities (`same_count / year_count`, or `same_count / denominator_count`).

> **Notes.** `uint8` caps at 255 comparable years — ample for 2015–2025; widen the dtype if you ever exceed it. Peak memory per block is `ROW_BLOCK_SIZE × N × sizeof(uint8)`.

### 3. `run_stability_pipeline.py` — components, maps, long CSV

Detects **persistent connected components** — groups of pixels that stay in the same cluster for at least a threshold fraction of years — and plots them.

**What it does**

- Loads cluster columns plus `lon`/`lat` from `kmeans_classified_points_all_years.csv`.
- Rebuilds the `(N, N)` same-cluster matrix in memory (row-blocked) and unions pixel pairs whose same-cluster count meets `min_count = ceil(threshold × year_count)` for each value in `THRESHOLDS`.
- Emits, per threshold:
  - **Size-histogram CSV** (raw data) and **PNG** (log-log size distribution).
  - **Component map PNG** showing the `TOP_COMPONENTS_TO_PLOT` (default 9) largest persistent components; all other clustered pixels are light grey.
  - **Long-format CSV** `co-occurrence-clusters.csv` when exactly one threshold is set. One row per (pixel, year) in original NetCDF order: `lat, lon, time, emission, gosif, component_1…component_9` (one-hot membership; ocean/unclustered pixels are all-zero).

**Key settings**

| Setting | Default | Purpose |
|---|---|---|
| `THRESHOLDS` | `(1.0,)` | Probability thresholds; `1.0` = same cluster every year |
| `YEARS` | 2015–2024 | Years included in the long CSV |
| `TOP_COMPONENTS_TO_PLOT` | 9 | Number of top components to colour / export |
| `SKIP_COMPONENTS` | `False` | Skip the whole analysis |
| `SAVE_IMAGE_FILES` | `True` | Write PNGs (requires `matplotlib` / `cartopy`) |

---

## Quick start

```bash
# 1. Cluster each year, select k automatically
python k_means.py

# 2. Stream the same-cluster co-occurrence matrix to disk
python cre_matrix.py

# 3. Detect persistent components, plot them, export the long CSV
python run_stability_pipeline.py
```

All numeric knobs live at the top of each file — no CLI flags are required. Each stage prints a progress bar.

---

## Configuration reference

Every setting is a module-level constant at the top of its script. The most frequently changed ones:

**`k_means.py`**

| Setting | Default | Purpose |
|---|---|---|
| `KMEANS_YEARS` | 2015–2025 | Years to cluster |
| `KMEANS_K_MIN` / `KMEANS_K_MAX` | 2 / 12 | Search range for `k` |
| `KMEANS_K_SELECTION_METHOD` | `"elbow"` | `"elbow"` or `"silhouette"` |
| `KMEANS_FEATURES` | `("emission", "gosif")` | Clustering features |
| `LOG_TRANSFORM_FEATURES` | `{"emission"}` | Features to `log1p` |
| `INCLUDE_SEA_POINTS` | `False` | Include ocean cells |
| `KMEANS_N_JOBS` | `0` (auto) | Worker processes for k-search |
| `SILHOUETTE_SAMPLE_SIZE` | `None` (all) | Subsample for silhouette |
| `REBUILD_K_SELECTION_CACHE` | `False` | Force recompute of k-selection scores |

**`cre_matrix.py`**

| Setting | Default | Purpose |
|---|---|---|
| `ROW_BLOCK_SIZE` | 1024 | Rows processed per streamed block |
| `CLUSTER_PREFIX` | `"kmeans_cluster_"` | Cluster column prefix to detect |

**`run_stability_pipeline.py`**

| Setting | Default | Purpose |
|---|---|---|
| `THRESHOLDS` | `(1.0,)` | Probability thresholds |
| `YEARS` | 2015–2024 | Years in the long CSV |
| `TOP_COMPONENTS_TO_PLOT` | 9 | Top components to colour / export |
| `SKIP_COMPONENTS` | `False` | Skip the analysis |
| `SAVE_IMAGE_FILES` | `True` | Write PNGs |

---

## Output files

Everything data-related lands in `Output/clustering/`; every figure lands in `Figures_and_maps/clustering/`.

| File | Written by | Description |
|---|---|---|
| `kmeans_classified_points_all_years.csv` | `k_means.py` | Per-pixel cluster label for each year (`lon, lat, kmeans_cluster_YYYY…`) |
| `kmeans_cluster_barycenters_k<k>_<year>.csv` | `k_means.py` | Cluster barycenters and point counts |
| `kmeans_elbow_scores_<year>.csv` / `kmeans_silhouette_scores_<year>.csv` | `k_means.py` | Cached k-selection scores |
| `pixel_cluster_stability_upper_uint8.bin` | `cre_matrix.py` | Upper-triangle same-cluster counts |
| `pixel_cluster_stability_upper_denominator_uint8.bin` | `cre_matrix.py` | Comparable-years denominator (only if missing values exist) |
| `pixel_cluster_stability_metadata.json` | `cre_matrix.py` | Format, shape, dtype, storage order, probability formula |
| `pixel_index.csv` | `cre_matrix.py` | `pixel_id → (lon, lat)` mapping (1-based) |
| `component_size_histogram_raw_t_<t>.csv` | `run_stability_pipeline.py` | Raw size distribution per threshold |
| `co-occurrence-clusters.csv` | `run_stability_pipeline.py` | Long per-(pixel, year) component membership |
| `kmeans_k<k>_<year>.png` | `k_means.py` | Per-year cluster map |
| `kmeans_cluster_means_k<k>_<year>.png` | `k_means.py` | Cluster mean-profile chart |
| `component_size_histogram_t_<t>.png` | `run_stability_pipeline.py` | Log-log component size distribution |
| `component_map_t_<t>.png` | `run_stability_pipeline.py` | Top-N persistent components on a world map |

---

## How the stability analysis works

1. **Per-year clustering.** Each grid cell is assigned a cluster label 1…k for each year independently.
2. **Co-occurrence.** For every pair of pixels, count the years in which they share a cluster. A high count means the two cells move together through the year-to-year cluster landscape.
3. **Thresholding.** A pair is considered "stably co-occurring" if its count meets `ceil(threshold × year_count)`. For `THRESHOLDS = (1.0,)` that means "every single year".
4. **Connected components.** Union-find merges every stably co-occurring pair. The resulting components are groups of pixels that stay spatially coherent across the study period; the largest ones are what the maps and the long CSV highlight.

Because `uint8` counts cap at 255, the pipeline comfortably covers the 2015–2025 window and scales to any study length under 255 years without changing the storage format.
