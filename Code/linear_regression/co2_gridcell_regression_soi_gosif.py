#!/usr/bin/env python3
"""
Per-LAND-cell regression of the CO2 growth rate on two drivers, with a switch
between the two predictor sets used in the paper.

    MODE = 0  ->  SOI model      (DSS): growth ~ b0 + bE*E_z    + bSOI*SOI_z
    MODE = 1  ->  GOSIF model    (DSb): growth ~ b0 + bdE*dE_z  + bdSIF*dSIF_z

NOTE on MODE 1: following the paper, the GOSIF specification uses the
YEAR-TO-YEAR DIFFERENCES of the surface drivers (delta_emission, delta_gosif),
not their absolute values. MODE 0 uses the absolute emission field together
with the global lagged SOI index. The switch therefore changes both the second
predictor AND whether the drivers are differenced.

PATHS
-----
All paths are resolved RELATIVE TO THE REPOSITORY, never to the machine, so
this runs unchanged for anyone who clones complexity72_2026. The script walks
up from its own location to find the repo root, so it also keeps working if it
is moved to a different depth inside Code/.

  Reads   Input/unified_annual_carbon_dataset_2015_2024.nc
          Input/SOI.txt                      (only when MODE = 0)
  Writes  Figures_and_maps/*.pdf             (figures, vector)
          Output/*_beta_maps.nc              (regression coefficients)

Run it from anywhere:
    python Code/linear_regression/co2_gridcell_regression_soi_gosif.py

Outputs (all figures vector PDF; beta maps also written as NetCDF):

  MODE = 0 (SOI)
    Figures_and_maps/soi_main_obs_pred_resid.pdf         Fig 3 of Main
    Figures_and_maps/soi_variants.pdf                    Fig 1 of Supplementary
    Figures_and_maps/soi_beta_r2_2x2.pdf                 Fig 2 of Supplementary
    Figures_and_maps/soi_city_contribution_4x3.pdf       Fig 4 of Main, updated
    Output/soi_beta_maps.nc                  beta0, bE, bSOI, R2 as NetCDF

  MODE = 1 (GOSIF)
    Figures_and_maps/delta_sif_main_obs_pred_resid.pdf   Fig 3 of Supplementary
    Figures_and_maps/delta_sif_variants.pdf              Fig 4 of Supplementary
    Figures_and_maps/delta_sif_beta_r2_2x2.pdf           Fig 5 of Supplementary
    Figures_and_maps/delta_sif_city_contribution_4x3.pdf Fig 6 of Supplementary
    Output/delta_sif_beta_maps.nc            beta0, bdE, bdSIF, R2 as NetCDF
"""

import re
from pathlib import Path
import math

import numpy as np
import pandas as pd
import xarray as xr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import cartopy.crs as ccrs
    HAS_CARTOPY = True
except Exception:
    HAS_CARTOPY = False

DRAW_COAST = False
if HAS_CARTOPY:
    try:
        _f = plt.figure()
        _ax = _f.add_subplot(projection=ccrs.PlateCarree())
        _ax.coastlines()
        _f.canvas.draw()
        plt.close(_f)
        DRAW_COAST = True
    except Exception:
        plt.close("all")
        print("note: coastlines offline; maps drawn without them")


# ============================== CONFIG ===================================

# -------------------------------------------------------------------------
#  MODE = 0 : use SOI   (DSS) as second predictor, absolute emission
#  MODE = 1 : use GOSIF (DSb) as second predictor, differenced drivers
# -------------------------------------------------------------------------
MODE = 0
# -------------------------------------------------------------------------

# --- locate the repository root -------------------------------------------
# The script finds the repo root by walking up from its own location until it
# sees the expected folders (or the .git directory). This means it works from
# ANY depth inside Code/ and for anyone who clones the repository, with no
# machine-specific paths to edit.

def find_repo_root(start: Path) -> Path:
    """Walk up from `start` until a folder containing Input/ and Output/
    (or .git) is found. Raises a clear error if the layout is unexpected."""
    for folder in [start, *start.parents]:
        if (folder / "Input").is_dir() and (folder / "Output").is_dir():
            return folder
        if (folder / ".git").exists():
            return folder
    raise FileNotFoundError(
        f"Could not locate the repository root above {start}.\n"
        "Expected to find a folder containing Input/ and Output/. "
        "Run this script from inside a clone of complexity72_2026."
    )


BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = find_repo_root(BASE_DIR)

INPUT_DIR = REPO_ROOT / "Input"
INPUT_FILE = INPUT_DIR / "unified_annual_carbon_dataset_2015_2024.nc"
SOI_FILE   = INPUT_DIR / "SOI.txt"

FIGDIR = REPO_ROOT / "Figures_and_maps"
NCDIR  = REPO_ROOT / "Output"
FIGDIR.mkdir(parents=True, exist_ok=True)
NCDIR.mkdir(parents=True, exist_ok=True)

FIG_EXT = "pdf"          # vector output
SAVE_NETCDF = True       # also write beta maps as NetCDF

# YEARS = [2016, 2018, 2020, 2022, 2024]
YEARS = [2015, 2017, 2019, 2021, 2023]

SOI_LAG_MONTHS = 7
MIN_OBS = 6
LAMBDAS = np.array([0.0, 0.1, 0.3, 1.0, 3.0, 10.0])

SIF_VAR = "gosif"
DELTA_SIF_VAR = "delta_gosif"

# Visual controls
YEAR_LABEL_X = -0.13
MAP_YEAR_FONTSIZE = 14

BETA0_CMAP = "viridis"    # beta0 heatmap
BETA_CMAP = "RdBu_r"      # shared diverging heatmap for the two driver betas
R2_CMAP = "viridis"       # separate heatmap for R2

plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 11,
    "pdf.fonttype": 42,   # editable text in vector output
    "ps.fonttype": 42,
})

# City list: 4 rows x 3 columns.
# Row 4 added: Delhi, Doha, New York.
CITIES = [
    ("Beijing, China",       39.9, 116.4,  "industrialized"),
    ("Ruhr Valley, Germany", 51.4,   7.2,  "industrialized"),
    ("Tokyo, Japan",         35.7, 139.7,  "industrialized"),
    ("Los Angeles, USA",     34.0, -118.2, "industrialized"),
    ("Amazon, Brazil",       -3.5, -62.0,  "forest"),
    ("Borneo, Indonesia",    -1.0, 113.5,  "forest"),
    ("Patagonia, Argentina", -45.0, -70.0, "steppe"),
    ("Siberia, Russia",       62.0, 100.0, "boreal"),
    ("Australian Outback",   -25.0, 133.0, "desert"),
    ("Delhi, India",          28.6,  77.2, "industrialized"),
    ("Doha, Qatar",           25.3,  51.5, "arid / industrialized"),
    ("New York, USA",         40.7, -74.0, "industrialized"),
]

print(f"MODE = {MODE}  ->  " + ("SOI (DSS), absolute emission"
                                if MODE == 0 else
                                "GOSIF (DSb), differenced drivers"))
print("INPUT_FILE:", INPUT_FILE, "| exists?", INPUT_FILE.exists())
if MODE == 0:
    print("SOI_FILE:", SOI_FILE, "| exists?", SOI_FILE.exists())
print("FIGDIR:", FIGDIR, "| NCDIR:", NCDIR)

if not INPUT_FILE.exists():
    raise FileNotFoundError(f"NetCDF file not found: {INPUT_FILE}")
if MODE == 0 and not SOI_FILE.exists():
    raise FileNotFoundError(f"SOI file not found: {SOI_FILE} (needed for MODE=0)")
if MODE not in (0, 1):
    raise ValueError("MODE must be 0 (SOI) or 1 (GOSIF)")


# --------------------- mode-dependent labels / names ---------------------

if MODE == 0:
    PREFIX = "soi"
    P1_KEY, P2_KEY = "bE", "bSOI"                    # NetCDF variable names
    P1_MATH = r"$\beta_E\,E_z$"
    P2_MATH = r"$\beta_{SOI}\,SOI_z$"
    P1_TITLE = r"$\beta_E$ (ppm/yr per 1 SD emission): emission sensitivity"
    P2_TITLE = r"$\beta_{SOI}$ (ppm/yr per 1 SD SOI): ENSO sensitivity"
    P1_SHORT, P2_SHORT = r"\beta_E", r"\beta_{SOI}"
    MAIN_PRED_LABEL = r"Predicted $\Delta CO_2$ (ppm/yr): OLS (DSe + DSs)"
    VAR2_LABEL = r"Predicted $\Delta CO_2$ (ppm/yr): Ridge (DSe + DSs)"
    VAR3_LABEL = r"Predicted $\Delta CO_2$ (ppm/yr): Ridge ($\Delta$DSe + DSs)"
    BAR_COLORS = ["tab:orange", "tab:blue"]
else:
    PREFIX = "delta_sif"
    P1_KEY, P2_KEY = "bdE", "bdSIF"
    P1_MATH = r"$\beta_{\Delta E}\,\Delta E_z$"
    P2_MATH = r"$\beta_{\Delta SIF}\,\Delta SIF_z$"
    P1_TITLE = (r"$\beta_{\Delta E}$ (ppm/yr per 1 SD $\Delta$emission): "
                r"emission-change sensitivity")
    P2_TITLE = (r"$\beta_{\Delta SIF}$ (ppm/yr per 1 SD $\Delta$SIF): "
                r"SIF-change sensitivity")
    P1_SHORT, P2_SHORT = r"\beta_{\Delta E}", r"\beta_{\Delta SIF}"
    MAIN_PRED_LABEL = (r"Predicted $\Delta CO_2$ (ppm/yr): "
                       r"OLS ($\Delta$DSe + $\Delta$DSb)")
    VAR2_LABEL = (r"Predicted $\Delta CO_2$ (ppm/yr): "
                  r"Ridge ($\Delta$DSe + $\Delta$DSb)")
    VAR3_LABEL = r"Predicted $\Delta CO_2$ (ppm/yr): OLS (DSe + DSb)"
    BAR_COLORS = ["tab:orange", "tab:green"]

BETA0_TITLE = r"$\beta_0$ (ppm/yr): local baseline growth"
R2_TITLE = r"$R^2$"


def outpath(stem, ext=None):
    """Figures go to Figures_and_maps/, derived NetCDF to Output/."""
    ext = ext or FIG_EXT
    folder = NCDIR if ext == "nc" else FIGDIR
    return folder / f"{PREFIX}_{stem}.{ext}"


# =========================== HELPERS =====================================

def read_standardized_soi(path):
    text = Path(path).read_text(errors="ignore")
    if "STANDARDIZED" not in text:
        raise ValueError("No STANDARDIZED section in SOI file.")

    std = text.split("STANDARDIZED", 1)[1]
    rows = []

    for line in std.splitlines():
        nums = re.findall(r"-?\d+(?:\.\d+)?", line)
        if len(nums) < 13:
            continue

        year = int(float(nums[0]))
        if year < 1800 or year > 2200:
            continue

        for month, value in enumerate([float(x) for x in nums[1:13]], start=1):
            rows.append({
                "year": year,
                "month": month,
                "soi": np.nan if value <= -900 else value,
            })

    return pd.DataFrame(rows)


def annual_lag_soi(soi_monthly, lag=SOI_LAG_MONTHS, min_months=12):
    df = soi_monthly.dropna(subset=["soi"]).copy()
    df["date"] = pd.to_datetime(dict(year=df.year, month=df.month, day=1))
    df["resp"] = df["date"] + pd.DateOffset(months=lag)
    df["co2_year"] = df["resp"].dt.year

    grouped = df.groupby("co2_year")["soi"]
    annual_mean = grouped.mean()
    annual_count = grouped.count()

    return annual_mean[annual_count >= min_months]


def build_land_mask(lat, lon):
    LON, LAT = np.meshgrid(lon, lat)
    LONq = np.where(LON > 180, LON - 360, LON)

    try:
        from global_land_mask import globe
        return globe.is_land(LAT, LONq)

    except Exception:
        import cartopy.io.shapereader as shp
        from shapely.geometry import Point
        from shapely.ops import unary_union
        from shapely.prepared import prep

        geom = unary_union(list(shp.Reader(shp.natural_earth(
            resolution="110m",
            category="physical",
            name="land",
        )).geometries()))

        prepared_geom = prep(geom)
        flat = np.array([
            prepared_geom.contains(Point(x, y))
            for x, y in zip(LONq.ravel(), LAT.ravel())
        ])

        return flat.reshape(LAT.shape)


def zscore_time(a):
    mu = np.nanmean(a, axis=0)
    sd = np.nanstd(a, axis=0)
    return (a - mu) / np.where(sd > 0, sd, np.nan)


def finite_percentile(a, q, fallback=np.nan):
    vals = a[np.isfinite(a)]
    if vals.size == 0:
        return fallback
    return np.nanpercentile(vals, q)


def sym_limits(a, pct=96):
    vals = a[np.isfinite(a)]
    if vals.size == 0:
        return -1.0, 1.0
    lim = np.nanpercentile(np.abs(vals), pct)
    if not np.isfinite(lim) or lim == 0:
        return -1.0, 1.0
    return -lim, lim


def fmt_coord(latv, lonv):
    """'39.9N, 116.4E' style label for the subplot titles."""
    ns = "N" if latv >= 0 else "S"
    ew = "E" if lonv >= 0 else "W"
    return f"{abs(latv):.1f}\u00b0{ns}, {abs(lonv):.1f}\u00b0{ew}"


# ========================== LOAD DATA ====================================

ds = xr.open_dataset(INPUT_FILE)

for name in ["growth_rate", "emission", SIF_VAR]:
    if name not in ds:
        available = ", ".join(list(ds.data_vars))
        raise KeyError(f"Variable {name!r} not found. Available: {available}")

lat = ds["lat"].values
lon = ds["lon"].values
tvals = ds["time"].values

ds_years = (
    tvals.astype(int)
    if np.issubdtype(tvals.dtype, np.integer)
    else pd.to_datetime(tvals).year.values
)

T, nlat, nlon = ds["growth_rate"].shape

gr = ds["growth_rate"].transpose("time", "lat", "lon").values.astype("float64")

land2d = build_land_mask(lat, lon)
land3d = np.broadcast_to(land2d[None], (T, nlat, nlon))
print(f"land cells: {int(land2d.sum())} ({100 * land2d.mean():.1f}%)")

Ytarget = np.where(land3d, gr, np.nan)

if "growth_err" in ds:
    growth_err = ds["growth_err"].transpose("time", "lat", "lon").values.astype("float64")
    Yerr = np.where(land3d, growth_err, np.nan)
    print("Using growth_err from NetCDF for observed city error bars.")
else:
    Yerr = np.full_like(Ytarget, np.nan)
    print("WARNING: growth_err not found. City error bars omitted.")

# --- absolute driver fields ---
emission_raw = ds["emission"].transpose("time", "lat", "lon").values.astype("float64")
sif_raw = ds[SIF_VAR].transpose("time", "lat", "lon").values.astype("float64")

E_z = np.where(land3d, zscore_time(emission_raw), np.nan)
SIF_z = np.where(land3d, zscore_time(sif_raw), np.nan)

# --- differenced driver fields ---
if "delta_emission" in ds:
    dE_raw = ds["delta_emission"].transpose("time", "lat", "lon").values.astype("float64")
    print("Using 'delta_emission' from NetCDF.")
else:
    dE_raw = np.full_like(emission_raw, np.nan)
    dE_raw[1:] = emission_raw[1:] - emission_raw[:-1]
    print("'delta_emission' not found; computed first differences.")

if DELTA_SIF_VAR in ds:
    dSIF_raw = ds[DELTA_SIF_VAR].transpose("time", "lat", "lon").values.astype("float64")
    print(f"Using {DELTA_SIF_VAR!r} from NetCDF.")
else:
    dSIF_raw = np.full_like(sif_raw, np.nan)
    dSIF_raw[1:] = sif_raw[1:] - sif_raw[:-1]
    print(f"{DELTA_SIF_VAR!r} not found; computed first differences.")

dE_z = np.where(land3d, zscore_time(dE_raw), np.nan)
dSIF_z = np.where(land3d, zscore_time(dSIF_raw), np.nan)

# --- SOI predictor (MODE 0 only) ---
soi_z1d = None
if MODE == 0:
    soi_ann = annual_lag_soi(read_standardized_soi(SOI_FILE))
    soi_raw = np.array([soi_ann.loc[int(y)] for y in ds_years], dtype="float64")
    soi_z1d = (soi_raw - np.nanmean(soi_raw)) / np.nanstd(soi_raw)
    soi_grid = np.broadcast_to(soi_z1d[:, None, None], (T, nlat, nlon)).copy()
    print("SOI lag-7 annual z:",
          dict(zip(ds_years.tolist(), np.round(soi_z1d, 2).tolist())))


# ========================== REGRESSION ===================================

def fit(Y, X1, X2, lambdas, min_obs=MIN_OBS):
    """
    X1 = first driver predictor (emission or delta_emission).
    X2 = second driver predictor (SOI or delta_SIF).

    Ridge penalty is applied to b1 and b2 only, not to beta0.
    If lambdas = [0.0], this is OLS.
    """
    nc = nlat * nlon

    Yf = Y.reshape(T, nc)
    A1 = X1.reshape(T, nc)
    A2 = X2.reshape(T, nc)

    valid = np.isfinite(Yf) & np.isfinite(A1) & np.isfinite(A2)
    nval = valid.sum(axis=0)

    Z1 = np.where(valid, A1, 0.0)
    Z2 = np.where(valid, A2, 0.0)
    ones = np.where(valid, 1.0, 0.0)

    X = np.stack([ones, Z1, Z2], axis=2)
    Yz = np.where(valid, Yf, 0.0)

    XtX = np.einsum("tni,tnj->nij", X, X)
    Xty = np.einsum("tni,tn->ni", X, Yz)

    ridge_penalty = np.diag([0.0, 1.0, 1.0])
    small_eye = 1e-8 * np.eye(3)

    ybar = Yz.sum(axis=0) / np.maximum(nval, 1)
    ss_tot = (np.where(valid, Yz - ybar, 0.0) ** 2).sum(axis=0)

    idx = np.where(nval >= min_obs)[0]

    beta = np.full((nc, 3), np.nan)
    r2 = np.full(nc, np.nan)

    if idx.size:
        best = np.full(idx.size, np.inf)
        best_beta = np.full((idx.size, 3), np.nan)
        best_r2 = np.full(idx.size, np.nan)

        Xi = X[:, idx, :]
        Yi = Yz[:, idx]
        Vi = valid[:, idx]

        for lam in lambdas:
            Minv = np.linalg.inv(XtX[idx] + lam * ridge_penalty + small_eye)
            b = np.einsum("nij,nj->ni", Minv, Xty[idx])

            yhat = np.einsum("tni,ni->tn", Xi, b)

            # Leave-one-out error approximation.
            lev = np.einsum("tni,nij,tnj->tn", Xi, Minv, Xi)
            denom = 1.0 - lev
            good = Vi & (denom > 1e-6)

            loo = np.where(good, (Yi - yhat) / np.where(good, denom, 1.0), 0.0)
            crit = (np.where(good, loo ** 2, 0.0).sum(axis=0)
                    / np.maximum(good.sum(axis=0), 1))

            ss_res = (np.where(Vi, Yi - yhat, 0.0) ** 2).sum(axis=0)
            r2_candidate = 1.0 - ss_res / np.where(ss_tot[idx] > 0, ss_tot[idx], np.nan)

            improve = crit < best
            best = np.where(improve, crit, best)
            for c in range(3):
                best_beta[:, c] = np.where(improve, b[:, c], best_beta[:, c])
            best_r2 = np.where(improve, r2_candidate, best_r2)

        beta[idx] = best_beta
        r2[idx] = best_r2

    A1p = np.where(valid, A1, np.nan)
    A2p = np.where(valid, A2, np.nan)

    pred = (
        beta[:, 0][None, :]
        + beta[:, 1][None, :] * A1p
        + beta[:, 2][None, :] * A2p
    ).reshape(T, nlat, nlon)

    reshape_map = lambda a: a.reshape(nlat, nlon)

    return {
        "beta0": reshape_map(beta[:, 0]),
        "b1": reshape_map(beta[:, 1]),
        "b2": reshape_map(beta[:, 2]),
        "r2": reshape_map(r2),
        "pred": pred,
    }


if MODE == 0:
    # MAIN     = OLS(emission + SOI)
    # VARIANTS = Ridge(emission + SOI) and Ridge(delta_emission + SOI)
    P1_MAIN, P2_MAIN = E_z, soi_grid
    print("fitting OLS:   emission + SOI [main] ...")
    main = fit(Ytarget, E_z, soi_grid, np.array([0.0]))
    print("fitting Ridge: emission + SOI [variant] ...")
    var_a = fit(Ytarget, E_z, soi_grid, LAMBDAS)
    print("fitting Ridge: delta_emission + SOI [variant] ...")
    var_b = fit(Ytarget, dE_z, soi_grid, LAMBDAS)
else:
    # MAIN     = OLS(delta_emission + delta_SIF)
    # VARIANTS = Ridge(delta_emission + delta_SIF) and OLS(emission + SIF)
    P1_MAIN, P2_MAIN = dE_z, dSIF_z
    print("fitting OLS:   delta_emission + delta_SIF [main] ...")
    main = fit(Ytarget, dE_z, dSIF_z, np.array([0.0]))
    print("fitting Ridge: delta_emission + delta_SIF [variant] ...")
    var_a = fit(Ytarget, dE_z, dSIF_z, LAMBDAS)
    print("fitting OLS:   emission + SIF [variant] ...")
    var_b = fit(Ytarget, E_z, SIF_z, np.array([0.0]))


# ======================= EXPORT BETA MAPS AS NetCDF ======================

def save_beta_netcdf(res, path):
    """Write beta0, the two driver betas and R2 as a CF-ish NetCDF file."""
    out = xr.Dataset(
        {
            "beta0": (("lat", "lon"), res["beta0"]),
            P1_KEY:  (("lat", "lon"), res["b1"]),
            P2_KEY:  (("lat", "lon"), res["b2"]),
            "r2":    (("lat", "lon"), res["r2"]),
        },
        coords={"lat": lat, "lon": lon},
    )
    out["beta0"].attrs = {"long_name": "local baseline growth rate",
                          "units": "ppm yr-1"}
    out[P1_KEY].attrs = {
        "long_name": ("emission sensitivity" if MODE == 0
                      else "emission-change sensitivity"),
        "units": ("ppm yr-1 per 1 SD emission" if MODE == 0
                  else "ppm yr-1 per 1 SD delta-emission")}
    out[P2_KEY].attrs = {
        "long_name": ("ENSO (SOI) sensitivity" if MODE == 0
                      else "SIF-change sensitivity"),
        "units": ("ppm yr-1 per 1 SD SOI" if MODE == 0
                  else "ppm yr-1 per 1 SD delta-SIF")}
    out["r2"].attrs = {"long_name": "coefficient of determination (in-sample)",
                       "units": "1"}
    out["lat"].attrs = {"units": "degrees_north"}
    out["lon"].attrs = {"units": "degrees_east"}
    out.attrs = {
        "title": "Per-grid-cell regression coefficients for CO2 growth rate",
        "mode": int(MODE),
        "model": ("growth_rate ~ beta0 + bE*emission_z + bSOI*SOI_z"
                  if MODE == 0 else
                  "growth_rate ~ beta0 + bdE*delta_emission_z "
                  "+ bdSIF*delta_SIF_z"),
        "estimator": "ordinary least squares (per land grid cell)",
        "predictors_standardized": "yes (per cell, over time)",
        "period": f"{int(ds_years.min())}-{int(ds_years.max())}",
        "source_dataset": INPUT_FILE.name,
        "note": "Ocean cells are NaN (excluded from the fit).",
    }
    out.to_netcdf(path)
    return out


if SAVE_NETCDF:
    nc_path = outpath("beta_maps", "nc")
    save_beta_netcdf(main, nc_path)
    print("saved", nc_path.name)


# ========================== MAP HELPERS ==================================

landmask = np.isfinite(main["pred"])
obs_field = np.where(landmask, Ytarget, np.nan)

vmin = finite_percentile(obs_field, 2, 0.0)
vmax = finite_percentile(obs_field, 98, 1.0)

main_resid = obs_field - main["pred"]
rlim = finite_percentile(np.abs(main_resid[np.isfinite(main_resid)]), 96, 1.0)
if not np.isfinite(rlim) or rlim == 0:
    rlim = 1.0

EXTENT = [lon.min(), lon.max(), lat.min(), lat.max()]


def add_map(ax, data, cmap, vmn, vmx):
    cm = plt.get_cmap(cmap).copy()
    cm.set_bad("0.85")

    kwargs = dict(origin="lower", extent=EXTENT, cmap=cm,
                  vmin=vmn, vmax=vmx, aspect="auto")
    if HAS_CARTOPY:
        kwargs["transform"] = ccrs.PlateCarree()

    im = ax.imshow(np.ma.masked_invalid(data), **kwargs)

    if HAS_CARTOPY:
        if DRAW_COAST:
            try:
                ax.coastlines(linewidth=0.5)
            except Exception:
                pass
        ax.set_global()
    return im


def add_year_label(ax, yr):
    ax.text(YEAR_LABEL_X, 0.5, str(yr), transform=ax.transAxes, rotation=90,
            va="center", ha="right", fontsize=MAP_YEAR_FONTSIZE,
            fontweight="bold", clip_on=False)


def yr_index(yr):
    return int(np.where(ds_years == yr)[0][0])


def add_colorbar(fig, im, ax, label=None):
    return fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)


# ============================ FIGURE 1 ===================================
# observed | main OLS prediction | residual

fig = plt.figure(figsize=(15.8, 3.25 * len(YEARS)))

for row, yr in enumerate(YEARS):
    ti = yr_index(yr)
    obs = np.where(landmask[ti], Ytarget[ti], np.nan)
    pred = main["pred"][ti]

    panels = [
        (obs, "viridis", vmin, vmax, r"Observed $\Delta CO_2$ (ppm/yr)"),
        (pred, "viridis", vmin, vmax, MAIN_PRED_LABEL),
        (obs - pred, "PuOr_r", -rlim, rlim, r"Residual $\Delta CO_2$ (ppm/yr)"),
    ]

    for col, (data, cmap, lo, hi, cblabel) in enumerate(panels):
        ax = fig.add_subplot(len(YEARS), 3, row * 3 + col + 1,
                             projection=ccrs.PlateCarree() if HAS_CARTOPY else None)
        im = add_map(ax, data, cmap, lo, hi)
        if row == 0:
            ax.set_title(cblabel, fontsize=12, fontweight="bold")
        if col == 0:
            add_year_label(ax, yr)
        add_colorbar(fig, im, ax, cblabel)

fig.tight_layout(rect=[0.04, 0, 1, 0.99])
fig.savefig(outpath("main_obs_pred_resid"), bbox_inches="tight")
plt.close(fig)
print("saved", outpath("main_obs_pred_resid").name)


# ============================ FIGURE 2 ===================================
# observed | variant A | variant B

fig = plt.figure(figsize=(15.8, 3.25 * len(YEARS)))

for row, yr in enumerate(YEARS):
    ti = yr_index(yr)
    obs = np.where(landmask[ti], Ytarget[ti], np.nan)

    panels = [
        (obs, "viridis", vmin, vmax, r"Observed $\Delta CO_2$ (ppm/yr)"),
        (var_a["pred"][ti], "viridis", vmin, vmax, VAR2_LABEL),
        (var_b["pred"][ti], "viridis", vmin, vmax, VAR3_LABEL),
    ]

    for col, (data, cmap, lo, hi, cblabel) in enumerate(panels):
        ax = fig.add_subplot(len(YEARS), 3, row * 3 + col + 1,
                             projection=ccrs.PlateCarree() if HAS_CARTOPY else None)
        im = add_map(ax, data, cmap, lo, hi)
        if row == 0:
            ax.set_title(cblabel, fontsize=12, fontweight="bold")
        if col == 0:
            add_year_label(ax, yr)
        add_colorbar(fig, im, ax, cblabel)

fig.tight_layout(rect=[0.04, 0, 1, 0.99])
fig.savefig(outpath("variants"), bbox_inches="tight")
plt.close(fig)
print("saved", outpath("variants").name)


# ============================ FIGURE 3 ===================================
# beta maps + R2 for the MAIN OLS model

b0 = main["beta0"]
b1m = main["b1"]
b2m = main["b2"]
r2m = main["r2"]

b0_finite = b0[np.isfinite(b0)]
if b0_finite.size == 0:
    b0_vmin, b0_vmax = 0.0, 1.0
else:
    b0_vmin = np.nanpercentile(b0_finite, 2)
    b0_vmax = np.nanpercentile(b0_finite, 98)

specs = [
    (b0, BETA0_TITLE, BETA0_CMAP, b0_vmin, b0_vmax),
    (b2m, P2_TITLE, BETA_CMAP, *sym_limits(b2m)),
    (b1m, P1_TITLE, BETA_CMAP, *sym_limits(b1m)),
    (r2m, R2_TITLE, R2_CMAP, 0.0, 1.0),
]

fig = plt.figure(figsize=(14.5, 8.5))

for k, (data, title, cmap, lo, hi) in enumerate(specs):
    ax = fig.add_subplot(2, 2, k + 1,
                         projection=ccrs.PlateCarree() if HAS_CARTOPY else None)
    im = add_map(ax, data, cmap, lo, hi)
    ax.set_title(title, fontsize=12, fontweight="bold")
    add_colorbar(fig, im, ax, None)

fig.tight_layout()
fig.savefig(outpath("beta_r2_2x2"), bbox_inches="tight")
plt.close(fig)
print("saved", outpath("beta_r2_2x2").name)


# ============================ FIGURE 4 ===================================
# city contribution decomposition, 4 rows x 3 columns.
# The beta0 contribution is not drawn as a bar, but is included in `pred`.

def nearest_valid(beta0_map, target_lat, target_lon):
    finite = np.isfinite(beta0_map)
    LON, LAT = np.meshgrid(lon, lat)
    dlon = np.abs(LON - target_lon)
    dlon = np.minimum(dlon, 360 - dlon)
    d2 = (LAT - target_lat) ** 2 + dlon ** 2
    d2 = np.where(finite, d2, np.inf)
    return np.unravel_index(np.argmin(d2), d2.shape)


def global_band(field, latitudes):
    """Area-weighted spatial mean +/- 1 SD across land cells for each year."""
    lat_weights = np.cos(np.deg2rad(latitudes))[:, None]

    mean = np.full(field.shape[0], np.nan)
    sd = np.full(field.shape[0], np.nan)

    for t in range(field.shape[0]):
        values = field[t]
        valid = np.isfinite(values)
        if not np.any(valid):
            continue
        weights = np.broadcast_to(lat_weights, values.shape)[valid]
        values_valid = values[valid]
        weight_sum = weights.sum()
        mean[t] = (weights * values_valid).sum() / weight_sum
        sd[t] = np.sqrt((weights * (values_valid - mean[t]) ** 2).sum() / weight_sum)

    return mean, mean - sd, mean + sd


def stacked_signed_bar(ax, x, comps, labels, facecolors, width=0.6):
    pos_base = np.zeros(len(x))
    neg_base = np.zeros(len(x))
    handles = []

    for comp, label, fc in zip(comps, labels, facecolors):
        comp = np.asarray(comp, dtype=float)
        pos = np.where(comp > 0, comp, 0.0)
        neg = np.where(comp < 0, comp, 0.0)

        h_pos = ax.bar(x, pos, bottom=pos_base, width=width, color=fc,
                       label=label, edgecolor="0.3", linewidth=0.6)
        ax.bar(x, neg, bottom=neg_base, width=width, color=fc,
               edgecolor="0.3", linewidth=0.6)

        handles.append(h_pos)
        pos_base += pos
        neg_base += neg

    return handles


n_cities = len(CITIES)
ncols = 3
nrows = math.ceil(n_cities / ncols)

fig, axes = plt.subplots(nrows, ncols, figsize=(20, 4.8 * nrows),
                         sharex=True, squeeze=False)
city_axes = axes.ravel()

legend_handles = None
legend_labels = None

global_mean, global_lo, global_hi = global_band(Ytarget, lat)

for ax, (name, city_lat, city_lon, category) in zip(city_axes, CITIES):
    i, j = nearest_valid(main["beta0"], city_lat, city_lon)

    b0v = main["beta0"][i, j]
    b1v = main["b1"][i, j]
    b2v = main["b2"][i, j]
    r2v = main["r2"][i, j]

    # driver contributions at this cell
    c1 = b1v * P1_MAIN[:, i, j]
    c2 = b2v * (soi_z1d if MODE == 0 else P2_MAIN[:, i, j])

    # Prediction includes beta0, which is not plotted as a bar.
    pred = b0v + c1 + c2

    obs = Ytarget[:, i, j]
    obs_err = Yerr[:, i, j]

    ax.fill_between(ds_years, global_lo, global_hi, color="0.75", alpha=0.45,
                    label="Global mean \u00b1 1 SD", zorder=0)
    ax.plot(ds_years, global_mean, color="0.45", lw=1.4)

    stacked_signed_bar(ax, ds_years, [c1, c2],
                       labels=[P1_MATH, P2_MATH], facecolors=BAR_COLORS)

    yerr_plot = np.where(np.isfinite(obs_err), obs_err, np.nan)

    ax.errorbar(ds_years, obs, yerr=yerr_plot, fmt="k-o", lw=2.0, ms=4.8,
                capsize=3.0, elinewidth=1.0, label="CAMS \u00b1 growth_err",
                zorder=7)
    ax.plot(ds_years, pred, "--s", color="tab:red", lw=1.8, ms=4.2,
            label="predicted", zorder=7)
    ax.axhline(0, color="black", lw=0.8)

    ax.set_title(
        f"{name} ({category})\n"
        f"{fmt_coord(city_lat, city_lon)}\n"
        rf"$\beta_0$={b0v:.2f}, ${P1_SHORT}$={b1v:.2f}, "
        rf"${P2_SHORT}$={b2v:.2f}, $R^2$={r2v:.2f}",
        fontsize=10.5, fontweight="bold",
    )

    ax.grid(True, axis="y", alpha=0.25)
    ax.set_xticks(ds_years)
    ax.tick_params(axis="x", rotation=45, labelsize=9)

    if legend_handles is None:
        legend_handles, legend_labels = ax.get_legend_handles_labels()

for ax in city_axes[n_cities:]:
    ax.axis("off")

for ax in axes[:, 0]:
    ax.set_ylabel("growth rate / contribution (ppm/yr)")
for ax in axes[-1, :]:
    ax.set_xlabel("year")

if legend_handles is not None:
    fig.legend(legend_handles, legend_labels, loc="lower center", ncol=5,
               fontsize=11, bbox_to_anchor=(0.5, -0.01))

fig.tight_layout(rect=[0, 0.035, 1, 0.99])
fig.savefig(outpath(f"city_contribution_{nrows}x{ncols}"), bbox_inches="tight")
plt.close(fig)
print("saved", outpath(f"city_contribution_{nrows}x{ncols}").name)

print("ALL DONE")