#!/usr/bin/env python3
"""
GLOBAL-MEAN regression of the atmospheric CO2 growth rate.

Companion to `co2_gridcell_regression_soi_gosif.py`. Instead of fitting one
regression per grid cell, the whole globe is first aggregated to a single
annual time series, and ONE regression is fitted on those annual points (2015-2025):

    MODE = 0  ->  SOI model   : G(t) ~ b0 + bE*E_z(t)  + bSOI*SOI_z(t) (the good fit)
    MODE = 1  ->  GOSIF model : G(t) ~ b0 + bdE*dE_z(t) + bdSIF*dSIF_z(t)

where G(t) is the area-weighted global mean growth rate, E(t) the global TOTAL
anthropogenic emission, and SIF(t) the globally integrated GOSIF. As in the
paper, the GOSIF specification uses the year-to-year DIFFERENCES of the
drivers; the SOI specification uses their absolute values.

HOW THE AGGREGATION IS DONE  (this matters, see AGGREGATION below)
  growth rate : area-weighted mean  (cos lat)   -- an intensive quantity
  emission    : plain sum over cells            -- total mass emitted
  GOSIF       : area-weighted sum (integral)    -- an intensity, so raw
                summing would over-weight small polar cells
  SOI         : already a single global index, 7-month lag as elsewhere

Because there are only 11 annual samples (2015-2025; 10 in MODE = 1, where the
first year is lost to differencing), this fit has 8 (7) degrees of freedom.
The script therefore reports standard errors, t- and p-values, and a
leave-one-year-out cross-validated R^2 (LOO R2) alongside the in-sample R^2 --
the gap between them is the honest measure of how much is overfitting.

PATHS
-----

  Reads   Input/unified_annual_carbon_dataset_2015_2025.nc
          Input/SOI.txt                          (only when MODE = 0)
  Writes  Figures_and_maps/global_regression_<prefix>.pdf
  (coefficients, SE, t, p, R2, adjusted R2 and LOO R2 are printed to screen)

"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================== CONFIG ===================================

# -------------------------------------------------------------------------
#  MODE = 0 : total emission + lagged SOI      (absolute values)
#  MODE = 1 : total emission + global GOSIF    (year-to-year differences)
# -------------------------------------------------------------------------
MODE = 0
# -------------------------------------------------------------------------

# Domain over which the growth rate is averaged.
#   "global" : every grid cell (CO2 is well mixed; comparable to Fig. 1)
#   "land"   : land cells only (comparable to the per-cell regressions)
GROWTH_DOMAIN = "global"

# How GOSIF is aggregated: "area_weighted_sum" (global integral, recommended)
# or "area_weighted_mean" (average intensity) or "raw_sum" (literal sum).
SIF_AGGREGATION = "area_weighted_sum"

# Emissions are always summed over land cells (total mass emitted).

SOI_LAG_MONTHS = 7
FIG_EXT = "pdf"

plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# --- locate the repository root -------------------------------------------

def find_repo_root(start: Path) -> Path:
    """Walk up from `start` until a folder containing Input/ and Output/
    (or .git) is found."""
    for folder in [start, *start.parents]:
        if (folder / "Input").is_dir() and (folder / "Output").is_dir():
            return folder
        if (folder / ".git").exists():
            return folder
    raise FileNotFoundError(
        f"Could not locate the repository root above {start}.\n"
        "Expected a folder containing Input/ and Output/. "
        "Run this script from inside a clone of complexity72_2026."
    )


BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = find_repo_root(BASE_DIR)

INPUT_DIR = REPO_ROOT / "Input"
INPUT_FILE = INPUT_DIR / "unified_annual_carbon_dataset_2015_2025.nc"
SOI_FILE = INPUT_DIR / "SOI.txt"

FIGDIR = REPO_ROOT / "Figures_and_maps"
FIGDIR.mkdir(parents=True, exist_ok=True)

if MODE not in (0, 1):
    raise ValueError("MODE must be 0 (SOI) or 1 (GOSIF)")
if not INPUT_FILE.exists():
    raise FileNotFoundError(f"NetCDF file not found: {INPUT_FILE}")
if MODE == 0 and not SOI_FILE.exists():
    raise FileNotFoundError(f"SOI file not found: {SOI_FILE} (needed for MODE=0)")

PREFIX = "soi" if MODE == 0 else "delta_sif"
print(f"MODE = {MODE}  ->  " + ("total emission + lagged SOI (absolute)"
                                if MODE == 0 else
                                "total emission + global GOSIF (differenced)"))
print("INPUT_FILE:", INPUT_FILE, "| exists?", INPUT_FILE.exists())


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
            rows.append({"year": year, "month": month,
                         "soi": np.nan if value <= -900 else value})
    return pd.DataFrame(rows)


def annual_lag_soi(soi_monthly, lag=SOI_LAG_MONTHS, min_months=12):
    df = soi_monthly.dropna(subset=["soi"]).copy()
    df["date"] = pd.to_datetime(dict(year=df.year, month=df.month, day=1))
    df["resp"] = df["date"] + pd.DateOffset(months=lag)
    df["co2_year"] = df["resp"].dt.year
    grouped = df.groupby("co2_year")["soi"]
    return grouped.mean()[grouped.count() >= min_months]


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
            resolution="110m", category="physical", name="land")).geometries()))
        pg = prep(geom)
        flat = np.array([pg.contains(Point(x, y))
                         for x, y in zip(LONq.ravel(), LAT.ravel())])
        return flat.reshape(LAT.shape)


def zscore(a):
    a = np.asarray(a, float)
    return (a - np.nanmean(a)) / np.nanstd(a)


# ====================== LOAD AND AGGREGATE ===============================

ds = xr.open_dataset(INPUT_FILE)
lat = ds["lat"].values
lon = ds["lon"].values
years = ds["time"].values.astype(int)
T = len(years)

gr = ds["growth_rate"].transpose("time", "lat", "lon").values.astype(float)
emis = ds["emission"].transpose("time", "lat", "lon").values.astype(float)
sif = ds["gosif"].transpose("time", "lat", "lon").values.astype(float)

land2d = build_land_mask(lat, lon)
print(f"land cells: {int(land2d.sum())} ({100 * land2d.mean():.1f}%)")

# cos(lat) area weights, broadcast to the grid
w2d = np.cos(np.deg2rad(lat))[:, None] * np.ones((lat.size, lon.size))


def area_weighted_mean(field2d, mask=None):
    v = field2d.ravel()
    w = w2d.ravel()
    m = np.isfinite(v)
    if mask is not None:
        m &= mask.ravel()
    return np.sum(w[m] * v[m]) / np.sum(w[m])


def area_weighted_sum(field2d, mask=None):
    v = field2d.ravel()
    w = w2d.ravel()
    m = np.isfinite(v)
    if mask is not None:
        m &= mask.ravel()
    return np.sum(w[m] * v[m])


def plain_sum(field2d, mask=None):
    v = field2d.ravel()
    m = np.isfinite(v)
    if mask is not None:
        m &= mask.ravel()
    return np.sum(v[m])


growth_mask = None if GROWTH_DOMAIN == "global" else land2d

G = np.array([area_weighted_mean(gr[t], growth_mask) for t in range(T)])
E_tot = np.array([plain_sum(emis[t], land2d) for t in range(T)])

if SIF_AGGREGATION == "area_weighted_sum":
    S_tot = np.array([area_weighted_sum(sif[t], land2d) for t in range(T)])
elif SIF_AGGREGATION == "area_weighted_mean":
    S_tot = np.array([area_weighted_mean(sif[t], land2d) for t in range(T)])
else:
    S_tot = np.array([plain_sum(sif[t], land2d) for t in range(T)])

# SOI (MODE 0)
SOI = np.full(T, np.nan)
if MODE == 0:
    soi_ann = annual_lag_soi(read_standardized_soi(SOI_FILE))
    SOI = np.array([soi_ann.loc[int(y)] for y in years], dtype=float)

print(f"\nglobal mean growth rate ({GROWTH_DOMAIN}): "
      f"{G.min():.2f} to {G.max():.2f} ppm/yr, mean {G.mean():.2f}")


# ====================== BUILD THE DESIGN MATRIX ==========================

if MODE == 0:
    x1_raw, x2_raw = E_tot, SOI
    x1_name, x2_name = "total_emission", "SOI_lag7"
    x1_lab, x2_lab = r"$\beta_E\,E_z$", r"$\beta_{SOI}\,SOI_z$"
    x1_sym, x2_sym = r"\beta_E", r"\beta_{SOI}"
    bar_colors = ["tab:orange", "tab:blue"]
    fit_years = years
    used = np.ones(T, bool)
else:
    # differenced drivers -> first year drops out
    x1_raw = np.full(T, np.nan)
    x2_raw = np.full(T, np.nan)
    x1_raw[1:] = E_tot[1:] - E_tot[:-1]
    x2_raw[1:] = S_tot[1:] - S_tot[:-1]
    x1_name, x2_name = "delta_total_emission", "delta_total_gosif"
    x1_lab, x2_lab = (r"$\beta_{\Delta E}\,\Delta E_z$",
                      r"$\beta_{\Delta SIF}\,\Delta SIF_z$")
    x1_sym, x2_sym = r"\beta_{\Delta E}", r"\beta_{\Delta SIF}"
    bar_colors = ["tab:orange", "tab:green"]
    used = np.isfinite(x1_raw) & np.isfinite(x2_raw)
    fit_years = years[used]

x1_z = np.full(T, np.nan)
x2_z = np.full(T, np.nan)
x1_z[used] = zscore(x1_raw[used])
x2_z[used] = zscore(x2_raw[used])

y = G[used]
X = np.column_stack([np.ones(used.sum()), x1_z[used], x2_z[used]])
n, k = X.shape
dof = n - k


# ============================ FIT ========================================

beta, *_ = np.linalg.lstsq(X, y, rcond=None)
yhat_used = X @ beta
resid = y - yhat_used

ss_res = np.sum(resid ** 2)
ss_tot = np.sum((y - y.mean()) ** 2)
r2 = 1 - ss_res / ss_tot
adj_r2 = 1 - (1 - r2) * (n - 1) / dof

XtX_inv = np.linalg.inv(X.T @ X)
sigma2 = ss_res / dof
se = np.sqrt(np.diag(XtX_inv) * sigma2)
tvals = beta / se

# two-sided p-values from the t distribution
try:
    from scipy import stats
    pvals = 2 * stats.t.sf(np.abs(tvals), dof)
except Exception:                       # scipy optional
    pvals = np.full(3, np.nan)

# leave-one-year-out R2 via the hat-matrix shortcut
H = X @ XtX_inv @ X.T
h = np.diag(H)
loo_resid = resid / (1 - h)
loo_r2 = 1 - np.sum(loo_resid ** 2) / ss_tot

names = ["intercept", x1_name, x2_name]
print("\n" + "=" * 68)
print(f"GLOBAL REGRESSION  (n = {n}, dof = {dof})")
print("=" * 68)
print(f"{'term':<24}{'coef':>9}{'se':>9}{'t':>8}{'p':>9}")
for nm, b, s, t_, p_ in zip(names, beta, se, tvals, pvals):
    print(f"{nm:<24}{b:>9.3f}{s:>9.3f}{t_:>8.2f}{p_:>9.3f}")
print("-" * 68)
print(f"R2 = {r2:.3f}   adjusted R2 = {adj_r2:.3f}   LOO R2 = {loo_r2:.3f}")
print("=" * 68)

# full-length prediction for plotting
pred_full = np.full(T, np.nan)
pred_full[used] = yhat_used
c1 = np.full(T, np.nan)
c2 = np.full(T, np.nan)
c1[used] = beta[1] * x1_z[used]
c2[used] = beta[2] * x2_z[used]


# =========================== FIGURE ======================================

fig, (ax, axb) = plt.subplots(
    2, 1, figsize=(9, 7.2), sharex=True,
    gridspec_kw={"height_ratios": [2.0, 1.0], "hspace": 0.08})

# --- top: observed vs predicted global mean growth rate ---
ax.plot(years, G, "k-o", lw=2, ms=5, label="Observed (CAMS global mean)",
        zorder=5)
ax.plot(years, pred_full, "--s", color="tab:red", lw=1.8, ms=4.5,
        label="Predicted", zorder=6)
ax.set_ylabel("CO$_2$ growth rate (ppm/yr)")
ax.grid(True, alpha=0.25)
ax.legend(loc="upper left", framealpha=0.9)
ax.set_title(
    f"Global-mean CO$_2$ growth rate regression "
    f"({'total emission + lagged SOI' if MODE == 0 else 'Δtotal emission + ΔGOSIF'})\n"
    rf"$\beta_0$={beta[0]:.2f}, ${x1_sym}$={beta[1]:.2f}, ${x2_sym}$={beta[2]:.2f},"
    rf"  $R^2$={r2:.2f}, LOO $R^2$={loo_r2:.2f}",
    fontsize=11, fontweight="bold")

# --- bottom: driver contributions (intercept excluded) ---
pos = np.zeros(T)
neg = np.zeros(T)
for comp, lab, col in [(c1, x1_lab, bar_colors[0]), (c2, x2_lab, bar_colors[1])]:
    cc = np.nan_to_num(comp)
    p = np.where(cc > 0, cc, 0.0)
    m = np.where(cc < 0, cc, 0.0)
    axb.bar(years, p, bottom=pos, width=0.6, color=col, label=lab,
            edgecolor="0.3", linewidth=0.6)
    axb.bar(years, m, bottom=neg, width=0.6, color=col,
            edgecolor="0.3", linewidth=0.6)
    pos += p
    neg += m

axb.axhline(0, color="k", lw=0.8)
axb.set_ylabel("contribution (ppm/yr)")
axb.set_xlabel("year")
axb.set_xticks(years)
axb.tick_params(axis="x", rotation=45)
axb.grid(True, axis="y", alpha=0.25)
axb.legend(loc="upper left", ncol=2, framealpha=0.9)

fig_path = FIGDIR / f"global_regression_{PREFIX}.{FIG_EXT}"
fig.savefig(fig_path, bbox_inches="tight")
plt.close(fig)
print("saved", fig_path.name)
print("ALL DONE")