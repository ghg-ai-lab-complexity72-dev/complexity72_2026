#!/usr/bin/env python3
"""
Per-LAND-cell regression of the RAW CO2 growth rate on a global lagged annual SOI index and
gridded emission, with emission standardized per cell and ocean cells masked out.

Require in the same directory:
  - unified_annual_carbon_dataset_2015_2024.nc 
  - SOI.txt 

Outputs:
  outputs/soi_main_obs_pred_resid.png: Fig 3 of Main
  outputs/soi_variants.png: Fig 1 of Supplementary Information
  outputs/soi_beta_r2_2x2.png: Fig 2 of Supplementary Information
  outputs/soi_city_contribution_3x3.png: Fig 4 of Main updated
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


# ----------------------------- CONFIG ------------------------------------

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = BASE_DIR / "../Input/unified_annual_carbon_dataset_2015_2024.nc"
SOI_FILE   = BASE_DIR / "../Input/SOI.txt"

OUTDIR = BASE_DIR / "../Output"
OUTDIR.mkdir(parents=True, exist_ok=True)

#YEARS = [2016, 2018, 2020, 2022, 2024]
YEARS = [2015, 2017, 2019, 2021, 2023]

SOI_LAG_MONTHS = 7
MIN_OBS = 6
LAMBDAS = np.array([0.0, 0.1, 0.3, 1.0, 3.0, 10.0])

# Visual controls
YEAR_LABEL_X = -0.13
MAP_YEAR_FONTSIZE = 14

BETA0_CMAP = "viridis"    # beta0 heatmap, as in the reference beta-map script
BETA_CMAP = "RdBu_r"      # shared diverging heatmap for betaSOI and betaE
R2_CMAP = "viridis"       # separate heatmap for R2

# Titles for the beta/R2 heatmaps, matching the reference style
BETA0_TITLE = r"$\beta_0$ (ppm/yr): local baseline growth"
BETA_SOI_TITLE = r"$\beta_{SOI}$ (ppm/yr per 1 SD SOI): ENSO sensitivity"
BETA_E_TITLE = r"$\beta_E$ (ppm/yr per 1 SD emission): emission sensitivity"
R2_TITLE = r"$R^2$"

plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 11,
})

# City list.
# Congo was removed; Tokyo is kept once. Australian Outback is added to complete the 3x3 grid.
CITIES = [
    ("Beijing, China",       39.9, 116.4,  "industrialized"),
    ("Ruhr Valley, Germany", 51.4,   7.2,  "industrialized"),
    ("Tokyo, Japan",         35.7, 139.7,  "industrialized"),
    ("Los Angeles, USA",     34.0, -118.2, "industrialized"),
    ("Amazon, Brazil",       -3.5, -62.0,  "forest"),
    ("Borneo, Indonesia",    -1.0, 113.5,  "forest"),
    ("Patagonia, Argentina", -45.0, -70.0, "steppe"),
    ("Siberia, Russia",       62.0, 100.0,  "boreal"),
    ("Australian Outback",   -25.0, 133.0,  "desert"),
]

print("INPUT_FILE:", INPUT_FILE)
print("INPUT exists?", INPUT_FILE.exists())
print("SOI_FILE:", SOI_FILE)
print("SOI exists?", SOI_FILE.exists())
print("OUTDIR:", OUTDIR)

if not INPUT_FILE.exists():
    raise FileNotFoundError(f"NetCDF file not found: {INPUT_FILE}")

if not SOI_FILE.exists():
    raise FileNotFoundError(f"SOI file not found: {SOI_FILE}")

# -------------------------------------------------------------------------


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


# ----------------------------- LOAD DATA ---------------------------------

ds = xr.open_dataset(INPUT_FILE)

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

if "growth_err" in ds:
    growth_err = ds["growth_err"].transpose("time", "lat", "lon").values.astype("float64")
    print("Using growth_err from NetCDF for observed city error bars.")
else:
    growth_err = np.full_like(gr, np.nan)
    print("WARNING: growth_err not found in NetCDF. City error bars will be omitted.")

Ytarget = gr.copy()

land2d = build_land_mask(lat, lon)
land3d = np.broadcast_to(land2d[None], (T, nlat, nlon))

print(f"land cells: {int(land2d.sum())} ({100 * land2d.mean():.1f}%)")

emission_raw = ds["emission"].transpose("time", "lat", "lon").values.astype("float64")

emission_z = zscore_time(emission_raw)

delta_emission = np.full_like(emission_raw, np.nan)
delta_emission[1:] = emission_raw[1:] - emission_raw[:-1]
delta_emission_z = zscore_time(delta_emission)

# Apply land mask: ocean -> NaN so it is excluded from fitting and display.
Ytarget = np.where(land3d, Ytarget, np.nan)
Yerr = np.where(land3d, growth_err, np.nan)
emission_z = np.where(land3d, emission_z, np.nan)
delta_emission_z = np.where(land3d, delta_emission_z, np.nan)

# SOI predictor.
soi_ann = annual_lag_soi(read_standardized_soi(SOI_FILE))
soi_raw = np.array([soi_ann.loc[int(y)] for y in ds_years], dtype="float64")
soi_z1d = (soi_raw - np.nanmean(soi_raw)) / np.nanstd(soi_raw)
soi_grid = np.broadcast_to(soi_z1d[:, None, None], (T, nlat, nlon)).copy()

print("SOI lag-7 annual z:", dict(zip(ds_years.tolist(), np.round(soi_z1d, 2).tolist())))


# ----------------------------- REGRESSION --------------------------------

def fit(Y, X1, X2, lambdas, min_obs=MIN_OBS):
    """
    X1 = emission or delta_emission predictor.
    X2 = SOI predictor.

    Returns:
        beta0, b1, b2, r2, pred

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

            loo = np.where(
                good,
                (Yi - yhat) / np.where(good, denom, 1.0),
                0.0,
            )

            crit = np.where(good, loo ** 2, 0.0).sum(axis=0) / np.maximum(good.sum(axis=0), 1)

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


# MAIN = OLS(SOI + emission)
# VARIANTS = Ridge(SOI + emission) and Ridge(SOI + delta_emission)
print("fitting OLS:   SOI + emission [main] ...")
main = fit(Ytarget, emission_z, soi_grid, np.array([0.0]))

print("fitting Ridge: SOI + emission [variant] ...")
ridge_emis = fit(Ytarget, emission_z, soi_grid, LAMBDAS)

print("fitting Ridge: SOI + delta_emission [variant] ...")
ridge_delta = fit(Ytarget, delta_emission_z, soi_grid, LAMBDAS)


# ----------------------------- MAP HELPERS -------------------------------

landmask = np.isfinite(main["pred"])
obs_field = np.where(landmask, Ytarget, np.nan)

vmin = np.nanpercentile(obs_field, 2)
vmax = np.nanpercentile(obs_field, 98)

main_resid = obs_field - main["pred"]
rlim = np.nanpercentile(np.abs(main_resid[np.isfinite(main_resid)]), 96)

EXTENT = [lon.min(), lon.max(), lat.min(), lat.max()]


def add_map(ax, data, cmap, vmn, vmx):
    cm = plt.get_cmap(cmap).copy()
    cm.set_bad("0.85")

    kwargs = dict(
        origin="lower",
        extent=EXTENT,
        cmap=cm,
        vmin=vmn,
        vmax=vmx,
        aspect="auto",
    )

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
    ax.text(
        YEAR_LABEL_X,
        0.5,
        str(yr),
        transform=ax.transAxes,
        rotation=90,
        va="center",
        ha="right",
        fontsize=MAP_YEAR_FONTSIZE,
        fontweight="bold",
        clip_on=False,
    )


def yr_index(yr):
    return int(np.where(ds_years == yr)[0][0])


def sym_limits(a, pct=96):
    vals = a[np.isfinite(a)]
    if vals.size == 0:
        return -1.0, 1.0
    lim = np.nanpercentile(np.abs(vals), pct)
    return -lim, lim


def add_colorbar(fig, im, ax, label=None):
    cb = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    return cb


# ----------------------------- FIGURE 1 ----------------------------------
# observed | OLS(SOI + emission) prediction | residual

fig = plt.figure(figsize=(15.8, 3.25 * len(YEARS)))

for row, yr in enumerate(YEARS):
    ti = yr_index(yr)
    obs = np.where(landmask[ti], Ytarget[ti], np.nan)
    pred = main["pred"][ti]

    panels = [
        (obs, "viridis", vmin, vmax, "Observed $\Delta{CO}2$" + " (ppm/yr)"),
        (pred, "viridis", vmin, vmax, "Predicted  $\Delta{CO}2$" + " (ppm/yr): LS (DSS + DSE)"),
        (obs - pred, "PuOr_r", -rlim, rlim, "Residual $\Delta{CO}2$" + " (ppm/yr)"),
    ]

    for col, (data, cmap, lo, hi, cblabel) in enumerate(panels):
        ax = fig.add_subplot(
            len(YEARS),
            3,
            row * 3 + col + 1,
            projection=ccrs.PlateCarree() if HAS_CARTOPY else None,
        )

        im = add_map(ax, data, cmap, lo, hi)
        if row == 0:
            ax.set_title(cblabel, fontsize=12, fontweight="bold")
        if col == 0:
            add_year_label(ax, yr)

        add_colorbar(fig, im, ax, cblabel)

fig.tight_layout(rect=[0.04, 0, 1, 0.99])
fig.savefig(OUTDIR / "soi_main_obs_pred_resid.png", dpi=105, bbox_inches="tight")
plt.close(fig)
print("saved soi_main_obs_pred_resid.png")


# ----------------------------- FIGURE 2 ----------------------------------
# observed | Ridge(SOI + emission) | Ridge(SOI + delta_emission)

fig = plt.figure(figsize=(15.8, 3.25 * len(YEARS)))

for row, yr in enumerate(YEARS):
    ti = yr_index(yr)
    obs = np.where(landmask[ti], Ytarget[ti], np.nan)

    panels = [
        (obs, "viridis", vmin, vmax, "Observed $\Delta{CO}2$" + " (ppm/yr)"),
        (ridge_emis["pred"][ti], "viridis", vmin, vmax, "Predicted $\Delta{CO}2$" + " (ppm/yr): Ridge (DSS + DSE)"),
        (ridge_delta["pred"][ti], "viridis", vmin, vmax, r"Predicted $\Delta{CO}2$" + " (ppm/yr): Ridge (DSS + DS $\Delta_{E}$)"),
    ]

    for col, (data, cmap, lo, hi, cblabel) in enumerate(panels):
        ax = fig.add_subplot(
            len(YEARS),
            3,
            row * 3 + col + 1,
            projection=ccrs.PlateCarree() if HAS_CARTOPY else None,
        )

        im = add_map(ax, data, cmap, lo, hi)
        if row == 0:
            ax.set_title(cblabel, fontsize=12, fontweight="bold")
        if col == 0:
            add_year_label(ax, yr)

        add_colorbar(fig, im, ax, cblabel)

fig.tight_layout(rect=[0.04, 0, 1, 0.99])
fig.savefig(OUTDIR / "soi_variants.png", dpi=105, bbox_inches="tight")
plt.close(fig)
print("saved soi_variants.png")


# ----------------------------- FIGURE 3 ----------------------------------
# beta maps + R2 for MAIN OLS model.
# Draw all beta maps in the same style as the reference beta-map script:
#   - beta0: viridis + 2/98 percentile limits
#   - betaSOI and betaE: RdBu_r + symmetric 96th-percentile limits
#   - R2 kept as a fourth panel

b0 = main["beta0"]
bE = main["b1"]
bSOI = main["b2"]
r2m = main["r2"]

b0_finite = b0[np.isfinite(b0)]
if b0_finite.size == 0:
    b0_vmin, b0_vmax = 0.0, 1.0
else:
    # Reference-style robust limits for beta0
    b0_vmin = np.nanpercentile(b0_finite, 2)
    b0_vmax = np.nanpercentile(b0_finite, 98)

specs = [
    (b0, BETA0_TITLE, BETA0_CMAP, b0_vmin, b0_vmax),
    (bSOI, BETA_SOI_TITLE, BETA_CMAP, *sym_limits(bSOI)),
    (bE, BETA_E_TITLE, BETA_CMAP, *sym_limits(bE)),
    (r2m, R2_TITLE, R2_CMAP, 0.0, 1.0),
]

fig = plt.figure(figsize=(14.5, 8.5))

for k, (data, title, cmap, lo, hi) in enumerate(specs):
    ax = fig.add_subplot(
        2,
        2,
        k + 1,
        projection=ccrs.PlateCarree() if HAS_CARTOPY else None,
    )

    im = add_map(ax, data, cmap, lo, hi)
    ax.set_title(title, fontsize=12, fontweight="bold")
    add_colorbar(fig, im, ax, None)

fig.tight_layout()
fig.savefig(OUTDIR / "soi_beta_r2_2x2.png", dpi=115, bbox_inches="tight")
plt.close(fig)
print("saved soi_beta_r2_2x2.png")


# ----------------------------- FIGURE 4 ----------------------------------
# city contribution decomposition for MAIN OLS model.
# No white beta0 bars are shown.
# Prediction still includes beta0.

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
        sd[t] = np.sqrt(
            (weights * (values_valid - mean[t]) ** 2).sum() / weight_sum
        )

    return mean, mean - sd, mean + sd


def stacked_signed_bar(ax, x, comps, labels, facecolors, width=0.6):
    pos_base = np.zeros(len(x))
    neg_base = np.zeros(len(x))

    handles = []

    for comp, label, fc in zip(comps, labels, facecolors):
        comp = np.asarray(comp, dtype=float)

        pos = np.where(comp > 0, comp, 0.0)
        neg = np.where(comp < 0, comp, 0.0)

        h_pos = ax.bar(
            x,
            pos,
            bottom=pos_base,
            width=width,
            color=fc,
            label=label,
            edgecolor="0.3",
            linewidth=0.6,
        )

        ax.bar(
            x,
            neg,
            bottom=neg_base,
            width=width,
            color=fc,
            edgecolor="0.3",
            linewidth=0.6,
        )

        handles.append(h_pos)

        pos_base += pos
        neg_base += neg

    return handles


n_cities = len(CITIES)
ncols = 3
nrows = math.ceil(n_cities / ncols)

fig, axes = plt.subplots(
    nrows,
    ncols,
    figsize=(20, 4.8 * nrows),
    sharex=True,
    squeeze=False,
)

city_axes = axes.ravel()

legend_handles = None
legend_labels = None

global_mean, global_lo, global_hi = global_band(Ytarget, lat)

for ax, (name, city_lat, city_lon, category) in zip(city_axes, CITIES):
    i, j = nearest_valid(main["beta0"], city_lat, city_lon)

    b0v = main["beta0"][i, j]
    bEv = main["b1"][i, j]
    bSv = main["b2"][i, j]
    r2v = main["r2"][i, j]

    cS = bSv * soi_z1d
    cE = bEv * emission_z[:, i, j]

    # Prediction includes beta0, but the beta0 contribution is not plotted as a bar.
    pred = b0v + cS + cE

    obs = Ytarget[:, i, j]
    obs_err = Yerr[:, i, j]

    ax.fill_between(
        ds_years,
        global_lo,
        global_hi,
        color="0.75",
        alpha=0.45,
        label="Global mean ± 1 SD",
        zorder=0,
    )
    ax.plot(
        ds_years,
        global_mean,
        color="0.45",
        lw=1.4,
       # label="global land mean",
       # zorder=1,
    )

    stacked_signed_bar(
        ax,
        ds_years,
        [cS, cE],
        labels=[r"$\beta_{SOI}\,SOI_z$", r"$\beta_E\,E_z$"],
        facecolors=["tab:blue", "tab:orange"],
    )

    yerr_plot = np.where(np.isfinite(obs_err), obs_err, np.nan)

    ax.errorbar(
        ds_years,
        obs,
        yerr=yerr_plot,
        fmt="k-o",
        lw=2.0,
        ms=4.8,
        capsize=3.0,
        elinewidth=1.0,
        label="observed ± growth_err",
        zorder=7,
    )

    ax.plot(
        ds_years,
        pred,
        "--s",
        color="tab:red",
        lw=1.8,
        ms=4.2,
        label="predicted",
        zorder=7,
    )

    ax.axhline(0, color="black", lw=0.8)

    ax.set_title(
        f"{name} ({category})\n"
        rf"$\beta_0$={b0v:.2f}, $\beta_{{SOI}}$={bSv:.2f}, "
        rf"$\beta_E$={bEv:.2f}, R²={r2v:.2f}",
        fontsize=10.5,
        fontweight="bold",
    )

    ax.grid(True, axis="y", alpha=0.25)
    ax.set_xticks(ds_years)
    ax.tick_params(axis="x", rotation=45, labelsize=9)

    if legend_handles is None:
        legend_handles, legend_labels = ax.get_legend_handles_labels()

# Hide unused panels if the city list does not exactly fill the grid.
for ax in city_axes[n_cities:]:
    ax.axis("off")

for ax in axes[:, 0]:
    ax.set_ylabel("growth rate / contribution (ppm/yr)")

for ax in axes[-1, :]:
    ax.set_xlabel("year")

if legend_handles is not None:
    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        ncol=4,
        fontsize=11,
        bbox_to_anchor=(0.5, -0.01),
    )

fig.tight_layout(rect=[0, 0.035, 1, 0.99])
fig.savefig(OUTDIR / "soi_city_contribution_3x3.png", dpi=115, bbox_inches="tight")
plt.close(fig)

print("saved soi_city_contribution_3x3.png")
print("ALL DONE")
