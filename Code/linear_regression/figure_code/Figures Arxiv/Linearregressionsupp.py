#!/usr/bin/env python3
"""
Land-only per-cell regression predicting the RAW CO2 growth rate from
delta_emission and delta_gosif / delta_SIF, standardized per cell.

Require in the same directory:
  - unified_annual_carbon_dataset_2015_2024.nc 

This keeps the regression structure:

    growth_rate(t,r) = beta0(r)
                     + beta_dE(r)   * delta_emission_z(t,r)
                     + beta_dSIF(r) * delta_SIF_z(t,r)
                     + error(t,r)

                     

                     
Outputs:
  outputs/delta_sif_main_obs_pred_resid.png: Fig 3 of Supplementary Information
  outputs/delta_sif_variants.png: Fig 4 of Supplementary Information
  outputs/delta_sif_beta_r2_2x2.png: Fig 5 of Supplementary Information
  outputs/delta_sif_city_contribution_3x3.png: Fig 6 of Supplementary Information
"""

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

OUTDIR = BASE_DIR / "../Output"
OUTDIR.mkdir(parents=True, exist_ok=True)

YEARS = [2016, 2018, 2020, 2022, 2024]
YEARS = [2015, 2017, 2019, 2021, 2023]


MIN_OBS = 6
LAMBDAS = np.array([0.0, 0.1, 0.3, 1.0, 3.0, 10.0])

SIF_VAR = "gosif"
DELTA_SIF_VAR = "delta_gosif"

# Visual controls
YEAR_LABEL_X = -0.13
MAP_YEAR_FONTSIZE = 14

# Heatmap style matching the reference script
BETA0_CMAP = "viridis"
BETA_CMAP = "RdBu_r"
R2_CMAP = "viridis"

BETA0_TITLE = r"$\beta_0$ (ppm/yr): local baseline growth"
BETA_DE_TITLE = r"$\beta_{\Delta E}$ (ppm/yr per 1 SD $\Delta$emission): emission-change sensitivity"
BETA_DSIF_TITLE = r"$\beta_{\Delta SIF}$ (ppm/yr per 1 SD $\Delta$SIF): SIF-change sensitivity"
R2_TITLE = r"$R^2$"

plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 11,
})

# 3x3 city/region list.
# Congo is omitted and Australian Outback is included to complete the grid.
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
]

print("INPUT_FILE:", INPUT_FILE)
print("INPUT exists?", INPUT_FILE.exists())
print("OUTDIR:", OUTDIR)

if not INPUT_FILE.exists():
    raise FileNotFoundError(f"NetCDF file not found: {INPUT_FILE}")


# -------------------------------------------------------------------------


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


# ----------------------------- LOAD DATA ---------------------------------

ds = xr.open_dataset(INPUT_FILE)

for name in ["growth_rate", "emission", SIF_VAR]:
    if name not in ds:
        available = ", ".join(list(ds.data_vars))
        raise KeyError(f"Variable {name!r} not found in NetCDF. Available variables: {available}")

lat = ds["lat"].values
lon = ds["lon"].values
tvals = ds["time"].values

ds_years = (
    tvals.astype(int)
    if np.issubdtype(tvals.dtype, np.integer)
    else pd.to_datetime(tvals).year.values
)

T, nlat, nlon = ds["growth_rate"].shape

land2d = build_land_mask(lat, lon)
land3d = np.broadcast_to(land2d[None], (T, nlat, nlon))

print(f"land cells: {int(land2d.sum())} ({100 * land2d.mean():.1f}%)")

gr = ds["growth_rate"].transpose("time", "lat", "lon").values.astype("float64")
Ytarget = np.where(land3d, gr, np.nan)

if "growth_err" in ds:
    growth_err = ds["growth_err"].transpose("time", "lat", "lon").values.astype("float64")
    Yerr = np.where(land3d, growth_err, np.nan)
    print("Using growth_err from NetCDF for city error bars.")
else:
    Yerr = np.full_like(Ytarget, np.nan)
    print("WARNING: growth_err not found in NetCDF. City error bars will be omitted.")

emission_raw = ds["emission"].transpose("time", "lat", "lon").values.astype("float64")
sif_raw = ds[SIF_VAR].transpose("time", "lat", "lon").values.astype("float64")

E_z = np.where(land3d, zscore_time(emission_raw), np.nan)
SIF_z = np.where(land3d, zscore_time(sif_raw), np.nan)

if "delta_emission" in ds:
    dE_raw = ds["delta_emission"].transpose("time", "lat", "lon").values.astype("float64")
    print("Using 'delta_emission' from NetCDF.")
else:
    dE_raw = np.full_like(emission_raw, np.nan)
    dE_raw[1:] = emission_raw[1:] - emission_raw[:-1]
    print("'delta_emission' not found; computed first differences from 'emission'.")

if DELTA_SIF_VAR in ds:
    dSIF_raw = ds[DELTA_SIF_VAR].transpose("time", "lat", "lon").values.astype("float64")
    print(f"Using {DELTA_SIF_VAR!r} from NetCDF.")
else:
    dSIF_raw = np.full_like(sif_raw, np.nan)
    dSIF_raw[1:] = sif_raw[1:] - sif_raw[:-1]
    print(f"{DELTA_SIF_VAR!r} not found; computed first differences from {SIF_VAR!r}.")

dE_z = np.where(land3d, zscore_time(dE_raw), np.nan)
dSIF_z = np.where(land3d, zscore_time(dSIF_raw), np.nan)


# ----------------------------- REGRESSION --------------------------------

def fit(Y, X1, X2, lambdas, min_obs=MIN_OBS):
    """
    X1 = delta_emission or emission predictor.
    X2 = delta_SIF or SIF predictor.

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


# MAIN estimator = OLS(delta_emission + delta_SIF).
# Variants keep the regression logic from the original delta script.
print("fitting OLS:   delta_emission + delta_SIF [main] ...")
main = fit(Ytarget, dE_z, dSIF_z, np.array([0.0]))

print("fitting Ridge: delta_emission + delta_SIF [variant] ...")
ridge_delta = fit(Ytarget, dE_z, dSIF_z, LAMBDAS)

print("fitting OLS:   emission + SIF [variant] ...")
ols_abs = fit(Ytarget, E_z, SIF_z, np.array([0.0]))


# ----------------------------- MAP HELPERS -------------------------------

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


def add_colorbar(fig, im, ax, label=None):
    # Keep colorbars unlabeled. Titles are shown only above panels/columns.
    cb = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    return cb


# ----------------------------- FIGURE 1 ----------------------------------
# observed | OLS(delta_emission + delta_SIF) prediction | residual

fig = plt.figure(figsize=(15.8, 3.25 * len(YEARS)))

for row, yr in enumerate(YEARS):
    ti = yr_index(yr)
    obs = np.where(landmask[ti], Ytarget[ti], np.nan)
    pred = main["pred"][ti]

    panels = [
        (obs, "viridis", vmin, vmax, r"Observed $\Delta CO_2$ (ppm/yr)"),
        (pred, "viridis", vmin, vmax, r"Predicted $\Delta CO_2$ (ppm/yr): OLS ($\Delta $DSe + $\Delta$DSb)"),
        (obs - pred, "PuOr_r", -rlim, rlim, r"Residual $\Delta CO_2$ (ppm/yr)"),
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

        add_colorbar(fig, im, ax, None)

fig.tight_layout(rect=[0.04, 0, 1, 0.99])
fig.savefig(OUTDIR / "delta_sif_main_obs_pred_resid.png", dpi=105, bbox_inches="tight")
plt.close(fig)
print("saved delta_sif_main_obs_pred_resid.png")


# ----------------------------- FIGURE 2 ----------------------------------
# observed | Ridge(delta_emission + delta_SIF) | OLS(emission + SIF)

fig = plt.figure(figsize=(15.8, 3.25 * len(YEARS)))

for row, yr in enumerate(YEARS):
    ti = yr_index(yr)
    obs = np.where(landmask[ti], Ytarget[ti], np.nan)

    panels = [
        (obs, "viridis", vmin, vmax, r"Observed $\Delta CO_2$ (ppm/yr)"),
        (ridge_delta["pred"][ti], "viridis", vmin, vmax, r"Predicted $\Delta CO_2$ (ppm/yr): Ridge ($\Delta$DSe + $\Delta$DSb)"),
        (ols_abs["pred"][ti], "viridis", vmin, vmax, r"Predicted $\Delta CO_2$ (ppm/yr): OLS (DSe + DSb)"),
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

        add_colorbar(fig, im, ax, None)

fig.tight_layout(rect=[0.04, 0, 1, 0.99])
fig.savefig(OUTDIR / "delta_sif_variants.png", dpi=105, bbox_inches="tight")
plt.close(fig)
print("saved delta_sif_variants.png")


# ----------------------------- FIGURE 3 ----------------------------------
# beta maps + R2 for MAIN OLS model.
# Reference-style beta maps:
#   - beta0: viridis + 2/98 percentile limits
#   - beta_deltaE and beta_deltaSIF: RdBu_r + symmetric 96th-percentile limits
#   - R2: viridis

b0 = main["beta0"]
bDE = main["b1"]
bDSIF = main["b2"]
r2m = main["r2"]

b0_finite = b0[np.isfinite(b0)]
if b0_finite.size == 0:
    b0_vmin, b0_vmax = 0.0, 1.0
else:
    b0_vmin = np.nanpercentile(b0_finite, 2)
    b0_vmax = np.nanpercentile(b0_finite, 98)

specs = [
    (b0, BETA0_TITLE, BETA0_CMAP, b0_vmin, b0_vmax),
    (bDE, BETA_DE_TITLE, BETA_CMAP, *sym_limits(bDE)),
    (bDSIF, BETA_DSIF_TITLE, BETA_CMAP, *sym_limits(bDSIF)),
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
fig.savefig(OUTDIR / "delta_sif_beta_r2_2x2.png", dpi=115, bbox_inches="tight")
plt.close(fig)
print("saved delta_sif_beta_r2_2x2.png")


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

for ax, (name, city_lat, city_lon, category) in zip(city_axes, CITIES):
    i, j = nearest_valid(main["beta0"], city_lat, city_lon)

    b0v = main["beta0"][i, j]
    bDEv = main["b1"][i, j]
    bDSIFv = main["b2"][i, j]
    r2v = main["r2"][i, j]

    cDE = bDEv * dE_z[:, i, j]
    cDSIF = bDSIFv * dSIF_z[:, i, j]

    # Prediction includes beta0, but the beta0 contribution is not plotted as a bar.
    pred = b0v + cDE + cDSIF

    obs = Ytarget[:, i, j]
    obs_err = Yerr[:, i, j]

    stacked_signed_bar(
        ax,
        ds_years,
        [cDE, cDSIF],
        labels=[
            r"$\beta_{\Delta E}\,\Delta E_z$",
            r"$\beta_{\Delta SIF}\,\Delta SIF_z$",
        ],
        facecolors=["tab:orange", "tab:green"],
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
        rf"$\beta_0$={b0v:.2f}, "
        rf"$\beta_{{\Delta E}}$={bDEv:.2f}, "
        rf"$\beta_{{\Delta SIF}}$={bDSIFv:.2f}, "
        rf"R²={r2v:.2f}",
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
fig.savefig(OUTDIR / "delta_sif_city_contribution_3x3.png", dpi=115, bbox_inches="tight")
plt.close(fig)

print("saved delta_sif_city_contribution_3x3.png")
print("ALL DONE")
