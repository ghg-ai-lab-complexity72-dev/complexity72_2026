#!/usr/bin/env python3
"""
Updated Figure 4 -- ground-based variant.

For every TCCON station: locate the nearest CAMS grid cell in the unified
dataset and, in one panel per station, overlay

    * the area-weighted GLOBAL-MEAN growth rate as a shaded band (grey),
      so each station can be read against the global spread;
    * the CAMS reanalysis growth rate at the station pixel (red dashed),
      with its own growth_err as a faint band;
    * the measured TCCON growth rate (black) with growth_rate_error bars;
      partial years (year_is_partial=1) drawn as open markers.

Only the CAMS window (2015-2024) is shown, since that is where the two can
be compared. Stations differ in how many of those years they cover -- some
have all ten, some only a few. Plot them all, then decide which to keep.

Inputs (place both NetCDFs next to this script, or edit the paths):
    unified_annual_carbon_dataset_2015_2024.nc   (CAMS growth_rate, growth_err)
    tccon_all_stations_annual_growth_rates.nc    (station growth_rate + error)

Output:
    outputs/figure4_tccon_variant.png
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap

# ----------------------------- CONFIG ------------------------------------
BASE_DIR = Path(__file__).resolve().parent
UNIFIED_FILE = BASE_DIR / "../Input/unified_annual_carbon_dataset_2015_2024.nc"
TCCON_FILE   = BASE_DIR / "../Input/tccon_all_stations_annual_growth_rates.nc"
OUTDIR = BASE_DIR / "../Output"
OUTDIR.mkdir(parents=True, exist_ok=True)

YEAR_MIN, YEAR_MAX = 2015, 2024      # CAMS comparison window
MIN_OVERLAP_YEARS  = 1               # drop stations with fewer TCCON years in window
NCOLS              = 5
BAND_SPREAD        = "sd"            # "sd" -> mean +/- 1 SD ; "iqr" -> 25-75th pct
SHOW_CAMS_ERR      = True            # faint band for CAMS growth_err
SORT_BY            = "coverage"      # "coverage" (most data first) or "latitude"

plt.rcParams.update({"font.size": 11, "axes.titlesize": 9,
                     "xtick.labelsize": 8, "ytick.labelsize": 8})

# Publication-ready TCCON station names, keyed by 2-letter short code.
# Falls back to the raw long_name for any code not listed here.
STATION_NAMES = {
    "ny": "Ny-\u00c5lesund", "so": "Sodankyl\u00e4", "ka": "Karlsruhe",
    "pr": "Paris", "or": "Orl\u00e9ans", "gm": "Garmisch",
    "pa": "Park Falls", "rj": "Rikubetsu", "oc": "Lamont",
    "df": "Edwards", "ci": "Pasadena", "db": "Darwin",
    "wg": "Wollongong", "hf": "Hefei", "iz": "Iza\u00f1a",
    "et": "East Trout Lake", "br": "Bremen", "js": "Saga",
    "tk": "Tsukuba", "eu": "Eureka", "xh": "Xianghe",
    "bu": "Burgos", "lr": "Lauder", "ni": "Nicosia",
    "ra": "R\u00e9union", "bi": "Bia\u0142ystok", "ll": "Lauder",
    "hw": "Harwell", "jf": "JPL",
}


def clean_name(short, long_name):
    return STATION_NAMES.get(short, long_name)


# ----------------------------- LOADERS -----------------------------------
def load_unified(path):
    ds = xr.open_dataset(path)
    gr = ds["growth_rate"].transpose("time", "lat", "lon")
    err = ds["growth_err"].transpose("time", "lat", "lon") if "growth_err" in ds else None
    yrs = ds["time"].values.astype(int)
    sel = (yrs >= YEAR_MIN) & (yrs <= YEAR_MAX)
    gr = gr.isel(time=np.where(sel)[0])
    if err is not None:
        err = err.isel(time=np.where(sel)[0])
    return gr, err, gr["time"].values.astype(int)


def load_tccon(path):
    ds = xr.open_dataset(path)
    return ds


# ------------------------- SPATIAL MATCHING ------------------------------
def nearest_pixel(gr, lat, lon):
    """(series over window, matched_lat, matched_lon) for nearest CAMS cell."""
    glat, glon = gr["lat"].values, gr["lon"].values
    lon_q = lon % 360 if glon.max() > 180 else ((lon + 180) % 360) - 180
    i = int(np.abs(glat - lat).argmin())
    j = int((np.abs((glon - lon_q + 180) % 360 - 180)).argmin())
    return gr.isel(lat=i, lon=j), (float(glat[i]), float(glon[j])), (i, j)


def global_band(gr, spread=BAND_SPREAD):
    """Area-weighted (cos lat) global-mean growth rate per year, with a band."""
    w2d = np.cos(np.deg2rad(gr["lat"].values))[:, None] * np.ones(
        (gr["lat"].size, gr["lon"].size))
    rows = []
    for k, y in enumerate(gr["time"].values):
        v = gr.isel(time=k).values.ravel(); w = w2d.ravel()
        m = np.isfinite(v); v, w = v[m], w[m]; ws = w.sum()
        mean = (w * v).sum() / ws
        if spread == "iqr":
            o = np.argsort(v); cw = np.cumsum(w[o]) / ws
            lo, hi = np.interp([0.25, 0.75], cw, v[o])
        else:
            sd = np.sqrt((w * (v - mean) ** 2).sum() / ws)
            lo, hi = mean - sd, mean + sd
        rows.append((int(y), mean, lo, hi))
    return pd.DataFrame(rows, columns=["year", "mean", "lo", "hi"]).set_index("year")


# ------------------------- STATION SELECTION -----------------------------
def station_frame(tds, years_win):
    ns = tds.sizes["station"]
    tyr = tds["year"].values.astype(int)
    inwin = np.isin(tyr, years_win)
    gr = tds["growth_rate"].values
    rows = []
    for s in range(ns):
        la = float(tds["station_latitude"].values[s])
        lo = float(tds["station_longitude"].values[s])
        if not (np.isfinite(la) and np.isfinite(lo)):
            continue
        n_win = int(np.isfinite(gr[s][inwin]).sum())
        if n_win < MIN_OVERLAP_YEARS:
            continue
        rows.append({
            "idx": s,
            "short": str(tds["station_short_name"].values[s]),
            "long": str(tds["station_long_name"].values[s]),
            "lat": la, "lon": lo, "n_win": n_win})
    df = pd.DataFrame(rows)
    if SORT_BY == "latitude":
        df = df.sort_values("lat", ascending=False)
    else:
        df = df.sort_values(["n_win", "lat"], ascending=[False, False])
    return df.reset_index(drop=True)


# ----------------------------- FIGURE ------------------------------------
def make_figure(gr, err, gyears, tds, band):
    sf = station_frame(tds, gyears)
    n = len(sf)
    nrows = int(np.ceil(n / NCOLS))
    fig, axes = plt.subplots(nrows, NCOLS, figsize=(3.4 * NCOLS, 2.5 * nrows),
                             squeeze=False, sharex=True, sharey=True)
    tyr = tds["year"].values.astype(int)
    win_mask = np.isin(tyr, gyears)
    tyr_win = tyr[win_mask]
    tgr = tds["growth_rate"].values
    terr = tds["growth_rate_error"].values
    tpar = tds["year_is_partial"].values if "year_is_partial" in tds else None

    for ax, row in zip(axes.ravel(), sf.itertuples()):
        # global-mean band
        ax.fill_between(band.index, band["lo"], band["hi"], color="0.82",
                        lw=0, zorder=1)
        ax.plot(band.index, band["mean"], color="0.5", lw=1.0, zorder=2)

        # CAMS pixel
        cams, (mlat, mlon), _ = nearest_pixel(gr, row.lat, row.lon)
        cy = cams["time"].values.astype(int); cv = cams.values
        if SHOW_CAMS_ERR and err is not None:
            ev = nearest_pixel(err, row.lat, row.lon)[0].values
            ax.fill_between(cy, cv - ev, cv + ev, color="tab:red", alpha=0.12,
                            lw=0, zorder=3)
        ax.plot(cy, cv, "--", color="tab:red", lw=1.3, marker="o", ms=3,
                zorder=4)

        # TCCON measured
        g = tgr[row.idx][win_mask]; e = terr[row.idx][win_mask]
        fin = np.isfinite(g)
        ax.plot(tyr_win[fin], g[fin], "-", color="k", lw=1.2, zorder=5)
        ax.errorbar(tyr_win[fin], g[fin], yerr=e[fin], fmt="none",
                    ecolor="k", elinewidth=0.8, capsize=2, zorder=6)
        if tpar is not None:
            par = tpar[row.idx][win_mask].astype(bool) & fin
            full = (~par) & fin
            ax.plot(tyr_win[full], g[full], "o", color="k", ms=4, zorder=7)
            ax.plot(tyr_win[par], g[par], "o", mfc="white", mec="k",
                    mew=1.0, ms=4, zorder=7)
        else:
            ax.plot(tyr_win[fin], g[fin], "o", color="k", ms=4, zorder=7)

        ax.set_title(f"{clean_name(row.short, row.long)} ({row.short})\n"
                     f"{row.lat:.1f}, {row.lon:.1f}  \u2022  n={row.n_win}")
        ax.grid(True, axis="y", alpha=0.25)
        ax.margins(x=0.03)

    for ax in axes.ravel()[n:]:
        ax.set_visible(False)
    for ax in axes[-1, :]:
        ax.set_xlabel("year")
        ax.set_xticks(list(gyears)[::2])
    for ax in axes[:, 0]:
        ax.set_ylabel("growth rate (ppm/yr)")

    legend = [
        Patch(facecolor="0.82", label="global mean \u00b11 SD"),
        Line2D([], [], color="tab:red", ls="--", marker="o", ms=4,
               label="CAMS (station pixel)"),
        Line2D([], [], color="k", marker="o", ms=4, label="TCCON (measured)"),
        Line2D([], [], color="k", marker="o", mfc="white", ms=4, ls="none",
               label="TCCON partial year"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.012), fontsize=9)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig, sf


def land_background(gr):
    """1 = land, 0 = ocean on the CAMS grid. Uses global_land_mask if present,
    otherwise falls back to the finite GOSIF footprint."""
    lat, lon = gr["lat"].values, gr["lon"].values
    LON, LAT = np.meshgrid(lon, lat)
    LONq = np.where(LON > 180, LON - 360, LON)
    try:
        from global_land_mask import globe
        return globe.is_land(LAT, LONq).astype(float), (lon.min(), lon.max(),
                                                         lat.min(), lat.max())
    except Exception:
        g = gr.isel(time=0).notnull().values.astype(float)  # crude fallback
        return g, (lon.min(), lon.max(), lat.min(), lat.max())


def make_station_map(gr, tds):
    """World map of the plotted TCCON stations, coloured by data coverage."""
    sf = station_frame(tds, gr["time"].values.astype(int))
    land, extent = land_background(gr)
    fig, ax = plt.subplots(figsize=(13, 6.6))
    ax.imshow(land, origin="lower", extent=extent, aspect="auto",
              cmap=ListedColormap(["#f4f8fb", "#e6e6e6"]), vmin=0, vmax=1,
              zorder=0)
    for v in range(-150, 151, 50):
        ax.axvline(v, color="0.9", lw=0.6, zorder=1)
    for h in range(-60, 61, 30):
        ax.axhline(h, color="0.9", lw=0.6, zorder=1)

    sc = ax.scatter(sf["lon"], sf["lat"], c=sf["n_win"], cmap="plasma",
                    vmin=1, vmax=10, s=70, edgecolor="k", linewidth=0.7,
                    zorder=5)
    for r in sf.itertuples():
        txt = ax.annotate(r.short, (r.lon, r.lat),
                          xytext=(4, 4), textcoords="offset points",
                          fontsize=7.5, fontweight="bold", zorder=6)
        txt.set_path_effects([pe.withStroke(linewidth=2.2, foreground="white")])

    cb = fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.02)
    cb.set_label("years with TCCON data in 2015\u20132024")
    cb.set_ticks(range(1, 11))
    ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
    ax.set_title(f"TCCON stations used in the CAMS comparison "
                 f"(n = {len(sf)})", fontweight="bold")
    ax.set_xticks(range(-180, 181, 60)); ax.set_yticks(range(-90, 91, 30))
    fig.tight_layout()
    return fig, sf


if __name__ == "__main__":
    gr, err, gyears = load_unified(UNIFIED_FILE)
    tds = load_tccon(TCCON_FILE)
    band = global_band(gr)
    fig, sf = make_figure(gr, err, gyears, tds, band)
    out = OUTDIR / "figure4_tccon_variant.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")

    fig_map, _ = make_station_map(gr, tds)
    out_map = OUTDIR / "figure_station_map.png"
    fig_map.savefig(out_map, dpi=150, bbox_inches="tight")
    print("wrote", out_map)
    print(f"stations plotted: {len(sf)}")
    print(sf[["short", "long", "lat", "lon", "n_win"]].to_string(index=False))
    print("global-mean band:")
    print(band.round(2).to_string())
    print("wrote", out)
