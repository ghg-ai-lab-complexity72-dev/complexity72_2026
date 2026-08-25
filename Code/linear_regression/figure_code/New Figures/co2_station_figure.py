"""
co2_station_figure.py
=====================

Builds the ground-based variant of Figure 4 (supervisor request):

  * match each TCCON station to its nearest CAMS DSg grid cell,
  * overlay the CAMS pixel growth-rate series and the measured TCCON series
    in one panel per station (Fig.-4 layout),
  * draw the AREA-WEIGHTED GLOBAL-MEAN growth rate as a shaded band behind
    every panel, so a station can be read against the global spread.

The same `overlay_panels` routine regenerates the original Figure 4 (city
points of interest) with the global band added -- just pass the city
coordinates instead of the station table.

------------------------------------------------------------------------
EXPECTED INPUTS  (adapt the two loaders to the real files)
------------------------------------------------------------------------
TCCON growth rates -- long-format table (CSV / xlsx), one row per
station-year, with columns (rename via `cols=` if yours differ):
    station | lat | lon | year | growth | growth_err
    Karlsruhe | 49.10 | 8.44 | 2016 | 2.61 | 0.20
Short and long records are both fine; missing years are simply absent.

CAMS DSg -- gridded annual growth rate as an xarray DataArray with dims
(year, lat, lon), ppm/yr, ocean = NaN. (Same object used elsewhere in the
pipeline; load your NetCDF with xr.open_dataarray / open_dataset[var].)
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# loaders  (edit these two to point at the real files)
# ----------------------------------------------------------------------
def load_tccon_table(path: str, cols: dict | None = None) -> pd.DataFrame:
    """Read the TCCON station growth-rate table into a canonical long frame."""
    cols = cols or {}
    reader = pd.read_excel if str(path).lower().endswith((".xls", ".xlsx")) \
        else pd.read_csv
    df = reader(path)
    rename = {cols.get("station", "station"): "station",
              cols.get("lat", "lat"): "lat",
              cols.get("lon", "lon"): "lon",
              cols.get("year", "year"): "year",
              cols.get("growth", "growth"): "growth",
              cols.get("growth_err", "growth_err"): "growth_err"}
    df = df.rename(columns=rename)
    if "growth_err" not in df:
        df["growth_err"] = np.nan
    keep = ["station", "lat", "lon", "year", "growth", "growth_err"]
    return df[keep].dropna(subset=["lat", "lon", "year", "growth"])


def load_dsg(path: str, var: str | None = None) -> xr.DataArray:
    """Load CAMS DSg gridded growth rate as (year, lat, lon)."""
    obj = xr.open_dataset(path)
    da = obj[var] if var else obj[list(obj.data_vars)[0]]
    return da.transpose("year", "lat", "lon")


# ----------------------------------------------------------------------
# spatial matching + global band
# ----------------------------------------------------------------------
def nearest_pixel_series(dsg: xr.DataArray, lat: float, lon: float
                         ) -> tuple[pd.Series, float, float]:
    """CAMS growth-rate series at the grid cell nearest (lat, lon).

    Handles either [-180,180] or [0,360] longitude conventions. Returns
    (series indexed by year, matched_lat, matched_lon)."""
    glat = dsg["lat"].values
    glon = dsg["lon"].values
    lon_q = lon % 360 if glon.max() > 180 else ((lon + 180) % 360) - 180
    i = int(np.abs(glat - lat).argmin())
    dlon = np.abs((glon - lon_q + 180) % 360 - 180)  # circular distance
    j = int(dlon.argmin())
    ts = dsg.isel(lat=i, lon=j).to_series()
    return ts, float(glat[i]), float(glon[j])


def global_mean_band(dsg: xr.DataArray, spread: str = "sd"
                     ) -> pd.DataFrame:
    """Area-weighted global-mean growth rate per year, with a spread band.

    spread='sd'  -> weighted mean +/- 1 weighted SD (default)
    spread='iqr' -> weighted 25th-75th percentile envelope
    Returns a frame indexed by year with columns mean, lo, hi."""
    w2d = np.cos(np.deg2rad(dsg["lat"].values))[:, None] * np.ones(
        (dsg["lat"].size, dsg["lon"].size))
    out = []
    for y in dsg["year"].values:
        v = dsg.sel(year=y).values.ravel()
        w = w2d.ravel()
        m = np.isfinite(v)
        v, w = v[m], w[m]
        wsum = w.sum()
        mean = np.sum(w * v) / wsum
        if spread == "iqr":
            order = np.argsort(v)
            cw = np.cumsum(w[order]) / wsum
            lo = np.interp(0.25, cw, v[order])
            hi = np.interp(0.75, cw, v[order])
        else:
            var = np.sum(w * (v - mean) ** 2) / wsum
            sd = np.sqrt(var)
            lo, hi = mean - sd, mean + sd
        out.append((int(y), mean, lo, hi))
    return pd.DataFrame(out, columns=["year", "mean", "lo", "hi"]).set_index("year")


# ----------------------------------------------------------------------
# the figure
# ----------------------------------------------------------------------
def overlay_panels(points: pd.DataFrame, dsg: xr.DataArray,
                   band: pd.DataFrame | None = None, ncols: int = 3,
                   title: str = "", share_y: bool = True, save: str | None = None):
    """One panel per point: global band (grey), CAMS-at-pixel (red dashed),
    measured series (black, with error bars where available).

    `points` is a long frame with columns
    station, lat, lon[, year, growth, growth_err]. If year/growth are
    present they are drawn as the measured (TCCON) series; if absent the
    panel shows only CAMS vs the global band (use this for the city POIs)."""
    if band is None:
        band = global_mean_band(dsg)
    stations = list(dict.fromkeys(points["station"]))
    n = len(stations)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 2.8 * nrows),
                             squeeze=False, sharey=share_y)
    yrs = band.index.values
    for ax, st in zip(axes.ravel(), stations):
        sub = points[points["station"] == st]
        lat, lon = float(sub["lat"].iloc[0]), float(sub["lon"].iloc[0])
        ax.fill_between(yrs, band["lo"], band["hi"], color="0.8", alpha=0.7,
                        lw=0, label="global mean $\\pm$1 SD")
        ax.plot(yrs, band["mean"], color="0.55", lw=1.0, zorder=2)
        cams, mlat, mlon = nearest_pixel_series(dsg, lat, lon)
        ax.plot(cams.index, cams.values, "--o", color="tab:red", ms=3, lw=1.2,
                label="CAMS (pixel)", zorder=4)
        has_meas = {"year", "growth"}.issubset(sub.columns) and sub["growth"].notna().any()
        if has_meas:
            s = sub.sort_values("year")
            ax.errorbar(s["year"], s["growth"], yerr=s["growth_err"],
                        fmt="-o", color="k", ms=3, lw=1.2, capsize=2,
                        label="TCCON", zorder=5)
            nlab = f"n={s['growth'].notna().sum()}"
        else:
            nlab = ""
        ax.set_title(f"{st}  ({lat:.1f}, {lon:.1f})  {nlab}", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.margins(x=0.02)
    for ax in axes.ravel()[n:]:
        ax.set_visible(False)
    axes[0, 0].legend(fontsize=6, loc="upper left", framealpha=0.9)
    for ax in axes[-1, :]:
        ax.set_xlabel("year", fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel("growth rate (ppm/yr)", fontsize=8)
    fig.suptitle(title or "CAMS pixel vs measured growth rate", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight")
    return fig


# ----------------------------------------------------------------------
# synthetic demo -> renders the exact layout without the real data
# ----------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(1)
    years = np.arange(2015, 2025)
    lat = np.arange(-89.5, 90, 1.0)
    lon = np.arange(-179.5, 180, 1.0)
    soi = rng.normal(size=len(years))
    base = 2.5 + 0.9 * soi
    grid = (base[:, None, None]
            + 0.15 * np.sin(np.deg2rad(lat))[None, :, None]
            + rng.normal(scale=0.25, size=(len(years), lat.size, lon.size)))
    land = rng.random((lat.size, lon.size)) < 0.3
    grid = np.where(land, grid, np.nan)
    dsg = xr.DataArray(grid, coords={"year": years, "lat": lat, "lon": lon},
                       dims=("year", "lat", "lon"))

    # a handful of real TCCON-like sites, some long records, some short
    sites = [("Karlsruhe", 49.1, 8.4, 2015), ("Lamont", 36.6, -97.5, 2015),
             ("Bialystok", 53.2, 23.0, 2015), ("Wollongong", -34.4, 150.9, 2016),
             ("Sodankyla", 67.4, 26.6, 2017), ("Darwin", -12.4, 130.9, 2015),
             ("Tsukuba", 36.0, 140.1, 2018), ("ParkFalls", 45.9, -90.3, 2015),
             ("Reunion", -21.0, 55.5, 2020)]
    rows = []
    for name, la, lo, y0 in sites:
        cams, _, _ = nearest_pixel_series(dsg, la, lo)
        for y in range(y0, 2025):
            g = float(cams.loc[y]) + rng.normal(scale=0.2)  # measured ~ near pixel
            rows.append((name, la, lo, y, g, 0.2))
    tccon = pd.DataFrame(rows, columns=["station", "lat", "lon", "year",
                                        "growth", "growth_err"])

    band = global_mean_band(dsg)
    overlay_panels(tccon, dsg, band=band,
                   title="Figure 4 (ground-based variant): CAMS pixel vs TCCON",
                   save="fig4_station_variant_DEMO.png")
    print("global-mean band (demo):")
    print(band.round(2))
    print("wrote fig4_station_variant_DEMO.png")
