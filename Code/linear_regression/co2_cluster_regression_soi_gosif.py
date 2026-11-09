#!/usr/bin/env python3
"""
Cluster-level (persistent co-occurrence components) regression of the CO2
growth rate, and the per-cluster predictor-contribution figure (Fig. 7, Main).

For every persistent cluster c (C1-C5) and year t the growth rate, emission and
GOSIF are averaged (unweighted) over the grid cells that belong to c in year t,
and two OLS models are fitted to the 10-year cluster-mean time series:

    Model SOI   : y_c(t) = b0 + bE*E_z^c(t) + bSOI *SOI_z(t)
    Model GOSIF : y_c(t) = b0 + bE*E_z^c(t) + bSIF *GOSIF_z^c(t)

Predictors are standardised per cluster over the years present in the cluster CSV 

The figure uses the same style as the city-contribution figure (Fig. 4, Main,
produced by co2_gridcell_regression_soi_gosif.py):
  * grey band  : area-weighted global land mean +/- 1 SD of the growth rate
  * black line : observed cluster-mean growth rate with propagated error bars
  * red dashed : model prediction (beta0 included)
  * bars       : driver contributions beta_E*E_z and beta_2*P_z (beta0 omitted)

ERROR PROPAGATION
-----------------
Each cell carries a 1-sigma uncertainty `growth_err` (sigma_i). CAMS errors are
strongly correlated between neighbouring cells, so the error of the cluster
mean  y_bar = (1/N) sum_i y_i  is propagated assuming fully correlated cell
errors:

    sigma(y_bar) = (1/N) sum_i sigma_i

Cells with a missing growth_err are excluded, and N is then the number of
cells with a finite error.

Repository layout:
  Reads   Input/unified_annual_carbon_dataset_2015_2025.nc
          Input/SOI.txt
          Output/clustering/co-occurrence-clusters.csv  (Output/)
  Writes  Figures_and_maps/cluster_contribution_5x2.pdf          (Fig. 7, Main)
          Output/cluster_regression_coefficients.csv
          Output/cluster_mean_timeseries.csv

"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================== CONFIG ===================================

SOI_LAG_MONTHS = 7
MIN_OBS = 6

CLUSTERS = [                      # column in the CSV, label, regime (Table 2)
    ("component_1", "C1", "carbon inactive"),
    ("component_2", "C2", "carbon limited"),
    ("component_3", "C3", "carbon transition"),
    ("component_4", "C4", "carbon source"),
    ("component_5", "C5", "carbon sink"),
]

MODELS = {                        # key: (column title, 2nd-predictor tex, bar colour)
    "soi":   ("Model SOI: emission + SOI",
              r"\beta_{SOI}", r"$\beta_{SOI}\,SOI_z$", "tab:blue"),
    "gosif": ("Model GOSIF: emission + GOSIF",
              r"\beta_{SIF}", r"$\beta_{SIF}\,SIF_z$", "tab:green"),
}
EMISSION_COLOR = "tab:orange"     # same colours as Fig. 4 / Supp. Fig. 6

plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 11,
    "pdf.fonttype": 42,           # editable text in vector output
    "ps.fonttype": 42,
})


# ============================== PATHS ====================================

# Gridded dataset: the same file used by co2_gridcell_regression_soi_gosif.py.
# Only the years present in the cluster CSV are analysed.
DATA_NAME = "unified_annual_carbon_dataset_2015_2025.nc"
SOI_NAME = "SOI.txt"
CLUSTER_NAME = "co-occurrence-clusters.csv"

# Input and output locations, relative to the root of the complexity72_2026
# clone. The cluster CSV is in Output/clustering/ (Output/ in old clones).
INPUTS = ([f"Input/{DATA_NAME}"],
          [f"Input/{SOI_NAME}"],
          [f"Output/clustering/{CLUSTER_NAME}", f"Output/{CLUSTER_NAME}"])
FIG_SUBDIR = "Figures_and_maps"
OUT_SUBDIR = "Output"


def resolve_inputs(folder: Path):
    """First existing candidate for each input, or None if any is missing."""
    found = []
    for candidates in INPUTS:
        hit = next((folder / c for c in candidates if (folder / c).is_file()), None)
        if hit is None:
            return None
        found.append(hit)
    return found


def find_root(explicit=None):
    """
    Locate the root of the complexity72_2026 clone.

    Order: --root argument, then the script's folder and its parents, then the
    current working directory and its parents. A candidate is accepted only if
    it actually contains the three input files (an unrelated .git higher up,
    e.g. in $HOME, is never mistaken for the repository).
    """
    if explicit is not None:
        root = Path(explicit).expanduser().resolve()
        if resolve_inputs(root) is None:
            raise FileNotFoundError(
                f"--root {root} does not contain Input/{DATA_NAME}, "
                f"Input/{SOI_NAME} and Output/clustering/{CLUSTER_NAME}.")
        return root

    candidates = []
    for start in (Path(__file__).resolve().parent, Path.cwd().resolve()):
        candidates += [start, *start.parents]
    for folder in candidates:
        if resolve_inputs(folder) is not None:
            return folder

    raise FileNotFoundError(
        f"Could not find the repository root (a folder containing "
        f"Input/{DATA_NAME}, Input/{SOI_NAME} and "
        f"Output/clustering/{CLUSTER_NAME}).\n"
        "Place this script in Code/linear_regression/ of your clone of "
        "complexity72_2026, or pass the clone explicitly:\n"
        "    python co2_cluster_regression_soi_gosif.py --root /path/to/complexity72_2026"
    )


def set_paths(root: Path):
    """Define all input/output paths relative to the repository root."""
    global REPO_ROOT, INPUT_FILE, SOI_FILE, CLUSTER_FILE, FIGDIR, OUTDIR
    REPO_ROOT = root
    INPUT_FILE, SOI_FILE, CLUSTER_FILE = resolve_inputs(root)
    FIGDIR = root / FIG_SUBDIR
    OUTDIR = root / OUT_SUBDIR


# ============================== HELPERS ==================================

def read_standardized_soi(path):
    """Monthly SOI from the STANDARDIZED block of the NOAA/CPC file."""
    text = Path(path).read_text(errors="ignore")
    if "STANDARDIZED" not in text:
        raise ValueError("No STANDARDIZED section in SOI file.")
    rows = []
    for line in text.split("STANDARDIZED", 1)[1].splitlines():
        nums = re.findall(r"-?\d+(?:\.\d+)?", line)
        if len(nums) < 13:
            continue
        year = int(float(nums[0]))
        if not 1800 <= year <= 2200:
            continue
        for month, value in enumerate(map(float, nums[1:13]), start=1):
            rows.append({"year": year, "month": month,
                         "soi": np.nan if value <= -900 else value})
    return pd.DataFrame(rows)


def annual_lag_soi(soi_monthly, lag=SOI_LAG_MONTHS, min_months=12):
    """Shift monthly SOI forward by `lag` months, then average per year."""
    df = soi_monthly.dropna(subset=["soi"]).copy()
    date = pd.to_datetime(dict(year=df.year, month=df.month, day=1))
    df["co2_year"] = (date + pd.DateOffset(months=lag)).dt.year
    g = df.groupby("co2_year")["soi"]
    return g.mean()[g.count() >= min_months]


def zscore(a):
    sd = np.nanstd(a)
    return (a - np.nanmean(a)) / sd if sd > 1e-12 else np.zeros_like(a)


def build_land_mask(lat, lon):
    """Same land mask as co2_gridcell_regression_soi_gosif.py (Fig. 4 band)."""
    LON, LAT = np.meshgrid(lon, lat)
    LONq = np.where(LON > 180, LON - 360, LON)
    try:
        from global_land_mask import globe
        return globe.is_land(LAT, LONq)
    except ImportError:
        import cartopy.io.shapereader as shp
        from shapely.geometry import Point
        from shapely.ops import unary_union
        from shapely.prepared import prep
        geom = prep(unary_union(list(shp.Reader(shp.natural_earth(
            resolution="110m", category="physical", name="land")).geometries())))
        flat = np.array([geom.contains(Point(x, y))
                         for x, y in zip(LONq.ravel(), LAT.ravel())])
        return flat.reshape(LAT.shape)


def global_band(field, latitudes):
    """Area-weighted (cos lat) spatial mean +/- 1 SD for each year."""
    w2d = np.cos(np.deg2rad(latitudes))[:, None]
    mean = np.full(field.shape[0], np.nan)
    sd = np.full(field.shape[0], np.nan)
    for t in range(field.shape[0]):
        valid = np.isfinite(field[t])
        w = np.broadcast_to(w2d, field[t].shape)[valid]
        v = field[t][valid]
        mean[t] = (w * v).sum() / w.sum()
        sd[t] = np.sqrt((w * (v - mean[t]) ** 2).sum() / w.sum())
    return mean, mean - sd, mean + sd


def fit_ols(y, x1, x2, min_obs=MIN_OBS):
    """OLS y ~ b0 + b1*x1 + b2*x2 with in-sample and exact LOO R^2."""
    ok = np.isfinite(y) & np.isfinite(x1) & np.isfinite(x2)
    out = dict(beta0=np.nan, beta1=np.nan, beta2=np.nan,
               r2=np.nan, loo_r2=np.nan, n_obs=int(ok.sum()))
    pred = np.full_like(y, np.nan)
    if ok.sum() < min_obs:
        return out, pred
    X = np.column_stack([np.ones(ok.sum()), x1[ok], x2[ok]])
    yy = y[ok]
    Xp = np.linalg.pinv(X)
    beta = Xp @ yy
    fit = X @ beta
    res = yy - fit
    ss_tot = np.sum((yy - yy.mean()) ** 2)
    lev = np.sum(X * Xp.T, axis=1)                 # hat-matrix diagonal
    loo_res = res / (1.0 - lev)                    # exact LOO residuals
    out.update(beta0=beta[0], beta1=beta[1], beta2=beta[2],
               r2=1 - np.sum(res ** 2) / ss_tot,
               loo_r2=1 - np.sum(loo_res ** 2) / ss_tot)
    pred[ok] = fit
    return out, pred


def stacked_signed_bar(ax, x, comps, labels, colors, width=0.6):
    """Positive and negative parts stacked separately (as in Fig. 4)."""
    pos_base = np.zeros(len(x))
    neg_base = np.zeros(len(x))
    for comp, label, fc in zip(comps, labels, colors):
        comp = np.asarray(comp, dtype=float)
        pos = np.where(comp > 0, comp, 0.0)
        neg = np.where(comp < 0, comp, 0.0)
        ax.bar(x, pos, bottom=pos_base, width=width, color=fc, label=label,
               edgecolor="0.3", linewidth=0.6)
        ax.bar(x, neg, bottom=neg_base, width=width, color=fc,
               edgecolor="0.3", linewidth=0.6)
        pos_base += pos
        neg_base += neg


# ============================== MAIN =====================================

def main():
    FIGDIR.mkdir(parents=True, exist_ok=True)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    print(f"REPO_ROOT = {REPO_ROOT}")
    for f in (INPUT_FILE, SOI_FILE, CLUSTER_FILE):
        print(f"  reads   {f.relative_to(REPO_ROOT)}")

    # ---------------- cluster labels ----------------
    # One row per (lat, lon, year) with 0/1 membership flags; pixels outside
    # every component (e.g. ocean) are all-zero and simply never selected.
    cl = pd.read_csv(CLUSTER_FILE)
    missing_cols = [col for col, _, _ in CLUSTERS if col not in cl.columns]
    if missing_cols:
        raise ValueError(f"{CLUSTER_FILE.name} lacks columns {missing_cols}")
    years = np.sort(cl["time"].unique()).astype(int)

    # ---------------- gridded data (restricted to those years) ----------------
    # Growth rate and its error are taken from the unified NetCDF, which is the
    # same source used for the grid-cell regression and for Fig. 4.
    ds = xr.open_dataset(INPUT_FILE)
    nc_years = ds["time"].values.astype(int)
    if not np.isin(years, nc_years).all():
        raise ValueError(f"Years {sorted(set(years) - set(nc_years))} of the "
                         f"cluster CSV are not in {INPUT_FILE.name}.")
    ds = ds.sel(time=years)
    lat, lon = ds["lat"].values, ds["lon"].values
    grid = {v: ds[v].transpose("time", "lat", "lon").values.astype("float64")
            for v in ("growth_rate", "growth_err", "emission", "gosif")}

    # ---------------- cluster labels → long table ----------------
    ilat = {v: i for i, v in enumerate(np.round(lat, 3))}
    ilon = {v: i for i, v in enumerate(np.round(lon, 3))}
    ti = cl["time"].map({y: k for k, y in enumerate(years)}).to_numpy()
    ii = cl["lat"].round(3).map(ilat).to_numpy()
    jj = cl["lon"].round(3).map(ilon).to_numpy()
    if np.isnan(ti.astype(float)).any() or pd.isna(ii).any() or pd.isna(jj).any():
        raise ValueError("Cluster CSV contains coordinates/years not in the NetCDF.")
    for v in grid:
        cl[v] = grid[v][ti, ii, jj]

    # ---------------- cluster-year means + propagated error ----------------
    rows = []
    for col, lab, _ in CLUSTERS:
        for yr, g in cl[cl[col] == 1].groupby("time"):
            gr = g["growth_rate"].dropna()
            err = g.loc[g["growth_rate"].notna(), "growth_err"].dropna()
            n_err = len(err)
            sig = err.mean() if n_err else np.nan      # fully correlated errors
            rows.append(dict(cluster=lab, year=int(yr),
                             n_cells=len(gr), n_cells_with_err=n_err,
                             growth_rate=gr.mean(),
                             growth_err=sig,
                             emission=g["emission"].mean(),
                             gosif=g["gosif"].mean()))
    ts = pd.DataFrame(rows)
    ts.to_csv(OUTDIR / "cluster_mean_timeseries.csv", index=False)

    # ---------------- SOI ----------------
    soi_ann = annual_lag_soi(read_standardized_soi(SOI_FILE))
    soi_z = zscore(np.array([soi_ann.loc[y] for y in years], dtype=float))

    # ---------------- regressions ----------------
    fits, coef_rows = {}, []
    for _, lab, _ in CLUSTERS:
        d = ts[ts.cluster == lab].set_index("year").reindex(years)
        y = d["growth_rate"].to_numpy()
        e_z = zscore(d["emission"].to_numpy())
        p2 = {"soi": soi_z, "gosif": zscore(d["gosif"].to_numpy())}
        for m in MODELS:
            res, pred = fit_ols(y, e_z, p2[m])
            fits[(lab, m)] = dict(res=res, pred=pred, y=y,
                                  err=d["growth_err"].to_numpy(),
                                  c1=res["beta1"] * e_z,
                                  c2=res["beta2"] * p2[m])
            coef_rows.append(dict(cluster=lab, model=m, **res))
    coefs = pd.DataFrame(coef_rows)
    coefs.to_csv(OUTDIR / "cluster_regression_coefficients.csv", index=False)
    print(coefs.round(3).to_string(index=False))

    # ---------------- global land band (identical to Fig. 4) ----------------
    land = build_land_mask(lat, lon)
    gr_land = np.where(land[None], grid["growth_rate"], np.nan)
    g_mean, g_lo, g_hi = global_band(gr_land, lat)

    # ---------------- figure: rows = clusters, cols = models ----------------
    nrows, ncols = len(CLUSTERS), len(MODELS)
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4.4 * nrows),
                             sharex=True, squeeze=False)
    err_label = "CAMS cluster mean $\\pm$ propagated growth_err"
    handles = {}

    for r, (_, lab, regime) in enumerate(CLUSTERS):
        for c, (m, (col_title, b2_tex, c2_label, c2_color)) in enumerate(MODELS.items()):
            ax = axes[r, c]
            f = fits[(lab, m)]
            res = f["res"]

            ax.fill_between(years, g_lo, g_hi, color="0.75", alpha=0.45,
                            label="Global mean $\\pm$ 1 SD", zorder=0)
            ax.plot(years, g_mean, color="0.45", lw=1.4, zorder=1)

            stacked_signed_bar(ax, years, [f["c1"], f["c2"]],
                               [r"$\beta_E\,E_z$", c2_label],
                               [EMISSION_COLOR, c2_color])

            ax.errorbar(years, f["y"], yerr=f["err"], fmt="k-o", lw=2.0,
                        ms=4.8, capsize=3.0, elinewidth=1.0,
                        label=err_label, zorder=7)
            ax.plot(years, f["pred"], "--s", color="tab:red", lw=1.8, ms=4.2,
                    label="predicted", zorder=7)
            ax.axhline(0, color="black", lw=0.8)

            ax.set_title(
                (f"{col_title}\n" if r == 0 else "")
                + f"{lab} ({regime})\n"
                + rf"$\beta_0$={res['beta0']:.2f}, $\beta_E$={res['beta1']:.2f}, "
                + rf"${b2_tex}$={res['beta2']:.2f}, "
                + rf"$R^2$={res['r2']:.2f}, LOO $R^2$={res['loo_r2']:.2f}",
                fontsize=10.5, fontweight="bold")
            ax.grid(True, axis="y", alpha=0.25)
            ax.set_xticks(years)
            ax.tick_params(axis="x", rotation=45, labelsize=9)

            for h, l in zip(*ax.get_legend_handles_labels()):
                handles.setdefault(l, h)

    for ax in axes[:, 0]:
        ax.set_ylabel("growth rate / contribution (ppm/yr)")
    for ax in axes[-1, :]:
        ax.set_xlabel("year")

    order = ["Global mean $\\pm$ 1 SD", "predicted", r"$\beta_E\,E_z$",
             MODELS["soi"][2], MODELS["gosif"][2], err_label]
    fig.legend([handles[k] for k in order], order, loc="lower center",
               ncol=3, fontsize=11, bbox_to_anchor=(0.5, -0.005))
    fig.tight_layout(rect=[0, 0.035, 1, 0.995])

    out = FIGDIR / f"cluster_contribution_{nrows}x{ncols}.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("saved", out.relative_to(REPO_ROOT))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--root", default=None,
                    help="path to the complexity72_2026 clone "
                         "(default: found automatically)")
    args = ap.parse_args()
    set_paths(find_root(args.root))
    main()