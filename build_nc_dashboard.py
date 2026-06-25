from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr


def detect_spatial_dims(da: xr.DataArray) -> tuple[str, str]:
    candidates = [
        ("lat", "lon"),
        ("latitude", "longitude"),
        ("y", "x"),
    ]
    for lat_dim, lon_dim in candidates:
        if lat_dim in da.dims and lon_dim in da.dims:
            return lat_dim, lon_dim
    raise ValueError(f"Unsupported spatial dims for {da.name}: {da.dims}")


def time_to_year_label(t: Any) -> str:
    t_arr = np.array(t)
    if np.issubdtype(t_arr.dtype, np.integer):
        return str(int(t))

    try:
        s = np.datetime_as_string(np.datetime64(t), unit="D")
    except Exception:
        s = str(t)

    if len(s) >= 4 and s[:4].isdigit():
        return s[:4]
    return s


def extract_year_points(
    da: xr.DataArray,
    time_index: int,
    max_points: int,
    rng: np.random.Generator,
) -> dict[str, Any] | None:
    lat_dim, lon_dim = detect_spatial_dims(da)

    da_t = da.isel(time=time_index)
    try:
        da_t = da_t.transpose(lat_dim, lon_dim)
    except ValueError:
        return None

    da_t = da_t.squeeze(drop=True)
    if da_t.ndim != 2:
        return None

    lat_vals = np.asarray(da_t[lat_dim].values)
    lon_vals = np.asarray(da_t[lon_dim].values)
    vals = np.asarray(da_t.values, dtype=np.float64)

    lon_grid, lat_grid = np.meshgrid(lon_vals, lat_vals)
    mask = np.isfinite(vals)
    if not np.any(mask):
        return {
            "lat": [],
            "lon": [],
            "val": [],
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "source_count": 0,
            "sample_count": 0,
        }

    lat_flat = lat_grid[mask].astype(np.float32)
    lon_flat = lon_grid[mask].astype(np.float32)
    val_flat = vals[mask].astype(np.float32)

    source_count = int(val_flat.size)
    if val_flat.size > max_points:
        idx = rng.choice(val_flat.size, size=max_points, replace=False)
        lat_flat = lat_flat[idx]
        lon_flat = lon_flat[idx]
        val_flat = val_flat[idx]

    return {
        "lat": np.round(lat_flat, 4).tolist(),
        "lon": np.round(lon_flat, 4).tolist(),
        "val": np.round(val_flat, 6).tolist(),
        "min": float(np.nanmin(val_flat)),
        "max": float(np.nanmax(val_flat)),
        "mean": float(np.nanmean(val_flat)),
        "source_count": source_count,
        "sample_count": int(val_flat.size),
    }


def build_payload(ds: xr.Dataset, max_points: int, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)

    payload: dict[str, Any] = {
        "meta": {
            "source": "xarray netCDF",
            "max_points_per_variable_year": int(max_points),
        },
        "variables": {},
    }

    for var_name, da in ds.data_vars.items():
        if "time" not in da.dims:
            continue

        try:
            lat_dim, lon_dim = detect_spatial_dims(da)
        except ValueError:
            continue

        years: list[str] = []
        seen_years: set[str] = set()
        cells: dict[str, dict[str, Any]] = {}

        for time_idx, t in enumerate(da["time"].values):
            year = time_to_year_label(t)
            if year in seen_years:
                continue

            cell = extract_year_points(da, time_idx, max_points=max_points, rng=rng)
            if cell is None:
                continue

            years.append(year)
            seen_years.add(year)
            cells[year] = cell

        if not years:
            continue

        trend: list[dict[str, Any]] = []
        try:
            mean_series = da.mean(dim=[lat_dim, lon_dim], skipna=True)
            for time_idx, t in enumerate(da["time"].values):
                year = time_to_year_label(t)
                if year not in cells:
                    continue
                val = mean_series.isel(time=time_idx).item()
                mean_val = float(val) if np.isfinite(val) else None
                trend.append({"year": year, "mean": mean_val})
        except Exception:
            trend = [{"year": y, "mean": None} for y in years]

        payload["variables"][var_name] = {
            "lat_dim": lat_dim,
            "lon_dim": lon_dim,
            "years": years,
            "cells": cells,
            "trend": trend,
        }

    payload["meta"]["variable_names"] = list(payload["variables"].keys())
    payload["meta"]["variable_count"] = len(payload["variables"])
    return payload


def render_html(payload: dict[str, Any], title: str) -> str:
    data_json = json.dumps(payload, separators=(",", ":"))

    html = """<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>__TITLE__</title>
  <script src=\"https://cdn.plot.ly/plotly-2.35.2.min.js\"></script>
  <style>
    :root {
      --bg: #f4f7fb;
      --panel: #ffffff;
      --ink: #11243a;
      --muted: #516476;
      --accent: #146c94;
      --border: #d6e0ea;
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      font-family: \"Segoe UI\", Tahoma, Geneva, Verdana, sans-serif;
      color: var(--ink);
      background:
        radial-gradient(1200px 600px at 15% -20%, #d4e8ff 0%, rgba(212, 232, 255, 0) 60%),
        radial-gradient(900px 500px at 95% 0%, #c6f0dd 0%, rgba(198, 240, 221, 0) 58%),
        var(--bg);
      min-height: 100vh;
    }

    .container {
      max-width: 1300px;
      margin: 0 auto;
      padding: 16px;
    }

    .header {
      display: grid;
      gap: 8px;
      margin-bottom: 12px;
    }

    .title {
      margin: 0;
      font-size: 1.5rem;
      line-height: 1.2;
    }

    .subtitle {
      margin: 0;
      color: var(--muted);
      font-size: 0.95rem;
    }

    .controls {
      display: grid;
      grid-template-columns: repeat(3, minmax(180px, 1fr));
      gap: 12px;
      padding: 12px;
      border: 1px solid var(--border);
      border-radius: 12px;
      background: var(--panel);
      margin-bottom: 12px;
    }

    .control {
      display: grid;
      gap: 6px;
      align-content: start;
    }

    label {
      font-size: 0.85rem;
      color: var(--muted);
    }

    select,
    input[type=\"range\"] {
      width: 100%;
    }

    .dashboard {
      display: grid;
      grid-template-columns: 2fr 1fr;
      gap: 12px;
    }

    .panel {
      border: 1px solid var(--border);
      border-radius: 12px;
      background: var(--panel);
      overflow: hidden;
      min-height: 280px;
    }

    .stats {
      padding: 10px 12px;
      font-size: 0.9rem;
      color: var(--muted);
      border-bottom: 1px solid var(--border);
      background: #f8fbff;
    }

    .plot {
      width: 100%;
      height: 100%;
      min-height: 260px;
    }

    .right-column {
      display: grid;
      gap: 12px;
    }

    @media (max-width: 980px) {
      .dashboard {
        grid-template-columns: 1fr;
      }
      .controls {
        grid-template-columns: 1fr;
      }
    }
  </style>
</head>
<body>
  <div class=\"container\">
    <header class=\"header\">
      <h1 class=\"title\">__TITLE__</h1>
      <p class=\"subtitle\">Interactive xarray dashboard from netCDF. Choose variable and year to update all plots.</p>
    </header>

    <section class=\"controls\">
      <div class=\"control\">
        <label for=\"variableSelect\">Data Variable</label>
        <select id=\"variableSelect\"></select>
      </div>
      <div class=\"control\">
        <label for=\"yearSelect\">Year</label>
        <select id=\"yearSelect\"></select>
      </div>
      <div class=\"control\">
        <label for=\"yearSlider\">Year Slider</label>
        <input id=\"yearSlider\" type=\"range\" min=\"0\" max=\"0\" step=\"1\" value=\"0\" />
      </div>
    </section>

    <section class=\"dashboard\">
      <div class=\"panel\">
        <div id=\"stats\" class=\"stats\"></div>
        <div id=\"mapPlot\" class=\"plot\"></div>
      </div>
      <div class=\"right-column\">
        <div class=\"panel\"><div id=\"histPlot\" class=\"plot\"></div></div>
        <div class=\"panel\"><div id=\"trendPlot\" class=\"plot\"></div></div>
      </div>
    </section>
  </div>

  <script>
    const DASHBOARD_DATA = __DATA_JSON__;

    const varSelect = document.getElementById("variableSelect");
    const yearSelect = document.getElementById("yearSelect");
    const yearSlider = document.getElementById("yearSlider");
    const statsEl = document.getElementById("stats");

    function variableNames() {
      return Object.keys(DASHBOARD_DATA.variables);
    }

    function updateYearOptions(selectedVariable) {
      const years = DASHBOARD_DATA.variables[selectedVariable].years;

      yearSelect.innerHTML = "";
      years.forEach((year, i) => {
        const opt = document.createElement("option");
        opt.value = year;
        opt.textContent = year;
        opt.dataset.index = String(i);
        yearSelect.appendChild(opt);
      });

      yearSlider.min = "0";
      yearSlider.max = String(Math.max(0, years.length - 1));
      yearSlider.value = "0";
      yearSelect.selectedIndex = 0;
    }

    function currentSelection() {
      const variable = varSelect.value;
      const year = yearSelect.value;
      const variableData = DASHBOARD_DATA.variables[variable];
      const cell = variableData.cells[year];
      return { variable, year, variableData, cell };
    }

    function renderAll() {
      const { variable, year, variableData, cell } = currentSelection();

      statsEl.textContent =
        `${variable} | ${year} | sampled ${cell.sample_count.toLocaleString()} of ${cell.source_count.toLocaleString()} points | min ${cell.min.toFixed(4)} | max ${cell.max.toFixed(4)} | mean ${cell.mean.toFixed(4)}`;

      const mapTrace = {
        type: "scattergeo",
        mode: "markers",
        lat: cell.lat,
        lon: cell.lon,
        marker: {
          size: 4,
          opacity: 0.8,
          color: cell.val,
          colorscale: "Turbo",
          showscale: true,
          colorbar: { title: variable }
        },
        hovertemplate: "lat=%{lat:.2f}<br>lon=%{lon:.2f}<br>value=%{marker.color:.4f}<extra></extra>"
      };

      const mapLayout = {
        margin: { t: 8, r: 8, b: 8, l: 8 },
        geo: {
          projection: { type: "equirectangular" },
          showland: true,
          landcolor: "#f1f4f7",
          showocean: true,
          oceancolor: "#ddebf7",
          showcountries: true,
          countrycolor: "#b0bac4"
        }
      };

      Plotly.react("mapPlot", [mapTrace], mapLayout, { responsive: true, displaylogo: false });

      const histTrace = {
        type: "histogram",
        x: cell.val,
        nbinsx: 50,
        marker: { color: "#146c94", line: { color: "#0e4f6d", width: 1 } },
        opacity: 0.9,
      };

      Plotly.react(
        "histPlot",
        [histTrace],
        {
          title: `Value Distribution: ${variable} (${year})`,
          margin: { t: 40, r: 12, b: 40, l: 48 },
          xaxis: { title: "Value" },
          yaxis: { title: "Count" },
        },
        { responsive: true, displaylogo: false }
      );

      const trendYears = variableData.trend.map(d => d.year);
      const trendVals = variableData.trend.map(d => d.mean);
      const idx = trendYears.indexOf(year);

      const trendTrace = {
        type: "scatter",
        mode: "lines+markers",
        x: trendYears,
        y: trendVals,
        line: { color: "#23395d", width: 2 },
        marker: { size: 7, color: "#23395d" },
        name: "Spatial mean"
      };

      const highlightTrace = {
        type: "scatter",
        mode: "markers",
        x: idx >= 0 ? [trendYears[idx]] : [],
        y: idx >= 0 ? [trendVals[idx]] : [],
        marker: { size: 12, color: "#d65a31", symbol: "diamond" },
        name: "Selected year"
      };

      Plotly.react(
        "trendPlot",
        [trendTrace, highlightTrace],
        {
          title: `Spatial Mean by Year: ${variable}`,
          margin: { t: 40, r: 12, b: 40, l: 48 },
          xaxis: { title: "Year" },
          yaxis: { title: "Mean value" },
          showlegend: true,
          legend: { orientation: "h" }
        },
        { responsive: true, displaylogo: false }
      );
    }

    function init() {
      const vars = variableNames();
      if (vars.length === 0) {
        statsEl.textContent = "No time + lat/lon variables found in the dataset.";
        return;
      }

      varSelect.innerHTML = "";
      vars.forEach((name, i) => {
        const opt = document.createElement("option");
        opt.value = name;
        opt.textContent = name;
        if (i === 0) opt.selected = true;
        varSelect.appendChild(opt);
      });

      updateYearOptions(varSelect.value);
      renderAll();

      varSelect.addEventListener("change", () => {
        updateYearOptions(varSelect.value);
        renderAll();
      });

      yearSelect.addEventListener("change", () => {
        yearSlider.value = String(yearSelect.selectedIndex);
        renderAll();
      });

      yearSlider.addEventListener("input", () => {
        yearSelect.selectedIndex = Number(yearSlider.value);
        renderAll();
      });
    }

    init();
  </script>
</body>
</html>
"""
    return html.replace("__DATA_JSON__", data_json).replace("__TITLE__", title)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build an interactive HTML dashboard from a netCDF file using xarray."
    )
    parser.add_argument(
        "--nc",
        type=Path,
        default=Path("unified_annual_carbon_dataset_2015_2024.nc"),
        help="Path to netCDF file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("nc_interactive_dashboard.html"),
        help="Output HTML file path",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=4000,
        help="Maximum sampled points per variable-year to keep HTML responsive",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Sampling seed for reproducible dashboards",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="NetCDF Carbon Data Dashboard",
        help="Dashboard title",
    )

    args = parser.parse_args()

    if args.max_points < 200:
        raise ValueError("--max-points must be at least 200")

    ds = xr.open_dataset(args.nc)
    payload = build_payload(ds, max_points=args.max_points, seed=args.seed)

    if payload["meta"]["variable_count"] == 0:
        raise RuntimeError(
            "No compatible variables found. Need variables with time + (lat/lon or latitude/longitude)."
        )

    html = render_html(payload=payload, title=args.title)
    args.output.write_text(html, encoding="utf-8")

    print(f"Dashboard written to: {args.output}")
    print(f"Variables included: {payload['meta']['variable_count']}")


if __name__ == "__main__":
    main()
