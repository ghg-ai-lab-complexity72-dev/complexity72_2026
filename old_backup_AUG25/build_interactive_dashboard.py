from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import xarray as xr


def detect_spatial_dims(da: xr.DataArray) -> tuple[str, str]:
    if "lat" in da.dims and "lon" in da.dims:
        return "lat", "lon"
    if "latitude" in da.dims and "longitude" in da.dims:
        return "latitude", "longitude"
    raise ValueError(f"Unsupported spatial dims for {da.name}: {da.dims}")


def time_to_label(t: object) -> str:
    t_arr = np.array(t)
    if np.issubdtype(t_arr.dtype, np.integer):
        return str(int(t))
    try:
        return np.datetime_as_string(np.datetime64(t), unit="D")
    except Exception:
        return str(t)


def sample_points(
    lat: np.ndarray,
    lon: np.ndarray,
    val: np.ndarray,
    max_points: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if val.size <= max_points:
        return lat, lon, val
    idx = rng.choice(val.size, size=max_points, replace=False)
    return lat[idx], lon[idx], val[idx]


def build_payload(
    ds: xr.Dataset,
    max_points: int = 25000,
    seed: int = 42,
) -> dict:
    rng = np.random.default_rng(seed)
    payload: dict[str, dict] = {"variables": {}, "meta": {}}

    years = [time_to_label(t) for t in ds["time"].values]
    payload["meta"]["years"] = years

    for var_name, da in ds.data_vars.items():
        if "time" not in da.dims:
            continue

        lat_dim, lon_dim = detect_spatial_dims(da)
        lat_vals = da[lat_dim].values
        lon_vals = da[lon_dim].values
        lon_grid, lat_grid = np.meshgrid(lon_vals, lat_vals)

        series: dict[str, dict] = {}
        global_min = np.inf
        global_max = -np.inf

        for t in ds["time"].values:
            label = time_to_label(t)
            da_t = da.sel(time=t).values
            da_t = np.asarray(da_t, dtype=np.float64)

            flat_lat = lat_grid.ravel()
            flat_lon = lon_grid.ravel()
            flat_val = da_t.reshape(-1)

            finite_mask = np.isfinite(flat_val)
            flat_lat = flat_lat[finite_mask]
            flat_lon = flat_lon[finite_mask]
            flat_val = flat_val[finite_mask]

            if flat_val.size == 0:
                series[label] = {
                    "lat": [],
                    "lon": [],
                    "val": [],
                    "stats": {
                        "count": 0,
                        "min": None,
                        "max": None,
                        "mean": None,
                        "p05": None,
                        "p95": None,
                    },
                }
                continue

            global_min = min(global_min, float(flat_val.min()))
            global_max = max(global_max, float(flat_val.max()))

            samp_lat, samp_lon, samp_val = sample_points(
                flat_lat,
                flat_lon,
                flat_val,
                max_points=max_points,
                rng=rng,
            )

            stats = {
                "count": int(flat_val.size),
                "min": float(np.min(flat_val)),
                "max": float(np.max(flat_val)),
                "mean": float(np.mean(flat_val)),
                "p05": float(np.percentile(flat_val, 5)),
                "p95": float(np.percentile(flat_val, 95)),
            }

            series[label] = {
                "lat": samp_lat.round(5).tolist(),
                "lon": samp_lon.round(5).tolist(),
                "val": samp_val.round(8).tolist(),
                "stats": stats,
            }

        if not series:
            continue

        if not np.isfinite(global_min) or not np.isfinite(global_max):
            global_min, global_max = 0.0, 1.0
        if global_min == global_max:
            global_max = global_min + 1e-9

        payload["variables"][var_name] = {
            "label": var_name,
            "range": {"min": float(global_min), "max": float(global_max)},
            "series": series,
        }

    payload["meta"]["variables"] = sorted(payload["variables"].keys())
    return payload


def render_html(payload: dict, title: str) -> str:
    payload_json = json.dumps(payload, separators=(",", ":"))

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{title}</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    :root {{
      --bg: #f4f7fb;
      --panel: #ffffff;
      --ink: #1e2a39;
      --muted: #5a6a7d;
      --accent: #1565c0;
      --border: #d7e0ea;
    }}
    body {{
      margin: 0;
      font-family: "Segoe UI", "Helvetica Neue", sans-serif;
      color: var(--ink);
      background: radial-gradient(circle at 15% 20%, #ffffff 0%, var(--bg) 45%, #e8eef7 100%);
    }}
    .page {{
      max-width: 1320px;
      margin: 24px auto;
      padding: 0 16px 24px;
    }}
    .head {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      flex-wrap: wrap;
      gap: 12px;
      margin-bottom: 14px;
    }}
    h1 {{
      margin: 0;
      font-size: 1.4rem;
      letter-spacing: 0.2px;
    }}
    .controls {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 10px;
    }}
    .control {{
      display: grid;
      gap: 4px;
      min-width: 180px;
    }}
    label {{
      font-size: 0.8rem;
      color: var(--muted);
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }}
    select {{
      border: 1px solid var(--border);
      border-radius: 9px;
      padding: 8px 10px;
      font-size: 0.95rem;
      background: #fff;
      color: var(--ink);
    }}
    input[type="range"] {{
      accent-color: var(--accent);
      width: 100%;
    }}
    .range-value {{
      color: var(--muted);
      font-size: 0.85rem;
      font-weight: 600;
    }}
    .stats {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
      gap: 10px;
      margin: 12px 0 14px;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 11px;
      padding: 10px 12px;
    }}
    .card .name {{
      color: var(--muted);
      font-size: 0.8rem;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      margin-bottom: 3px;
    }}
    .card .value {{
      font-size: 1rem;
      font-weight: 700;
    }}
    .grid {{
      display: grid;
      grid-template-columns: 2fr 1fr;
      gap: 14px;
    }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 12px;
      overflow: hidden;
      min-height: 420px;
    }}
    .plot {{
      width: 100%;
      height: 100%;
      min-height: 420px;
    }}
    .note {{
      margin-top: 10px;
      color: var(--muted);
      font-size: 0.9rem;
    }}
    @media (max-width: 960px) {{
      .grid {{ grid-template-columns: 1fr; }}
      .panel, .plot {{ min-height: 360px; }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <div class="head">
      <h1>{title}</h1>
      <div class="controls">
        <div class="control">
          <label for="variableSelect">Data Variable</label>
          <select id="variableSelect"></select>
        </div>
        <div class="control">
          <label for="yearSelect">Time</label>
          <select id="yearSelect"></select>
        </div>
        <div class="control">
          <label for="sizeScale">Point Size Scale</label>
          <input id="sizeScale" type="range" min="0.5" max="3" step="0.1" value="1.4" />
          <span id="sizeScaleValue" class="range-value">1.4x</span>
        </div>
      </div>
    </div>
    <div class="stats" id="stats"></div>
    <div class="grid">
      <div class="panel"><div id="mapPlot" class="plot"></div></div>
      <div class="panel"><div id="histPlot" class="plot"></div></div>
    </div>
    <div class="note">Map points are sampled for responsiveness; marker size and color both scale with the selected variable values.</div>
  </div>
  <script>
    const DATA = {payload_json};

    const varSelect = document.getElementById("variableSelect");
    const yearSelect = document.getElementById("yearSelect");
    const sizeScale = document.getElementById("sizeScale");
    const sizeScaleValue = document.getElementById("sizeScaleValue");
    const statsEl = document.getElementById("stats");

    function populateSelect(el, items) {{
      el.innerHTML = "";
      for (const it of items) {{
        const opt = document.createElement("option");
        opt.value = it;
        opt.textContent = it;
        el.appendChild(opt);
      }}
    }}

    function fmt(val, digits = 3) {{
      if (val === null || val === undefined || Number.isNaN(val)) return "N/A";
      const abs = Math.abs(val);
      if (abs >= 1e6 || (abs > 0 && abs < 1e-3)) return Number(val).toExponential(2);
      return Number(val).toLocaleString(undefined, {{ maximumFractionDigits: digits }});
    }}

    function renderStats(stats) {{
      const items = [
        ["Count", stats.count],
        ["Min", fmt(stats.min)],
        ["Max", fmt(stats.max)],
        ["Mean", fmt(stats.mean)],
        ["P05", fmt(stats.p05)],
        ["P95", fmt(stats.p95)],
      ];
      statsEl.innerHTML = items
        .map(([k, v]) => `<div class="card"><div class="name">${{k}}</div><div class="value">${{v}}</div></div>`)
        .join("");
    }}

    function computeMarkerSizes(values, stats, scaleFactor) {{
      if (!values.length) return [];
      const baseMin = 3.0;
      const baseMax = 16.0;

      const p05 = stats.p05;
      const p95 = stats.p95;

      const lo = (p05 !== null && p05 !== undefined) ? p05 : Math.min(...values);
      const hi = (p95 !== null && p95 !== undefined) ? p95 : Math.max(...values);
      const span = Math.max(1e-12, hi - lo);

      return values.map((v) => {{
        const clipped = Math.max(lo, Math.min(hi, v));
        const norm = (clipped - lo) / span;
        return (baseMin + (baseMax - baseMin) * norm) * scaleFactor;
      }});
    }}

    function updatePlots() {{
      const variable = varSelect.value;
      const year = yearSelect.value;
      const spec = DATA.variables[variable];
      const block = spec.series[year];
      const zmin = spec.range.min;
      const zmax = spec.range.max;
      const scaleFactor = Number(sizeScale.value || "1");
      sizeScaleValue.textContent = `${{scaleFactor.toFixed(1)}}x`;

      renderStats(block.stats);

      const values = block.val;
      const markerSize = computeMarkerSizes(values, block.stats, scaleFactor);

      const mapTrace = {{
        type: "scattergeo",
        mode: "markers",
        lat: block.lat,
        lon: block.lon,
        text: values.map(v => `${{variable}}: ${{fmt(v, 5)}}`),
        hovertemplate: "%{{text}}<br>lat=%{{lat:.2f}}, lon=%{{lon:.2f}}<extra></extra>",
        marker: {{
          color: values,
          cmin: zmin,
          cmax: zmax,
          colorscale: "Turbo",
          size: markerSize,
          opacity: 0.78,
          colorbar: {{ title: variable }},
        }},
      }};

      const mapLayout = {{
        title: `${{variable}} map (${{year}})`,
        margin: {{ l: 10, r: 10, t: 42, b: 8 }},
        geo: {{
          projection: {{ type: "equirectangular" }},
          showland: true,
          landcolor: "#f4f6f8",
          showcountries: true,
          countrycolor: "#8ea0b3",
          showcoastlines: true,
          coastlinecolor: "#8ea0b3",
          bgcolor: "#ffffff",
        }},
      }};

      Plotly.react("mapPlot", [mapTrace], mapLayout, {{ responsive: true, displaylogo: false }});

      const histTrace = {{
        type: "histogram",
        x: values,
        nbinsx: 50,
        marker: {{ color: "#1565c0", line: {{ color: "#ffffff", width: 0.4 }} }},
        opacity: 0.9,
      }};

      const histLayout = {{
        title: `${{variable}} distribution (${{year}})`,
        margin: {{ l: 48, r: 14, t: 42, b: 48 }},
        xaxis: {{ title: variable }},
        yaxis: {{ title: "Count" }},
        bargap: 0.02,
      }};

      Plotly.react("histPlot", [histTrace], histLayout, {{ responsive: true, displaylogo: false }});
    }}

    const variables = DATA.meta.variables;
    populateSelect(varSelect, variables);
    const firstVar = variables[0];
    populateSelect(yearSelect, Object.keys(DATA.variables[firstVar].series));

    varSelect.addEventListener("change", () => {{
      const years = Object.keys(DATA.variables[varSelect.value].series);
      const currentYear = yearSelect.value;
      populateSelect(yearSelect, years);
      if (years.includes(currentYear)) yearSelect.value = currentYear;
      updatePlots();
    }});

    yearSelect.addEventListener("change", updatePlots);
    sizeScale.addEventListener("input", updatePlots);
    updatePlots();
  </script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build an interactive HTML dashboard from a netCDF dataset."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("unified_annual_carbon_dataset_2015_2024.nc"),
        help="Path to netCDF file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("multi_variable_dashboard.html"),
        help="Output HTML file path.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=25000,
        help="Maximum sampled points per variable/time for plotting.",
    )
    args = parser.parse_args()

    ds = xr.open_dataset(args.input)
    payload = build_payload(ds, max_points=args.max_points)

    if not payload["variables"]:
        raise ValueError("No time-dependent variables with lat/lon dimensions were found.")

    html = render_html(payload, title="Global Carbon Dataset Explorer")
    args.output.write_text(html, encoding="utf-8")
    print(f"Wrote dashboard to {args.output}")


if __name__ == "__main__":
    main()
