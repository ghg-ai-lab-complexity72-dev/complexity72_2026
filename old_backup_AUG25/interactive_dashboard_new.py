from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import geopandas as gpd
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


def load_geo_boundaries() -> dict[str, list[float | None]]:
    """Loads continent, country, and state outlines using GeoPandas and converts them to Plotly line coordinates."""
    lons: list[float | None] = []
    lats: list[float | None] = []

    def add_geometry(geom) -> None:
        if geom is None or geom.is_empty:
            return
        if geom.geom_type == "Polygon":
            x, y = geom.exterior.coords.xy
            lons.extend(np.round(x, 3).tolist())
            lats.extend(np.round(y, 3).tolist())
            lons.append(None)
            lats.append(None)
        elif geom.geom_type == "MultiPolygon":
            for poly in geom.geoms:
                add_geometry(poly)

    try:
        world_url = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_110m_admin_0_countries.geojson"
        states_url = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_110m_admin_1_states_provinces.geojson"

        gdf_world = gpd.read_file(world_url)
        gdf_states = gpd.read_file(states_url)

        for geom in gdf_world.geometry:
            add_geometry(geom)

        for geom in gdf_states.geometry:
            add_geometry(geom)

    except Exception as e:
        print(f"Warning: Could not fetch GeoPandas geographic boundaries ({e}). Map will render without overlays.")

    return {"lon": lons, "lat": lats}


def build_payload(ds: xr.Dataset, input_path: Path) -> dict:
    payload: dict[str, dict] = {"variables": {}, "meta": {}}

    years = [time_to_label(t) for t in ds["time"].values]
    payload["meta"]["years"] = years

    # --- EMBED NETCDF FILE AS BASE64 FOR CLIENT-SIDE DOWNLOAD ---
    print(f"Encoding '{input_path.name}' for embedded download...")
    nc_bytes = input_path.read_bytes()
    payload["meta"]["nc_base64"] = base64.b64encode(nc_bytes).decode("ascii")
    payload["meta"]["nc_filename"] = input_path.name

    print("Extracting geographic boundaries using GeoPandas...")
    payload["meta"]["geo_lines"] = load_geo_boundaries()

    for var_name, da in ds.data_vars.items():
        if "time" not in da.dims:
            continue

        lat_dim, lon_dim = detect_spatial_dims(da)
        lat_vals = da[lat_dim].values.round(3).tolist()
        lon_vals = da[lon_dim].values.round(3).tolist()

        series: dict[str, dict] = {}
        global_min = np.inf
        global_max = -np.inf

        for t in ds["time"].values:
            label = time_to_label(t)
            da_t = da.sel(time=t).values
            da_t = np.asarray(da_t, dtype=np.float64)

            # --- FILTER 'growth_rate' TO VALUES BETWEEN 0 AND 5 ONLY ---
            if "growth_rate" in var_name.lower():
                da_t = np.where((da_t >= 0.0) & (da_t <= 5.0), da_t, np.nan)

            finite_vals = da_t[np.isfinite(da_t)]

            if finite_vals.size == 0:
                series[label] = {
                    "grid": [],
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

            global_min = min(global_min, float(finite_vals.min()))
            global_max = max(global_max, float(finite_vals.max()))

            stats = {
                "count": int(finite_vals.size),
                "min": float(np.min(finite_vals)),
                "max": float(np.max(finite_vals)),
                "mean": float(np.mean(finite_vals)),
                "p05": float(np.percentile(finite_vals, 5)),
                "p95": float(np.percentile(finite_vals, 95)),
            }

            clean_grid = np.where(np.isfinite(da_t), np.round(da_t, 3), None).tolist()

            series[label] = {
                "grid": clean_grid,
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
            "lat": lat_vals,
            "lon": lon_vals,
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
      align-items: flex-end;
      gap: 10px;
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 10px;
    }}
    .control {{
      display: grid;
      gap: 4px;
      min-width: 150px;
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
    .btn {{
      border: 1px solid var(--accent);
      background: var(--accent);
      color: #ffffff;
      border-radius: 9px;
      padding: 8px 12px;
      font-size: 0.9rem;
      font-weight: 600;
      cursor: pointer;
      transition: background 0.2s, transform 0.1s;
      height: 38px;
    }}
    .btn:hover {{
      background: #0d47a1;
    }}
    .btn-secondary {{
      background: #ffffff;
      color: var(--ink);
      border-color: var(--border);
    }}
    .btn-secondary:hover {{
      background: #eef3f9;
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
      min-height: 440px;
    }}
    .plot {{
      width: 100%;
      height: 100%;
      min-height: 440px;
    }}
    .note {{
      margin-top: 10px;
      color: var(--muted);
      font-size: 0.9rem;
    }}
    @media (max-width: 960px) {{
      .grid {{ grid-template-columns: 1fr; }}
      .panel, .plot {{ min-height: 380px; }}
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
          <label for="scaleSelect">Map Colorscale</label>
          <select id="scaleSelect">
            <option value="linear">Linear Scale</option>
            <option value="log">Logarithmic Scale</option>
          </select>
        </div>
        <div class="control">
          <label for="profileXScaleSelect">Profile X Scale</label>
          <select id="profileXScaleSelect">
            <option value="linear">Linear</option>
            <option value="log">Logarithmic</option>
          </select>
        </div>
        <div class="control">
          <label>Data Export</label>
          <button id="downloadNcBtn" class="btn">Download Full (.nc)</button>
        </div>
        <div class="control">
          <label>&nbsp;</label>
          <button id="downloadCsvBtn" class="btn btn-secondary">Download Slice (.csv)</button>
        </div>
      </div>
    </div>
    <div class="stats" id="stats"></div>
    <div class="grid">
      <div class="panel"><div id="mapPlot" class="plot"></div></div>
      <div class="panel"><div id="profilePlot" class="plot"></div></div>
    </div>
    <div class="note">Raster heatmap alongside Zonal Latitude Profile (showing mean & 5th–95th percentile spread by latitude).</div>
<div class="note">The underlying dataset is derived from CAMS Greenhouse Gas Reanalysis (Agustí-Panareda et al., 2023), the EDGAR v8.0 emissions database (Crippa et al., 2024), and global GOSIF solar-induced chlorophyll fluorescence data (Li & Xiao, 2019).</div>  
    <div class="note">Code and additional analyisis can be found at <a href="https://github.com/ghg-ai-lab-complexity72-dev/complexity72_2026" target="_blank">https://github.com/cams-gas-reanalysis</a>.</div>

  </div>
  <script>
    const DATA = {payload_json};

    const varSelect = document.getElementById("variableSelect");
    const yearSelect = document.getElementById("yearSelect");
    const scaleSelect = document.getElementById("scaleSelect");
    const profileXScaleSelect = document.getElementById("profileXScaleSelect");
    const statsEl = document.getElementById("stats");
    const downloadNcBtn = document.getElementById("downloadNcBtn");
    const downloadCsvBtn = document.getElementById("downloadCsvBtn");

    const PALETTE_TURBO = [
      [0.188, 0.071, 0.231],
      [0.243, 0.235, 0.651],
      [0.263, 0.424, 0.918],
      [0.208, 0.608, 0.973],
      [0.118, 0.769, 0.882],
      [0.118, 0.882, 0.686],
      [0.286, 0.953, 0.447],
      [0.553, 0.980, 0.224],
      [0.788, 0.925, 0.129],
      [0.949, 0.784, 0.133],
      [0.988, 0.557, 0.118],
      [0.918, 0.306, 0.078],
      [0.729, 0.098, 0.031],
      [0.478, 0.016, 0.012]
    ];

    function interpolateColor(palette, t) {{
      t = Math.max(0, Math.min(1, t));
      const n = palette.length - 1;
      const idx = t * n;
      const i = Math.floor(idx);
      const frac = idx - i;
      if (i >= n) {{
        const c = palette[n];
        return `rgb(${{Math.round(c[0]*255)}},${{Math.round(c[1]*255)}},${{Math.round(c[2]*255)}})`;
      }}
      const c1 = palette[i];
      const c2 = palette[i + 1];
      const r = Math.round((c1[0] + (c2[0] - c1[0]) * frac) * 255);
      const g = Math.round((c1[1] + (c2[1] - c1[1]) * frac) * 255);
      const b = Math.round((c1[2] + (c2[2] - c1[2]) * frac) * 255);
      return `rgb(${{r}},${{g}},${{b}})`;
    }}

    function generateColorscale(mode, zMin, zMax, flatVals) {{
      const N = 100;
      if (mode !== "log" || zMax <= 0) {{
        const cs = [];
        for (let i = 0; i < N; i++) {{
          const t = i / (N - 1);
          cs.push([t, interpolateColor(PALETTE_TURBO, t)]);
        }}
        return cs;
      }}

      let posMin = zMin > 0 ? zMin : null;
      if (!posMin) {{
        const posVals = flatVals.filter(v => v > 0);
        posMin = posVals.length > 0 ? Math.min(...posVals) : zMax * 1e-4;
      }}
      if (posMin >= zMax) posMin = zMax * 1e-4;

      const logMin = Math.log10(posMin);
      const logMax = Math.log10(zMax);
      const range = zMax - zMin;

      const cs = [];
      cs.push([0.0, interpolateColor(PALETTE_TURBO, 0)]);

      for (let i = 1; i < N - 1; i++) {{
        const p = i / (N - 1);
        const logVal = logMin + p * (logMax - logMin);
        const val = Math.pow(10, logVal);
        let x = (val - zMin) / (range || 1);
        x = Math.max(0, Math.min(1, x));
        cs.push([x, interpolateColor(PALETTE_TURBO, p)]);
      }}

      cs.push([1.0, interpolateColor(PALETTE_TURBO, 1)]);

      cs.sort((a, b) => a[0] - b[0]);
      for (let i = 1; i < cs.length; i++) {{
        if (cs[i][0] < cs[i-1][0]) {{
          cs[i][0] = cs[i-1][0];
        }}
      }}
      return cs;
    }}

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

    function computeLatProfile(grid, lats) {{
      const means = [];
      const p05s = [];
      const p95s = [];
      const validLats = [];

      for (let i = 0; i < grid.length; i++) {{
        const row = grid[i];
        if (!row) continue;

        const vals = row.filter(v => v !== null && !isNaN(v));
        if (vals.length === 0) continue;

        vals.sort((a, b) => a - b);
        const sum = vals.reduce((a, b) => a + b, 0);
        const mean = sum / vals.length;

        const p05Idx = Math.floor(vals.length * 0.05);
        const p95Idx = Math.min(vals.length - 1, Math.floor(vals.length * 0.95));

        means.push(mean);
        p05s.push(vals[p05Idx]);
        p95s.push(vals[p95Idx]);
        validLats.push(lats[i]);
      }}

      return {{ lats: validLats, means, p05s, p95s }};
    }}

    function updatePlots() {{
      const variable = varSelect.value;
      const year = yearSelect.value;
      const scaleType = scaleSelect.value;
      const profileXScale = profileXScaleSelect.value;

      const spec = DATA.variables[variable];
      const block = spec.series[year];

      renderStats(block.stats);

      const rawGrid = block.grid;
      const flatValues = rawGrid.flat().filter(v => v !== null && !isNaN(v));
      const minVal = spec.range.min;
      const maxVal = spec.range.max;

      const latMin = Math.min(...spec.lat);
      const latMax = Math.max(...spec.lat);
      const latPad = (latMax - latMin) * 0.02 || 1.0;
      const latRange = [latMin - latPad, latMax + latPad];

      const customColorscale = generateColorscale(scaleType, minVal, maxVal, flatValues);
      const titleSuffix = scaleType === "log" ? " (Log Colorscale)" : " (Linear Colorscale)";

      const mapTrace = {{
        type: "heatmap",
        z: rawGrid,
        x: spec.lon,
        y: spec.lat,
        zmin: minVal,
        zmax: maxVal,
        colorscale: customColorscale,
        colorbar: {{ title: variable }},
        hovertemplate: "Lon: %{{x:.2f}}<br>Lat: %{{y:.2f}}<br>Value: %{{z}}<extra></extra>",
        hoverongaps: false
      }};

      const geoLines = DATA.meta.geo_lines || {{ lon: [], lat: [] }};
      const outlineTrace = {{
        type: "scatter",
        mode: "lines",
        x: geoLines.lon,
        y: geoLines.lat,
        line: {{ color: "rgba(35, 35, 35, 0.7)", width: 0.8 }},
        hoverinfo: "skip",
        showlegend: false
      }};

      const mapLayout = {{
        title: `${{variable}} raster (${{year}})${{titleSuffix}}`,
        margin: {{ l: 40, r: 10, t: 42, b: 40 }},
        xaxis: {{ title: "Longitude" }},
        yaxis: {{ title: "Latitude", range: latRange }},
        template: "plotly_white"
      }};

      Plotly.react("mapPlot", [mapTrace, outlineTrace], mapLayout, {{ responsive: true, displaylogo: false }});

      const profile = computeLatProfile(rawGrid, spec.lat);

      const bandTrace = {{
        type: "scatter",
        mode: "lines",
        x: [...profile.p05s, ...profile.p95s.slice().reverse()],
        y: [...profile.lats, ...profile.lats.slice().reverse()],
        fill: "toself",
        fillcolor: "rgba(21, 101, 192, 0.18)",
        line: {{ color: "transparent" }},
        name: "5th-95th Percentile Range",
        hoverinfo: "skip"
      }};

      const meanTrace = {{
        type: "scatter",
        mode: "lines",
        x: profile.means,
        y: profile.lats,
        line: {{ color: "#1565c0", width: 2.5 }},
        name: "Zonal Mean",
        hovertemplate: "Lat: %{{y:.2f}}<br>Mean " + variable + ": %{{x:.3f}}<extra></extra>"
      }};

      const profileLayout = {{
        title: `${{variable}} vs Latitude (${{year}})`,
        margin: {{ l: 45, r: 14, t: 42, b: 40 }},
        xaxis: {{ 
          title: variable, 
          type: profileXScale === "log" ? "log" : "linear" 
        }},
        yaxis: {{ 
          title: "Latitude", 
          range: latRange 
        }},
        showlegend: true,
        legend: {{ x: 0.05, y: 0.98, bgcolor: "rgba(255,255,255,0.7)" }},
        template: "plotly_white"
      }};

      Plotly.react("profilePlot", [bandTrace, meanTrace], profileLayout, {{ responsive: true, displaylogo: false }});
    }}

    // --- DOWNLOAD FULL NETCDF (.nc) FILE ---
    downloadNcBtn.addEventListener("click", () => {{
      const b64 = DATA.meta.nc_base64;
      const fileName = DATA.meta.nc_filename || "unified_annual_carbon_dataset_2015_2024.nc";
      
      const binaryString = atob(b64);
      const len = binaryString.length;
      const bytes = new Uint8Array(len);
      for (let i = 0; i < len; i++) {{
        bytes[i] = binaryString.charCodeAt(i);
      }}

      const blob = new Blob([bytes.buffer], {{ type: "application/x-netcdf" }});
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = fileName;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    }});

    // --- DOWNLOAD ACTIVE SLICE AS CSV (.csv) ---
    downloadCsvBtn.addEventListener("click", () => {{
      const variable = varSelect.value;
      const year = yearSelect.value;
      const spec = DATA.variables[variable];
      const grid = spec.series[year].grid;

      let csv = `latitude,longitude,${{variable}}\n`;

      for (let i = 0; i < spec.lat.length; i++) {{
        const lat = spec.lat[i];
        const row = grid[i];
        if (!row) continue;
        for (let j = 0; j < spec.lon.length; j++) {{
          const lon = spec.lon[j];
          const val = row[j];
          if (val !== null && val !== undefined && !isNaN(val)) {{
            csv += `${{lat}},${{lon}},${{val}}\n`;
          }}
        }}
      }}

      const blob = new Blob([csv], {{ type: "text/csv;charset=utf-8;" }});
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `${{variable}}_${{year}}.csv`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    }});

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
    scaleSelect.addEventListener("change", updatePlots);
    profileXScaleSelect.addEventListener("change", updatePlots);
    updatePlots();
  </script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build an interactive pixel-raster HTML dashboard with GeoPandas boundaries and NetCDF/CSV dataset downloads."
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
        default=Path("index.html"),
        help="Output HTML file path.",
    )
    args = parser.parse_args()

    ds = xr.open_dataset(args.input)
    payload = build_payload(ds, args.input)

    if not payload["variables"]:
        raise ValueError("No time-dependent variables with lat/lon dimensions were found.")

    html = render_html(payload, title="Global Carbon Dataset Explorer")
    args.output.write_text(html, encoding="utf-8")
    print(f"Wrote dashboard to {args.output}")


if __name__ == "__main__":
    main()