from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import xarray as xr
from bokeh.io import output_file, save
from bokeh.layouts import column, row
from bokeh.resources import INLINE
from bokeh.models import (
	ColorBar,
	ColumnDataSource,
	CustomJS,
	Div,
	GeoJSONDataSource,
	HoverTool,
	LinearColorMapper,
	NumeralTickFormatter,
	Select,
)
from bokeh.palettes import Turbo256
from bokeh.plotting import figure


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
	if max_points <= 0:
		return lat, lon, val
	if val.size <= max_points:
		return lat, lon, val
	idx = rng.choice(val.size, size=max_points, replace=False)
	return lat[idx], lon[idx], val[idx]


def infer_cell_step(coords: np.ndarray) -> float:
	arr = np.asarray(coords, dtype=float)
	if arr.size < 2:
		return 1.0
	diffs = np.diff(arr)
	diffs = np.abs(diffs[np.isfinite(diffs)])
	if diffs.size == 0:
		return 1.0
	step = float(np.median(diffs))
	return step if step > 0 else 1.0


def build_payload(ds: xr.Dataset, max_points: int = 25000, seed: int = 42) -> dict:
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
		dlat = infer_cell_step(lat_vals)
		dlon = infer_cell_step(lon_vals)
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
				"dlat": dlat,
				"dlon": dlon,
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


def load_world_geojson(borders_path: Path) -> str:
	if not borders_path.exists():
		world = gpd.read_file(
			"https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
		)
		world.to_file(borders_path, driver="GPKG")
	else:
		world = gpd.read_file(borders_path)

	world = world.to_crs("EPSG:4326")
	world = world[world.geometry.notna() & ~world.geometry.is_empty].copy()
	return world.to_json()


def compute_kde(
	values: list[float],
	grid_points: int = 160,
	max_samples: int = 6000,
) -> tuple[list[float], list[float]]:
	if not values:
		return [], []

	arr = np.asarray(values, dtype=float)
	arr = arr[np.isfinite(arr)]
	if arr.size == 0:
		return [], []

	if arr.size > max_samples:
		idx = np.linspace(0, arr.size - 1, max_samples, dtype=int)
		arr = arr[idx]

	min_v = float(np.min(arr))
	max_v = float(np.max(arr))

	if min_v == max_v:
		xs = np.linspace(min_v - 1.0, max_v + 1.0, grid_points)
		ys = np.zeros_like(xs)
		ys[len(xs) // 2] = 1.0
		return xs.tolist(), ys.tolist()

	std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
	q25, q75 = np.percentile(arr, [25, 75])
	iqr = float(q75 - q25)
	sigma = min(std, iqr / 1.34) if iqr > 0 else std
	if not np.isfinite(sigma) or sigma <= 0:
		sigma = (max_v - min_v) / 6.0
	if sigma <= 0:
		sigma = 1.0

	bw = 1.06 * sigma * (arr.size ** (-1.0 / 5.0))
	if not np.isfinite(bw) or bw <= 0:
		bw = max((max_v - min_v) / 80.0, 1e-12)

	pad = 3.0 * bw
	xs = np.linspace(min_v - pad, max_v + pad, grid_points)
	diff = (xs[:, None] - arr[None, :]) / bw
	coef = 1.0 / (arr.size * bw * np.sqrt(2.0 * np.pi))
	ys = coef * np.exp(-0.5 * diff * diff).sum(axis=1)
	return xs.tolist(), ys.tolist()


def format_stats_html(stats: dict, variable: str, year: str) -> str:
	return (
		f"<b>{variable}</b> | <b>{year}</b> &nbsp; "
		f"count={stats['count']:,} &nbsp; "
		f"min={stats['min']:.4g} &nbsp; "
		f"max={stats['max']:.4g} &nbsp; "
		f"mean={stats['mean']:.4g} &nbsp; "
		f"p05={stats['p05']:.4g} &nbsp; "
		f"p95={stats['p95']:.4g}"
	)


def build_dashboard(payload: dict, world_geojson: str, title: str):
	variables = payload["meta"]["variables"]
	if not variables:
		raise ValueError("No variables were found to visualize.")

	first_var = variables[0]
	years = list(payload["variables"][first_var]["series"].keys())
	if not years:
		raise ValueError(f"No years found for variable {first_var}.")
	first_year = years[0]

	block = payload["variables"][first_var]["series"][first_year]
	zmin = payload["variables"][first_var]["range"]["min"]
	zmax = payload["variables"][first_var]["range"]["max"]
	p05 = block["stats"]["p05"]
	p95 = block["stats"]["p95"]

	vals = np.asarray(block["val"], dtype=float)
	color_vals = vals.copy()
    
	cell_w = float(block.get("dlon", 1.0))
	cell_h = float(block.get("dlat", 1.0))

	point_source = ColumnDataSource(
		data={
			"lat": block["lat"],
			"lon": block["lon"],
			"val": block["val"],
			"color_val": color_vals.tolist(),
			"cell_w": [cell_w] * len(block["val"]),
			"cell_h": [cell_h] * len(block["val"]),
		}
	)

	kde_x, kde_y = compute_kde(block["val"])
	hist_source = ColumnDataSource(data={"x": kde_x, "y": kde_y})

	world_source = GeoJSONDataSource(geojson=world_geojson)

	mapper = LinearColorMapper(palette=Turbo256, low=zmin, high=zmax, nan_color="#00000000")

	map_fig = figure(
		width=900,
		height=500,
		title=f"{first_var} map ({first_year})",
		x_axis_label="Longitude",
		y_axis_label="Latitude",
		tools="pan,wheel_zoom,box_zoom,reset,save",
		active_scroll="wheel_zoom",
		x_range=(-180, 180),
		y_range=(-60, 85),
		match_aspect=True,
	)

	map_fig.patches(
		"xs",
		"ys",
		source=world_source,
		fill_alpha=0.08,
		fill_color="#d9e2ec",
		line_color="#4a5568",
		line_width=0.6,
		line_alpha=0.85,
		level="underlay",
	)

	circles = map_fig.rect(
		x="lon",
		y="lat",
		width="cell_w",
		height="cell_h",
		source=point_source,
		alpha=0.88,
		line_alpha=0,
		fill_color={"field": "color_val", "transform": mapper},
	)

	# Draw outlines again above points so coastlines/country borders remain visible.
	map_fig.patches(
		"xs",
		"ys",
		source=world_source,
		fill_alpha=0.0,
		line_color="#1f2937",
		line_width=1.0,
		line_alpha=0.95,
		level="overlay",
	)

	map_fig.add_tools(
		HoverTool(
			renderers=[circles],
			tooltips=[
				("lon", "@lon{0.00}"),
				("lat", "@lat{0.00}"),
				("value", "@val{0.000}"),
			],
		)
	)

	map_fig.grid.grid_line_alpha = 0.25

	color_bar = ColorBar(
		color_mapper=mapper,
		formatter=NumeralTickFormatter(format="0.00a"),
		label_standoff=8,
		title=first_var,
		width=12,
	)
	map_fig.add_layout(color_bar, "right")

	hist_fig = figure(
		width=420,
		height=500,
		title=f"{first_var} KDE ({first_year})",
		x_axis_label=first_var,
		y_axis_label="Density",
		tools="pan,wheel_zoom,box_zoom,reset,save",
	)
	hist_fig.line(
		x="x",
		y="y",
		source=hist_source,
		line_color="#1565c0",
		line_width=2,
		line_alpha=0.95,
	)

	stats_div = Div(
		text=format_stats_html(block["stats"], first_var, first_year),
		styles={
			"font-family": "Segoe UI, Arial, sans-serif",
			"font-size": "13px",
			"color": "#1e2a39",
			"background-color": "#ffffff",
			"border": "1px solid #d7e0ea",
			"border-radius": "8px",
			"padding": "8px 10px",
			"margin-bottom": "8px",
		},
	)

	var_select = Select(title="Data Variable", value=first_var, options=variables)
	year_select = Select(title="Time", value=first_year, options=years)
	color_scale_select = Select(title="Color Scale", value="normal", options=["normal", "log"])

	data_json = json.dumps(payload)

	callback = CustomJS(
		args=dict(
			data_json=data_json,
			var_select=var_select,
			year_select=year_select,
			color_scale_select=color_scale_select,
			point_source=point_source,
			hist_source=hist_source,
			stats_div=stats_div,
			map_fig=map_fig,
			hist_fig=hist_fig,
			mapper=mapper,
			color_bar=color_bar,
		),
		code="""
const data = JSON.parse(data_json);

function kdeFromValues(values, gridN = 160, maxSamples = 6000) {
	const finite = values.filter((v) => Number.isFinite(v));
	if (!finite.length) return {x: [], y: []};

	let sample = finite;
	if (sample.length > maxSamples) {
		const step = Math.ceil(sample.length / maxSamples);
		const out = [];
		for (let i = 0; i < sample.length; i += step) out.push(sample[i]);
		sample = out;
	}

	const sorted = sample.slice().sort((a, b) => a - b);
	const n = sorted.length;
	const minV = sorted[0];
	const maxV = sorted[n - 1];

	if (minV === maxV) {
		return {x: [minV - 1, minV, minV + 1], y: [0, 1, 0]};
	}

	const mean = sorted.reduce((acc, v) => acc + v, 0) / n;
	const varSum = sorted.reduce((acc, v) => {
		const d = v - mean;
		return acc + d * d;
	}, 0);
	const std = n > 1 ? Math.sqrt(varSum / (n - 1)) : 0;

	const q = (p) => {
		const idx = (n - 1) * p;
		const lo = Math.floor(idx);
		const hi = Math.ceil(idx);
		if (lo === hi) return sorted[lo];
		const w = idx - lo;
		return sorted[lo] * (1 - w) + sorted[hi] * w;
	};

	const iqr = q(0.75) - q(0.25);
	let sigma = iqr > 0 ? Math.min(std, iqr / 1.34) : std;
	if (!Number.isFinite(sigma) || sigma <= 0) sigma = (maxV - minV) / 6;
	if (!Number.isFinite(sigma) || sigma <= 0) sigma = 1;

	let bw = 1.06 * sigma * Math.pow(n, -1 / 5);
	if (!Number.isFinite(bw) || bw <= 0) bw = Math.max((maxV - minV) / 80, 1e-12);

	const pad = 3 * bw;
	const x0 = minV - pad;
	const x1 = maxV + pad;
	const step = (x1 - x0) / (gridN - 1);
	const x = [];
	const y = [];

	const norm = 1 / (n * bw * Math.sqrt(2 * Math.PI));

	for (let i = 0; i < gridN; i++) {
		const xv = x0 + i * step;
		let sum = 0;
		for (let j = 0; j < n; j++) {
			const u = (xv - sorted[j]) / bw;
			sum += Math.exp(-0.5 * u * u);
		}
		x.push(xv);
		y.push(norm * sum);
	}

	return {x, y};
}

function fmt(v) {
  if (v === null || v === undefined || Number.isNaN(v)) return 'N/A';
  const av = Math.abs(v);
  if (av >= 1e6 || (av > 0 && av < 1e-3)) return Number(v).toExponential(2);
  return Number(v).toLocaleString(undefined, {maximumFractionDigits: 4});
}

const variable = var_select.value;
let years = Object.keys(data.variables[variable].series);

if (year_select.options.length !== years.length || year_select.options[0] !== years[0]) {
  year_select.options = years;
  if (!years.includes(year_select.value)) {
	year_select.value = years[0];
  }
}

const year = year_select.value;
const colorMode = color_scale_select.value;
const spec = data.variables[variable];
const block = spec.series[year];
const values = block.val;
const dlon = block.dlon ?? 1.0;
const dlat = block.dlat ?? 1.0;

let colorVals = [];
let low = spec.range.min;
let high = spec.range.max;
let colorTitle = `${variable} (linear)`;

if (colorMode === 'log') {
	let foundPositive = false;
	let minLog = Infinity;
	let maxLog = -Infinity;

	for (const v of values) {
		if (v > 0) {
			foundPositive = true;
			const lv = Math.log10(v);
			colorVals.push(lv);
			if (lv < minLog) minLog = lv;
			if (lv > maxLog) maxLog = lv;
		} else {
			colorVals.push(NaN);
		}
	}

	if (foundPositive) {
		low = minLog;
		high = maxLog;
		colorTitle = `${variable} (log10)`;
	} else {
		colorVals = values.slice();
		colorTitle = `${variable} (linear; no positive values for log)`;
	}
} else {
	colorVals = values.slice();
}

if (low === high) {
	high = low + 1e-9;
}

point_source.data = {
  lat: block.lat,
  lon: block.lon,
  val: values,
	color_val: colorVals,
	cell_w: Array(values.length).fill(dlon),
	cell_h: Array(values.length).fill(dlat),
};
point_source.change.emit();

mapper.low = low;
mapper.high = high;
color_bar.title = colorTitle;

const kde = kdeFromValues(values);
hist_source.data = {x: kde.x, y: kde.y};
hist_source.change.emit();

map_fig.title.text = `${variable} map (${year})`;
hist_fig.title.text = `${variable} KDE (${year})`;
hist_fig.xaxis[0].axis_label = variable;
hist_fig.yaxis[0].axis_label = 'Density';

stats_div.text =
  `<b>${variable}</b> | <b>${year}</b> &nbsp; ` +
  `count=${block.stats.count.toLocaleString()} &nbsp; ` +
  `min=${fmt(block.stats.min)} &nbsp; ` +
  `max=${fmt(block.stats.max)} &nbsp; ` +
  `mean=${fmt(block.stats.mean)} &nbsp; ` +
  `p05=${fmt(block.stats.p05)} &nbsp; ` +
  `p95=${fmt(block.stats.p95)}`;
""",
	)

	var_select.js_on_change("value", callback)
	year_select.js_on_change("value", callback)
	color_scale_select.js_on_change("value", callback)

	controls = row(var_select, year_select, color_scale_select, sizing_mode="stretch_width")
	plots = row(map_fig, hist_fig)
	header = Div(
		text=f"<h2 style='margin:0 0 8px 0; font-family: Segoe UI, Arial, sans-serif;'>{title}</h2>",
	)

	return column(header, controls, stats_div, plots)


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Build an interactive Bokeh dashboard from a netCDF dataset."
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
		default=0,
		help="Maximum sampled points per variable/time for plotting. Use 0 for full native resolution.",
	)
	parser.add_argument(
		"--borders",
		type=Path,
		default=Path("ne_110m_admin_0_countries.gpkg"),
		help="Path to country borders geopackage.",
	)
	parser.add_argument(
		"--title",
		type=str,
		default="Global Carbon Dataset Explorer",
		help="Dashboard title.",
	)
	parser.add_argument(
		"--offline",
		action="store_true",
		help="Embed Bokeh JS/CSS inline for a fully offline-shareable HTML.",
	)
	args = parser.parse_args()

	ds = xr.open_dataset(args.input)
	payload = build_payload(ds, max_points=args.max_points)
	if not payload["variables"]:
		raise ValueError("No time-dependent variables with lat/lon dimensions were found.")

	world_geojson = load_world_geojson(args.borders)
	dashboard = build_dashboard(payload, world_geojson, title=args.title)

	if args.offline:
		output_file(args.output, title=args.title, mode="inline")
		save(dashboard, resources=INLINE)
	else:
		output_file(args.output, title=args.title)
		save(dashboard)
	print(f"Wrote dashboard to {args.output}")


if __name__ == "__main__":
	main()

