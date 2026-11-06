#!/usr/bin/env python3
"""
Cluster annual carbon data from a NetCDF file with KMeans and plot the classes
on a world map.

Input:
  Input/unified_annual_carbon_dataset_2015_2025.nc

Required packages:
  pip install numpy xarray netcdf4 matplotlib cartopy shapely

Usage:
  python k_means.py
"""

from __future__ import annotations

import csv
import math
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path


# Configuration.
def find_repo_root(start: Path) -> Path:
    """Walk up from `start` until a folder containing Input/ and Output/
    (or .git) is found. Raises a clear error if the layout is unexpected."""
    for folder in [start, *start.parents]:
        if (folder / "Input").is_dir() and (folder / "Output").is_dir():
            return folder
        if (folder / ".git").exists():
            return folder
    raise FileNotFoundError(
        f"Could not locate the repository root above {start}.\n"
        "Expected to find a folder containing Input/ and Output/. "
        "Run this script from inside a clone of complexity72_2026."
    )


BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = find_repo_root(BASE_DIR)

INPUT_DIR = REPO_ROOT / "Input"
NETCDF_FILE = INPUT_DIR / "unified_annual_carbon_dataset_2015_2025.nc"

# Data artifacts (CSV/TXT/JSON) go under Output/clustering
DATA_OUTPUT_DIR = REPO_ROOT / "Output" / "clustering"
# Figures (PNG/SVG/PDF) go under Figures_and_maps/clustering
FIGURE_OUTPUT_DIR = REPO_ROOT / "Figures_and_maps" / "clustering"

KMEANS_YEARS = [i for i in range(2015, 2026)]
KMEANS_K_MIN = 2
KMEANS_K_MAX = 12
KMEANS_K_SELECTION_METHOD = "elbow"  # "elbow" or "silhouette"
KMEANS_FEATURES = ("emission", "gosif")
INCLUDE_SEA_POINTS = False  # True: land + sea, False: land only.
WRITE_LAND_SEA_MASK = False
MISSING_VALUE_FEATURES = (
    "growth_rate",
    "emission",
    "gosif",
    "delta_emission",
    "delta_gosif",
    "cams_xco2",
    "growth_err",
    "mean_co2",
)
MISSING_COUNT_MIN = 0
MISSING_COUNT_MAX = 8
PLOT_MISSING_VALUES = False
LOG_TRANSFORM_FEATURES = {"emission"}
MAX_ITERATIONS = 100
CONVERGENCE_TOLERANCE = 1e-4
RANDOM_SEED = 72
KMEANS_N_JOBS = 0  # 0 means auto, 1 disables multiprocessing.
PROGRESS_BAR_WIDTH = 32
SILHOUETTE_SAMPLE_SIZE: int | None = None
REBUILD_K_SELECTION_CACHE = False
SHOW = False
DPI = 160
FIGSIZE = (13, 7)
PROFILE_FIGSIZE = (13, 8)
PROFILE_BAR_HEIGHT = 0.72
CLUSTER_COLORS = (
    "#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f",
    "#edc949", "#af7aa1", "#ff9da7", "#9c755f", "#bab0ab",
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
)
MISSING_COUNT_COLORS = (
    "#f7fbff", "#deebf7", "#c6dbef", "#9ecae1", "#6baed6",
    "#4292c6", "#2171b5", "#08519c", "#08306b",
)
LAND_COLOR = "#f2f2f2"
OCEAN_COLOR = "#eaf3fb"
COASTLINE_RESOLUTION = "110m"
COASTLINE_WIDTH = 0.8
BORDER_WIDTH = 0.4
BORDER_ALPHA = 0.7
GRIDLINE_WIDTH = 0.35
GRIDLINE_COLOR = "gray"
GRIDLINE_ALPHA = 0.5
GRIDLINE_STYLE = "--"
TITLE_SIZE = 16
COLORBAR_PAD = 0.06
COLORBAR_SHRINK = 0.82
BARYCENTER_MARKER_SIZE = 115
BARYCENTER_EDGE_WIDTH = 0.85
ELBOW_CACHE_FIELDS = (
    "year",
    "source_size_bytes",
    "source_mtime_ns",
    "k",
    "intra_variance",
    "iterations",
    "point_count",
    "features",
    "log_transform_features",
    "random_seed",
    "max_iterations",
    "convergence_tolerance",
)
SILHOUETTE_CACHE_FIELDS = (
    "year",
    "source_size_bytes",
    "source_mtime_ns",
    "k",
    "silhouette_score",
    "iterations",
    "point_count",
    "scored_point_count",
    "features",
    "log_transform_features",
    "random_seed",
    "max_iterations",
    "convergence_tolerance",
)


# ---------------------------------------------------------------------------
# Validation / utilities
# ---------------------------------------------------------------------------
def validate_config() -> None:
    if KMEANS_K_SELECTION_METHOD not in {"elbow", "silhouette"}:
        raise ValueError(
            'KMEANS_K_SELECTION_METHOD must be either "elbow" or "silhouette"'
        )
    if KMEANS_K_MIN < 2:
        raise ValueError("KMEANS_K_MIN must be at least 2 for K selection")
    if KMEANS_K_MAX < KMEANS_K_MIN:
        raise ValueError("KMEANS_K_MAX must be >= KMEANS_K_MIN")
    if KMEANS_K_MAX > len(CLUSTER_COLORS):
        raise ValueError(
            f"KMEANS_K_MAX={KMEANS_K_MAX} is larger than the number "
            f"of available cluster colors={len(CLUSTER_COLORS)}"
        )
    if KMEANS_N_JOBS < 0:
        raise ValueError("KMEANS_N_JOBS must be 0 or greater")
    if SILHOUETTE_SAMPLE_SIZE is not None and SILHOUETTE_SAMPLE_SIZE < 2:
        raise ValueError("SILHOUETTE_SAMPLE_SIZE must be at least 2 when set")


def resolved_worker_count(task_count: int) -> int:
    if task_count <= 1:
        return 1
    if KMEANS_N_JOBS == 0:
        available_cpus = os.cpu_count() or 1
        worker_count = max(1, available_cpus - 1)
    else:
        worker_count = KMEANS_N_JOBS
    return max(1, min(worker_count, task_count))


def print_progress(current: int, total: int, label: str) -> None:
    if total <= 0:
        return
    ratio = current / total
    filled_width = int(PROGRESS_BAR_WIDTH * ratio)
    bar = "#" * filled_width + "-" * (PROGRESS_BAR_WIDTH - filled_width)
    percent = ratio * 100.0
    print(f"\r{label}: [{bar}] {current}/{total} ({percent:5.1f}%)", end="", flush=True)
    if current == total:
        print()


def finish_progress_if_needed(current: int, total: int, label: str) -> None:
    if current < total:
        print_progress(total, total, label)


def cell_extent(values: list[float]) -> tuple[float, float]:
    if len(values) == 1:
        return values[0] - 0.5, values[0] + 0.5
    step = min(abs(values[i + 1] - values[i]) for i in range(len(values) - 1))
    return values[0] - step / 2, values[-1] + step / 2


def transform_feature(name: str, value: float) -> float:
    if name in LOG_TRANSFORM_FEATURES:
        return math.log1p(max(value, 0.0))
    return value


# ---------------------------------------------------------------------------
# Land mask
# ---------------------------------------------------------------------------
def import_land_mask_modules():
    try:
        from cartopy.io import shapereader
        from shapely.geometry import Point
        from shapely.ops import unary_union
        from shapely.prepared import prep
    except ModuleNotFoundError as exc:
        missing = exc.name or "a required land mask package"
        raise SystemExit(
            f"Missing Python package: {missing}\n"
            "Install the land mask dependencies with:\n"
            "  pip install cartopy shapely"
        ) from exc
    return shapereader, Point, unary_union, prep


_LAND_MASK_POINT = None
_LAND_MASK_PREPARED_GEOMETRY = None


def land_mask_resources():
    global _LAND_MASK_POINT, _LAND_MASK_PREPARED_GEOMETRY
    if _LAND_MASK_POINT is not None and _LAND_MASK_PREPARED_GEOMETRY is not None:
        return _LAND_MASK_POINT, _LAND_MASK_PREPARED_GEOMETRY
    shapereader, Point, unary_union, prep = import_land_mask_modules()
    land_path = shapereader.natural_earth(
        resolution=COASTLINE_RESOLUTION, category="physical", name="land",
    )
    land_geometry = unary_union(list(shapereader.Reader(land_path).geometries()))
    _LAND_MASK_POINT = Point
    _LAND_MASK_PREPARED_GEOMETRY = prep(land_geometry)
    return _LAND_MASK_POINT, _LAND_MASK_PREPARED_GEOMETRY


@lru_cache(maxsize=None)
def is_land_point(lat: float, lon: float) -> bool:
    Point, prepared_land = land_mask_resources()
    return bool(prepared_land.intersects(Point(lon, lat)))


def keep_point_for_clustering(lat: float, lon: float) -> bool:
    return INCLUDE_SEA_POINTS or is_land_point(lat, lon)


# ---------------------------------------------------------------------------
# NetCDF loading
# ---------------------------------------------------------------------------
def load_points(
    dataset,
    year: int,
) -> tuple[
    list[list[float]],
    list[list[float]],
    list[tuple[float, float]],
    list[float],
    list[float],
]:
    """
    Read one year's emission/gosif from the NetCDF and return:
        (points, raw_points, locations, lats, lons)
    Matches the old CSV row order (Latitude from -89.5 to 89.5, Bottom-Up).
    """
    import numpy as np

    selection = dataset.sel(time=year, drop=True) if "time" in dataset.dims else dataset

    emission_name = "emission"
    gosif_name = "gosif"

    for name in (emission_name, gosif_name):
        if name not in selection:
            raise ValueError(
                f"Missing variable {name!r} in NetCDF for year {year}. "
                f"Available: {list(selection.data_vars)}"
            )

    lat_axis = np.asarray(dataset["lat"].values, dtype=float).ravel()
    lon_axis = np.asarray(dataset["lon"].values, dtype=float).ravel()

    emission_raw = np.asarray(selection[emission_name].values, dtype=float)
    gosif_raw = np.asarray(selection[gosif_name].values, dtype=float)
    
    # Replace missing values with 0 so those cells are still clustered
    # instead of being silently dropped by the isfinite check below.
    emission_raw = np.where(np.isfinite(emission_raw), emission_raw, 0.0)
    gosif_raw = np.where(np.isfinite(gosif_raw), gosif_raw, 0.0)

    # The original CSV data goes from -89.5 (South) to +89.5 (North).
    # NetCDF files usually go from +89.5 (North) to -89.5 (South).
    # We must flip the latitudes to match the original pipeline.
    if lat_axis[0] > lat_axis[-1]:
        lat_axis = lat_axis[::-1]
        emission_raw = emission_raw[::-1, :]
        gosif_raw = gosif_raw[::-1, :]

    if emission_raw.ndim == 2 and emission_raw.shape == (lat_axis.size, lon_axis.size):
        # Generate lat/lon grids in the correct order
        lat_grid, lon_grid = np.meshgrid(lat_axis, lon_axis, indexing='ij')
        lat_values = lat_grid.ravel()
        lon_values = lon_grid.ravel()
        emission_values = emission_raw.ravel()
        gosif_values = gosif_raw.ravel()
    else:
        lat_values = lat_axis
        lon_values = lon_axis
        emission_values = emission_raw.ravel()
        gosif_values = gosif_raw.ravel()

    if not (lat_values.size == lon_values.size == emission_values.size == gosif_values.size):
        raise ValueError(
            "Coordinate and feature arrays are not the same length: "
            f"lat={lat_values.size}, lon={lon_values.size}, "
            f"emission={emission_values.size}, gosif={gosif_values.size}"
        )

    points: list[list[float]] = []
    raw_points: list[list[float]] = []
    locations: list[tuple[float, float]] = []

    for lat, lon, emission, gosif in zip(lat_values, lon_values, emission_values, gosif_values):
        if not math.isfinite(emission) or not math.isfinite(gosif):
            continue
        if not keep_point_for_clustering(float(lat), float(lon)):
            continue

        feature_values = {"emission": float(emission), "gosif": float(gosif)}
        raw_point = [feature_values[f] for f in KMEANS_FEATURES]
        point = [transform_feature(f, v) for f, v in zip(KMEANS_FEATURES, raw_point)]
        points.append(point)
        raw_points.append(raw_point)
        locations.append((float(lat), float(lon)))

    return (
        points,
        raw_points,
        locations,
        sorted(set(lat_axis.tolist())),
        sorted(set(lon_axis.tolist())),
    )

# ---------------------------------------------------------------------------
# Standardization / KMeans
# ---------------------------------------------------------------------------
def standardize(points: list[list[float]]) -> list[list[float]]:
    if not points:
        raise ValueError("No points to cluster")
    point_count = len(points)
    dimension_count = len(points[0])
    means = [
        sum(point[d] for point in points) / point_count
        for d in range(dimension_count)
    ]
    variances = [
        sum((point[d] - means[d]) ** 2 for point in points) / point_count
        for d in range(dimension_count)
    ]
    stds = [math.sqrt(v) or 1.0 for v in variances]
    return [
        [(point[d] - means[d]) / stds[d] for d in range(dimension_count)]
        for point in points
    ]


def squared_distance(left: list[float], right: list[float]) -> float:
    return sum((left[i] - right[i]) ** 2 for i in range(len(left)))


def nearest_center(point: list[float], centers: list[list[float]]) -> int:
    return min(range(len(centers)), key=lambda i: squared_distance(point, centers[i]))


def initialize_centers(points: list[list[float]], k: int) -> list[list[float]]:
    if len(points) < k:
        raise ValueError(f"K={k} is larger than the number of points")
    rng = random.Random(RANDOM_SEED)
    centers = [points[rng.randrange(len(points))][:]]
    while len(centers) < k:
        distances = [
            min(squared_distance(point, center) for center in centers)
            for point in points
        ]
        total_distance = sum(distances)
        if total_distance == 0:
            centers.append(points[len(centers)][:])
            continue
        target = rng.random() * total_distance
        cumulative = 0.0
        for point, distance in zip(points, distances):
            cumulative += distance
            if cumulative >= target:
                centers.append(point[:])
                break
    return centers


def farthest_point(points: list[list[float]], centers: list[list[float]]) -> list[float]:
    return max(
        points,
        key=lambda point: min(squared_distance(point, center) for center in centers),
    )[:]


def assignment_to_new_centers(
    points: list[list[float]],
    centers: list[list[float]],
    sums: list[list[float]],
    counts: list[int],
) -> list[list[float]]:
    new_centers: list[list[float]] = []
    for label, count in enumerate(counts):
        if count == 0:
            new_centers.append(farthest_point(points, centers))
            continue
        new_centers.append([value / count for value in sums[label]])
    return new_centers


def fit_kmeans(
    points: list[list[float]],
    k: int,
) -> tuple[list[int], list[list[float]], int]:
    if k < 1:
        raise ValueError("K must be at least 1")
    centers = initialize_centers(points, k)
    labels = [-1 for _ in points]
    dimension_count = len(points[0])
    for iteration in range(1, MAX_ITERATIONS + 1):
        changed = False
        sums = [[0.0 for _ in range(dimension_count)] for _ in range(k)]
        counts = [0 for _ in range(k)]
        for point_index, point in enumerate(points):
            label = nearest_center(point, centers)
            if labels[point_index] != label:
                changed = True
            labels[point_index] = label
            counts[label] += 1
            for d in range(dimension_count):
                sums[label][d] += point[d]
        new_centers = assignment_to_new_centers(points, centers, sums, counts)
        shift = max(
            math.sqrt(squared_distance(old_center, new_center))
            for old_center, new_center in zip(centers, new_centers)
        )
        centers = new_centers
        if not changed or shift < CONVERGENCE_TOLERANCE:
            return labels, centers, iteration
    return labels, centers, MAX_ITERATIONS


def kmeans(points: list[list[float]], k: int) -> tuple[list[int], int]:
    labels, _, iterations = fit_kmeans(points, k)
    return labels, iterations


def kmeans_candidate_values(point_count: int) -> list[int]:
    max_k = min(KMEANS_K_MAX, point_count - 1)
    if max_k < KMEANS_K_MIN:
        raise ValueError(f"Need at least {KMEANS_K_MIN + 1} points to select KMeans K")
    return list(range(KMEANS_K_MIN, max_k + 1))


# ---------------------------------------------------------------------------
# K-selection worker plumbing
# ---------------------------------------------------------------------------
_WORKER_POINTS: list[list[float]] | None = None
_WORKER_SCORED_INDICES: list[int] | None = None
_PRINTED_K_SELECTION_WORKER_COUNT = False


def initialize_k_selection_worker(
    points: list[list[float]],
    scored_indices: list[int] | None,
) -> None:
    global _WORKER_POINTS, _WORKER_SCORED_INDICES
    _WORKER_POINTS = points
    _WORKER_SCORED_INDICES = scored_indices


def worker_points() -> list[list[float]]:
    if _WORKER_POINTS is None:
        raise RuntimeError("K selection worker was not initialized")
    return _WORKER_POINTS


def worker_scored_indices() -> list[int]:
    if _WORKER_SCORED_INDICES is None:
        raise RuntimeError("Silhouette worker was not initialized")
    return _WORKER_SCORED_INDICES


def print_k_selection_worker_count_once(worker_count: int) -> None:
    global _PRINTED_K_SELECTION_WORKER_COUNT
    if _PRINTED_K_SELECTION_WORKER_COUNT:
        return
    print(f"Using {worker_count} worker process(es) for K selection")
    _PRINTED_K_SELECTION_WORKER_COUNT = True


def compute_elbow_candidate(k: int) -> tuple[int, float, int, float]:
    points = worker_points()
    labels, centers, iterations = fit_kmeans(points, k)
    inertia = inertia_from_centers(points, labels, centers)
    intra_variance = inertia / len(points)
    return k, intra_variance, iterations, inertia


def compute_silhouette_candidate(k: int) -> tuple[int, float, int]:
    points = worker_points()
    scored_indices = worker_scored_indices()
    labels, _, iterations = fit_kmeans(points, k)
    score = silhouette_score(points, labels, k, scored_indices)
    return k, score, iterations


# ---------------------------------------------------------------------------
# Cache paths / metadata
# ---------------------------------------------------------------------------
def elbow_cache_path(year: int) -> Path:
    return DATA_OUTPUT_DIR / f"kmeans_elbow_scores_{year}.csv"


def source_file_metadata(year: int) -> dict[str, str]:
    try:
        st = NETCDF_FILE.stat()
        return {
            "source_size_bytes": str(st.st_size),
            "source_mtime_ns": str(st.st_mtime_ns),
        }
    except OSError:
        return {"source_size_bytes": "", "source_mtime_ns": ""}


def base_cache_metadata(year: int, point_count: int) -> dict[str, str]:
    return {
        **source_file_metadata(year),
        "point_count": str(point_count),
        "features": "|".join(KMEANS_FEATURES),
        "log_transform_features": "|".join(sorted(LOG_TRANSFORM_FEATURES)),
        "random_seed": str(RANDOM_SEED),
        "max_iterations": str(MAX_ITERATIONS),
        "convergence_tolerance": str(CONVERGENCE_TOLERANCE),
    }


def silhouette_cache_path(year: int) -> Path:
    return DATA_OUTPUT_DIR / f"kmeans_silhouette_scores_{year}.csv"


def silhouette_cache_metadata(
    year: int,
    point_count: int,
    scored_point_count: int,
) -> dict[str, str]:
    return {
        **base_cache_metadata(year, point_count),
        "scored_point_count": str(scored_point_count),
    }


def load_elbow_score_cache(
    year: int,
    point_count: int,
) -> list[tuple[int, float, int]] | None:
    if REBUILD_K_SELECTION_CACHE:
        return None
    cache_path = elbow_cache_path(year)
    if not cache_path.is_file():
        return None
    expected_year = str(year)
    expected_metadata = base_cache_metadata(year, point_count)
    expected_ks = kmeans_candidate_values(point_count)
    try:
        with cache_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != list(ELBOW_CACHE_FIELDS):
                return None
            rows: list[tuple[int, float, int]] = []
            for row in reader:
                if row.get("year") != expected_year:
                    return None
                if any(row.get(k) != v for k, v in expected_metadata.items()):
                    return None
                rows.append((int(row["k"]), float(row["intra_variance"]), int(row["iterations"])))
    except (OSError, ValueError):
        return None
    rows.sort(key=lambda row: row[0])
    if [k for k, _, _ in rows] != expected_ks:
        return None
    print(f"Loading cached KMeans elbow scores {cache_path}")
    return rows


def save_elbow_score_cache(
    year: int,
    point_count: int,
    scores: list[tuple[int, float, int]],
) -> Path:
    cache_path = elbow_cache_path(year)
    metadata = base_cache_metadata(year, point_count)
    DATA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ELBOW_CACHE_FIELDS)
        writer.writeheader()
        for k, intra_variance, iterations in scores:
            writer.writerow({
                "year": str(year),
                "k": k,
                "intra_variance": f"{intra_variance:.12g}",
                "iterations": iterations,
                **metadata,
            })
    return cache_path


def load_silhouette_score_cache(
    year: int,
    point_count: int,
    scored_point_count: int,
) -> list[tuple[int, float, int]] | None:
    if REBUILD_K_SELECTION_CACHE:
        return None
    cache_path = silhouette_cache_path(year)
    if not cache_path.is_file():
        return None
    expected_year = str(year)
    expected_metadata = silhouette_cache_metadata(year, point_count, scored_point_count)
    expected_ks = kmeans_candidate_values(point_count)
    try:
        with cache_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != list(SILHOUETTE_CACHE_FIELDS):
                return None
            rows: list[tuple[int, float, int]] = []
            for row in reader:
                if row.get("year") != expected_year:
                    return None
                if any(row.get(k) != v for k, v in expected_metadata.items()):
                    return None
                rows.append((int(row["k"]), float(row["silhouette_score"]), int(row["iterations"])))
    except (OSError, ValueError):
        return None
    rows.sort(key=lambda row: row[0])
    if [k for k, _, _ in rows] != expected_ks:
        return None
    print(f"Loading cached KMeans silhouette scores {cache_path}")
    return rows


def save_silhouette_score_cache(
    year: int,
    point_count: int,
    scored_point_count: int,
    scores: list[tuple[int, float, int]],
) -> Path:
    cache_path = silhouette_cache_path(year)
    metadata = silhouette_cache_metadata(year, point_count, scored_point_count)
    DATA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SILHOUETTE_CACHE_FIELDS)
        writer.writeheader()
        for k, score, iterations in scores:
            writer.writerow({
                "year": str(year),
                "k": k,
                "silhouette_score": f"{score:.12g}",
                "iterations": iterations,
                **metadata,
            })
    return cache_path


# ---------------------------------------------------------------------------
# K-selection scoring
# ---------------------------------------------------------------------------
def compute_elbow_scores(
    points: list[list[float]],
    candidate_values: list[int],
) -> list[tuple[int, float, int]]:
    worker_count = resolved_worker_count(len(candidate_values))
    print_k_selection_worker_count_once(worker_count)
    total_tasks = len(candidate_values)
    completed_tasks = 0
    progress_label = "K selection"
    print_progress(completed_tasks, total_tasks, progress_label)

    if worker_count == 1:
        initialize_k_selection_worker(points, None)
        results = []
        for k in candidate_values:
            results.append(compute_elbow_candidate(k))
            completed_tasks += 1
            print_progress(completed_tasks, total_tasks, progress_label)
    else:
        results = []
        with ProcessPoolExecutor(
            max_workers=worker_count,
            initializer=initialize_k_selection_worker,
            initargs=(points, None),
        ) as executor:
            futures = [executor.submit(compute_elbow_candidate, k) for k in candidate_values]
            for future in as_completed(futures):
                results.append(future.result())
                completed_tasks += 1
                print_progress(completed_tasks, total_tasks, progress_label)

    finish_progress_if_needed(completed_tasks, total_tasks, progress_label)
    results.sort(key=lambda row: row[0])
    for k, intra_variance, iterations, inertia in results:
        print(
            f"k={k}: intra_variance={intra_variance:.6f}, "
            f"inertia={inertia:.6f}, iterations={iterations}"
        )
    return [(k, intra_variance, iterations) for k, intra_variance, iterations, _ in results]


def compute_silhouette_scores(
    points: list[list[float]],
    candidate_values: list[int],
    scored_indices: list[int],
) -> list[tuple[int, float, int]]:
    worker_count = resolved_worker_count(len(candidate_values))
    print_k_selection_worker_count_once(worker_count)
    total_tasks = len(candidate_values)
    completed_tasks = 0
    progress_label = "K selection"
    print_progress(completed_tasks, total_tasks, progress_label)

    if worker_count == 1:
        initialize_k_selection_worker(points, scored_indices)
        results = []
        for k in candidate_values:
            results.append(compute_silhouette_candidate(k))
            completed_tasks += 1
            print_progress(completed_tasks, total_tasks, progress_label)
    else:
        results = []
        with ProcessPoolExecutor(
            max_workers=worker_count,
            initializer=initialize_k_selection_worker,
            initargs=(points, scored_indices),
        ) as executor:
            futures = [executor.submit(compute_silhouette_candidate, k) for k in candidate_values]
            for future in as_completed(futures):
                results.append(future.result())
                completed_tasks += 1
                print_progress(completed_tasks, total_tasks, progress_label)

    finish_progress_if_needed(completed_tasks, total_tasks, progress_label)
    results.sort(key=lambda row: row[0])
    for k, score, iterations in results:
        print(f"k={k}: silhouette={score:.6f}, iterations={iterations}")
    return results


def inertia_from_centers(
    points: list[list[float]],
    labels: list[int],
    centers: list[list[float]],
) -> float:
    return sum(
        squared_distance(point, centers[label])
        for point, label in zip(points, labels)
    )


def silhouette_score(
    points: list[list[float]],
    labels: list[int],
    cluster_count: int,
    scored_indices: list[int],
) -> float:
    if len(scored_indices) < 2:
        return -math.inf
    cluster_counts = [0 for _ in range(cluster_count)]
    for label in labels:
        if label < 0 or label >= cluster_count:
            return -math.inf
        cluster_counts[label] += 1
    if sum(1 for c in cluster_counts if c > 0) < 2:
        return -math.inf
    score_sum = 0.0
    for point_index in scored_indices:
        point = points[point_index]
        own_label = labels[point_index]
        distance_sums = [0.0 for _ in range(cluster_count)]
        for other_point, other_label in zip(points, labels):
            distance_sums[other_label] += math.sqrt(squared_distance(point, other_point))
        if cluster_counts[own_label] <= 1:
            continue
        own_average_distance = distance_sums[own_label] / (cluster_counts[own_label] - 1)
        nearest_other_average = math.inf
        for label, count in enumerate(cluster_counts):
            if label == own_label or count == 0:
                continue
            nearest_other_average = min(nearest_other_average, distance_sums[label] / count)
        denominator = max(own_average_distance, nearest_other_average)
        if denominator > 0.0 and nearest_other_average < math.inf:
            score_sum += (nearest_other_average - own_average_distance) / denominator
    return score_sum / len(scored_indices)


def elbow_distance(
    k: int,
    value: float,
    min_k: int,
    max_k: int,
    min_value: float,
    max_value: float,
) -> float:
    if max_k == min_k or max_value == min_value:
        return 0.0
    x = (k - min_k) / (max_k - min_k)
    y = (value - min_value) / (max_value - min_value)
    return abs(x + y - 1.0) / math.sqrt(2.0)


def choose_elbow_k(scores: list[tuple[int, float, int]]) -> tuple[int, float]:
    if not scores:
        raise ValueError("No elbow scores available")
    min_k = scores[0][0]
    max_k = scores[-1][0]
    values = [value for _, value, _ in scores]
    min_value = min(values)
    max_value = max(values)
    selected_k, selected_value, _ = max(
        scores,
        key=lambda row: (
            elbow_distance(row[0], row[1], min_k, max_k, min_value, max_value),
            -row[0],
        ),
    )
    return selected_k, selected_value


def select_optimal_k_by_elbow(
    year: int,
    points: list[list[float]],
) -> tuple[int, float, list[tuple[int, float, int]]]:
    cached_scores = load_elbow_score_cache(year, len(points))
    if cached_scores is not None:
        scores = cached_scores
    else:
        candidate_values = kmeans_candidate_values(len(points))
        print(
            "Computing KMeans elbow scores for "
            f"k={candidate_values[0]}..{candidate_values[-1]} on all {len(points)} points"
        )
        scores = compute_elbow_scores(points, candidate_values)
        cache_path = save_elbow_score_cache(year, len(points), scores)
        print(f"Saved {cache_path}")
    selected_k, selected_value = choose_elbow_k(scores)
    return selected_k, selected_value, scores


def select_optimal_k_by_silhouette(
    year: int,
    points: list[list[float]],
) -> tuple[int, float, list[tuple[int, float, int]]]:
    scored_indices = silhouette_scored_indices(len(points))
    cached_scores = load_silhouette_score_cache(year, len(points), len(scored_indices))
    if cached_scores is not None:
        scores = cached_scores
    else:
        candidate_values = kmeans_candidate_values(len(points))
        print(
            "Computing KMeans silhouette scores for "
            f"k={candidate_values[0]}..{candidate_values[-1]} on {len(scored_indices)} scored points"
        )
        scores = compute_silhouette_scores(points, candidate_values, scored_indices)
        cache_path = save_silhouette_score_cache(year, len(points), len(scored_indices), scores)
        print(f"Saved {cache_path}")
    selected_k, selected_score, _ = max(scores, key=lambda row: (row[1], -row[0]))
    return selected_k, selected_score, scores


def silhouette_scored_indices(point_count: int) -> list[int]:
    if SILHOUETTE_SAMPLE_SIZE is None or SILHOUETTE_SAMPLE_SIZE >= point_count:
        return list(range(point_count))
    rng = random.Random(RANDOM_SEED)
    return sorted(rng.sample(range(point_count), SILHOUETTE_SAMPLE_SIZE))


def select_optimal_k(
    year: int,
    points: list[list[float]],
) -> tuple[int, float, list[tuple[int, float, int]]]:
    if KMEANS_K_SELECTION_METHOD == "elbow":
        return select_optimal_k_by_elbow(year, points)
    return select_optimal_k_by_silhouette(year, points)


def selection_score_text(score: float) -> str:
    if KMEANS_K_SELECTION_METHOD == "elbow":
        return f"intra_variance={score:.6f}"
    return f"silhouette={score:.4f}"


# ---------------------------------------------------------------------------
# Output writers / helpers
# ---------------------------------------------------------------------------
def labels_to_grid(
    labels: list[int],
    locations: list[tuple[float, float]],
    lats: list[float],
    lons: list[float],
) -> list[list[float]]:
    lat_index = {lat: i for i, lat in enumerate(lats)}
    lon_index = {lon: i for i, lon in enumerate(lons)}
    grid = [[math.nan for _ in lons] for _ in lats]
    for label, (lat, lon) in zip(labels, locations):
        grid[lat_index[lat]][lon_index[lon]] = label
    return grid


def cluster_feature_barycenter_locations(
    points: list[list[float]],
    labels: list[int],
    locations: list[tuple[float, float]],
    cluster_count: int,
) -> list[tuple[int, float, float]]:
    if not points:
        return []
    dimension_count = len(points[0])
    sums = [[0.0 for _ in range(dimension_count)] for _ in range(cluster_count)]
    counts = [0 for _ in range(cluster_count)]
    for point, label in zip(points, labels):
        if label < 0 or label >= cluster_count:
            continue
        counts[label] += 1
        for d in range(dimension_count):
            sums[label][d] += point[d]
    centers: list[list[float] | None] = []
    for label, count in enumerate(counts):
        if count == 0:
            centers.append(None)
            continue
        centers.append([value / count for value in sums[label]])
    best_distances = [math.inf for _ in range(cluster_count)]
    representatives: list[tuple[int, float, float] | None] = [None for _ in range(cluster_count)]
    for point, label, (lat, lon) in zip(points, labels, locations):
        if label < 0 or label >= cluster_count:
            continue
        center = centers[label]
        if center is None:
            continue
        distance = squared_distance(point, center)
        if distance < best_distances[label]:
            best_distances[label] = distance
            representatives[label] = (label, lat, lon)
    return [r for r in representatives if r is not None]


def cluster_feature_means(
    points: list[list[float]],
    labels: list[int],
    cluster_count: int,
) -> tuple[list[list[float]], list[int]]:
    if not points:
        return [], []
    dimension_count = len(points[0])
    sums = [[0.0 for _ in range(dimension_count)] for _ in range(cluster_count)]
    counts = [0 for _ in range(cluster_count)]
    for point, label in zip(points, labels):
        if label < 0 or label >= cluster_count:
            continue
        counts[label] += 1
        for d in range(dimension_count):
            sums[label][d] += point[d]
    means: list[list[float]] = []
    for label, count in enumerate(counts):
        if count == 0:
            means.append([math.nan for _ in range(dimension_count)])
            continue
        means.append([value / count for value in sums[label]])
    return means, counts


def classified_points_all_years_output_path() -> Path:
    return DATA_OUTPUT_DIR / "kmeans_classified_points_all_years.csv"


def cluster_barycenters_output_path(year: int, cluster_count: int) -> Path:
    return DATA_OUTPUT_DIR / f"kmeans_cluster_barycenters_k{cluster_count}_{year}.csv"


def write_classified_points_all_years_csv(
    classified_points_by_year: dict[int, dict[tuple[float, float], int]],
) -> Path:
    output_path = classified_points_all_years_output_path()
    years = sorted(classified_points_by_year)
    locations = sorted(
        {
            location
            for classified_points in classified_points_by_year.values()
            for location in classified_points
        },
        key=lambda location: (location[1], location[0]),
    )
    DATA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("lon", "lat", *(f"kmeans_cluster_{year}" for year in years)))
        for lon, lat in locations:
            writer.writerow(
                (
                    lon,
                    lat,
                    *(
                        classified_points_by_year[year].get((lon, lat), "")
                        for year in years
                    ),
                )
            )
    return output_path


def write_cluster_barycenters_csv(
    year: int,
    means: list[list[float]],
    counts: list[int],
    cluster_count: int,
) -> Path:
    output_path = cluster_barycenters_output_path(year, cluster_count)
    DATA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("cluster", "point_count", *KMEANS_FEATURES))
        for label in range(cluster_count):
            writer.writerow((label + 1, counts[label], *means[label]))
    return output_path


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def import_plotting_modules():
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import BoundaryNorm, ListedColormap
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
    except ModuleNotFoundError as exc:
        missing = exc.name or "a required plotting package"
        raise SystemExit(
            f"Missing Python package: {missing}\n"
            "Install the plotting dependencies with:\n"
            "  pip install matplotlib cartopy"
        ) from exc
    return plt, BoundaryNorm, ListedColormap, ccrs, cfeature


def add_world_overlay(ax, cfeature) -> None:
    ax.set_global()
    ax.add_feature(cfeature.LAND, facecolor=LAND_COLOR, edgecolor="none", zorder=0)
    ax.add_feature(cfeature.OCEAN, facecolor=OCEAN_COLOR, edgecolor="none", zorder=0)
    ax.coastlines(resolution=COASTLINE_RESOLUTION, linewidth=COASTLINE_WIDTH, color="black", zorder=2)
    ax.add_feature(
        cfeature.BORDERS, linewidth=BORDER_WIDTH, edgecolor="black",
        alpha=BORDER_ALPHA, zorder=2,
    )


def add_gridlines(ax) -> None:
    gridlines = ax.gridlines(
        draw_labels=True, linewidth=GRIDLINE_WIDTH, color=GRIDLINE_COLOR,
        alpha=GRIDLINE_ALPHA, linestyle=GRIDLINE_STYLE,
    )
    gridlines.top_labels = False
    gridlines.right_labels = False


def add_barycenter_markers(
    ax,
    ccrs,
    marker_locations: list[tuple[int, float, float]],
) -> None:
    for label, lat, lon in marker_locations:
        ax.scatter(
            [lon], [lat], marker="*", s=BARYCENTER_MARKER_SIZE,
            color=CLUSTER_COLORS[label], edgecolors="black",
            linewidths=BARYCENTER_EDGE_WIDTH, transform=ccrs.PlateCarree(), zorder=4,
        )


def plot_kmeans_map(
    year: int,
    grid: list[list[float]],
    lats: list[float],
    lons: list[float],
    marker_locations: list[tuple[int, float, float]],
    cluster_count: int,
    iterations: int,
    selection_score: float,
) -> Path:
    output_path = FIGURE_OUTPUT_DIR / f"kmeans_k{cluster_count}_{year}.png"

    plt, BoundaryNorm, ListedColormap, ccrs, cfeature = import_plotting_modules()
    lon_min, lon_max = cell_extent(lons)
    lat_min, lat_max = cell_extent(lats)
    cmap = ListedColormap(CLUSTER_COLORS[:cluster_count])
    norm = BoundaryNorm([label - 0.5 for label in range(cluster_count + 1)], cluster_count)

    fig, ax = plt.subplots(
        figsize=FIGSIZE,
        subplot_kw={"projection": ccrs.PlateCarree()},
        constrained_layout=True,
    )
    add_world_overlay(ax, cfeature)
    image = ax.imshow(
        grid, origin="lower", extent=(lon_min, lon_max, lat_min, lat_max),
        transform=ccrs.PlateCarree(), cmap=cmap, norm=norm,
        interpolation="nearest", zorder=1,
    )
    add_gridlines(ax)
    add_barycenter_markers(ax, ccrs, marker_locations)
    ax.set_title(
        f"KMeans world map - {year} - k={cluster_count} - "
        f"{KMEANS_K_SELECTION_METHOD} {selection_score_text(selection_score)} - "
        f"{iterations} iterations",
        fontsize=TITLE_SIZE,
    )
    colorbar = fig.colorbar(
        image, ax=ax, orientation="horizontal",
        pad=COLORBAR_PAD, shrink=COLORBAR_SHRINK, ticks=list(range(cluster_count)),
    )
    colorbar.set_ticklabels([f"Cluster {label + 1}" for label in range(cluster_count)])
    colorbar.set_label(f"KMeans classes from: {', '.join(KMEANS_FEATURES)}")

    FIGURE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=DPI)
    if SHOW:
        plt.show()
    else:
        plt.close(fig)
    return output_path


def plot_cluster_mean_profiles(
    year: int,
    means: list[list[float]],
    counts: list[int],
    cluster_count: int,
) -> Path:
    output_path = FIGURE_OUTPUT_DIR / f"kmeans_cluster_means_k{cluster_count}_{year}.png"

    plt, _, _, _, _ = import_plotting_modules()
    cluster_labels = [f"Cluster {l+1} (n={counts[l]})" for l in range(cluster_count)]
    y_positions = list(range(cluster_count))
    colors = CLUSTER_COLORS[:cluster_count]

    fig, axes = plt.subplots(
        1, len(KMEANS_FEATURES), figsize=PROFILE_FIGSIZE,
        sharey=True, constrained_layout=True,
    )
    if len(KMEANS_FEATURES) == 1:
        axes = [axes]

    for dimension, (ax, feature) in enumerate(zip(axes, KMEANS_FEATURES)):
        values = [cluster_means[dimension] for cluster_means in means]
        ax.barh(
            y_positions, values, height=PROFILE_BAR_HEIGHT,
            color=colors, edgecolor="black", linewidth=0.45,
        )
        ax.set_title(feature, fontsize=TITLE_SIZE - 2)
        ax.set_xlabel("Mean value")
        ax.grid(axis="x", linewidth=0.35, alpha=0.45)
        ax.set_axisbelow(True)

    axes[0].set_yticks(y_positions)
    axes[0].set_yticklabels(cluster_labels)
    axes[0].invert_yaxis()
    fig.suptitle(f"KMeans cluster mean profiles - {year} - k={cluster_count}", fontsize=TITLE_SIZE)

    FIGURE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=DPI)
    if SHOW:
        plt.show()
    else:
        plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# Per-year pipeline + main
# ---------------------------------------------------------------------------
def process_year(dataset, year: int) -> tuple[int, dict[tuple[float, float], int]]:
    print(f"Loading year {year} from {NETCDF_FILE.name}")
    points, raw_points, locations, lats, lons = load_points(dataset, year)
    scaled_points = standardize(points)

    selected_k, selected_score, _ = select_optimal_k(year, scaled_points)
    print(
        f"Selected K: {selected_k} by {KMEANS_K_SELECTION_METHOD} "
        f"with {selection_score_text(selected_score)}"
    )

    labels, iterations = kmeans(scaled_points, selected_k)

    grid = labels_to_grid(labels, locations, lats, lons)
    marker_locations = cluster_feature_barycenter_locations(
        scaled_points, labels, locations, selected_k,
    )
    output_path = plot_kmeans_map(
        year, grid, lats, lons, marker_locations, selected_k, iterations, selected_score,
    )
    print(f"Saved {output_path}")

    means, counts = cluster_feature_means(raw_points, labels, selected_k)
    barycenters_output = write_cluster_barycenters_csv(year, means, counts, selected_k)
    print(f"Saved {barycenters_output}")

    profile_output_path = plot_cluster_mean_profiles(year, means, counts, selected_k)
    print(f"Saved {profile_output_path}")

    if PLOT_MISSING_VALUES:
        print("note: PLOT_MISSING_VALUES=True is not implemented for the NetCDF path yet")

    return year, {
        (lon, lat): label + 1
        for (lat, lon), label in zip(locations, labels)
    }


def main() -> int:
    try:
        validate_config()
        if not NETCDF_FILE.is_file():
            raise FileNotFoundError(f"NetCDF not found: {NETCDF_FILE}")
    except (FileNotFoundError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 1

    import xarray as xr

    classified_points_by_year: dict[int, dict[tuple[float, float], int]] = {}
    with xr.open_dataset(NETCDF_FILE) as dataset:
        if "time" in dataset.dims:
            available = [int(y) for y in dataset["time"].values]
        else:
            available = list(KMEANS_YEARS)
        missing = [y for y in KMEANS_YEARS if y not in available]
        if missing:
            raise SystemExit(
                f"Years missing from NetCDF: {missing}. Available: {available}"
            )

        total_years = len(KMEANS_YEARS)
        for index, year in enumerate(KMEANS_YEARS, start=1):
            print(f"\n=== KMeans year {year} ({index}/{total_years}) ===")
            year_label, classified_points = process_year(dataset, year)
            classified_points_by_year[year_label] = classified_points

    classified_points_output = write_classified_points_all_years_csv(
        classified_points_by_year
    )
    print(f"Saved {classified_points_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
