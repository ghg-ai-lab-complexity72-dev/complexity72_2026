from __future__ import annotations

import csv
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

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
   
# ---------------------------------------------------------------------------
# Edit these values directly before running the script.
# ---------------------------------------------------------------------------
#NETCDF_FILE = BASE_DIR / "Input" / "unified_annual_carbon_dataset_2015_2025.nc"
BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = find_repo_root(BASE_DIR)
INPUT_DIR = REPO_ROOT / "Input"
# Data artifacts (CSV/JSON) go under Output/clustering
DATA_OUTPUT_DIR = REPO_ROOT / "Output" / "clustering"
# Figures (PNG/SVG) go under Figures_and_maps/clustering
FIGURE_OUTPUT_DIR = REPO_ROOT / "Figures_and_maps" / "clustering"

# Input produced by k_means.py — lives in DATA_OUTPUT_DIR now
CLUSTER_CSV = DATA_OUTPUT_DIR / "kmeans_classified_points_all_years.csv"
THRESHOLDS = (1.0,)
WORKERS = 15                      # kept for API compatibility; NetCDF path is single-process
SKIP_COMPONENTS = False
SAVE_IMAGE_FILES = True
TOP_COMPONENTS_TO_PLOT = 9

YEARS = tuple(range(2015, 2025))  # 2015..2024 inclusive

PROGRESS_BAR_WIDTH = 40
MAP_EXTENT = (-180, 180, -90, 90)
MAP_OCEAN_COLOR = "#e9f3fb"
MAP_LAND_COLOR = "#f7f7f2"
KMEANS_STYLE_COMPONENT_COLORS = (
    "#808080", "#DDD69C", "#F5E4AD", "#E47E53", "#1F642F",
    "#DA70D6", "#000000", "#008080", "#7CFC00",
)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MatrixMetadata:
    pixel_count: int
    year_count: int
    cluster_columns: tuple[str, ...]
    lon: np.ndarray
    lat: np.ndarray
    clusters: np.ndarray  # shape (pixel_count, year_count), int16, -1 = missing


class DisjointSet:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.size = [1] * size

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left == root_right:
            return
        if self.size[root_left] < self.size[root_right]:
            root_left, root_right = root_right, root_left
        self.parent[root_right] = root_left
        self.size[root_left] += self.size[root_right]

    def component_sizes_by_root(self) -> dict[int, int]:
        counts: Counter[int] = Counter()
        for index in range(len(self.parent)):
            counts[self.find(index)] += 1
        return dict(counts)

    def groups(self, min_size: int = 1) -> list[list[int]]:
        grouped: dict[int, list[int]] = defaultdict(list)
        for index in range(len(self.parent)):
            grouped[self.find(index)].append(index)
        return [group for group in grouped.values() if len(group) >= min_size]


# ---------------------------------------------------------------------------
# NetCDF loading
# ---------------------------------------------------------------------------
def load_metadata(csv_path: Path) -> MatrixMetadata:
    """
    Read cluster columns + lon/lat from k_means.py's output CSV.
    Columns expected: lon, lat, kmeans_cluster_YYYY, ...
    """
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Cluster CSV not found: {csv_path}\n"
            "Run k_means.py first."
        )

    lon_list: list[float] = []
    lat_list: list[float] = []
    cluster_cols: list[str] = []
    cluster_rows: list[list[int]] = []

    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        if "lon" not in header or "lat" not in header:
            raise ValueError(f"{csv_path} must contain 'lon' and 'lat' columns")

        lon_idx = header.index("lon")
        lat_idx = header.index("lat")
        cluster_cols = [c for c in header if c.startswith("kmeans_cluster_")]
        if not cluster_cols:
            raise ValueError(f"No kmeans_cluster_* columns in {csv_path}")
        cluster_idx = [header.index(c) for c in cluster_cols]

        for row in reader:
            lon_list.append(float(row[lon_idx]))
            lat_list.append(float(row[lat_idx]))
            cluster_rows.append(
                [int(row[i]) if row[i] != "" else -1 for i in cluster_idx]
            )

    if not cluster_rows:
        raise ValueError(f"{csv_path} has no data rows")

    pixel_count = len(cluster_rows)
    clusters = np.asarray(cluster_rows, dtype=np.int16)  # (N, year_count), -1 = missing

    return MatrixMetadata(
        pixel_count=pixel_count,
        year_count=len(cluster_cols),
        cluster_columns=tuple(cluster_cols),
        lon=np.asarray(lon_list, dtype=float),
        lat=np.asarray(lat_list, dtype=float),
        clusters=clusters,
    )


def build_same_count_matrix(metadata: MatrixMetadata) -> np.ndarray:
    """
    Build the (N, N) uint8 matrix where entry [i, j] = number of years in which
    pixel i and pixel j share the same cluster.

    Pixels whose cluster is -1 for a given year contribute 0 for that year
    (they are treated as not-comparable). With k_means.py's output this never
    happens because missing values are dropped upstream, but the guard keeps
    the denominator well-defined.
    """
    n = metadata.pixel_count
    clusters = metadata.clusters            # (N, year_count)
    year_count = metadata.year_count
    same = np.zeros((n, n), dtype=np.uint8)

    # Row-block-wise build to keep peak temporaries small.
    block = 2048
    total_blocks = (n + block - 1) // block
    for b_idx, start in enumerate(range(0, n, block)):
        end = min(n, start + block)
        block_clusters = clusters[start:end]                      # (B, Y)
        # (B, 1, Y) == (1, N, Y) -> (B, N, Y), then sum over years
        eq = block_clusters[:, None, :] == clusters[None, :, :]   # (B, N, Y)
        valid = (block_clusters[:, None, :] >= 0) & (clusters[None, :, :] >= 0)
        counts = (eq & valid).sum(axis=2).astype(np.uint8)        # (B, N)
        same[start:end, :] = counts
        _print_progress("Building same-count matrix", b_idx + 1, total_blocks)

    same = np.maximum(same, same.T)
    np.fill_diagonal(same, 0)
    return same

# ---------------------------------------------------------------------------
# Threshold helpers (unchanged semantics)
# ---------------------------------------------------------------------------
def validate_thresholds(thresholds: Iterable[float]) -> tuple[float, ...]:
    clean = tuple(sorted(set(float(t) for t in thresholds)))
    if not clean:
        raise ValueError("At least one threshold is required")
    for threshold in clean:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"Threshold must be between 0 and 1: {threshold}")
    return clean


def min_count_for_probability(threshold: float, denominator: int) -> int:
    return math.ceil(threshold * denominator - 1e-12)


def threshold_token(threshold: float) -> str:
    return f"{threshold:g}".replace(".", "_")


def _print_progress(label: str, done: int, total: int) -> None:
    ratio = 0.0 if total <= 0 else min(1.0, done / total)
    filled = int(PROGRESS_BAR_WIDTH * ratio)
    bar = "#" * filled + "-" * (PROGRESS_BAR_WIDTH - filled)
    print(f"\r{label}: [{bar}] {ratio * 100:5.1f}%", end="", flush=True)
    if done >= total:
        print()


# ---------------------------------------------------------------------------
# Component detection (replaces the old binary scan)
# ---------------------------------------------------------------------------
def build_components(
    same: np.ndarray,
    metadata: MatrixMetadata,
    thresholds: tuple[float, ...],
) -> dict[float, DisjointSet]:
    n = metadata.pixel_count
    year_count = metadata.year_count

    dsu_by_threshold: dict[float, DisjointSet] = {t: DisjointSet(n) for t in thresholds}
    min_count_by_threshold = {
        t: min_count_for_probability(t, year_count) for t in thresholds
    }

    total_pairs = n * (n - 1) // 2
    done = 0
    _print_progress("Building components", done, total_pairs)

    # Iterate upper triangle row by row; vectorized per row for speed.
    for i in range(n - 1):
        row = same[i, i + 1:]                       # length n - i - 1
        j_indices = np.arange(i + 1, n)
        for threshold in thresholds:
            min_count = min_count_by_threshold[threshold]
            hit = np.nonzero(row >= min_count)[0]
            if hit.size:
                dsu = dsu_by_threshold[threshold]
                for offset in hit:
                    dsu.union(i, int(j_indices[offset]))
        done += row.size
        if i % 256 == 0 or i == n - 2:
            _print_progress("Building components", done, total_pairs)

    return dsu_by_threshold


# ---------------------------------------------------------------------------
# Plot / output helpers (identical to original)
# ---------------------------------------------------------------------------
def load_pyplot():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "SAVE_IMAGE_FILES=True requires matplotlib. Install matplotlib or set "
            "SAVE_IMAGE_FILES=False at the top of this script."
        ) from exc
    return plt


def load_cartopy():
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
    except ModuleNotFoundError as exc:
        raise RuntimeError("Component maps require Cartopy. Install cartopy first.") from exc
    return ccrs, cfeature


def color_for_component(root: int, top_roots: list[int]) -> str:
    if root not in top_roots:
        return "#d9d9d9"
    return KMEANS_STYLE_COMPONENT_COLORS[top_roots.index(root) % len(KMEANS_STYLE_COMPONENT_COLORS)]


def write_component_size_histograms(
    output_dir: Path,
    metadata: MatrixMetadata,
    thresholds: tuple[float, ...],
    dsu_by_threshold: dict[float, DisjointSet],
) -> list[Path]:
    plt = load_pyplot()
    from matplotlib.ticker import FuncFormatter, LogLocator
    written: list[Path] = []

    for threshold in thresholds:
        dsu = dsu_by_threshold[threshold]
        sizes = list(dsu.component_sizes_by_root().values())
        size_counts = Counter(sizes)
        size_values = sorted(size_counts)
        count_values = [size_counts[s] for s in size_values]
        token = threshold_token(threshold)
        min_count = min_count_for_probability(threshold, metadata.year_count)
        raw_path = DATA_OUTPUT_DIR / f"component_size_histogram_raw_t_{token}.csv"
        path = FIGURE_OUTPUT_DIR / f"component_size_histogram_t_{token}.png"
        written += [raw_path, path]

        with raw_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(("threshold", "min_same_count", "component_size",
                             "component_count", "total_pixels_at_size",
                             "pixel_fraction_at_size"))
            for size, count in zip(size_values, count_values):
                total = size * count
                writer.writerow((threshold, min_count, size, count, total,
                                 total / metadata.pixel_count))

        fig, ax = plt.subplots(figsize=(min(15, max(10, 4 + 0.22 * len(size_values))), 6), dpi=170)
        if sizes:
            y_max = max(count_values)
            bar_log_width = 0.01
            left = [s / (10 ** (bar_log_width / 2)) for s in size_values]
            widths = [s * (10 ** (bar_log_width / 2) - 10 ** (-bar_log_width / 2)) for s in size_values]
            ax.bar(left, count_values, color="#4f7fad", edgecolor="white",
                   linewidth=0.7, width=widths, align="edge")
            ax.set_xscale("log")
            ax.set_xlim(left[0] * 0.95, (left[-1] + widths[-1]) * 1.05)
            top_sizes = sorted(set(sorted(sizes, reverse=True)[:TOP_COMPONENTS_TO_PLOT]))
            ax.set_xticks(top_sizes)
            ax.set_xticklabels([f"{s}\n({s / metadata.pixel_count:.2%})" for s in top_sizes],
                               rotation=45, ha="right", fontsize=8)
            ax.set_yscale("log")
            ax.set_ylim(0.8, max(1.5, y_max * 1.6))
            ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v):d}" if v >= 1 else ""))
            ax.yaxis.set_minor_formatter(FuncFormatter(lambda _v, _p: ""))

        ax.set_title(f"Component size distribution at threshold {threshold:g}")
        ax.set_xlabel("Component size (number of pixels)")
        ax.set_ylabel("Number of components (log scale)")
        ax.grid(axis="both", color="#d0d7de", linewidth=0.7, alpha=0.65)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        fig.savefig(path, format="png", bbox_inches="tight")
        plt.close(fig)

    return written


def write_component_images(
    output_dir: Path,
    metadata: MatrixMetadata,
    thresholds: tuple[float, ...],
    dsu_by_threshold: dict[float, DisjointSet],
) -> list[Path]:
    plt = load_pyplot()
    ccrs, cfeature = load_cartopy()
    import matplotlib.patches as mpatches

    written: list[Path] = []
    lons = metadata.lon
    lats = metadata.lat

    for threshold in thresholds:
        dsu = dsu_by_threshold[threshold]
        sizes_by_root = dsu.component_sizes_by_root()
        sorted_roots = sorted(sizes_by_root, key=lambda r: (-sizes_by_root[r], r))
        token = threshold_token(threshold)
        FIGURE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        map_path = FIGURE_OUTPUT_DIR / f"component_map_t_{token}.png"
        written.append(map_path)

        top_roots = sorted_roots[:TOP_COMPONENTS_TO_PLOT]
        pixel_roots = [dsu.find(i) for i in range(metadata.pixel_count)]
        top_root_set = set(top_roots)

        top_mask = np.array([r in top_root_set for r in pixel_roots])
        other_mask = ~top_mask
        component_colors = [color_for_component(r, top_roots) for r in pixel_roots if r in top_root_set]

        fig = plt.figure(figsize=(16, 8.5), dpi=170)
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        ax.set_extent(MAP_EXTENT, crs=ccrs.PlateCarree())
        ax.set_facecolor(MAP_OCEAN_COLOR)
        ax.add_feature(cfeature.OCEAN, facecolor=MAP_OCEAN_COLOR, zorder=0)
        ax.add_feature(cfeature.LAND, facecolor=MAP_LAND_COLOR, edgecolor="none", zorder=0)
        ax.add_feature(cfeature.COASTLINE, edgecolor="black", linewidth=0.65, zorder=3)
        ax.add_feature(cfeature.BORDERS, edgecolor="#2f2f2f", linewidth=0.35, zorder=3)

        if other_mask.any():
            ax.scatter(lons[other_mask], lats[other_mask], c="#d9d9d9", s=7, marker="s",
                       linewidths=0, alpha=0.65, transform=ccrs.PlateCarree(), zorder=1)
        ax.scatter(lons[top_mask], lats[top_mask], c=component_colors, s=11, marker="s",
                   linewidths=0, transform=ccrs.PlateCarree(), zorder=2)

        ax.set_title(f"Top {TOP_COMPONENTS_TO_PLOT} persistent components at threshold {threshold:g}")
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.45,
                          color="#8a98a8", alpha=0.35, linestyle="--")
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {"size": 10}
        gl.ylabel_style = {"size": 10}

        handles = [
            mpatches.Patch(color=color_for_component(r, top_roots),
                           label=f"Component {i + 1}: {sizes_by_root[r]:,} pixels")
            for i, r in enumerate(top_roots)
        ]
        if metadata.pixel_count - sum(sizes_by_root[r] for r in top_root_set):
            handles.append(mpatches.Patch(color="#d9d9d9", label="Remaining"))
        ax.legend(handles=handles, title="Persistent components", loc="center left",
                  bbox_to_anchor=(1.005, 0.5), frameon=True, framealpha=1.0,
                  edgecolor="#2f2f2f", fontsize=12, title_fontsize=13,
                  borderpad=0.7, labelspacing=0.65, handlelength=1.4, handleheight=1.1)

        fig.subplots_adjust(left=0.02, right=0.80, top=0.94, bottom=0.06)
        fig.savefig(map_path, format="png", bbox_inches="tight", pad_inches=0.04)
        plt.close(fig)

    return written


# ---------------------------------------------------------------------------
# Long-format component CSV export
# ---------------------------------------------------------------------------
# Columns: lat, lon, time, emission, gosif, component_1, ..., component_N
# One row per (pixel, year) in the original NetCDF grid order:
#   lat from -89.5 (south) to +89.5 (north), lon from -179.5 to +179.5.
# NaN emission/gosif are written as 0 (same convention as k_means.py).
# ---------------------------------------------------------------------------
def load_year_arrays(
    dataset,
    year: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (lat, lon, emission, gosif) as 1-D arrays in the original
    CSV order (south-to-north lat, west-to-east lon). NaN/inf -> 0."""
    sel = dataset.sel(time=year, drop=True) if "time" in dataset.dims else dataset

    lat_axis = np.asarray(dataset["lat"].values, dtype=float).ravel()
    lon_axis = np.asarray(dataset["lon"].values, dtype=float).ravel()

    emission = np.asarray(sel["emission"].values, dtype=float)
    gosif = np.asarray(sel["gosif"].values, dtype=float)

    if lat_axis[0] > lat_axis[-1]:
        lat_axis = lat_axis[::-1]
        emission = emission[::-1, :]
        gosif = gosif[::-1, :]

    emission = np.where(np.isfinite(emission), emission, 0.0)
    gosif = np.where(np.isfinite(gosif), gosif, 0.0)

    lat_grid, lon_grid = np.meshgrid(lat_axis, lon_axis, indexing="ij")
    return lat_grid.ravel(), lon_grid.ravel(), emission.ravel(), gosif.ravel()


def build_location_index(metadata: MatrixMetadata) -> dict[tuple[float, float], int]:
    return {
        (round(float(lat), 4), round(float(lon), 4)): i
        for i, (lat, lon) in enumerate(zip(metadata.lat, metadata.lon))
    }


def write_long_component_csv(
    output_path: Path,
    metadata: MatrixMetadata,
    dataset,
    threshold: float,
    dsu: DisjointSet,
    top_roots: list[int],
) -> Path:
    """Write the long per-year component CSV for one threshold."""
    n_components = len(top_roots)
    root_to_index = {root: i + 1 for i, root in enumerate(top_roots)}

    # Precompute each pixel's component index (0 = not in top-N).
    pixel_component = np.zeros(metadata.pixel_count, dtype=np.int16)
    for i in range(metadata.pixel_count):
        root = dsu.find(i)
        if root in root_to_index:
            pixel_component[i] = root_to_index[root]

    location_index = build_location_index(metadata)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["lat", "lon", "time", "emission", "gosif"]
            + [f"component_{k}" for k in range(1, n_components + 1)]
        )

        for year in YEARS:
            lat, lon, emission, gosif = load_year_arrays(dataset, year)
            for la, lo, em, gs in zip(lat, lon, emission, gosif):
                key = (round(float(la), 4), round(float(lo), 4))
                idx = location_index.get(key)
                if idx is None:
                    components = [0] * n_components  # ocean / unclustered
                else:
                    comp = int(pixel_component[idx])
                    components = [1 if comp == k + 1 else 0
                                  for k in range(n_components)]
                writer.writerow([la, lo, year, em, gs, *components])

    return output_path
    
    

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> int:
    metadata = load_metadata(CLUSTER_CSV)
    thresholds = validate_thresholds(THRESHOLDS)

    DATA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    if not SKIP_COMPONENTS and SAVE_IMAGE_FILES:
        same = build_same_count_matrix(metadata)
        dsu_by_threshold = build_components(same, metadata, thresholds)
        del same

        # Both helpers now write to their respective directories internally.
        written += write_component_size_histograms(
           DATA_OUTPUT_DIR, metadata, thresholds, dsu_by_threshold
          )
        written += write_component_images(
           FIGURE_OUTPUT_DIR, metadata, thresholds, dsu_by_threshold
          )

        # Long-format CSV. Requires a NetCDF file and exactly one threshold.
        try:
            import xarray as xr
            netcdf_path = REPO_ROOT / "Input" / "unified_annual_carbon_dataset_2015_2025.nc"
            if len(thresholds) != 1:
                print(
                    f"Skipping long CSV: needs exactly one threshold, got {thresholds}"
                )
            elif not netcdf_path.is_file():
                print(f"Skipping long CSV: NetCDF not found at {netcdf_path}")
            else:
                threshold = thresholds[0]
                dsu = dsu_by_threshold[threshold]
                sizes_by_root = dsu.component_sizes_by_root()
                top_roots = sorted(
                    sizes_by_root, key=lambda r: (-sizes_by_root[r], r)
                )[:TOP_COMPONENTS_TO_PLOT]

                csv_path = DATA_OUTPUT_DIR / "co-occurrence-clusters.csv"
                with xr.open_dataset(netcdf_path) as dataset:
                    write_long_component_csv(
                        csv_path, metadata, dataset, threshold, dsu, top_roots
                    )
                written.append(csv_path)
        except ImportError as exc:
            print(f"Skipping long CSV: {exc}")

    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
