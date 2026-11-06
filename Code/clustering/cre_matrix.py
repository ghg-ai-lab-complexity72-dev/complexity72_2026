from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Locate repository root (works from any depth inside the repo).
# ---------------------------------------------------------------------------
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

# Data artifacts (CSV/JSON/bin) live under Output/clustering
DATA_OUTPUT_DIR = REPO_ROOT / "Output" / "clustering"

# Cluster labels produced by k_means.py (also under Output/clustering)
INPUT_CSV = DATA_OUTPUT_DIR / "kmeans_classified_points_all_years.csv"

# Intermediate artifacts consumed by run_stability_pipeline.py /
# plot_stability_heatmap.py / sophia.py
OUTPUT_MATRIX_BIN = DATA_OUTPUT_DIR / "pixel_cluster_stability_upper_uint8.bin"
OUTPUT_DENOMINATOR_BIN = DATA_OUTPUT_DIR / "pixel_cluster_stability_upper_denominator_uint8.bin"
OUTPUT_METADATA_JSON = DATA_OUTPUT_DIR / "pixel_cluster_stability_metadata.json"
OUTPUT_PIXELS_CSV = DATA_OUTPUT_DIR / "pixel_index.csv"

CLUSTER_PREFIX = "kmeans_cluster_"
ROW_BLOCK_SIZE = 1024
PROGRESS_BAR_WIDTH = 32


def print_progress(done: int, total: int) -> None:
    if total <= 0:
        return

    ratio = done / total
    filled_width = int(PROGRESS_BAR_WIDTH * ratio)
    bar = "#" * filled_width + "-" * (PROGRESS_BAR_WIDTH - filled_width)
    percent = ratio * 100.0
    print(
        f"\rMatrix creation: [{bar}] {done}/{total} rows ({percent:5.1f}%)",
        end="",
        flush=True,
    )

    if done == total:
        print()


def cluster_columns(df: pd.DataFrame) -> list[str]:
    columns = [column for column in df.columns if column.startswith(CLUSTER_PREFIX)]
    if not columns:
        raise ValueError(f"No columns starting with {CLUSTER_PREFIX!r} found")
    return sorted(columns)


def write_pixel_index(df: pd.DataFrame) -> None:
    with OUTPUT_PIXELS_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("pixel_id", "lon", "lat"))
        for pixel_index, row in enumerate(df[["lon", "lat"]].itertuples(index=False), start=1):
            writer.writerow((pixel_index, row.lon, row.lat))


def upper_triangle_value_count(pixel_count: int) -> int:
    return pixel_count * (pixel_count + 1) // 2


def write_metadata(
    columns: list[str],
    pixel_count: int,
    has_missing_values: bool,
) -> None:
    year_count = len(columns)
    metadata = {
        "format": "upper_triangle_row_major",
        "pixel_count": pixel_count,
        "stored_value_count": upper_triangle_value_count(pixel_count),
        "cluster_columns": columns,
        "year_count": year_count,
        "pixel_index_file": OUTPUT_PIXELS_CSV.name,
        "same_count_file": OUTPUT_MATRIX_BIN.name,
        "same_count_dtype": "uint8",
        "same_count_meaning": "number of comparable years where two pixels are in the same cluster",
        "indexing": "pixels are 1-based in pixel_index.csv, but binary offsets are 0-based",
        "storage_order": "for row i from 0 to N-1, values are stored for columns j=i..N-1",
    }

    if has_missing_values:
        metadata.update(
            {
                "denominator_file": OUTPUT_DENOMINATOR_BIN.name,
                "denominator_dtype": "uint8",
                "probability": "same_count / denominator_count; denominator_count=0 means unavailable",
            }
        )
    else:
        metadata.update(
            {
                "denominator_file": None,
                "probability": f"same_count / {year_count}",
            }
        )

    with OUTPUT_METADATA_JSON.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")


def same_cluster_probability_matrix_binary(
    clusters: np.ndarray,
    row_block_size: int = ROW_BLOCK_SIZE,
) -> None:
    pixel_count, year_count = clusters.shape
    if year_count > np.iinfo(np.uint8).max:
        raise ValueError("Too many years to store counts as uint8")

    valid = ~pd.isna(clusters)
    has_missing_values = not bool(valid.all())

    denominator_handle = (
        OUTPUT_DENOMINATOR_BIN.open("wb")
        if has_missing_values
        else None
    )

    with OUTPUT_MATRIX_BIN.open("wb") as matrix_handle:
        print_progress(0, pixel_count)

        for row_start in range(0, pixel_count, row_block_size):
            row_end = min(row_start + row_block_size, pixel_count)
            block = clusters[row_start:row_end]
            block_valid = valid[row_start:row_end]

            same_count = np.zeros((row_end - row_start, pixel_count), dtype=np.uint8)
            comparable_count = np.zeros((row_end - row_start, pixel_count), dtype=np.uint8)

            for year_index in range(year_count):
                comparable = block_valid[:, [year_index]] & valid[:, year_index]
                same = comparable & (block[:, [year_index]] == clusters[:, year_index])
                comparable_count += comparable
                same_count += same

            for offset in range(row_end - row_start):
                column_start = row_start + offset
                matrix_handle.write(same_count[offset, column_start:].tobytes())
                if denominator_handle is not None:
                    denominator_handle.write(
                        comparable_count[offset, column_start:].tobytes()
                    )

            matrix_handle.flush()
            if denominator_handle is not None:
                denominator_handle.flush()
            print_progress(row_end, pixel_count)

    if denominator_handle is not None:
        denominator_handle.close()


def main() -> int:
    DATA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_CSV.is_file():
        raise SystemExit(
            f"Cluster CSV not found: {INPUT_CSV}\n"
            "Run k_means.py first (it writes into Output/clustering/)."
        )

    df = pd.read_csv(INPUT_CSV)
    columns = cluster_columns(df)
    clusters = df[columns].to_numpy()

    write_pixel_index(df)
    same_cluster_probability_matrix_binary(clusters)
    write_metadata(
        columns=columns,
        pixel_count=len(df),
        has_missing_values=not bool((~pd.isna(clusters)).all()),
    )

    print(f"Wrote {OUTPUT_MATRIX_BIN}")
    print(f"Wrote {OUTPUT_PIXELS_CSV}")
    print(f"Wrote {OUTPUT_METADATA_JSON}")
    if OUTPUT_DENOMINATOR_BIN.exists():
        print(f"Wrote {OUTPUT_DENOMINATOR_BIN}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
