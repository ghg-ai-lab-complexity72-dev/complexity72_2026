from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import xarray as xr


def detect_spatial_dims(ds: xr.Dataset) -> tuple[str, str]:
    if "lat" in ds.coords and "lon" in ds.coords:
        return "lat", "lon"
    if "latitude" in ds.coords and "longitude" in ds.coords:
        return "latitude", "longitude"
    raise ValueError("Dataset must have lat/lon or latitude/longitude coordinates.")


def pick_country_column(world: gpd.GeoDataFrame) -> str:
    for candidate in ("ADMIN", "NAME", "NAME_EN", "SOVEREIGNT"):
        if candidate in world.columns:
            return candidate
    return world.columns[0]


def load_world(borders_path: Path) -> gpd.GeoDataFrame:
    if borders_path.exists():
        world = gpd.read_file(borders_path)
    else:
        world = gpd.read_file(
            "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
        )
        try:
            world.to_file(borders_path, driver="GPKG")
        except Exception:
            pass

    world = world.to_crs("EPSG:4326")
    world = world[world.geometry.notna() & ~world.geometry.is_empty].copy()
    return world


def build_country_grid(
    ds: xr.Dataset,
    world: gpd.GeoDataFrame,
    country_column: str,
) -> xr.DataArray:
    lat_dim, lon_dim = detect_spatial_dims(ds)

    lat_vals = ds[lat_dim].values
    lon_vals = ds[lon_dim].values

    lon_grid, lat_grid = np.meshgrid(lon_vals, lat_vals)

    points = gpd.GeoDataFrame(
        {
            "lat": lat_grid.ravel(),
            "lon": lon_grid.ravel(),
        },
        geometry=gpd.points_from_xy(lon_grid.ravel(), lat_grid.ravel()),
        crs="EPSG:4326",
    )

    joined = gpd.sjoin(
        points,
        world[[country_column, "geometry"]],
        how="left",
        predicate="within",
    )

    country_2d = joined[country_column].fillna("Ocean").to_numpy().reshape(lat_grid.shape)

    return xr.DataArray(
        country_2d,
        coords={lat_dim: lat_vals, lon_dim: lon_vals},
        dims=(lat_dim, lon_dim),
        name="country",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add a country name grid to a netCDF dataset."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("unified_annual_carbon_dataset_2015_2024.nc"),
        help="Input netCDF file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("unified_annual_carbon_dataset_2015_2024_with_countries.nc"),
        help="Output netCDF file.",
    )
    parser.add_argument(
        "--borders",
        type=Path,
        default=Path("ne_110m_admin_0_countries.gpkg"),
        help="Country borders GeoPackage path.",
    )
    parser.add_argument(
        "--ocean-label",
        type=str,
        default="Ocean",
        help="Label for grid points not inside any country.",
    )

    args = parser.parse_args()

    ds = xr.open_dataset(args.input)
    world = load_world(args.borders)
    country_column = pick_country_column(world)

    country_da = build_country_grid(ds, world, country_column)
    country_da = country_da.where(country_da != "Ocean", other=args.ocean_label)

    ds_out = ds.assign(country=country_da)
    ds_out.to_netcdf(args.output)

    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()