from collections.abc import Collection
from typing import Literal, Optional, Union

import geopandas
import pandas
from pyproj.aoi import AreaOfInterest
from pyproj.database import query_utm_crs_info

from demeter.raster import polaris, sentinel2, usgs


def fetch_point_data(
    points: Union[str, geopandas.GeoSeries, geopandas.GeoDataFrame],
    values_to_fetch: Collection[
        Literal[
            "polaris_carbon_stock",
            "sentinel2_ndvi",
            "usgs_hydrography",
            "usgs_topography",
        ]
    ],
    *,
    start_depth: int = 0,
    end_depth: Optional[int] = None,
    year: Optional[int] = None,
    month: Optional[int] = None,
) -> geopandas.GeoDataFrame:
    """
    Fetch data from one or more sources for the given points.

    `end_depth` (in cm) is required for POLARIS.

    `year` and `month` are required for Sentinel-2 NDVI.

    Example:

    ```python
    point_data = fetch_point_data(
        "points.geojson",
        values_to_fetch=["polaris_carbon_stock", "sentinel2_ndvi"],
        end_depth=30,
        year=2024,
        month=9,
    )
    ```
    """
    if isinstance(points, str):
        points = geopandas.read_file(points, columns=[])

    if isinstance(points, geopandas.GeoDataFrame):
        points = points.geometry

    assert isinstance(points, geopandas.GeoSeries)

    if points.empty:
        raise ValueError("No points provided")

    if set(points.geom_type) != {"Point"}:
        raise ValueError("Only points are supported")

    if points.crs is None:
        raise ValueError("Points must have a CRS")

    values_to_fetch = set(values_to_fetch)

    dataframes_to_merge = []

    if "polaris_carbon_stock" in values_to_fetch:
        if end_depth is None:
            raise ValueError("end_depth must be provided for POLARIS")

        dataframes_to_merge.append(
            _carbon_stock_from_polaris(points, start_depth, end_depth)
        )

    if "sentinel2_ndvi" in values_to_fetch:
        if year is None or month is None:
            raise ValueError("year and month must be provided for Sentinel-2 NDVI")

        dataframes_to_merge.append(_ndvi_from_sentinel2(points, year, month))

    if "usgs_hydrography" in values_to_fetch:
        dataframes_to_merge.append(_hydro_data_from_usgs(points))

    if "usgs_topography" in values_to_fetch:
        dataframes_to_merge.append(_topo_data_from_usgs(points))

    return geopandas.GeoDataFrame(
        pandas.concat(dataframes_to_merge, axis="columns", copy=False),
        geometry=points,
        crs=points.crs,
    )


def _carbon_stock_from_polaris(
    points: geopandas.GeoSeries, start_depth: int, end_depth: int
) -> pandas.DataFrame:
    points_in_polaris_crs = points.to_crs(polaris.RASTER_CRS)
    polaris_carbon_stock = polaris.estimate_carbon_stock(
        points_in_polaris_crs,
        start_depth=start_depth,
        end_depth=end_depth,
    )
    assert polaris_carbon_stock.stddev
    return pandas.DataFrame(
        {
            "polaris_carbon_stock_mean": [
                polaris_carbon_stock.mean.value_at(point.x, point.y)
                for point in points_in_polaris_crs
            ],
            "polaris_carbon_stock_stddev": [
                polaris_carbon_stock.stddev.value_at(point.x, point.y)
                for point in points_in_polaris_crs
            ],
        },
    )


def _ndvi_from_sentinel2(
    points: geopandas.GeoSeries, year: int, month: int
) -> pandas.DataFrame:
    points_with_utm_zones = geopandas.GeoDataFrame(geometry=points.to_crs("EPSG:4326"))

    point_utm_zones = []
    for index, point in enumerate(points_with_utm_zones.geometry):
        utm_zones = query_utm_crs_info(
            datum_name="WGS 84",
            area_of_interest=AreaOfInterest(point.x, point.y, point.x, point.y),
            contains=True,
        )
        assert len(utm_zones) == 1
        utm_zone = utm_zones[0]
        point_utm_zones.append(f"EPSG:{utm_zone.code}")
    points_with_utm_zones["utm_zone"] = point_utm_zones

    for utm_zone in points_with_utm_zones["utm_zone"].unique():
        index = points_with_utm_zones["utm_zone"] == utm_zone
        selection = points_with_utm_zones.loc[index].geometry
        points_with_utm_zones.loc[index, "coordinates_in_utm_zone"] = selection.to_crs(
            utm_zone
        )

    output = pandas.DataFrame(index=points_with_utm_zones.index)

    for rasters in sentinel2.ndvi.fetch_and_build_ndvi_rasters(points, year, month):
        index = points_with_utm_zones["utm_zone"] == rasters.crs
        selection = points_with_utm_zones.loc[index]

        assert rasters.mean
        assert rasters.min
        assert rasters.max
        assert rasters.stddev
        output.loc[index, "sentinel2_ndvi_mean"] = [
            rasters.mean.value_at(point.x, point.y)
            for point in selection["coordinates_in_utm_zone"]
        ]
        output.loc[index, "sentinel2_ndvi_min"] = [
            rasters.min.value_at(point.x, point.y)
            for point in selection["coordinates_in_utm_zone"]
        ]
        output.loc[index, "sentinel2_ndvi_max"] = [
            rasters.max.value_at(point.x, point.y)
            for point in selection["coordinates_in_utm_zone"]
        ]
        output.loc[index, "sentinel2_ndvi_stddev"] = [
            rasters.stddev.value_at(point.x, point.y)
            for point in selection["coordinates_in_utm_zone"]
        ]

    return output


def _hydro_data_from_usgs(points: geopandas.GeoSeries) -> pandas.DataFrame:
    points_in_usgs_hydro_crs = points.to_crs(usgs.hydrography.RASTER_CRS)

    rasters = (
        usgs.hydrography.fetch_and_merge_rasters(name, points_in_usgs_hydro_crs).raster
        for name in ("cat", "fac", "fdr")
    )
    point_values = (
        [raster.value_at(point.x, point.y) for point in points_in_usgs_hydro_crs]
        for raster in rasters
    )
    output = pandas.DataFrame(
        zip(*point_values),
        columns=["usgs_catchment_id", "usgs_flow_accumulation", "usgs_flow_direction"],
    )
    output["usgs_catchment_id"] = output["usgs_catchment_id"].astype("Int64")
    output["usgs_flow_direction"] = (
        output["usgs_flow_direction"]
        .apply(usgs.constants.FlowDirection)  # type: ignore
        .astype("category")
    )
    return output


def _topo_data_from_usgs(points: geopandas.GeoSeries) -> pandas.DataFrame:
    points_in_usgs_topo_crs = points.to_crs(usgs.topography.RASTER_CRS)
    elevation = usgs.topography.fetch_and_merge_rasters(points_in_usgs_topo_crs)
    return pandas.DataFrame(
        {
            "usgs_elevation": [
                elevation.value_at(point.x, point.y)
                for point in points_in_usgs_topo_crs
            ]
        }
    )
