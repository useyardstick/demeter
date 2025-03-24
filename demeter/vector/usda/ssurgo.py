"""
Tools for fetching Soil Survey (SSURGO) data from USDA:
https://www.nrcs.usda.gov/resources/data-and-reports/soil-survey-geographic-database-ssurgo
"""

from typing import Union

import geopandas
import numpy
import pandas
import requests
import sqlalchemy
from sqlalchemy import bindparam
from sqlalchemy.dialects import mssql

SOIL_DATA_ACCESS_API_URL = "https://sdmdataaccess.sc.egov.usda.gov/tabular/post.rest"

SQL_DIALECT = mssql.dialect()

PRIMARY_COMPONENTS_SQL = """
WITH
  intersecting_geometries AS (
    SELECT
      geometry::UnionAggregate(mupolygongeo).STAsText() AS geometry,
      mukey AS map_unit_key
    FROM
      mupolygon
    WHERE
      mupolygongeo.STIntersects(geometry::STGeomFromText(:wkt, :epsg)) = 1
    GROUP BY
      mukey
  ),
  intersecting_map_units AS (
    SELECT
      intersecting_geometries.*,
      mapunit.musym AS map_unit_symbol,
      mapunit.muname AS map_unit_name
    FROM
      intersecting_geometries
      LEFT JOIN mapunit ON mapunit.mukey = intersecting_geometries.map_unit_key
  ),
  primary_components AS (
    SELECT TOP 1 WITH TIES
      intersecting_map_units.*,
      cokey AS component_key,
      comppct_r AS component_percent,
      compname AS component_name,
      compkind AS component_kind,
      drainagecl AS drainage_class,
      taxclname AS taxonomic_class,
      taxorder AS taxonomic_order
    FROM
      intersecting_map_units
      LEFT JOIN component ON component.mukey = intersecting_map_units.map_unit_key
      AND component.majcompflag = 'Yes'
    ORDER BY
      ROW_NUMBER() OVER (PARTITION BY mukey ORDER BY comppct_r DESC)
  )
SELECT
  primary_components.*,
  pmgroupname AS parent_material
FROM
  primary_components
  LEFT JOIN copmgrp ON copmgrp.cokey = primary_components.component_key
  AND copmgrp.rvindicator = 'Yes'
ORDER BY
  map_unit_key
"""

HORIZONS_SQL = """
WITH
  horizons AS (
    SELECT
      chkey AS horizon_key,
      cokey AS component_key,
      hzdept_r AS top_depth_cm,
      hzdepb_r AS bottom_depth_cm,
      (100 - fraggt10_r - frag3to10_r) * (sieveno10_r / 100) AS fine_fraction_percent_by_weight,
      sandtotal_r AS sand_percent_of_fine_fraction_by_weight,
      silttotal_r AS silt_percent_of_fine_fraction_by_weight,
      claytotal_r AS clay_percent_of_fine_fraction_by_weight,
      om_r AS organic_matter_percent_of_fine_fraction_by_weight,
      dbovendry_r AS oven_dry_bulk_density_g_per_cm3
    FROM
      chorizon
    WHERE
      cokey IN :component_keys
      AND hzdepb_r > :top_depth_cm
      AND hzdept_r < :bottom_depth_cm
  )
SELECT
  *,
  100 - fine_fraction_percent_by_weight AS gravel_percent_by_weight
FROM
  horizons
"""

FRAGMENTS_SQL = """
SELECT
  chkey AS horizon_key,
  fragvol_r AS fragment_percent_by_volume,
  fragsize_r AS fragment_size,
  fragkind AS fragment_kind
FROM
  chfrags
WHERE
  chkey IN :horizon_keys
"""


def fetch_primary_soil_components(
    geometries: Union[str, geopandas.GeoDataFrame, geopandas.GeoSeries],
    *,
    top_depth_cm: int = 0,
    bottom_depth_cm: int,
    crop: bool = True,
) -> geopandas.GeoDataFrame:
    """
    Fetch all SSURGO map units that intersect with the given geometries. Return
    a GeoDataFrame with the primary component of each map unit, along with a
    depth-weighted average of that component's soil properties over the given
    depth range.

    Example:

    ```python
    fetch_primary_soil_components("path/to/geometries.geojson", bottom_depth_cm=100)
    ```
    """
    if bottom_depth_cm <= top_depth_cm:
        raise ValueError("bottom_depth_cm must be greater than top_depth_cm")

    if isinstance(geometries, str):
        geometries = geopandas.read_file(geometries)

    assert isinstance(geometries, (geopandas.GeoSeries, geopandas.GeoDataFrame))

    # First, find the primary components for the map units intersecting with
    # the given geometries:
    geometries_combined = geometries.geometry.union_all()
    primary_components = _send_query(
        PRIMARY_COMPONENTS_SQL,
        wkt=geometries_combined.wkt,
        epsg=geometries.crs.to_epsg(),
    )
    primary_components = geopandas.GeoDataFrame(
        primary_components,
        geometry=geopandas.GeoSeries.from_wkt(primary_components.geometry),
        crs="EPSG:4326",
    )

    # Fetch horizons for each primary component, and aggregate them over the
    # requested depth range:
    component_keys = primary_components["component_key"].tolist()
    horizons_aggregated = _fetch_and_aggregate_horizons_by_component(
        component_keys, top_depth_cm, bottom_depth_cm
    )

    # Merge the aggregated horizons into the primary components DataFrame:
    primary_components = primary_components.merge(
        horizons_aggregated,
        how="left",
        on="component_key",
        validate="one_to_one",
    )

    # Use best possible dtypes. Exclude numeric types here, as matplotlib seems
    # to struggle with them:
    primary_components = primary_components.convert_dtypes(
        convert_integer=False,
        convert_floating=False,
    )

    assert isinstance(primary_components, geopandas.GeoDataFrame)

    if crop:
        return primary_components.clip(geometries.to_crs("EPSG:4326"), sort=True)

    return primary_components


def _send_query(sql: str, *binds, **params) -> pandas.DataFrame:
    compiled_sql = _compile_sql(sql, *binds, **params)
    response = requests.post(
        SOIL_DATA_ACCESS_API_URL,
        data={"query": compiled_sql, "format": "JSON+COLUMNNAME"},
    )
    response.raise_for_status()
    columns, *rows = response.json()["Table"]
    dataframe = pandas.DataFrame(rows, columns=columns)

    # The API returns numeric values as strings. Convert them to numeric dtype:
    for column in columns:
        try:
            converted = pandas.to_numeric(dataframe[column], errors="raise")
        except ValueError:
            pass
        else:
            dataframe[column] = converted

    return dataframe


def _compile_sql(sql: str, *binds, **params) -> str:
    return str(
        sqlalchemy.text(sql)
        .bindparams(*binds, **params)
        .compile(
            compile_kwargs={"literal_binds": True},
            dialect=SQL_DIALECT,
        )
    )


def _fetch_and_aggregate_horizons_by_component(
    component_keys, top_depth_cm, bottom_depth_cm
) -> pandas.DataFrame:
    horizons = _send_query(
        HORIZONS_SQL,
        bindparam("component_keys", component_keys, expanding=True),
        top_depth_cm=top_depth_cm,
        bottom_depth_cm=bottom_depth_cm,
    )

    # Fetch fragments and aggregate them per-horizon:
    horizon_keys = horizons["horizon_key"].tolist()
    fragments = _send_query(
        FRAGMENTS_SQL,
        bindparam("horizon_keys", horizon_keys, expanding=True),
    )
    fragments_aggregated = (
        fragments.groupby("horizon_key")[["fragment_percent_by_volume"]]
        .sum()
        .join(
            fragments[fragments["fragment_kind"].notna()]
            .groupby("horizon_key")["fragment_kind"]
            .unique()
        )
        .reset_index()
    )
    horizons = horizons.merge(
        fragments_aggregated,
        how="left",
        on="horizon_key",
        validate="one_to_one",
    ).drop(columns=["horizon_key"])

    # Aggregate each soil property across all the horizons using a
    # depth-weighted average:
    columns_to_aggregate = horizons.columns.difference(
        ["component_key", "fragment_kind"], sort=False
    )
    horizons_aggregated = horizons.groupby("component_key")[columns_to_aggregate].apply(
        _depth_weighted_average,  # type: ignore
        top_depth_cm,
        bottom_depth_cm,
    )

    # Concatenate the different kinds of fragments at each horizon into a
    # single string:
    fragment_kinds_aggregated = (
        horizons[horizons["fragment_kind"].notna()]
        .groupby("component_key")["fragment_kind"]
        .agg(_concat_unique_values)
    )

    return horizons_aggregated.join(fragment_kinds_aggregated)


def _depth_weighted_average(horizons, top_depth_cm, bottom_depth_cm):
    """
    Calculate a depth-weighted average over the given depth range of the soil
    properties in the given horizons DataFrame. Ignore any missing values.
    """
    horizons = horizons.copy(deep=False)
    top_depths_clipped = horizons.pop("top_depth_cm").clip(lower=top_depth_cm)
    bottom_depths_clipped = horizons.pop("bottom_depth_cm").clip(upper=bottom_depth_cm)
    weights = bottom_depths_clipped - top_depths_clipped
    return pandas.Series(
        {
            column: _weighted_average_excluding_missing_values(
                horizons[column], weights
            )
            for column in horizons.columns
        }
    )


def _weighted_average_excluding_missing_values(series, weights):
    average = numpy.ma.average(_series_to_masked_array(series), weights=weights)

    # If the entire input series is masked, `numpy.ma.average` returns
    # `numpy.ma.masked`. This confuses pandas. Return `None` instead:
    if average is numpy.ma.masked:
        return None

    return average


def _series_to_masked_array(series: pandas.Series) -> numpy.ma.MaskedArray:
    return numpy.ma.masked_array(series, mask=series.isna())


def _concat_unique_values(strings: pandas.Series) -> str:
    return ", ".join(sorted(strings.explode().unique()))
