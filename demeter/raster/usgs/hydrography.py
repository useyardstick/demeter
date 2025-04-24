"""
Tools for fetching hydrography data from USGS:
https://www.usgs.gov/national-hydrography/access-national-hydrography-products

Example:

```python
catchments = fetch_and_merge_rasters("cat.tif", "path/to/boundaries.geojson")
pixels, transform, crs = catchments.raster

# USGS hydrography zip archives include a DBF file with pixel counts, which we
# parse and return as a dict:
pixel_counts = catchments.counts
catchment_areas_in_m2 = {
    catchment_id: pixel_count * 100
    for catchment_id, pixel_count in pixel_counts.items()
}
```
"""

import json
import math
import os
import re
import warnings
from collections import defaultdict
from collections.abc import Generator, Iterable, Mapping, Sequence
from dataclasses import dataclass
from tempfile import TemporaryDirectory
from typing import Optional, Union, overload
from zipfile import ZipFile

import geopandas
import numpy
import rasterio
import requests
import shapely.geometry
from dbfread import DBF

from demeter.raster import Raster
from demeter.raster.usgs.constants import S3_BUCKET_NAME
from demeter.raster.usgs.utils import (
    download_from_s3,
    merge_and_crop_rasters,
    s3_client,
)
from demeter.utils import Lazy

# USGS hydrography rasters are stored in S3 under:
S3_PREFIX = "StagedProducts/Hydrography/NHDPlusHR/VPU/Current/Raster/"

# USGS raster files for CONUS use the EPSG:5070 projection. Some rasters
# specify the ESRI:102039 projection, which is essentially the same thing.
RASTER_CRS = "EPSG:5070"


@dataclass
class USGSHydrographyRasterBase:
    raster_filename: str
    counts: Mapping[int, int]


@dataclass
class USGSHydrographyRaster(USGSHydrographyRasterBase):
    """
    A `Raster` instance containing the USGS hydrography raster, and a dict
    mapping of pixel counts from the sidecar DBF file.
    """

    raster: Raster


@dataclass
class USGSHydrographyRasterOnDisk(USGSHydrographyRasterBase):
    """
    Same as `USGSHydrographyRaster`, but the raster is stored on disk.
    """

    raster_path: str


class MissingCatchmentIDWarning(Warning):
    pass


@overload
def fetch_and_merge_rasters(
    raster_filename: str,
    geometries: Union[str, geopandas.GeoDataFrame, geopandas.GeoSeries],
    *,
    crop: bool = True,
    dst_path: None = None,
) -> USGSHydrographyRaster: ...


@overload
def fetch_and_merge_rasters(
    raster_filename: list[str],
    geometries: Union[str, geopandas.GeoDataFrame, geopandas.GeoSeries],
    *,
    crop: bool = True,
    dst_path: None = None,
) -> Iterable[USGSHydrographyRaster]: ...


@overload
def fetch_and_merge_rasters(
    raster_filename: str,
    geometries: Union[str, geopandas.GeoDataFrame, geopandas.GeoSeries],
    *,
    crop: bool = True,
    dst_path: str,
) -> USGSHydrographyRasterOnDisk: ...


@overload
def fetch_and_merge_rasters(
    raster_filename: list[str],
    geometries: Union[str, geopandas.GeoDataFrame, geopandas.GeoSeries],
    *,
    crop: bool = True,
    dst_path: str,
) -> Iterable[USGSHydrographyRasterOnDisk]: ...


def fetch_and_merge_rasters(
    raster_filename: Union[str, list[str]],
    geometries: Union[str, geopandas.GeoDataFrame, geopandas.GeoSeries],
    *,
    crop: bool = True,
    dst_path: Optional[str] = None,
) -> Union[USGSHydrographyRasterBase, Iterable[USGSHydrographyRasterBase]]:
    """
    Fetch the given raster (e.g. "cat.tif") from USGS for the given geometries.
    If the geometries span multiple HU4 regions, fetch all the necessary
    rasters and stitch them together.

    Pass multiple raster filenames as a list to fetch multiple rasters for the
    same geometries. For example:

    ```python
    cat, fdr = fetch_and_merge_rasters(["cat.tif", "fdr.tif"], "path/to/boundaries.geojson")
    ```

    If `crop` is True (the default), crop the output raster to the given
    geometries.

    If `dst_path` is provided, save the output raster to the given path. Use
    this for large geometries that don't fit in memory.
    """
    if isinstance(raster_filename, str):
        return next(
            _fetch_and_merge_rasters(
                [raster_filename], geometries, crop=crop, dst_path=dst_path
            )
        )

    return _fetch_and_merge_rasters(
        raster_filename, geometries, crop=crop, dst_path=dst_path
    )


def _fetch_and_merge_rasters(
    raster_filenames: list[str],
    geometries: Union[str, geopandas.GeoDataFrame, geopandas.GeoSeries],
    crop: bool = True,
    dst_path: Optional[str] = None,
) -> Generator[USGSHydrographyRasterBase]:
    if isinstance(geometries, str):
        geometries = geopandas.read_file(geometries)

    assert isinstance(geometries, (geopandas.GeoSeries, geopandas.GeoDataFrame))

    downloaded_archive_paths = Lazy(fetch_rasters(geometries))

    for raster_filename in raster_filenames:
        yield _extract_raster(
            raster_filename,
            downloaded_archive_paths,
            crop_to=geometries.to_crs(RASTER_CRS) if crop else None,
            dst_path=dst_path,
        )


def _extract_raster(
    raster_filename: str,
    archive_paths: Iterable[str],
    crop_to: Optional[Union[geopandas.GeoDataFrame, geopandas.GeoSeries]],
    dst_path: Optional[str] = None,
) -> USGSHydrographyRasterBase:
    if crop_to is not None:
        assert str(crop_to.crs) == RASTER_CRS

    if not raster_filename.endswith(".tif"):
        raster_filename = f"{raster_filename}.tif"

    with TemporaryDirectory() as tmpdir:
        # Extract rasters and sidecar .vat.dbf files to a temporary directory:
        raster_paths = []
        all_rasters_have_sidecar_dbf = True
        for archive_path in archive_paths:
            with ZipFile(archive_path) as zip_archive:
                raster_path = _find_raster_path_in_archive(zip_archive, raster_filename)
                print(f"Extracting {raster_path}")
                raster_paths.append(zip_archive.extract(raster_path, tmpdir))
                try:
                    zip_archive.extract(f"{raster_path}.vat.dbf", tmpdir)
                except KeyError:
                    all_rasters_have_sidecar_dbf = False

        counts: dict[int, int] = defaultdict(int)

        if raster_filename == "cat.tif":
            # Catchment rasters use a raster-specific grid code int for each
            # pixel, which maps to a global catchment area ID. The mapping is
            # stored in a sidecar .vat.dbf file; read it and use it to map each
            # pixel in the raster to the full catchment ID.
            if crop_to is not None:
                minx, miny, maxx, maxy = crop_to.total_bounds
                bounds = (
                    math.floor(minx),
                    math.floor(miny),
                    math.ceil(maxx),
                    math.ceil(maxy),
                )
            else:
                bounds = None

            for raster_path in raster_paths:
                value_to_catchment_id_mapping: dict[int, int] = {}
                with DBF(f"{raster_path}.vat.dbf", raw=True) as dbf:
                    for record in dbf:
                        try:
                            # Catchment IDs are encoded as floats for some
                            # reason, but they're really integers.
                            catchment_id = float(record["NHDPlusID"])
                            catchment_id = int(catchment_id)
                        except ValueError:
                            # FIXME: Some dbf records contain a sequence of
                            # null characters as NHDPlusID. Example:
                            #
                            #     {
                            #         'VALUE': b'        0',
                            #         'COUNT': b'  2134900',
                            #         'NHDPlusID': b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00',
                            #     }
                            #
                            # I inspected some of these areas on a map; they
                            # look like any other catchment area. Maybe it's an
                            # issue with the data?
                            #
                            # For now, just skip these records, set the
                            # corresponding pixels to nodata in the output, and
                            # log a warning if we try to access them.
                            continue

                        value, count = _extract_value_and_count_from_dbf_record(record)
                        counts[catchment_id] += count
                        value_to_catchment_id_mapping[value] = catchment_id

                with rasterio.open(raster_path) as src:
                    profile = src.profile
                    assert profile["count"] == 1

                    if bounds:
                        window = (
                            rasterio.windows.from_bounds(*bounds, src.transform)
                            .round_offsets()
                            .round_lengths()
                        )
                        # Expand by 1 pixel on each side to ensure we capture
                        # the full extent:
                        window = rasterio.windows.Window(
                            window.col_off - 1,
                            window.row_off - 1,
                            window.width + 2,
                            window.height + 2,
                        )
                        window = window.intersection(
                            rasterio.windows.Window(0, 0, src.width, src.height)
                        )
                    else:
                        window = None

                    pixels = src.read(1, window=window)

                values_not_in_catchment_id_mapping = set()

                @numpy.vectorize(
                    otypes=[numpy.int64]  # to fit 14-digit catchment IDs
                )  # type: ignore
                def mapper(value):
                    if value == src.nodata:
                        return value
                    try:
                        return value_to_catchment_id_mapping[value]
                    except KeyError:
                        values_not_in_catchment_id_mapping.add(value)
                        return src.nodata

                # Map pixel values to catchment IDs:
                mapped_pixels = mapper(pixels)

                if values_not_in_catchment_id_mapping:
                    warnings.warn(
                        f"Some values in {os.path.relpath(raster_path, tmpdir)} could not be mapped to catchment IDs: {values_not_in_catchment_id_mapping}",
                        category=MissingCatchmentIDWarning,
                    )

                # Our mapped numpy array uses dtype int64, but GIS software
                # like QGIS seems to fall over when it encounters 64-bit
                # integers. As a hacky workaround, use 64-bit floats instead:
                profile["dtype"] = "float64"

                # Overwrite the raster with the mapped pixels:
                with rasterio.open(raster_path, "w", **profile) as dst:
                    dst.write(mapped_pixels, 1, window=window)
        elif all_rasters_have_sidecar_dbf:
            for raster_path in raster_paths:
                with DBF(f"{raster_path}.vat.dbf") as dbf:
                    for record in dbf:
                        value, count = _extract_value_and_count_from_dbf_record(record)
                        counts[value] += count

        if dst_path is None:
            return USGSHydrographyRaster(
                raster_filename,
                counts,
                merge_and_crop_rasters(raster_paths, crop_to=crop_to),
            )

        return USGSHydrographyRasterOnDisk(
            raster_filename,
            counts,
            merge_and_crop_rasters(raster_paths, crop_to=crop_to, dst_path=dst_path),
        )


def fetch_rasters(
    geometries: Union[str, geopandas.GeoDataFrame, geopandas.GeoSeries],
) -> Iterable[str]:
    """
    Fetch all the raster archives that intersect with the given geometries.
    Yield the local path to each downloaded zip archive.
    """
    if isinstance(geometries, str):
        geometries = geopandas.read_file(geometries)

    assert isinstance(geometries, (geopandas.GeoSeries, geopandas.GeoDataFrame))

    if geometries.empty:
        raise ValueError("No geometries provided")

    hu4_codes = find_hu4_codes(geometries)
    return download_raster_archives(hu4_codes)


def find_hu4_codes(
    geometries: Union[str, geopandas.GeoDataFrame, geopandas.GeoSeries]
) -> Sequence[str]:
    """
    Return the HU4 codes for the regions that intersect with the given
    geometries.

    USGS rasters are organized by 4-digit Hydrologic Unit (HU4). To know which
    rasters to download, we need to identify which HU4 regions the input geometry
    intersects with.
    """
    if isinstance(geometries, str):
        geometries = geopandas.read_file(geometries)

    assert isinstance(geometries, (geopandas.GeoSeries, geopandas.GeoDataFrame))

    # Query the USGS Watershed Boundary Dataset (WBD) web service for the HU4
    # regions that intersect with the geometries. This web service errors out
    # if we try to query using a large geo file, so use the bounding box:
    geometries_in_4326 = geometries.to_crs("EPSG:4326")
    geometries_combined = geometries_in_4326.union_all()
    bounding_box = shapely.geometry.box(*geometries_combined.bounds)
    bounding_box_as_ersi_json = {"rings": bounding_box.__geo_interface__["coordinates"]}
    response = requests.get(
        "https://hydro.nationalmap.gov/arcgis/rest/services/wbd/MapServer/2/query",
        params={
            "geometry": json.dumps(bounding_box_as_ersi_json),
            "geometryType": "esriGeometryPolygon",
            "inSR": "4326",
            "spatialRel": "esriSpatialRelIntersects",
            "outFields": "HUC4,Name",
            "returnGeometry": "true",
            "f": "GeoJSON",
        },
    )
    response.raise_for_status()
    hu4_regions = geopandas.GeoDataFrame.from_features(response.json())

    # We fetched all HU4 regions that intersect with the bounding box of the
    # geometries, but we only need the ones that intersect with the geometries
    # themselves:
    hu4_regions = hu4_regions[hu4_regions.intersects(geometries_combined)]

    if hu4_regions.empty:
        raise ValueError("No HU4 regions found for geometries. Are they in CONUS?")

    geometries_without_hu4_region = geometries[
        geometries_in_4326.disjoint(hu4_regions.union_all())
    ]
    if not geometries_without_hu4_region.empty:
        raise ValueError(
            f"Can't find HU4 region for geometries at index: {geometries_without_hu4_region.index.tolist()}. Are they in CONUS?"
        )

    return hu4_regions["huc4"].tolist()


def download_raster_archives(hu4_codes: Iterable[str]) -> Iterable[str]:
    """
    Download the raster .zip files for the given HU4 codes.
    """
    raster_keys = raster_keys_by_hu4_code()

    for hu4_code in hu4_codes:
        key = raster_keys[hu4_code]
        yield download_from_s3(key)


_raster_keys_by_hu4_code: dict[str, str] = {}  # populate this lazily


def raster_keys_by_hu4_code() -> dict[str, str]:
    """
    Return a dict mapping HU4 codes to raster keys in S3.
    """
    if not _raster_keys_by_hu4_code:
        raster_keys = _fetch_raster_keys()

        for key in raster_keys:
            match = re.search(r"NHDPLUS_H_(\d{4})_HU4(?:_\d{8})?_RASTER\.zip$", key)
            if not match:
                continue

            hu4_code = match.group(1)
            _raster_keys_by_hu4_code[hu4_code] = key

    return _raster_keys_by_hu4_code


def _fetch_raster_keys() -> Sequence[str]:
    print("Fetching list of available rasters")
    response = s3_client.list_objects_v2(
        Bucket=S3_BUCKET_NAME,
        Prefix=S3_PREFIX,
    )

    return [
        item["Key"] for item in response["Contents"] if item["Key"].endswith(".zip")
    ]


def _find_raster_path_in_archive(zip_archive: ZipFile, raster_filename: str) -> str:
    raster_paths = [
        path
        for path in zip_archive.namelist()
        if os.path.basename(path) == raster_filename
    ]

    if not raster_paths:
        raise ValueError(
            f"Could not find raster '{raster_filename}' in {zip_archive.filename}"
        )
    if len(raster_paths) > 1:
        raise ValueError(
            f"Multiple '{raster_filename}' files found in {zip_archive.filename}"
        )

    return raster_paths[0]


def _extract_value_and_count_from_dbf_record(record) -> tuple[int, int]:
    try:
        value = record["VALUE"]
    except KeyError:
        value = record["Value"]

    try:
        count = record["COUNT"]
    except KeyError:
        count = record["Count"]

    # USGS DBF files sometimes encode values in scientific notation, like this:
    # b' 3.63000000000e+002'
    # To parse these correctly, decode as float first, then coerce to int:
    return int(float(value)), int(float(count))
