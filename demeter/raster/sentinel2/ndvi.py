"""
Tools for fetching Sentinel-2 rasters in the red and NIR bands, and using them
to calculate Normalized Difference Vegetation Index (NDVI) rasters.

Example:

```python
os.environ["COPERNICUS_AWS_ENDPOINT_URL"] = "https://eodata.dataspace.copernicus.eu/"
os.environ["COPERNICUS_AWS_ACCESS_KEY_ID"] = ...
os.environ["COPERNICUS_AWS_SECRET_ACCESS_KEY"] = ...

rasters = fetch_and_build_ndvi_rasters(
    "path/to/boundaries.geojson",
    year=2024,
    month=9,
    statistics=["mean", "min", "max", "stddev"],
)
```

Sentinel-2 rasters are projected using the Universal Transverse Mercator (UTM)
system. If the input geometries span multiple UTM zones, this function will
return a separate raster for each zone. You can use them separately, or project
to a common CRS if necessary. Note that projecting rasters involves resampling,
which is lossy.
"""

import os
from collections import defaultdict
from collections.abc import Collection, Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from itertools import groupby
from tempfile import TemporaryDirectory
from typing import Literal, Optional, Union, cast, overload

import geopandas
import numpy
import shapely.geometry
from pyproj import CRS

from demeter.raster import Raster
from demeter.raster.sentinel2.constants import CLOUD_VALUES, Band, Resolution
from demeter.raster.sentinel2.utils.download import download_keys
from demeter.raster.sentinel2.utils.rasters import (
    DetectorFootprintMaskMetadata,
    RasterMetadata,
    SafeMetadata,
    list_raster_keys,
)
from demeter.raster.sentinel2.utils.search import find_safe_files
from demeter.raster.sentinel2.utils.tiles import find_tiles_for_geometries
from demeter.raster.utils.mask import mask
from demeter.raster.utils.merge import check_for_overlapping_pixels, merge, merge_stddev

BANDS_NEEDED_FOR_NDVI = {Band.RED, Band.NIR, Band.SCL}


def _ndvi_band(value: str) -> Band:
    band = Band(value)
    if band not in BANDS_NEEDED_FOR_NDVI:
        raise ValueError(f"Band not needed to calculate NDVI: {band}")
    return band


def _max_workers() -> int:
    max_processes = os.environ.get("SENTINEL2_NDVI_MAX_PROCESSES")
    if max_processes is not None:
        return int(max_processes)

    cpu_count = os.cpu_count()
    if cpu_count is None:
        return 1

    # Default to half the number of cores, to avoid using too much memory.
    return cpu_count // 2


_pool = ProcessPoolExecutor(max_workers=_max_workers())


@dataclass
class NDVIRastersBase:
    crs: str


@dataclass
class NDVIRasters(NDVIRastersBase):
    mean: Optional[Raster] = None
    min: Optional[Raster] = None
    max: Optional[Raster] = None
    stddev: Optional[Raster] = None


@dataclass
class NDVIRastersOnDisk(NDVIRastersBase):
    mean: Optional[str] = None
    min: Optional[str] = None
    max: Optional[str] = None
    stddev: Optional[str] = None


@overload
def fetch_and_build_ndvi_rasters(
    geometries: Union[str, geopandas.GeoDataFrame, geopandas.GeoSeries],
    year: int,
    month: int,
    statistics: Optional[Collection[Literal["mean", "min", "max", "stddev"]]] = None,
    *,
    crop: bool = True,
    dst_path: str,
) -> Iterable[NDVIRastersOnDisk]: ...


@overload
def fetch_and_build_ndvi_rasters(
    geometries: Union[str, geopandas.GeoDataFrame, geopandas.GeoSeries],
    year: int,
    month: int,
    statistics: Optional[Collection[Literal["mean", "min", "max", "stddev"]]] = None,
    *,
    crop: bool = True,
    dst_path: None = None,
) -> Iterable[NDVIRasters]: ...


def fetch_and_build_ndvi_rasters(
    geometries: Union[str, geopandas.GeoDataFrame, geopandas.GeoSeries],
    year: int,
    month: int,
    statistics: Optional[Collection[Literal["mean", "min", "max", "stddev"]]] = None,
    *,
    crop: bool = True,
    dst_path: Optional[str] = None,
) -> Iterable[Union[NDVIRasters, NDVIRastersOnDisk]]:
    """
    Download red and NIR reflectance rasters from Sentinel-2 for the given
    geometries over the given month, use them to calculate NDVI, and merge the
    NDVI rasters together per the requested statistic.

    If `crop` is True (the default), crop the output raster to the given
    geometries. If `crop` if False, the output raster will cover the extent of
    the Sentinel-2 rasters intersecting with the given geometries.

    If `dst_path` is given, NDVI rasters will be saved to that directory. Use
    this for large geometries that don't fit in memory.
    """
    if isinstance(geometries, str):
        geometries = geopandas.read_file(geometries)

    assert isinstance(geometries, (geopandas.GeoSeries, geopandas.GeoDataFrame))

    tiles = list(find_tiles_for_geometries(geometries))
    print(
        f"Searching for rasters in tiles: {sorted({tile_id for tile_id, _ in tiles})}"
    )
    safe_keys = list(find_safe_files(tiles, year, month))

    raster_keys = list_raster_keys(
        safe_keys,
        [
            (Band.RED, Resolution.R10),
            (Band.NIR, Resolution.R10),
            (Band.SCL, Resolution.R20),
        ],
    )
    return fetch_and_build_ndvi_rasters_from_keys(
        raster_keys,
        statistics,
        crop_to=geometries if crop else None,
        dst_path=dst_path,
    )


def fetch_and_build_ndvi_rasters_from_keys(
    raster_keys: Iterable[str],
    statistics: Optional[Collection[Literal["mean", "min", "max", "stddev"]]] = None,
    crop_to: Optional[Union[geopandas.GeoDataFrame, geopandas.GeoSeries]] = None,
    dst_path: Optional[str] = None,
) -> Iterable[Union[NDVIRasters, NDVIRastersOnDisk]]:
    """
    Download the given rasters, use them to calculate NDVI, and merge the NDVI
    rasters together per the requested statistics.

    The given raster keys should be for red, NIR, and SCL bands. Red and NIR are
    needed to calculate NDVI, and SCL is used to mask out clouds.
    """

    # First, sort raster keys by UTM zone and datatake:
    def sort_key(raster_key):
        safe_metadata = SafeMetadata.from_filename(raster_key)
        return safe_metadata.utm_zone, safe_metadata.datatake_timestamp

    sorted_raster_keys = sorted(raster_keys, key=sort_key)

    # Then download them _in that order_, so we can process them one datatake
    # at a time as the rasters are being downloaded:
    print(f"Downloading {len(sorted_raster_keys)} rasters")
    download_paths = download_keys(sorted_raster_keys)

    # In case input geometries are not already in EPSG:4326, transform them so
    # we can split them by UTM zone:
    if crop_to is not None:
        crop_to = crop_to.to_crs("EPSG:4326")

    # Group input rasters by CRS, and yield a separate output raster for each:
    for crs, raster_paths in groupby(
        download_paths, lambda path: SafeMetadata.from_filename(path).crs
    ):
        yield build_ndvi_rasters_for_crs(
            crs, raster_paths, statistics, crop_to, dst_path
        )


def build_ndvi_rasters_for_crs(
    crs: str,
    raster_paths: Iterable[str],
    statistics: Optional[Collection[Literal["mean", "min", "max", "stddev"]]] = None,
    crop_to: Optional[Union[geopandas.GeoDataFrame, geopandas.GeoSeries]] = None,
    dst_path: Optional[str] = None,
) -> Union[NDVIRasters, NDVIRastersOnDisk]:
    """
    Build NDVI rasters for each datatake, then merge them together according to
    the requested statistic.

    Input rasters should all be in the given CRS, and sorted by datatake so we
    can process them lazily. The output raster will also in the same CRS as the
    input rasters.
    """
    if statistics is None:
        statistics = {"mean", "min", "max", "stddev"}
    else:
        statistics = set(statistics)

    calculate_stddev = "stddev" in statistics
    statistics.discard("stddev")
    statistics = cast(set[Literal["mean", "min", "max"]], statistics)
    if calculate_stddev and "mean" not in statistics:
        raise ValueError("Cannot calculate standard deviation without mean")

    print(f"Building {statistics} NDVI rasters for {crs}")

    if dst_path is None:
        rasters: Union[NDVIRasters, NDVIRastersOnDisk] = NDVIRasters(crs=crs)
    else:
        try:
            os.mkdir(dst_path)
        except FileExistsError:
            pass
        os.mkdir(os.path.join(dst_path, crs))
        rasters = NDVIRastersOnDisk(crs=crs)

    if crop_to is None:
        crop_to_projected = None
    else:
        # Crop geometry mask to the current UTM zone's area of use:
        if crop_to.crs.to_epsg() != 4326:
            raise ValueError("Input geometries must be in EPSG:4326")
        crs_area_of_use = CRS.from_string(crs).area_of_use
        assert crs_area_of_use
        crop_to_projected = crop_to.clip(crs_area_of_use.bounds).to_crs(crs)

    with TemporaryDirectory() as tmpdir:
        background_jobs = []
        processed_datatakes = set()
        for datatake, datatake_raster_paths in groupby(
            raster_paths,
            lambda path: SafeMetadata.from_filename(path).datatake_timestamp,
        ):
            # Input rasters should be sorted by datatake, so we can process
            # them lazily one datatake at a time:
            if datatake in processed_datatakes:
                raise ValueError(
                    f"Datatake already processed: {datatake}. Pass in sorted order."
                )

            processed_datatakes.add(datatake)

            future = _pool.submit(
                build_and_save_ndvi_raster_for_datatake,
                tmpdir,
                datatake,
                list(datatake_raster_paths),
                crop_to_projected,
            )
            background_jobs.append(future)

        ndvi_raster_paths = [future.result() for future in background_jobs]

        for statistic in statistics:
            print(f"Calculating {statistic} NDVI raster in {crs}")
            merged_ndvi_raster = merge(
                ndvi_raster_paths,
                method=statistic,
                allow_resampling=False,
                dst_path=(
                    os.path.join(dst_path, crs, f"{statistic}.tif")
                    if dst_path
                    else None
                ),
            )
            setattr(rasters, statistic, merged_ndvi_raster)

        if calculate_stddev:
            print(f"Calculating standard deviation NDVI raster in {crs}")
            assert rasters.mean
            stddev_ndvi_raster = merge_stddev(
                ndvi_raster_paths,
                rasters.mean,
                dst_path=(
                    os.path.join(dst_path, crs, "stddev.tif") if dst_path else None
                ),
            )
            rasters.stddev = stddev_ndvi_raster

    return rasters


def build_ndvi_raster_for_datatake(
    datatake_timestamp: str,
    raster_paths: Iterable[str],
    crop_to: Optional[Union[geopandas.GeoDataFrame, geopandas.GeoSeries]] = None,
) -> Raster:
    """
    From the given rasters from the same datatake, calculate and return an NDVI
    raster.
    """
    print(f"Calculating NDVI for datatake: {datatake_timestamp}")

    # First, extract red, NIR, and SCL rasters and masks from the datatake:
    rasters_by_band: dict[Band, list[str]] = defaultdict(list)
    masks_by_band: dict[Band, list[str]] = defaultdict(list)

    for raster_path in raster_paths:
        metadata = SafeMetadata.from_filename(raster_path)
        if metadata.datatake_timestamp != datatake_timestamp:
            raise ValueError(
                "Don't mix datatakes when building NDVI rasters: "
                f"{metadata.datatake_timestamp} != {datatake_timestamp}"
            )
        if crop_to is not None and str(crop_to.crs) != metadata.crs:
            raise ValueError(
                "Geometry does not match raster CRS: "
                f"{crop_to.crs} != {metadata.crs}"
            )

        # Check if this is a data raster or a mask:
        try:
            raster_metadata = RasterMetadata.from_filename(raster_path)
        except ValueError:
            mask_metadata = DetectorFootprintMaskMetadata.from_filename(raster_path)
            band = _ndvi_band(mask_metadata.band)
            masks_by_band[band].append(raster_path)
        else:
            band = _ndvi_band(raster_metadata.band)
            rasters_by_band[band].append(raster_path)

    # Then merge them together:
    merged_rasters: dict[Band, Raster] = {
        band: merge_and_crop_rasters(rasters, crop_to)
        for band, rasters in rasters_by_band.items()
    }
    merged_masks: dict[Band, Raster] = {
        band: merge_and_crop_rasters(masks, crop_to)
        for band, masks in masks_by_band.items()
    }

    # Apply detector footprint masks for each band.
    # See https://sentiwiki.copernicus.eu/web/s2-products#S2Products-ArtefactsattheedgeoftheswathduetoL2ANoDatamask
    for band, footprint_mask in merged_masks.items():
        raster = merged_rasters[band]
        assert footprint_mask.shape == raster.shape
        assert footprint_mask.transform == raster.transform
        assert footprint_mask.crs == raster.crs
        raster.pixels[footprint_mask.pixels.mask] = numpy.ma.masked

    red, nir, scl = (
        merged_rasters[Band.RED],
        merged_rasters[Band.NIR],
        merged_rasters[Band.SCL],
    )

    # The SCL mask is upscaled from 20m to 10m resolution, so we may need to
    # crop 10m off the edges for it to line up perfectly with the red and NIR
    # rasters:
    if scl.shape != red.shape:
        scl = mask(scl, [shapely.geometry.box(*red.bounds)], crop=True)

    # Check that all the rasters are aligned on the same pixel grid:
    assert red.shape == nir.shape == scl.shape
    assert red.transform == nir.transform == scl.transform
    assert red.crs == nir.crs == scl.crs

    # Apply cloud mask:
    cloud_mask = numpy.isin(scl.pixels, CLOUD_VALUES)
    red.pixels[cloud_mask] = numpy.ma.masked
    nir.pixels[cloud_mask] = numpy.ma.masked

    # Calculate NDVI:
    red_reflectance = extract_surface_reflectance(red.pixels)
    nir_reflectance = extract_surface_reflectance(nir.pixels)
    ndvi = calculate_ndvi(red_reflectance, nir_reflectance)
    return Raster(ndvi, red.transform, str(red.crs))


def build_and_save_ndvi_raster_for_datatake(
    output_directory: str, datatake_timestamp: str, *args, **kwargs
) -> str:
    """
    Same as `build_ndvi_raster_for_datatake`, but also save the NDVI raster to
    a file in the given directory and return its path.
    """
    ndvi_raster = build_ndvi_raster_for_datatake(datatake_timestamp, *args, **kwargs)
    raster_path = os.path.join(output_directory, f"{datatake_timestamp}.tif")
    ndvi_raster.save(raster_path)
    return raster_path


def merge_and_crop_rasters(
    raster_paths: Sequence[str],
    crop_to: Optional[Union[geopandas.GeoDataFrame, geopandas.GeoSeries]] = None,
) -> Raster:
    """
    Merge rasters, optionally cropping to the given geometries.

    Note that Sentinel-2 tiles are 100km x 100km, but the rasters are buffered
    to 110km x 110km, so there is some overlap between rasters at tile
    boundaries. Values from adjacent tiles should be identical within this
    boundary region - if not, log a warning.
    """
    if crop_to is None:
        return _merge_rasters(raster_paths)

    merged = _merge_rasters(raster_paths, bounds=tuple(crop_to.total_bounds))
    return mask(merged, crop_to, all_touched=True)


def _merge_rasters(sources: Sequence[str], **kwargs) -> Raster:
    return merge(
        sources,
        method=check_for_overlapping_pixels,
        res=10,
        allow_resampling=False,
        **kwargs,
    )


def extract_surface_reflectance(pixels: numpy.ndarray) -> numpy.ma.MaskedArray:
    """
    Sentinel-2 surface reflectance values are given in the 1-10000 range, with
    0 as the nodata value. Scale reflectance to the 0-1 range, and add a nodata
    mask.
    """
    return numpy.ma.masked_equal(pixels, 0) / 10000


def calculate_ndvi(red, nir):
    return (nir - red) / (nir + red)
