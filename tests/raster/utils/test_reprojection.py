import numpy
import pytest

from demeter.raster import Raster
from demeter.raster.utils.reprojection import (
    align,
    align_and_merge,
    reproject,
    reproject_and_merge,
)
from demeter.raster.utils.transform import aligned_pixel_grids


@pytest.fixture
def elevation_raster():
    return Raster.from_file("tests/raster/fixtures/humboldt_park_elevation.tif")


def test_reproject(elevation_raster):
    reprojected = reproject(
        elevation_raster, crs="EPSG:5070", resampling_method="average"
    )
    assert reprojected.crs == "EPSG:5070"
    assert reprojected.shape != elevation_raster.shape
    assert round(reprojected.pixels.mean()) == round(elevation_raster.pixels.mean())


def test_reproject_unmasked(elevation_raster):
    unmasked = numpy.ma.masked_array(elevation_raster.pixels.filled(numpy.nan))
    unmasked_raster = Raster(unmasked, elevation_raster.transform, elevation_raster.crs)
    reprojected = reproject(
        unmasked_raster, crs="EPSG:5070", resampling_method="average"
    )
    assert reprojected.crs == "EPSG:5070"
    assert reprojected.shape != elevation_raster.shape
    assert round(reprojected.pixels.mean()) == round(elevation_raster.pixels.mean())


def test_align(elevation_raster):
    flow_direction = Raster.from_file("tests/raster/fixtures/humboldt_park_fdr.tif")

    reprojected_elevation = reproject(
        elevation_raster, crs="EPSG:5070", resampling_method="average"
    )
    assert reprojected_elevation.crs == flow_direction.crs
    assert reprojected_elevation.transform != flow_direction.transform

    aligned_elevation = align(
        reprojected_elevation, to=flow_direction, resampling_method="average"
    )
    assert aligned_elevation.crs == flow_direction.crs
    assert aligned_elevation.resolution == flow_direction.resolution == (10, 10)
    assert aligned_pixel_grids(
        flow_direction.bounds, [flow_direction.transform, aligned_elevation.transform]
    )
    assert round(aligned_elevation.pixels.mean()) == round(
        elevation_raster.pixels.mean()
    )


def test_reproject_and_merge():
    ndvi_rasters = [
        Raster.from_file("tests/raster/fixtures/ndvi-utm-zone-13.tif"),
        Raster.from_file("tests/raster/fixtures/ndvi-utm-zone-14.tif"),
    ]
    reprojected = reproject_and_merge(
        ndvi_rasters,
        crs="EPSG:4326",
        resampling_method="average",
        merge_method="mean",
    )
    assert reprojected.crs == "EPSG:4326"

    original_mean = sum(raster.pixels.mean() for raster in ndvi_rasters) / len(
        ndvi_rasters
    )
    assert round(reprojected.pixels.mean(), 1) == round(original_mean, 1)


def test_align_and_merge():
    flow_direction = Raster.from_file("tests/raster/fixtures/texas_fdr.tif")
    ndvi_rasters = [
        Raster.from_file("tests/raster/fixtures/ndvi-utm-zone-13.tif"),
        Raster.from_file("tests/raster/fixtures/ndvi-utm-zone-14.tif"),
    ]
    reprojected = align_and_merge(
        ndvi_rasters,
        to=flow_direction,
        resampling_method="average",
        merge_method="mean",
    )
    assert reprojected.crs == flow_direction.crs

    original_mean = sum(raster.pixels.mean() for raster in ndvi_rasters) / len(
        ndvi_rasters
    )
    assert reprojected.resolution == flow_direction.resolution == (10, 10)
    assert aligned_pixel_grids(
        flow_direction.bounds, [flow_direction.transform, reprojected.transform]
    )
    assert round(reprojected.pixels.mean(), 1) == round(original_mean, 1)
