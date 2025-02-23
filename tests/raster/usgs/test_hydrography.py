import geopandas
import numpy
import pytest
import rasterio.transform

from demeter.raster.usgs.hydrography import (
    RASTER_CRS,
    fetch_and_merge_rasters,
    find_hu4_codes,
)


@pytest.fixture
def geometries():
    return geopandas.read_file("tests/fixtures/tango_oscar.geojson")


def test_find_hu4_codes(geometries, record_or_replay_requests):
    assert sorted(find_hu4_codes(geometries)) == ["1022", "1023"]


# TODO: save test fixtures
def test_fetch_and_merge_cat(geometries):
    catchments = fetch_and_merge_rasters("cat.tif", geometries)
    assert catchments.raster.shape == (1633, 8702)
    assert catchments.raster.pixels.count() == 41205
    assert catchments.raster.crs in {"EPSG:5070", "ESRI:102039"}
    assert numpy.array_equal(
        numpy.unique(catchments.raster.pixels).compressed(),
        [
            23000300005215,
            23000300015091,
            23000300020146,
            23000300020147,
            23002100001502,
            23002100001589,
            23002100004714,
            23002100004760,
            23002100004780,
            23002100005868,
            23002100015503,
            23002100016466,
            23002100019752,
            23002100024031,
        ],
    )

    # Check that raster bounds are within 10m of input geometry bounds:
    height, width = catchments.raster.shape
    raster_bounds = rasterio.transform.array_bounds(
        height, width, catchments.raster.transform
    )
    input_geometry_bounds = geometries.to_crs(RASTER_CRS).total_bounds
    assert all(abs(input_geometry_bounds - raster_bounds) < 10)


def test_fetch_and_merge_fdr(geometries):
    fdr = fetch_and_merge_rasters("fdr.tif", geometries)
    assert fdr.raster.shape == (1633, 8702)
    assert fdr.raster.pixels.count() == 41205
    assert fdr.raster.crs in {"EPSG:5070", "ESRI:102039"}
    assert numpy.array_equal(
        numpy.unique(fdr.raster.pixels).compressed(),
        [1, 2, 4, 8, 16, 32, 64, 128],
    )
