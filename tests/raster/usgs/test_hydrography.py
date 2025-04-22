import geopandas
import numpy
import pytest
import rasterio.transform

from demeter.raster import Raster
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
@pytest.mark.parametrize("filename", (None, "cat.tif"))
def test_fetch_and_merge_cat(geometries, tmp_path, filename):
    if filename is None:
        catchments = fetch_and_merge_rasters("cat.tif", geometries)
        raster = catchments.raster
    else:
        raster_path = str(tmp_path / filename)
        catchments = fetch_and_merge_rasters(
            "cat.tif", geometries, dst_path=raster_path
        )
        raster = Raster.from_file(catchments.raster_path)

    assert raster.shape == (1633, 8702)
    assert raster.pixels.count() == 41205
    assert raster.crs in {"EPSG:5070", "ESRI:102039"}
    assert numpy.array_equal(
        numpy.unique(raster.pixels).compressed(),
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
    height, width = raster.shape
    raster_bounds = rasterio.transform.array_bounds(height, width, raster.transform)
    input_geometry_bounds = geometries.to_crs(RASTER_CRS).total_bounds
    assert all(abs(input_geometry_bounds - raster_bounds) < 10)


@pytest.mark.parametrize("filename", (None, "fdr.tif"))
def test_fetch_and_merge_fdr(geometries, tmp_path, filename):
    if filename is None:
        fdr = fetch_and_merge_rasters("fdr.tif", geometries)
        raster = fdr.raster
    else:
        raster_path = str(tmp_path / filename)
        fdr = fetch_and_merge_rasters("fdr.tif", geometries, dst_path=raster_path)
        raster = Raster.from_file(fdr.raster_path)

    assert raster.shape == (1633, 8702)
    assert raster.pixels.count() == 41205
    assert raster.crs in {"EPSG:5070", "ESRI:102039"}
    assert numpy.array_equal(
        numpy.unique(raster.pixels).compressed(),
        [1, 2, 4, 8, 16, 32, 64, 128],
    )
