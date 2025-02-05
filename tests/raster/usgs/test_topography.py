import geopandas
import pytest
import rasterio.transform

from demeter.raster.usgs.topography import RASTER_CRS, fetch_and_merge_rasters


@pytest.fixture
def geometries():
    # Chosen to span 2 USGS elevation tiles:
    return geopandas.read_file("tests/fixtures/permian_basin.geojson")


@pytest.fixture
def geometries_different_pixel_grid():
    return geopandas.read_file("tests/fixtures/nebraska.geojson")


# TODO: save test fixtures
def test_fetch_and_merge_rasters(geometries):
    raster = fetch_and_merge_rasters(geometries)
    assert raster.shape == (14934, 4791)
    assert raster.crs == "EPSG:4269"
    assert raster.pixels.count() == 6019
    assert round(raster.pixels.mean()) == 979

    # Check that raster bounds are within 10m of input geometry bounds:
    height, width = raster.shape
    raster_bounds = rasterio.transform.array_bounds(height, width, raster.transform)
    input_geometry_bounds = geometries.to_crs(RASTER_CRS).total_bounds
    assert all(abs(input_geometry_bounds - raster_bounds) < 10)


def test_fetch_and_merge_rasters_different_pixel_grid(geometries_different_pixel_grid):
    raster = fetch_and_merge_rasters(geometries_different_pixel_grid)

    assert raster.pixels.count() == 52122
    assert round(raster.pixels.mean()) == 451
