import geopandas
import pytest
import rasterio.transform

from demeter.raster import Raster
from demeter.raster.sentinel2.ndvi import NDVIRasters, fetch_and_build_ndvi_rasters

SEARCH_RESPONSES_FIXTURE_DIR = "tests/fixtures/recorded_responses"


@pytest.fixture
def geometries():
    """
    Chosen to span multiple tiles.
    """
    return geopandas.read_file("tests/fixtures/fields_spanning_sentinel2_tiles.geojson")


@pytest.fixture
def geometries_spanning_datatake_edge():
    """
    For testing detector footprint masking.
    """
    return geopandas.read_file("tests/fixtures/texas_west.geojson")


@pytest.mark.parametrize("save_to_disk", (False, True))
def test_fetch_and_build_ndvi_rasters(
    geometries,
    record_or_replay_requests,
    use_sentinel2_fixtures,
    save_to_disk,
    tmp_path,
):
    use_sentinel2_fixtures(crop_to=geometries)

    rasters = list(
        fetch_and_build_ndvi_rasters(
            geometries, 2024, 9, dst_path=str(tmp_path) if save_to_disk else None
        )
    )
    assert len(rasters) == 1  # geometries are all in UTM zone 14

    raster = rasters[0]

    if save_to_disk:
        raster = NDVIRasters(
            crs=raster.crs,
            mean=Raster.from_file(raster.mean),
            min=Raster.from_file(raster.min),
            max=Raster.from_file(raster.max),
            stddev=Raster.from_file(raster.stddev),
        )

    assert raster.crs == "EPSG:32614"
    assert raster.mean

    assert raster.mean.shape == (1521, 319)
    assert raster.mean.pixels.count() == 12287
    assert round(raster.mean.pixels.mean(), 3) == 0.548
    assert raster.mean.crs == "EPSG:32614"

    assert raster.min
    assert round(raster.min.pixels.mean(), 3) == 0.458

    assert raster.max
    assert round(raster.max.pixels.mean(), 3) == 0.602

    assert raster.stddev
    assert round(raster.stddev.pixels.mean(), 3) == 0.049

    # Check that raster bounds are within 10m (1 pixel) of input geometry bounds:
    height, width = raster.mean.shape
    raster_bounds = rasterio.transform.array_bounds(
        height, width, raster.mean.transform
    )
    input_geometry_bounds = geometries.to_crs(raster.crs).total_bounds
    assert all(abs(input_geometry_bounds - raster_bounds) < 10)


def test_detector_footprint_mask(
    geometries_spanning_datatake_edge,
    record_or_replay_requests,
    use_sentinel2_fixtures,
):
    """
    Applying the detector footprint mask should prevent artifacts at the edges
    of satellite's field of view. These artifacts are most visible in the min
    NDVI raster.
    """
    use_sentinel2_fixtures(crop_to=geometries_spanning_datatake_edge)

    rasters = list(
        fetch_and_build_ndvi_rasters(
            geometries_spanning_datatake_edge, 2024, 9, statistics=["min"]
        )
    )
    assert len(rasters) == 1

    raster = rasters[0]
    assert raster.min

    pixels, transform, crs = raster.min

    # Without applying the detector footprint mask, this raster has a min value
    # of -0.715
    assert round(pixels.min(), 3) == 0.017
