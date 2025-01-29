import geopandas
import pytest
import rasterio.transform

from demeter.raster.sentinel2.ndvi import fetch_and_build_ndvi_rasters

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


def test_fetch_and_build_ndvi_rasters(
    geometries,
    sentinel2_rasters_in_s3,
    record_or_replay_requests,
):
    record_or_replay_requests(
        f"{SEARCH_RESPONSES_FIXTURE_DIR}/2024_09_fields_spanning_sentinel2_tiles.yaml"
    )

    rasters = list(fetch_and_build_ndvi_rasters(geometries, 2024, 9))
    assert len(rasters) == 1  # geometries are all in UTM zone 14

    raster = rasters[0]
    assert raster.crs == "EPSG:32614"
    assert raster.mean

    assert raster.mean.shape == (1521, 319)
    assert raster.mean.pixels.count() == 12287
    assert round(raster.mean.pixels.mean(), 3) == 0.548
    assert raster.mean.crs == "EPSG:32614"

    # Check that raster bounds are within 10m (1 pixel) of input geometry bounds:
    height, width = raster.mean.shape
    raster_bounds = rasterio.transform.array_bounds(
        height, width, raster.mean.transform
    )
    input_geometry_bounds = geometries.to_crs(raster.crs).total_bounds
    assert all(abs(input_geometry_bounds - raster_bounds) < 10)


def test_fetch_and_build_ndvi_rasters_min(
    geometries,
    sentinel2_rasters_in_s3,
    replay_requests,
):
    replay_requests(
        f"{SEARCH_RESPONSES_FIXTURE_DIR}/2024_09_fields_spanning_sentinel2_tiles.yaml"
    )

    rasters = list(
        fetch_and_build_ndvi_rasters(geometries, 2024, 9, statistics=["min"])
    )
    assert len(rasters) == 1

    raster = rasters[0]
    assert raster.min

    pixels, transform, crs = raster.min
    assert round(pixels.mean(), 3) == 0.458


def test_fetch_and_build_ndvi_rasters_max(
    geometries,
    sentinel2_rasters_in_s3,
    replay_requests,
):
    replay_requests(
        f"{SEARCH_RESPONSES_FIXTURE_DIR}/2024_09_fields_spanning_sentinel2_tiles.yaml"
    )

    rasters = list(
        fetch_and_build_ndvi_rasters(geometries, 2024, 9, statistics=["max"])
    )
    assert len(rasters) == 1

    raster = rasters[0]
    assert raster.max

    pixels, transform, crs = raster.max
    assert round(pixels.mean(), 3) == 0.602


def test_fetch_and_build_ndvi_rasters_stddev(
    geometries,
    sentinel2_rasters_in_s3,
    replay_requests,
):
    replay_requests(
        f"{SEARCH_RESPONSES_FIXTURE_DIR}/2024_09_fields_spanning_sentinel2_tiles.yaml"
    )

    with pytest.raises(ValueError):
        list(fetch_and_build_ndvi_rasters(geometries, 2024, 9, statistics=["stddev"]))

    rasters = list(
        fetch_and_build_ndvi_rasters(geometries, 2024, 9, statistics=["mean", "stddev"])
    )
    assert len(rasters) == 1

    raster = rasters[0]
    assert raster.mean
    assert raster.stddev

    pixels, transform, crs = raster.stddev
    assert round(pixels.mean(), 3) == 0.049


def test_detector_footprint_mask(
    geometries_spanning_datatake_edge,
    sentinel2_rasters_in_s3,
    record_or_replay_requests,
):
    """
    Applying the detector footprint mask should prevent artifacts at the edges
    of satellite's field of view. These artifacts are most visible in the min
    NDVI raster.
    """
    record_or_replay_requests(
        f"{SEARCH_RESPONSES_FIXTURE_DIR}/2024_09_agoro_shea_flanagan_west.yaml"
    )

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
