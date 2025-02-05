import os
import re
import shutil

import geopandas
import pytest
import rasterio.transform

from demeter.raster import polaris


@pytest.fixture
def remote_cache(monkeypatch, s3):
    bucket_name = "polaris-cache"
    s3.create_bucket(Bucket=bucket_name)
    monkeypatch.setenv("POLARIS_REMOTE_CACHE", f"s3://{bucket_name}")


@pytest.fixture
def geometries():
    return geopandas.GeoDataFrame.from_features(
        [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            (-88, 43),
                            (-88, 41.5),
                            (-87, 41.5),
                            (-87, 43),
                            (-88, 43),
                        ]
                    ],
                },
                "properties": {},
            }
        ]
    )


@pytest.fixture
def mock_polaris(requests_mock):
    requests_mock.add_callback(
        "GET",
        re.compile(re.escape("http://hydrology.cee.duke.edu/POLARIS/PROPERTIES/v1.0/")),
        callback=_mock_polaris_callback,
    )
    yield requests_mock


def _mock_polaris_callback(request):
    filename = os.path.basename(request.url)
    fixture_dir = "tests/raster/fixtures/polaris/sparse"
    fixture_path = os.path.join(fixture_dir, filename)
    with open(fixture_path, "rb") as file:
        return 200, {"Content-Type": "image/tiff"}, file.read()


def test_fetch_polaris_data_for_depth_range(mock_polaris, geometries):
    rasters = polaris.fetch_polaris_data_for_depth_range(
        geometries,
        soil_property="om",
        end_depth=100,
    )
    assert rasters.stddev
    assert rasters.mean.pixels.shape == rasters.stddev.pixels.shape
    assert rasters.mean.transform == rasters.stddev.transform
    assert rasters.mean.crs == "EPSG:4326"


def test_fetch_polaris_data_for_depth_range_below_ground(mock_polaris, geometries):
    rasters = polaris.fetch_polaris_data_for_depth_range(
        geometries,
        soil_property="om",
        start_depth=30,
        end_depth=100,
    )
    assert rasters.stddev
    assert rasters.mean.pixels.shape == rasters.stddev.pixels.shape
    assert rasters.mean.transform == rasters.stddev.transform

    # Check that we didn't request any POLARIS tiles above 30cm:
    for call in mock_polaris.calls:
        for depth in polaris.Depth.select_between(0, 30):
            start_depth, end_depth = depth.value
            assert f"{start_depth}_{end_depth}" not in call.request.path_url


def test_fetch_polaris_data_for_depth_range_arbitrary_depths(mock_polaris, geometries):
    rasters = polaris.fetch_polaris_data_for_depth_range(
        geometries,
        soil_property="om",
        start_depth=10,
        end_depth=45,
    )
    assert rasters.stddev
    assert rasters.mean.pixels.shape == rasters.stddev.pixels.shape
    assert rasters.mean.transform == rasters.stddev.transform

    # Check that we only fetched the necessary POLARIS depths:
    requested_depths = set()
    for call in mock_polaris.calls:
        match = re.search(r"/(\d+)_(\d+)/[^/]+\.tif", call.request.path_url)
        assert match
        requested_depths.add(match.groups())

    assert requested_depths == {("5", "15"), ("15", "30"), ("30", "60")}


def test_fetch_polaris_data_for_depth_range_median_and_mode(mock_polaris, geometries):
    rasters = polaris.fetch_polaris_data_for_depth_range(
        geometries,
        soil_property=polaris.SoilProperty.ORGANIC_MATTER,
        additional_statistics=[polaris.Statistic.MEDIAN, polaris.Statistic.MODE],
        end_depth=100,
    )
    assert rasters.stddev
    assert rasters.median
    assert rasters.mode
    assert (
        rasters.mean.pixels.shape
        == rasters.stddev.pixels.shape
        == rasters.median.pixels.shape
        == rasters.mode.pixels.shape
    )
    assert (
        rasters.mean.transform
        == rasters.stddev.transform
        == rasters.median.transform
        == rasters.mode.transform
    )


def test_fetch_polaris_data(mock_polaris, geometries):
    raster = polaris.fetch_polaris_data(
        geometries,
        soil_property=polaris.SoilProperty.ORGANIC_MATTER,
        statistic=polaris.Statistic.MEAN,
        depth=polaris.Depth.ZERO_TO_FIVE_CM,
    )

    # Check that raster bounds are within 10m of input geometry bounds:
    height, width = raster.shape
    raster_bounds = rasterio.transform.array_bounds(height, width, raster.transform)
    assert all(abs(geometries.total_bounds - raster_bounds) < 10)


def test_fetch_polaris_data_with_remote_cache(
    mock_polaris, geometries, polaris_cache_directory, remote_cache
):
    for _ in range(2):
        polaris.fetch_polaris_data(
            geometries,
            soil_property=polaris.SoilProperty.ORGANIC_MATTER,
            statistic=polaris.Statistic.MEAN,
            depth=polaris.Depth.ZERO_TO_FIVE_CM,
        )
        shutil.rmtree(polaris_cache_directory)  # clear local cache

    unique_paths_called = {call.request.path_url for call in mock_polaris.calls}
    assert len(unique_paths_called) == len(mock_polaris.calls)


def test_select_depths_for_polaris():
    depths = polaris.Depth.select_between(0, 30)
    assert {x.value for x in depths} == {(0, 5), (5, 15), (15, 30)}
    assert sum(x.thickness for x in depths) == 30

    depths = polaris.Depth.select_between(0, 100)
    assert {x.value for x in depths} == {
        (0, 5),
        (5, 15),
        (15, 30),
        (30, 60),
        (60, 100),
    }

    assert sum(x.thickness for x in depths) == 100

    depths = polaris.Depth.select_between(0, 200)
    assert {x.value for x in depths} == {
        (0, 5),
        (5, 15),
        (15, 30),
        (30, 60),
        (60, 100),
        (100, 200),
    }

    assert sum(x.thickness for x in depths) == 200

    with pytest.raises(Exception):
        polaris.Depth.select_between(0, 99)


def test_estimate_carbon_stock(mock_polaris, geometries):
    rasters = polaris.estimate_carbon_stock(
        geometries,
        end_depth=100,
    )
    assert rasters.stddev
    assert rasters.mean.pixels.shape == rasters.stddev.pixels.shape
    assert rasters.mean.transform == rasters.stddev.transform
