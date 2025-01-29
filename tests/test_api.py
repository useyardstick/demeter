import os
import re

import geopandas
import pytest
from pandas.testing import assert_frame_equal

from demeter import api
from demeter.raster import polaris
from demeter.raster.usgs.constants import FlowDirection


@pytest.fixture
def mock_polaris(requests_mock):
    requests_mock.add_callback(
        "GET",
        re.compile(re.escape(polaris.BASE_URL)),
        callback=_mock_polaris_callback,
    )
    yield requests_mock


def _mock_polaris_callback(request):
    fixture_dir = "tests/raster/fixtures/polaris/point_data"
    fixture_path = os.path.join(fixture_dir, request.url.removeprefix(polaris.BASE_URL))
    with open(fixture_path, "rb") as file:
        return 200, {"Content-Type": "image/tiff"}, file.read()


def test_fetch_point_data(
    mock_polaris,
    sentinel2_rasters_in_s3,
    record_or_replay_requests,
):
    record_or_replay_requests(
        "tests/fixtures/recorded_responses/2024_09_point_data.yaml"
    )

    point_data = api.fetch_point_data(
        "tests/fixtures/points.geojson",
        [
            "polaris_carbon_stock",
            "sentinel2_ndvi",
            "usgs_hydrography",
            "usgs_topography",
        ],
        end_depth=30,
        year=2024,
        month=9,
    )

    expected = geopandas.GeoDataFrame.from_features(
        {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [-101.994703345038246, 36.245087922194472],
                    },
                    "properties": {
                        "polaris_carbon_stock_mean": 0.43250282843338883,
                        "polaris_carbon_stock_stddev": 0.1382376404293934,
                        "sentinel2_ndvi_mean": 0.28177666160884057,
                        "sentinel2_ndvi_min": 0.14034555926038189,
                        "sentinel2_ndvi_max": 0.51770045385779129,
                        "sentinel2_ndvi_stddev": 0.14222214508384667,
                        "usgs_catchment_id": 21001200027800,
                        "usgs_flow_accumulation": 0,
                        "usgs_flow_direction": FlowDirection.NE,
                        "usgs_elevation": 1099.478759765625,
                    },
                },
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [-102.092656767059026, 36.2512340537711],
                    },
                    "properties": {
                        "polaris_carbon_stock_mean": 0.85669107154489887,
                        "polaris_carbon_stock_stddev": 0.24542622540846529,
                        "sentinel2_ndvi_mean": 0.2328074349884762,
                        "sentinel2_ndvi_min": 0.14356435643564355,
                        "sentinel2_ndvi_max": 0.37533753375337536,
                        "sentinel2_ndvi_stddev": 0.10184737379464784,
                        "usgs_catchment_id": 21001200027800,
                        "usgs_flow_accumulation": 8,
                        "usgs_flow_direction": FlowDirection.SE,
                        "usgs_elevation": 1121.6126708984375,
                    },
                },
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [-102.066431251323877, 36.244726369989777],
                    },
                    "properties": {
                        "polaris_carbon_stock_mean": 0.87177195221129,
                        "polaris_carbon_stock_stddev": 0.26388554859031843,
                        "sentinel2_ndvi_mean": 0.16163945715744241,
                        "sentinel2_ndvi_min": 0.13242009132420088,
                        "sentinel2_ndvi_max": 0.1866913123844732,
                        "sentinel2_ndvi_stddev": 0.020139062008694288,
                        "usgs_catchment_id": 21001200027800,
                        "usgs_flow_accumulation": 19,
                        "usgs_flow_direction": FlowDirection.S,
                        "usgs_elevation": 1117.9820556640625,
                    },
                },
            ],
        },
    ).astype(
        {
            "usgs_catchment_id": "Int64",
            "usgs_flow_direction": "category",
        }
    )

    assert_frame_equal(
        point_data,
        expected,
        check_like=True,  # ignore column order
    )
