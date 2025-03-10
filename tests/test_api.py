import geopandas
import pytest
from pandas.testing import assert_frame_equal

from demeter import api
from demeter.raster.usgs.constants import FlowDirection


@pytest.fixture
def points():
    return geopandas.read_file("tests/fixtures/points.geojson")


def test_fetch_point_data(
    points,
    record_or_replay_requests,
    use_polaris_fixtures,
    use_sentinel2_fixtures,
):
    use_polaris_fixtures(crop_to=points)
    use_sentinel2_fixtures(crop_to=points)

    point_data = api.fetch_point_data(
        points,
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
