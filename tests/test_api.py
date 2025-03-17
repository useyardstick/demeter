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
            "ssurgo_primary_component",
        ],
        end_depth=30,
        year=2024,
        month=9,
    )

    expected = (
        geopandas.GeoDataFrame.from_features(
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
                            "ssurgo_map_unit_key": 372099,
                            "ssurgo_component_key": 25640791,
                            "ssurgo_component_percent": 80,
                            "ssurgo_component_name": "Dalhart",
                            "ssurgo_component_kind": "Series",
                            "ssurgo_drainage_class": "Well drained",
                            "ssurgo_taxonomic_class": "Fine-loamy, mixed, superactive, mesic Aridic Haplustalfs",
                            "ssurgo_taxonomic_order": "Alfisols",
                            "ssurgo_parent_material": "loamy eolian deposits",
                            "ssurgo_fine_fraction_percent_by_weight": 100.0,
                            "ssurgo_sand_percent_of_fine_fraction_by_weight": 63.666666666666664,
                            "ssurgo_silt_percent_of_fine_fraction_by_weight": 20.0,
                            "ssurgo_clay_percent_of_fine_fraction_by_weight": 16.333333333333332,
                            "ssurgo_organic_matter_percent_of_fine_fraction_by_weight": 0.6666666666666666,
                            "ssurgo_oven_dry_bulk_density_g_per_cm3": 1.5516666666666665,
                            "ssurgo_gravel_percent_by_weight": 0.0,
                            "ssurgo_fragment_percent_by_volume": None,
                            "ssurgo_fragment_kind": None,
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
                            "ssurgo_map_unit_key": 372110,
                            "ssurgo_component_key": 25640753,
                            "ssurgo_component_percent": 90,
                            "ssurgo_component_name": "Sherm",
                            "ssurgo_component_kind": "Series",
                            "ssurgo_drainage_class": "Well drained",
                            "ssurgo_taxonomic_class": "Fine, mixed, superactive, mesic Torrertic Paleustolls",
                            "ssurgo_taxonomic_order": "Mollisols",
                            "ssurgo_parent_material": "silty and clayey loess",
                            "ssurgo_fine_fraction_percent_by_weight": 100.0,
                            "ssurgo_sand_percent_of_fine_fraction_by_weight": 28.5,
                            "ssurgo_silt_percent_of_fine_fraction_by_weight": 33.5,
                            "ssurgo_clay_percent_of_fine_fraction_by_weight": 38.0,
                            "ssurgo_organic_matter_percent_of_fine_fraction_by_weight": 1.275,
                            "ssurgo_oven_dry_bulk_density_g_per_cm3": 1.6849999999999998,
                            "ssurgo_gravel_percent_by_weight": 0.0,
                            "ssurgo_fragment_percent_by_volume": None,
                            "ssurgo_fragment_kind": None,
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
                            "ssurgo_map_unit_key": 372111,
                            "ssurgo_component_key": 25640805,
                            "ssurgo_component_percent": 80,
                            "ssurgo_component_name": "Spurlock",
                            "ssurgo_component_kind": "Series",
                            "ssurgo_drainage_class": "Well drained",
                            "ssurgo_taxonomic_class": "Coarse-loamy, carbonatic, mesic Aridic Calciustepts",
                            "ssurgo_taxonomic_order": "Inceptisols",
                            "ssurgo_parent_material": "late Pleistocene & Holocene age calcareous loamy eolian deposits",
                            "ssurgo_fine_fraction_percent_by_weight": 94.0,
                            "ssurgo_sand_percent_of_fine_fraction_by_weight": 44.6,
                            "ssurgo_silt_percent_of_fine_fraction_by_weight": 32.0,
                            "ssurgo_clay_percent_of_fine_fraction_by_weight": 23.4,
                            "ssurgo_organic_matter_percent_of_fine_fraction_by_weight": 0.84,
                            "ssurgo_oven_dry_bulk_density_g_per_cm3": 1.556,
                            "ssurgo_gravel_percent_by_weight": 6.0,
                            "ssurgo_fragment_percent_by_volume": 3.2,
                            "ssurgo_fragment_kind": "Carbonate nodules, Quartzite fragments",
                        },
                    },
                ],
            },
        )
        .astype(
            {
                "usgs_catchment_id": "Int64",
                "usgs_flow_direction": "category",
            }
        )
        .convert_dtypes(
            convert_integer=False,
            convert_floating=False,
        )
    )

    assert_frame_equal(
        point_data,
        expected,
        check_like=True,  # ignore column order
    )
