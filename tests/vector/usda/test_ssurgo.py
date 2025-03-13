import geopandas
import pandas
import pytest
from pandas.testing import assert_frame_equal

from demeter.vector.usda.ssurgo import fetch_primary_soil_components


@pytest.fixture
def geometries():
    return geopandas.read_file("tests/fixtures/central_valley.geojson")


def test_fetch_primary_soil_components(record_or_replay_requests, geometries):
    primary_soil_components = fetch_primary_soil_components(
        geometries, bottom_depth_cm=100
    ).sort_values("map_unit_key", ignore_index=True)

    # Check that the returned GeoDataFrame matches the input geometries:
    assert all(geometries.covered_by(primary_soil_components.geometry.union_all()))

    # Check that the returned GeoDataFrame has all the data we expect:
    assert_frame_equal(
        primary_soil_components.drop(columns="geometry"),
        pandas.DataFrame(
            {
                "map_unit_key": [462972, 462974, 463068, 463073, 463074, 463075],
                "component_key": [
                    25536052,
                    25536056,
                    25536410,
                    25536435,
                    25536442,
                    25536443,
                ],
                "component_percent": [85, 85, 85, 85, 85, 85],
                "component_name": [
                    "Burchell",
                    "Burchell",
                    "Landlow",
                    "Lewis",
                    "Lewis",
                    "Lewis",
                ],
                "component_kind": [
                    "Series",
                    "Series",
                    "Series",
                    "Series",
                    "Series",
                    "Series",
                ],
                "drainage_class": [
                    "Somewhat poorly drained",
                    "Somewhat poorly drained",
                    "Somewhat poorly drained",
                    "Moderately well drained",
                    "Moderately well drained",
                    "Moderately well drained",
                ],
                "taxonomic_class": [
                    "Fine-loamy, mixed, active, thermic Mollic Haploxeralfs",
                    "Fine-loamy, mixed, active, thermic Mollic Haploxeralfs",
                    "Fine, smectitic, thermic Aquic Haploxerolls",
                    "Fine, smectitic, thermic Natric Durixeralfs",
                    "Fine, smectitic, thermic Natric Durixeralfs",
                    "Fine, smectitic, thermic Natric Durixeralfs",
                ],
                "taxonomic_order": [
                    "Alfisols",
                    "Alfisols",
                    "Mollisols",
                    "Alfisols",
                    "Alfisols",
                    "Alfisols",
                ],
                "parent_material": [
                    "alluvium derived from igneous and sedimentary rock",
                    "alluvium derived from igneous and sedimentary rock",
                    "alluvium derived from igneous, metamorphic and sedimentary rock",
                    "alluvium derived from igneous, metamorphic and sedimentary rock",
                    "alluvium derived from igneous, metamorphic and sedimentary rock",
                    "alluvium derived from igneous, metamorphic and sedimentary rock",
                ],
                "fine_fraction_percent_by_weight": [
                    97.5,
                    97.5,
                    100.0,
                    99.19444444444444,
                    99.19444444444444,
                    99.19444444444444,
                ],
                "sand_percent_of_fine_fraction_by_weight": [
                    19.936,
                    19.936,
                    22.82,
                    28.25111111111111,
                    28.25111111111111,
                    28.25111111111111,
                ],
                "silt_percent_of_fine_fraction_by_weight": [
                    52.888999999999996,
                    52.888999999999996,
                    36.88,
                    32.193333333333335,
                    32.193333333333335,
                    32.193333333333335,
                ],
                "clay_percent_of_fine_fraction_by_weight": [
                    27.175,
                    27.175,
                    40.3,
                    39.55555555555556,
                    39.55555555555556,
                    39.55555555555556,
                ],
                "organic_matter_percent_of_fine_fraction_by_weight": [
                    1.1425,
                    1.1425,
                    0.625,
                    0.25277777777777777,
                    0.25277777777777777,
                    0.25277777777777777,
                ],
                "oven_dry_bulk_density_g_per_cm3": [
                    1.5738999999999999,
                    1.5738999999999999,
                    1.7309999999999999,
                    1.7952222222222225,
                    1.7952222222222225,
                    1.795222222222222,
                ],
                "gravel_percent_by_weight": [
                    2.5,
                    2.5,
                    0.0,
                    0.8055555555555556,
                    0.8055555555555556,
                    0.8055555555555556,
                ],
            }
        ),
    )
