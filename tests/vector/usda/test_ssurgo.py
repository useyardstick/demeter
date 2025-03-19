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
    )

    # Check that the returned GeoDataFrame matches the input geometries:
    geometries_projected = geometries.to_crs("EPSG:5070").geometry.union_all()
    output_projected = primary_soil_components.to_crs("EPSG:5070").geometry.union_all()
    intersection = geometries_projected.intersection(output_projected)
    assert intersection.area == pytest.approx(geometries_projected.area, rel=1e-3)

    # Check that the returned GeoDataFrame has all the data we expect:
    assert_frame_equal(
        primary_soil_components.drop(columns="geometry"),
        pandas.DataFrame(
            {
                "map_unit_key": [
                    462569,
                    462573,
                    462583,
                    462584,
                    462616,
                    462620,
                    462621,
                    462718,
                    3295885,
                ],
                "map_unit_symbol": [
                    "DrA",
                    "DwA",
                    "FrA",
                    "FsA",
                    "HfA",
                    "HkbA",
                    "HmA",
                    "TpA",
                    "HdA",
                ],
                "map_unit_name": [
                    "Dinuba sandy loam, 0 to 1 percent slopes",
                    "Dinuba sandy loam, slightly saline-alkali, 0 to 1 percent slopes",
                    "Fresno fine sandy loam, moderately saline-alkali, 0 to 1 percent slopes",
                    "Fresno fine sandy loam, strongly saline-alkali, 0 to 1 percent slopes",
                    "Hilmar loamy sand, 0 to 1 percent",
                    "Hilmar loamy sand, slightly saline-alkali, 0 to 1 percent slopes",
                    "Hilmar sand, 0 to 3 percent slopes",
                    "Traver sandy loam, slightly saline-alkali, 0 to 1 percent slopes",
                    "Hanford sandy loam, 0 to 3 percent slopes",
                ],
                "component_key": [
                    26039573,
                    26039579,
                    26039679,
                    26039644,
                    26039790,
                    26039753,
                    26039733,
                    26040127,
                    26040173,
                ],
                "component_percent": [
                    85,
                    85,
                    85,
                    85,
                    85,
                    85,
                    85,
                    85,
                    85,
                ],
                "component_name": [
                    "Dinuba",
                    "Dinuba",
                    "Fresno",
                    "Fresno",
                    "Hilmar",
                    "Hilmar",
                    "Hilmar",
                    "Traver",
                    "Hanford",
                ],
                "component_kind": [
                    "Series",
                    "Series",
                    "Series",
                    "Series",
                    "Series",
                    "Series",
                    "Series",
                    "Series",
                    "Series",
                ],
                "drainage_class": [
                    "Moderately well drained",
                    "Moderately well drained",
                    "Moderately well drained",
                    "Moderately well drained",
                    "Somewhat excessively drained",
                    "Somewhat excessively drained",
                    "Somewhat excessively drained",
                    "Moderately well drained",
                    "Well drained",
                ],
                "taxonomic_class": [
                    "Coarse-loamy, mixed, active, thermic Typic Haploxeralfs",
                    "Coarse-loamy, mixed, active, thermic Typic Haploxeralfs",
                    "Fine-loamy, mixed, thermic Natric Durixeralfs",
                    "Fine-loamy, mixed, thermic Natric Durixeralfs",
                    "Sandy over loamy, mixed (calcareous), active, thermic Aeric Halaquepts",
                    "Sandy over loamy, mixed (calcareous), active, thermic Aeric Halaquepts",
                    "Sandy over loamy, mixed (calcareous), active, thermic Aeric Halaquepts",
                    "Coarse-loamy, mixed, thermic Natric Haploxeralfs",
                    "Coarse-loamy, mixed, superactive, nonacid, thermic Typic Xerorthents",
                ],
                "taxonomic_order": [
                    "Alfisols",
                    "Alfisols",
                    "Alfisols",
                    "Alfisols",
                    "Inceptisols",
                    "Inceptisols",
                    "Inceptisols",
                    "Alfisols",
                    "Entisols",
                ],
                "parent_material": [
                    "alluvium derived from granite",
                    "alluvium derived from granite",
                    "alluvium derived from granite",
                    "alluvium derived from granite",
                    "wind modified sandy alluvium derived from granite over silty alluvium derived from granite",
                    "wind modified sandy alluvium derived from granite over silty alluvium derived from granite",
                    "wind modified sandy alluvium derived from granite over silty alluvium derived from granite",
                    "alluvium derived from granite",
                    "alluvium derived from granite",
                ],
                "fine_fraction_percent_by_weight": [
                    97.0,
                    97.0,
                    96.65979381443299,
                    96.65979381443299,
                    100.0,
                    100.0,
                    100.0,
                    97.0,
                    92.0,
                ],
                "sand_percent_of_fine_fraction_by_weight": [
                    57.864,
                    57.864,
                    39.74329896907216,
                    39.74329896907216,
                    69.736,
                    69.736,
                    72.88600000000001,
                    67.06,
                    68.0,
                ],
                "silt_percent_of_fine_fraction_by_weight": [
                    28.886,
                    28.886,
                    37.06082474226804,
                    37.06082474226804,
                    23.049,
                    23.049,
                    20.349,
                    19.39,
                    20.0,
                ],
                "clay_percent_of_fine_fraction_by_weight": [
                    13.25,
                    13.25,
                    23.195876288659793,
                    23.195876288659793,
                    7.215,
                    7.215,
                    6.765,
                    13.55,
                    12.0,
                ],
                "organic_matter_percent_of_fine_fraction_by_weight": [
                    0.315,
                    0.315,
                    0.3170103092783505,
                    0.3170103092783505,
                    0.275,
                    0.275,
                    0.275,
                    0.54,
                    0.4,
                ],
                "oven_dry_bulk_density_g_per_cm3": [
                    1.6704,
                    1.6324,
                    1.5904123711340208,
                    1.5904123711340208,
                    1.7299999999999998,
                    1.7299999999999998,
                    1.7299999999999998,
                    1.62,
                    1.65,
                ],
                "gravel_percent_by_weight": [
                    3.0,
                    3.0,
                    3.3402061855670104,
                    3.3402061855670104,
                    0.0,
                    0.0,
                    0.0,
                    3.0,
                    8.0,
                ],
                "fragment_percent_by_volume": [
                    2.0,
                    2.0,
                    2.0,
                    2.0,
                    None,
                    None,
                    None,
                    2.0,
                    5.0,
                ],
                "fragment_kind": [
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    "Igneous rock fragments",
                ],
            }
        ).convert_dtypes(
            convert_integer=False,
            convert_floating=False,
        ),
    )
