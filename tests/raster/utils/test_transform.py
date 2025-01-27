import pytest
from rasterio import Affine

from demeter.raster.utils.transform import (
    align_bounds_to_transform,
    extract_resolution_from_transform,
)


@pytest.fixture
def transform():
    return Affine(
        9.25925927753796e-05,
        0.0,
        -88.00055555649311,
        0.0,
        -9.259259269219641e-05,
        42.00055555599499,
    )


def test_extract_resolution_from_transform(transform):
    xres, yres = extract_resolution_from_transform(transform)
    assert xres == 9.25925927753796e-05
    assert yres == 9.259259269219641e-05


def test_align_bounds_to_transform(transform):
    left = -87.70684656839217
    bottom = 41.90305914410823
    right = -87.69698317121978
    top = 41.91004667088552

    (
        aligned_left,
        aligned_bottom,
        aligned_right,
        aligned_top,
    ) = align_bounds_to_transform((left, bottom, right, top), transform)

    # Check that the aligned bounds are larger than the original bounds:
    assert aligned_left < left
    assert aligned_bottom < bottom
    assert aligned_right > right
    assert aligned_top > top

    # Check that the bounds are aligned to the pixel grid:
    xres, yres = extract_resolution_from_transform(transform)

    left_col = (aligned_left - transform.xoff) / xres
    right_col = (aligned_right - transform.xoff) / xres
    assert left_col == pytest.approx(round(left_col))
    assert right_col == pytest.approx(round(right_col))

    top_row = (aligned_top - transform.yoff) / -yres
    bottom_row = (aligned_bottom - transform.yoff) / -yres
    assert top_row == pytest.approx(round(top_row))
    assert bottom_row == pytest.approx(round(bottom_row))
