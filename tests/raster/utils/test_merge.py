import warnings

import numpy
import pytest
import rasterio
from numpy.testing import assert_array_equal

from demeter.raster import Raster
from demeter.raster.utils.merge import (
    OverlappingPixelsWarning,
    check_for_overlapping_pixels,
    merge,
    merge_stddev,
    merge_variance,
)


@pytest.fixture
def int_rasters_with_zero_nodata(tmp_path):
    return _save_rasters(
        tmp_path,
        [
            numpy.array(
                [
                    [6, 0],
                    [9, 4],
                ]
            ),
            numpy.array(
                [
                    [4, 3],
                    [5, 5],
                ]
            ),
        ],
        nodata=0,
    )


@pytest.fixture
def int_rasters_with_nonzero_nodata(tmp_path):
    return _save_rasters(
        tmp_path,
        [
            numpy.array(
                [
                    [6, -9999],
                    [9, 4],
                ]
            ),
            numpy.array(
                [
                    [4, 3],
                    [5, 5],
                ]
            ),
        ],
        nodata=-9999,
    )


@pytest.fixture
def float_rasters(tmp_path):
    return _save_rasters(
        tmp_path,
        [
            numpy.array(
                [
                    [4.0, 3.0],
                    [5.0, 5.0],
                ]
            ),
            numpy.array(
                [
                    [6.0, numpy.nan],
                    [9.0, 4.0],
                ]
            ),
        ],
        nodata=numpy.nan,
    )


@pytest.fixture
def top_left_raster():
    return Raster(
        pixels=numpy.ma.masked_array(
            [
                [0, 1, 2],
                [4, 5, 6],
                [8, 9, 10],
            ],
        ),
        transform=rasterio.Affine(10, 0, -176010, 0, -10, 2390250),
        crs="EPSG:5070",
    )


@pytest.fixture
def bottom_right_raster():
    return Raster(
        pixels=numpy.ma.masked_array(
            [
                [5, 6, 7],
                [9, 10, 11],
                [13, 14, 15],
            ],
        ),
        transform=rasterio.Affine(10, 0, -176000, 0, -10, 2390240),
        crs="EPSG:5070",
    )


def test_merge_int_rasters_with_nonzero_nodata(int_rasters_with_nonzero_nodata):
    """
    There was a bug in `rasterio.merge` that caused it to return invalid data
    when merging int rasters with a nonzero nodata value. The bug was fixed in
    rasterio 1.4.3. This is a regression test to make sure it stays fixed.
    """
    merged = merge(int_rasters_with_nonzero_nodata)
    assert_array_equal(merged.pixels[0], numpy.ma.masked_array([[6, 3], [9, 4]]))
    assert merged.pixels.fill_value == -9999


def test_merge_int_rasters_with_nonzero_nodata_as_float(
    int_rasters_with_nonzero_nodata,
):
    """
    The rasterio bug above also happened when converting the merged output to a
    float dtype. Make sure that stays fixed too.
    """
    merged = merge(int_rasters_with_nonzero_nodata, dtype=float)
    assert_array_equal(merged.pixels[0], numpy.ma.masked_array([[6, 3], [9, 4]]))
    assert merged.pixels.fill_value == -9999


def test_merge_int_rasters_with_nonzero_nodata_passing_zero_nodata(
    int_rasters_with_nonzero_nodata,
):
    """
    Passing a zero nodata value to `merge` works.
    """
    merged = merge(int_rasters_with_nonzero_nodata, nodata=0)
    assert_array_equal(merged.pixels[0], numpy.ma.masked_array([[6, 3], [9, 4]]))
    assert merged.pixels.fill_value == 0


def test_merge_int_rasters_with_zero_nodata(int_rasters_with_zero_nodata):
    """
    Merging int rasters with a zero nodata value works as expected.
    """
    merged = merge(int_rasters_with_zero_nodata)
    assert_array_equal(merged.pixels[0], numpy.ma.masked_array([[6, 3], [9, 4]]))
    assert merged.pixels.fill_value == 0


def test_merge_int_rasters_with_zero_nodata_passing_nonzero_nodata(
    int_rasters_with_zero_nodata,
):
    """
    Merging int rasters with a zero nodata value works as expected, even when
    passing a nonzero nodata value to `merge`.
    """
    merged = merge(int_rasters_with_zero_nodata, nodata=-9999)
    assert_array_equal(merged.pixels[0], numpy.ma.masked_array([[6, 3], [9, 4]]))
    assert merged.pixels.fill_value == -9999


def test_merge_min(float_rasters):
    min_raster, _, _ = merge(float_rasters, method="min")
    assert_array_equal(
        min_raster[0],
        numpy.array(
            [
                [4.0, 3.0],
                [5.0, 4.0],
            ]
        ),
    )


def test_merge_max(float_rasters):
    max_raster, _, _ = merge(float_rasters, method="max")
    assert_array_equal(
        max_raster[0],
        numpy.array(
            [
                [6.0, 3.0],
                [9.0, 5.0],
            ]
        ),
    )


def test_merge_mean(float_rasters):
    mean_raster, _, _ = merge(float_rasters, method="mean")
    assert_array_equal(
        mean_raster[0],
        numpy.array(
            [
                [5.0, 3.0],
                [7.0, 4.5],
            ]
        ),
    )


def test_merge_variance(float_rasters):
    mean = merge(float_rasters, method="mean")
    variance_raster, _, _ = merge_variance(float_rasters, mean)
    assert_array_equal(
        variance_raster[0],
        numpy.array(
            [
                [1.0, 0.0],
                [4.0, 0.25],
            ]
        ),
    )


def test_merge_stddev(float_rasters):
    mean = merge(float_rasters, method="mean")
    stddev_raster, _, _ = merge_stddev(float_rasters, mean)
    assert_array_equal(
        stddev_raster[0],
        numpy.array(
            [
                [1.0, 0.0],
                [2.0, 0.5],
            ]
        ),
    )


def test_merge_aligned_rasters(top_left_raster, bottom_right_raster):
    merged = merge(
        [top_left_raster, bottom_right_raster],
        allow_resampling=False,
    )

    assert merged.transform == top_left_raster.transform
    assert merged.crs == "EPSG:5070"
    assert_array_equal(
        merged.pixels[0],
        numpy.ma.masked_equal(
            [
                [0, 1, 2, -9999],
                [4, 5, 6, 7],
                [8, 9, 10, 11],
                [-9999, 13, 14, 15],
            ],
            -9999,
        ),
    )


def test_merge_rasters_no_resampling(top_left_raster, bottom_right_raster):
    # Offset top-left raster down and to the right by a fraction of a pixel:
    original_transform = top_left_raster.transform
    top_left_raster.transform = rasterio.Affine(
        10, 0, original_transform.xoff + 1, 0, -10, original_transform.yoff + 1
    )
    with pytest.raises(ValueError):
        merge(
            [top_left_raster, bottom_right_raster],
            allow_resampling=False,
        )


def test_merge_snap_bounds_to_grid(top_left_raster, bottom_right_raster):
    merged = merge(
        [top_left_raster, bottom_right_raster],
        bounds=(-175995, 2390215, -175975, 2390235),
        allow_resampling=False,
    )
    assert merged.transform == bottom_right_raster.transform
    assert_array_equal(
        merged.pixels,
        bottom_right_raster.pixels,
    )


def test_merge_overlapping_rasters(top_left_raster, bottom_right_raster):
    # The input rasters have overlapping pixels, but the values are equal.
    # Check that we don't get a warning:
    with warnings.catch_warnings():
        warnings.simplefilter("error", category=OverlappingPixelsWarning)
        merged = merge(
            [top_left_raster, bottom_right_raster],
            method=check_for_overlapping_pixels,
        )

    expected_output = numpy.ma.masked_equal(
        [
            [0, 1, 2, -9999],
            [4, 5, 6, 7],
            [8, 9, 10, 11],
            [-9999, 13, 14, 15],
        ],
        -9999,
    )
    assert_array_equal(merged.pixels[0], expected_output)

    # Change the second raster so that it has a pixel with a different value.
    # Now we should get a warning when we merge:
    bottom_right_raster.pixels[0, 0, 0] = 42
    with pytest.warns(OverlappingPixelsWarning):
        merged = merge(
            [top_left_raster, bottom_right_raster],
            method=check_for_overlapping_pixels,
        )

    # The output should use the first value for the overlapping pixel, so the
    # result is the same as before:
    assert_array_equal(merged.pixels[0], expected_output)


def _save_rasters(tmp_path, arrays, nodata):
    for index, array in enumerate(arrays):
        height, width = array.shape
        with rasterio.open(
            tmp_path / f"raster_{index}.tif",
            "w",
            count=1,
            height=height,
            width=width,
            dtype=array.dtype,
            transform=rasterio.Affine(1, 0, 0, 0, -1, 0),
            crs="EPSG:4326",
            nodata=nodata,
        ) as dst:
            dst.write(array, indexes=1)

    return list(tmp_path.glob("*.tif"))
