import math
from collections.abc import Iterable

import numpy
import rasterio.transform
import rasterio.windows
from rasterio import Affine


def extract_resolution_from_transform(
    transform: Affine,
) -> tuple[float, float]:
    """
    Return (x, y) resolution of the given Affine transform.
    """
    return transform.a, -transform.e


def align_bounds_to_transform(
    bounds: tuple[float, float, float, float],
    transform: Affine,
) -> tuple[float, float, float, float]:
    """
    Expand the given bounds to align with the transform's pixel grid.
    """
    left, bottom, right, top = bounds

    # We should always snap up and left to avoid cropping the input bounds.  In
    # some cases however, the top-left corner of the bounds is *just* above or
    # to the left of a pixel, because of floating point/rounding issues. Snap
    # down/right such cases:
    top_row, left_col = rasterio.transform.rowcol(
        transform, left, top, op=_floor_unless_close
    )
    bottom_row, right_col = rasterio.transform.rowcol(transform, right, bottom)
    new_left, new_top = rasterio.transform.xy(transform, top_row, left_col, offset="ul")
    new_right, new_bottom = rasterio.transform.xy(
        transform, bottom_row, right_col, offset="lr"
    )
    return new_left, new_bottom, new_right, new_top


def aligned_pixel_grids(
    bounds: tuple[float, float, float, float],
    transforms: Iterable[rasterio.Affine],
) -> bool:
    """
    Check that the given transforms are on the same pixel grid within the
    given bounds, give or take a small tolerance.
    """
    windows = [
        rasterio.windows.from_bounds(*bounds, transform).round_lengths()
        for transform in transforms
    ]

    # First, check that the given transforms yield the same pixel offsets for
    # the top-left corner of the bounds:
    offsets = [(window.row_off, window.col_off) for window in windows]
    pixel_offsets_rounded = [tuple(row) for row in numpy.array(offsets).round(2) % 1]
    if len(set(pixel_offsets_rounded)) > 1:
        return False

    # Next, check that the given transforms yield the same number of pixels
    # over the given bounds:
    shapes = [(window.height, window.width) for window in windows]
    return len(set(shapes)) == 1


def _floor_unless_close(number):
    ceil = math.ceil(number)
    if math.isclose(number, ceil):
        return ceil
    return math.floor(number)
