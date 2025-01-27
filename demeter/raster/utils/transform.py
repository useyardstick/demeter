import math

import rasterio.transform
from rasterio import Affine


def extract_resolution_from_transform(
    transform: Affine,
) -> tuple[float, float]:
    """
    Return (x, y) resolution of the given Affine transform.
    """
    return transform.a, -transform.e


def extract_grid_offset_from_transform(transform: Affine) -> tuple[float, float]:
    """
    The (x, y) offset of given transform's origin point on a grid aligned with
    its resolution. For example: a transform with a resolution of (10, 10) and
    an offset of (16, 10) has a grid offset of (6, 0).
    """
    xres, yres = extract_resolution_from_transform(transform)
    return transform.xoff % xres, transform.yoff % yres


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


def _floor_unless_close(number):
    ceil = math.ceil(number)
    if math.isclose(number, ceil):
        return ceil
    return math.floor(number)
