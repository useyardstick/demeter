import os
import shutil
import warnings
from collections.abc import Callable, Iterable, Sequence
from contextlib import ExitStack, contextmanager, nullcontext
from tempfile import TemporaryDirectory
from typing import Literal, Optional, Union, overload

import numpy
import rasterio
import rasterio.merge
import rasterio.stack
from rasterio.merge import copy_count, copy_first, copy_sum

from demeter.raster import Raster
from demeter.raster.utils.transform import (
    align_bounds_to_transform,
    aligned_pixel_grids,
)


@overload
def merge(
    rasters: Sequence,
    *,
    method: Union[
        Literal["first", "last", "min", "max", "sum", "count", "mean"], Callable
    ] = "first",
    bounds: Optional[tuple[float, float, float, float]] = None,
    allow_resampling: bool = True,
    dst_path: str,
    **kwargs,
) -> str: ...


@overload
def merge(
    rasters: Sequence,
    *,
    method: Union[
        Literal["first", "last", "min", "max", "sum", "count", "mean"], Callable
    ] = "first",
    bounds: Optional[tuple[float, float, float, float]] = None,
    allow_resampling: bool = True,
    dst_path: None = None,
    **kwargs,
) -> Raster: ...


def merge(
    rasters,
    *,
    method: Union[
        Literal["first", "last", "min", "max", "sum", "count", "mean"], Callable
    ] = "first",
    bounds: Optional[tuple[float, float, float, float]] = None,
    allow_resampling: bool = True,
    dst_path: Optional[str] = None,
    **kwargs,
) -> Union[str, Raster]:
    """
    Wraps `rasterio.merge.merge` to operate on Raster instances as well as
    rasterio datasets.

    The `method` argument specifies how to handle overlapping pixels. See
    https://rasterio.readthedocs.io/en/stable/api/rasterio.merge.html for
    details on the available methods.

    In addition to rasterio's built-in methods listed above, this also supports
    a `mean` method that returns the mean of all valid overlapping pixels.

    If you only need a specific region of the merged raster, pass
    `bounds=(left, bottom, right, top)`. This will speed up the merge
    significantly.

    By default, this function will resample rasters if they don't align to a
    common pixel grid. To prevent this, set `allow_resampling=False`. This will
    raise an error if the input rasters don't align.

    If you pass `dst_path`, the merged raster will be written to disk and the
    path to the file will be returned. Use this for large rasters that don't
    fit in memory.
    """
    if isinstance(rasters[0], Raster):
        dataset_opener = _rasters_as_datasets
    else:
        dataset_opener = nullcontext  # type: ignore

    with dataset_opener(rasters) as sources:
        return _merge(
            sources,
            method=method,
            bounds=bounds,
            allow_resampling=allow_resampling,
            dst_path=dst_path,
            **kwargs,
        )


def merge_variance(
    rasters: Sequence,
    mean: Union[str, Raster],
    dst_path: Optional[str] = None,
    **kwargs,
) -> Union[str, Raster]:
    """
    Calculate the mean variance of rasters from the given mean.
    """
    with TemporaryDirectory() as tmpdir:
        if isinstance(mean, str) and isinstance(rasters[0], Raster):
            mean = Raster.from_file(mean)
        elif isinstance(mean, Raster) and isinstance(rasters[0], str):
            mean_path = os.path.join(tmpdir, "mean.tif")
            mean.save(mean_path)
            mean = mean_path

        # First, stack the input rasters with the mean raster.
        stacked_path = os.path.join(tmpdir, "stacked.tif")
        _stack([*rasters, mean], allow_resampling=False, dst_path=stacked_path)

        # Then, merge the stacked raster to calculate variance.
        return merge(
            [stacked_path],
            method=_copy_variance_from_stacked_mean,
            output_count=1,
            dst_path=dst_path,
            **kwargs,
        )


def merge_stddev(
    rasters: Sequence,
    mean: Union[str, Raster],
    dst_path: Optional[str] = None,
    **kwargs,
) -> Union[str, Raster]:
    """
    Calculate the mean standard deviation of rasters from the given mean.
    """
    variance_raster = merge_variance(rasters, mean, **kwargs)
    return merge(
        [variance_raster],
        method=_ufunc_as_merge_method(numpy.sqrt),
        dst_path=dst_path,
        **kwargs,
    )


@contextmanager
def _rasters_as_datasets(rasters: Iterable[Raster]):
    with ExitStack() as stack:
        datasets = [stack.enter_context(raster.as_dataset()) for raster in rasters]
        yield datasets


@contextmanager
def _dataset_opener(source):
    if isinstance(source, (str, os.PathLike)):
        with rasterio.open(source) as dataset:
            yield dataset
    else:
        yield source


def _merge(
    sources: Sequence,
    *,
    method,
    bounds=None,
    allow_resampling=True,
    output_count=None,
    dst_path: Optional[str] = None,
    **kwargs,
) -> Union[str, Raster]:
    # Get the CRS from the first raster. If any of the other rasters have a
    # different CRS, the call to `rasterio.merge.merge` below will raise an
    # exception, so we can safely assume this is the CRS to use for the output
    # raster.
    first_source = sources[0]

    with _dataset_opener(first_source) as dataset:
        crs = dataset.crs
        num_bands = dataset.count
        transform = dataset.transform

    if crs is None:
        raise ValueError("Rasters have no CRS")

    # To merge without resampling, all rasters must be on the same pixel grid.
    # Rasterio doesn't provide a way to enforce this when merging, so check
    # first:
    if not allow_resampling and not _aligned_pixel_grids(sources):
        raise ValueError(
            "Rasters must be on the same pixel grid to merge without resampling"
        )

    # If bounds are given, snap them to the first raster's pixel grid:
    if bounds:
        bounds = align_bounds_to_transform(bounds, transform)

    calculating_mean = method == "mean"
    if calculating_mean:
        if output_count is not None or num_bands > 1:
            raise ValueError(
                "Calculating mean for multi-band rasters not yet supported"
            )

        # Stack two numpy arrays, the first with the sum of valid values at
        # each pixel, and the second with the count of valid values at each
        # pixel. Then divide the first array by the second to get the mean.
        method = _copy_sum_and_count
        output_count = 2

    merged = rasterio.merge.merge(
        sources,
        masked=True,
        method=method,
        bounds=bounds,
        dst_path=dst_path,
        output_count=output_count,
        **kwargs,
    )

    if dst_path is None:
        pixels, transform = merged
        raster: Union[str, Raster] = Raster(pixels, transform, str(crs))
    else:
        raster = dst_path

    if calculating_mean:
        return _mean_from_sum_and_count(raster)
    return raster


def _mean_from_sum_and_count(raster: Union[str, Raster]) -> Union[str, Raster]:
    if isinstance(raster, str):
        with TemporaryDirectory() as tmpdir:
            tmp_path = os.path.join(tmpdir, "mean.tif")
            rasterio.merge.merge(
                [raster],
                masked=True,
                method=_copy_mean_from_sum_and_count,
                dst_path=tmp_path,
                output_count=1,
            )
            shutil.copyfile(tmp_path, raster)
        return raster

    pixels, transform, crs = raster
    pixels_sum, pixels_count = pixels
    pixels_mean = pixels_sum / pixels_count
    return Raster(pixels_mean, transform, crs)


def _ufunc_as_merge_method(fn):
    """
    Take numpy ufunc, and return a rasterio merge method that applies it.
    """

    def merge_method(merged_data, new_data, merged_mask, new_mask, **kwargs):
        copy_first(merged_data, fn(new_data), merged_mask, new_mask, **kwargs)

    return merge_method


def _copy_sum_and_count(merged_data, new_data, merged_mask, new_mask, **kwargs):
    """
    Combines rasterio's builtin `copy_sum` and `copy_count` functions.

    Expects a 3D array of length 2, which you can get by passing
    `output_count=2` to `rasterio.merge.merge`. We split this into two arrays,
    using the first for the sum and the second for the count.
    """
    assert merged_data.ndim == 3 and len(merged_data) == 2
    assert new_data.ndim == 3 and len(new_data) == 1
    merged_sum, merged_count = numpy.split(merged_data, 2)
    merged_sum_mask, merged_count_mask = numpy.split(merged_mask, 2)

    copy_sum(merged_sum, new_data, merged_sum_mask, new_mask, **kwargs)
    copy_count(merged_count, new_data, merged_count_mask, new_mask, **kwargs)


def _copy_mean_from_sum_and_count(
    merged_data, new_data, merged_mask, new_mask, **kwargs
):
    """
    Pass this as the `method` argument to `rasterio.merge.merge` to combine an
    2-band raster of sum and count values into a 1-band raster of mean values.

    Using `rasterio.merge.merge` is a bit of a hack for this, since we're not
    actually merging 2 rasters together. But it allows us to leverage existing
    code in rasterio for chunking the data, which is helpful for large rasters
    that don't fit in memory.
    """
    assert merged_data.ndim == 3 and len(merged_data) == 1
    assert new_data.ndim == 3 and len(new_data) == 2
    new_sum, new_count = numpy.split(new_data, 2)
    new_sum_mask, new_count_mask = numpy.split(new_mask, 2)

    new_mask = numpy.logical_and(new_sum_mask, new_count_mask)
    mean = new_sum / new_count
    copy_first(merged_data, mean, merged_mask, new_mask, **kwargs)


def _copy_variance_from_stacked_mean(
    merged_data, new_data, merged_mask, new_mask, **kwargs
):
    """
    Given a multi-band raster where the *last* raster is the mean of all the
    other bands, calculate and return the variance of all the other bands.
    """
    assert merged_data.ndim == 3 and len(merged_data) == 1
    assert new_data.ndim == 3 and len(new_data) > 1
    new_values, new_mean = numpy.split(new_data, [-1])
    _, new_mean_mask = numpy.split(new_mask, [-1])

    variance = (new_values - new_mean) ** 2
    mean_variance = variance.mean(axis=0, keepdims=True)
    copy_first(merged_data, mean_variance, merged_mask, new_mean_mask, **kwargs)


def _stack(rasters: Sequence, *, allow_resampling=True, **kwargs):
    if isinstance(rasters[0], Raster):
        dataset_opener = _rasters_as_datasets
    else:
        dataset_opener = nullcontext  # type: ignore

    with dataset_opener(rasters) as sources:
        if not allow_resampling and not _aligned_pixel_grids(sources):
            raise ValueError(
                "Rasters must be on the same pixel grid to stack without resampling"
            )

        return rasterio.stack.stack(sources, **kwargs)


def _aligned_pixel_grids(sources) -> bool:
    xs = []
    ys = []
    transforms = []
    for source in sources:
        with _dataset_opener(source) as dataset:
            transforms.append(dataset.transform)
            left, bottom, right, top = dataset.bounds
            xs += [left, right]
            ys += [bottom, top]

    extent = min(xs), min(ys), max(xs), max(ys)

    return aligned_pixel_grids(extent, transforms)


class OverlappingPixelsWarning(Warning):
    pass


def check_for_overlapping_pixels(
    merged_data, new_data, merged_mask, new_mask, **kwargs
):
    """
    When passed as the `method` argument to `rasterio.merge.merge`, this
    function checks whether any two rasters have data for the same pixel.
    If they do, it logs a warning.
    """
    # `merged_mask` and `new_mask` are boolean arrays with True values for
    # invalid pixels. For every pixel, one or both of `merged_mask` and
    # `new_mask` should be True. If *both* are False, it means two rasters have
    # valid data for the same pixel. If we see this, and the values are
    # different, log a warning.
    overlap_mask = ~(merged_mask | new_mask)
    if (merged_data[overlap_mask] != new_data[overlap_mask]).any():
        warnings.warn(
            "Input rasters have overlapping pixels with different values!",
            category=OverlappingPixelsWarning,
        )

    # Carry on with rasterio's default merge behavior:
    rasterio.merge.copy_first(merged_data, new_data, merged_mask, new_mask, **kwargs)
