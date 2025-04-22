from contextlib import nullcontext
from typing import Optional, Union, overload

import geopandas
import rasterio
import rasterio.features
import rasterio.mask
import rasterio.windows

from demeter.raster import Raster


@overload
def mask(
    raster,
    shapes,
    *,
    crop: bool = False,
    dst_path: str,
    **kwargs,
) -> str: ...


@overload
def mask(
    raster,
    shapes,
    *,
    crop: bool = False,
    dst_path: None = None,
    **kwargs,
) -> Raster: ...


def mask(
    raster,
    shapes,
    *,
    crop: bool = False,
    dst_path: Optional[str] = None,
    **kwargs,
) -> Union[str, Raster]:
    """
    Wraps `rasterio.mask.mask`, with the following differences:

    - Can accept a `Raster` instance as well as a rasterio dataset.
    - Returns a `Raster` instance instead of a (raster, transform) 2-tuple.
    - Alternatively, writes the masked raster to disk if `dst_path` is given.
      Useful for large rasters that don't fit in memory.
    """
    if isinstance(shapes, geopandas.GeoDataFrame):
        shapes = shapes.geometry

    if isinstance(raster, str):
        dataset_opener = rasterio.open
    elif isinstance(raster, Raster):
        dataset_opener = Raster.as_dataset
    else:
        dataset_opener = nullcontext

    with dataset_opener(raster) as src:
        crs = src.crs
        if crs is None:
            raise ValueError("Raster has no CRS")

        if dst_path is None:
            pixels, transform = rasterio.mask.mask(
                src, shapes, filled=False, crop=crop, **kwargs
            )
            return Raster(pixels, transform, str(crs))

        # If `dst_path` is given, write the masked raster to disk in chunks:
        profile = src.profile
        if crop:
            # TODO: padding
            window = rasterio.features.geometry_window(src, shapes)
            profile.update(
                width=window.width,
                height=window.height,
                transform=src.window_transform(window),
            )
        else:
            window = rasterio.windows.Window(0, 0, src.width, src.height)

        chunks = rasterio.windows.subdivide(
            window, 512, 512  # TODO: parametrize chunk size?
        )

        with rasterio.open(dst_path, "w", **profile) as dst:
            for chunk in chunks:
                pixels = src.read(window=chunk, masked=True)
                transform = src.window_transform(chunk)
                raster = Raster(pixels, transform, str(crs))
                masked = mask(raster, shapes, crop=False, **kwargs)
                dst.write(masked.pixels, window=chunk)

        return dst_path
