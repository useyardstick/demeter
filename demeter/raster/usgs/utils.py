import os

import boto3
from botocore import UNSIGNED
from botocore.client import Config
from filelock import FileLock

from demeter.raster import Raster
from demeter.raster.usgs.constants import CACHED_RASTER_FILES_DIRECTORY, S3_BUCKET_NAME
from demeter.raster.utils.mask import mask
from demeter.raster.utils.merge import check_for_overlapping_pixels, merge

# Bucket is public, don't send credentials:
s3_client = boto3.client("s3", config=Config(signature_version=UNSIGNED))


def download_from_s3(key: str) -> str:
    local_path = os.path.join(CACHED_RASTER_FILES_DIRECTORY, key)
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    with FileLock(f"{local_path}.lock", timeout=60):
        if os.path.exists(local_path):
            # TODO: check if file in cache is stale
            print(f"Cache hit: {local_path}")
        else:
            print(f"Downloading s3://{S3_BUCKET_NAME}/{key}")
            s3_client.download_file(S3_BUCKET_NAME, key, local_path)

    return local_path


def merge_and_crop_rasters(sources, crop_to=None) -> Raster:
    if crop_to is None:
        return _merge_rasters(sources)

    merged = _merge_rasters(sources, bounds=tuple(crop_to.total_bounds))
    return mask(merged, crop_to, all_touched=True)


def _merge_rasters(sources, **kwargs) -> Raster:
    # FIXME: merging datasets that are very far apart uses a huge amount of
    # memory, even if the data is very sparse. I think this is because numpy
    # arrays allocate memory for every pixel. Find a way to mitigate.
    print("Merging rasters")

    # USGS elevation tiles overlap their neighboring tiles by 6 pixels. The
    # data in this overlapping region should be the same for both tiles. If
    # not, log a warning.
    return merge(
        sources,
        method=check_for_overlapping_pixels,
        allow_resampling=False,
        **kwargs,
    )
