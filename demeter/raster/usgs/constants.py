import os
from enum import Enum, unique

# This is where USGS keeps its raster archives:
S3_BUCKET_NAME = "prd-tnm"

# Downloaded raster files are cached here:
# TODO: use tmpdir in tests
CACHED_RASTER_FILES_DIRECTORY = os.environ.get(
    "USGS_CACHED_RASTER_FILES_DIRECTORY", ".usgs_cache"
)


@unique
class FlowDirection(Enum):
    SINK = 0
    E = 1
    SE = 2
    S = 4
    SW = 8
    W = 16
    NW = 32
    N = 64
    NE = 128
