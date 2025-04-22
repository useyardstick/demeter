import geopandas
import numpy
import pytest
from rasterio import Affine

from demeter.raster import Raster
from demeter.raster.utils.mask import mask


@pytest.mark.parametrize("filename", (None, "raster.tif"))
def test_mask_raster(tmp_path, filename):
    matrix = numpy.ma.ones((4, 4))
    shapes = geopandas.GeoDataFrame.from_features(
        [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            (1, 1),
                            (3, 1),
                            (3, 3),
                            (1, 3),
                            (1, 1),
                        ]
                    ],
                },
                "properties": {},
            },
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            (2, 2),
                            (4, 2),
                            (4, 4),
                            (2, 4),
                            (2, 2),
                        ]
                    ],
                },
                "properties": {},
            },
        ]
    )
    input_raster = Raster(matrix, transform=Affine.identity(), crs="EPSG:4326")
    if filename is None:
        result = mask(input_raster, shapes=shapes)
    else:
        raster_path = tmp_path / filename
        mask(input_raster, shapes=shapes, dst_path=raster_path)
        result = Raster.from_file(raster_path)

    expected = numpy.ma.array(
        matrix,
        mask=~numpy.ma.make_mask(
            [
                [0, 0, 0, 0],
                [0, 1, 1, 0],
                [0, 1, 1, 1],
                [0, 0, 1, 1],
            ]
        ),
    )
    assert numpy.ma.allequal(result.pixels, expected)
