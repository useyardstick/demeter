# Table of Contents

* [demeter.api](#demeter.api)
  * [fetch\_point\_data](#demeter.api.fetch_point_data)
* [demeter.raster](#demeter.raster)
  * [Raster](#demeter.raster.Raster)
    * [pixels](#demeter.raster.Raster.pixels)
    * [transform](#demeter.raster.Raster.transform)
    * [crs](#demeter.raster.Raster.crs)
    * [from\_file](#demeter.raster.Raster.from_file)
    * [count](#demeter.raster.Raster.count)
    * [shape](#demeter.raster.Raster.shape)
    * [bounds](#demeter.raster.Raster.bounds)
    * [resolution](#demeter.raster.Raster.resolution)
    * [dtype](#demeter.raster.Raster.dtype)
    * [nodata](#demeter.raster.Raster.nodata)
    * [value\_at](#demeter.raster.Raster.value_at)
    * [values\_at](#demeter.raster.Raster.values_at)
    * [save](#demeter.raster.Raster.save)
    * [as\_dataset](#demeter.raster.Raster.as_dataset)
    * [\_\_post\_init\_\_](#demeter.raster.Raster.__post_init__)
    * [\_\_iter\_\_](#demeter.raster.Raster.__iter__)
* [demeter.raster.polaris](#demeter.raster.polaris)
  * [estimate\_carbon\_stock](#demeter.raster.polaris.estimate_carbon_stock)
  * [fetch\_polaris\_data\_for\_depth\_range](#demeter.raster.polaris.fetch_polaris_data_for_depth_range)
  * [fetch\_polaris\_data](#demeter.raster.polaris.fetch_polaris_data)
* [demeter.raster.slga](#demeter.raster.slga)
  * [estimate\_carbon\_stock](#demeter.raster.slga.estimate_carbon_stock)
  * [fetch\_slga\_data\_for\_depth\_range](#demeter.raster.slga.fetch_slga_data_for_depth_range)
* [demeter.raster.usgs.topography](#demeter.raster.usgs.topography)
  * [fetch\_and\_merge\_rasters](#demeter.raster.usgs.topography.fetch_and_merge_rasters)
  * [fetch\_rasters](#demeter.raster.usgs.topography.fetch_rasters)
* [demeter.raster.usgs.hydrography](#demeter.raster.usgs.hydrography)
  * [USGSHydrographyRaster](#demeter.raster.usgs.hydrography.USGSHydrographyRaster)
  * [USGSHydrographyRasterOnDisk](#demeter.raster.usgs.hydrography.USGSHydrographyRasterOnDisk)
  * [fetch\_and\_merge\_rasters](#demeter.raster.usgs.hydrography.fetch_and_merge_rasters)
  * [fetch\_rasters](#demeter.raster.usgs.hydrography.fetch_rasters)
  * [find\_hu4\_codes](#demeter.raster.usgs.hydrography.find_hu4_codes)
  * [download\_raster\_archives](#demeter.raster.usgs.hydrography.download_raster_archives)
  * [raster\_keys\_by\_hu4\_code](#demeter.raster.usgs.hydrography.raster_keys_by_hu4_code)
* [demeter.raster.sentinel2.ndvi](#demeter.raster.sentinel2.ndvi)
  * [fetch\_and\_build\_ndvi\_rasters](#demeter.raster.sentinel2.ndvi.fetch_and_build_ndvi_rasters)
  * [fetch\_and\_build\_ndvi\_rasters\_from\_keys](#demeter.raster.sentinel2.ndvi.fetch_and_build_ndvi_rasters_from_keys)
  * [build\_ndvi\_rasters\_for\_crs](#demeter.raster.sentinel2.ndvi.build_ndvi_rasters_for_crs)
  * [build\_ndvi\_raster\_for\_datatake](#demeter.raster.sentinel2.ndvi.build_ndvi_raster_for_datatake)
  * [build\_and\_save\_ndvi\_raster\_for\_datatake](#demeter.raster.sentinel2.ndvi.build_and_save_ndvi_raster_for_datatake)
  * [merge\_and\_crop\_rasters](#demeter.raster.sentinel2.ndvi.merge_and_crop_rasters)
  * [extract\_surface\_reflectance](#demeter.raster.sentinel2.ndvi.extract_surface_reflectance)
* [demeter.raster.utils.mask](#demeter.raster.utils.mask)
  * [mask](#demeter.raster.utils.mask.mask)
* [demeter.raster.utils.merge](#demeter.raster.utils.merge)
  * [merge](#demeter.raster.utils.merge.merge)
  * [merge\_variance](#demeter.raster.utils.merge.merge_variance)
  * [merge\_stddev](#demeter.raster.utils.merge.merge_stddev)
  * [check\_for\_overlapping\_pixels](#demeter.raster.utils.merge.check_for_overlapping_pixels)
* [demeter.raster.utils.reprojection](#demeter.raster.utils.reprojection)
  * [reproject](#demeter.raster.utils.reprojection.reproject)
  * [align](#demeter.raster.utils.reprojection.align)
  * [reproject\_and\_merge](#demeter.raster.utils.reprojection.reproject_and_merge)
  * [align\_and\_merge](#demeter.raster.utils.reprojection.align_and_merge)
* [demeter.vector.usda.ssurgo](#demeter.vector.usda.ssurgo)
  * [fetch\_primary\_soil\_components](#demeter.vector.usda.ssurgo.fetch_primary_soil_components)

<a id="demeter.api"></a>

# demeter.api

<a id="demeter.api.fetch_point_data"></a>

#### fetch\_point\_data

```python
def fetch_point_data(points: Union[str, geopandas.GeoSeries,
                                   geopandas.GeoDataFrame],
                     values_to_fetch: Collection[Literal[
                         "polaris_carbon_stock",
                         "sentinel2_ndvi",
                         "usgs_hydrography",
                         "usgs_topography",
                         "ssurgo_primary_component",
                     ]],
                     *,
                     start_depth: int = 0,
                     end_depth: Optional[int] = None,
                     year: Optional[int] = None,
                     month: Optional[int] = None) -> geopandas.GeoDataFrame
```

Fetch data from one or more sources for the given points.

`end_depth` (in cm) is required for POLARIS and SSURGO.

`year` and `month` are required for Sentinel-2 NDVI.

**Example**:

  
```python
point_data = fetch_point_data(
    "points.geojson",
    values_to_fetch=["polaris_carbon_stock", "sentinel2_ndvi"],
    end_depth=30,
    year=2024,
    month=9,
)
```

<a id="demeter.raster"></a>

# demeter.raster

<a id="demeter.raster.Raster"></a>

## Raster Objects

```python
@dataclass
class Raster()
```

Rasterio has a file-centric API. It has the concept of a MemoryFile for
in-memory processing, but it's a bit clunky. This is intended as a simpler
in-memory representation of raster data, with direct access to the raster
pixels as a numpy masked array.

<a id="demeter.raster.Raster.pixels"></a>

#### pixels

A 3-dimensional array: one 2D array per band.

<a id="demeter.raster.Raster.transform"></a>

#### transform

The transform from pixel coordinates to geographic coordinates in this
raster's CRS.

<a id="demeter.raster.Raster.crs"></a>

#### crs

This raster's CRS, as a string. For example: "EPSG:5070".

<a id="demeter.raster.Raster.from_file"></a>

#### from\_file

```python
@classmethod
def from_file(cls, path: str) -> "Raster"
```

Read the file at the given path into memory and return a Raster
instance.

<a id="demeter.raster.Raster.count"></a>

#### count

```python
@property
def count() -> int
```

Number of bands.

<a id="demeter.raster.Raster.shape"></a>

#### shape

```python
@property
def shape() -> tuple[int, int]
```

Height and width of the raster.

<a id="demeter.raster.Raster.bounds"></a>

#### bounds

```python
@property
def bounds() -> tuple[float, float, float, float]
```

The raster's bounds, in the raster's CRS.

<a id="demeter.raster.Raster.resolution"></a>

#### resolution

```python
@property
def resolution() -> tuple[float, float]
```

The raster's (x, y) resolution.

<a id="demeter.raster.Raster.dtype"></a>

#### dtype

```python
@property
def dtype()
```

The raster's data type.

<a id="demeter.raster.Raster.nodata"></a>

#### nodata

```python
@property
def nodata()
```

The raster's nodata value.

<a id="demeter.raster.Raster.value_at"></a>

#### value\_at

```python
def value_at(x: float, y: float)
```

Find the pixel corresponding to the given coordinates, and return its
value. Assumes a single-band raster.

<a id="demeter.raster.Raster.values_at"></a>

#### values\_at

```python
def values_at(x: float, y: float) -> list
```

Find all pixels at the given coordinates from all bands, and return
them as a list.

<a id="demeter.raster.Raster.save"></a>

#### save

```python
def save(path: str, **kwargs)
```

Save the raster to the given path.

<a id="demeter.raster.Raster.as_dataset"></a>

#### as\_dataset

```python
def as_dataset() -> rasterio.io.DatasetReader
```

Write this raster to a `MemoryFile`, then open it for reading.

<a id="demeter.raster.Raster.__post_init__"></a>

#### \_\_post\_init\_\_

```python
def __post_init__()
```

Runtime validation to ensure:

- All attributes are set, in case mypy doesn't catch a missing value.
- The `pixels` array is a 3-dimensional masked array.

<a id="demeter.raster.Raster.__iter__"></a>

#### \_\_iter\_\_

```python
def __iter__()
```

Raster used to be a NamedTuple. This provides backward-compatibility
for unpacking, as in:

```
pixels, transform, crs = raster
```

<a id="demeter.raster.polaris"></a>

# demeter.raster.polaris

Tools for fetching raster data from the POLARIS dataset:
http://hydrology.cee.duke.edu/POLARIS/PROPERTIES/v1.0/

**Example**:

  
```python
polaris_om = fetch_polaris_data_for_depth_range(
    "/path/to/geometries.geojson",
    soil_property="om",
    start_depth=0,
    end_depth=100,
)
mean_raster, transform, crs = polaris_om.mean
stddev_raster, _, _ = polaris_om.stddev
```

<a id="demeter.raster.polaris.estimate_carbon_stock"></a>

#### estimate\_carbon\_stock

```python
def estimate_carbon_stock(
        geometries: Union[str, geopandas.GeoSeries, geopandas.GeoDataFrame],
        *,
        start_depth: int = 0,
        end_depth: int,
        calculate_standard_deviation: bool = True) -> CombinedRasters
```

Convenience function for the common use case of fetching organic matter and
bulk density rasters from POLARIS, then combining them into a estimated
carbon stock raster.

<a id="demeter.raster.polaris.fetch_polaris_data_for_depth_range"></a>

#### fetch\_polaris\_data\_for\_depth\_range

```python
def fetch_polaris_data_for_depth_range(
        geometries: Union[str, geopandas.GeoSeries, geopandas.GeoDataFrame],
        soil_property: Union[str, SoilProperty],
        *,
        start_depth: int = 0,
        end_depth: int,
        calculate_standard_deviation: bool = True,
        additional_statistics: list[Statistic] = []) -> CombinedRasters
```

High-level interface to POLARIS.

Fetch all POLARIS tiles between the given `start_depth` and `end_depth`,
and return a depth-weighted average across the entire depth range.

If `calculate_standard_deviation` is True (default), also return a raster
showing the standard deviation at each pixel, inferred from the p5-p95
split (assuming normal distribution).

<a id="demeter.raster.polaris.fetch_polaris_data"></a>

#### fetch\_polaris\_data

```python
def fetch_polaris_data(geometries: Union[str, geopandas.GeoSeries,
                                         geopandas.GeoDataFrame],
                       soil_property: Union[str, SoilProperty],
                       statistic: Statistic, depth: Depth) -> Raster
```

Low-level interface to POLARIS.

Download raster images from POLARIS, merge them, and return a raster
containing the values from the merged images.

<a id="demeter.raster.slga"></a>

# demeter.raster.slga

Tools for fetching raster data from the Soil and Landscape Grid of Australia
(SLGA): https://esoil.io/TERNLandscapes/Public/Pages/SLGA/index.html

**Example**:

  
```python
slga_om = fetch_slga_data_for_depth_range(
    "/path/to/geometries.geojson",
    soil_property=SoilProperty.ORGANIC_CARBON,
    start_depth=0,
    end_depth=100,
)
mean_raster, transform, crs = slga_om.mean
stddev_raster, _, _ = slga_om.stddev
```

<a id="demeter.raster.slga.estimate_carbon_stock"></a>

#### estimate\_carbon\_stock

```python
def estimate_carbon_stock(
        geometries: Union[str, geopandas.GeoSeries, geopandas.GeoDataFrame],
        *,
        start_depth: int = 0,
        end_depth: int,
        calculate_standard_deviation: bool = True) -> CombinedRasters
```

Convenience function for the common use case of fetching organic matter and
bulk density rasters from SLGA, then combining them into a estimated carbon
stock raster.

<a id="demeter.raster.slga.fetch_slga_data_for_depth_range"></a>

#### fetch\_slga\_data\_for\_depth\_range

```python
def fetch_slga_data_for_depth_range(
        geometries: Union[str, geopandas.GeoSeries, geopandas.GeoDataFrame],
        soil_property: SoilProperty,
        *,
        start_depth: int = 0,
        end_depth: int,
        calculate_standard_deviation: bool = True) -> CombinedRasters
```

High-level interface to SLGA.

Fetch all SLGA pixels between the given `start_depth` and `end_depth`, and
return a depth-weighted average across the entire depth range.

If `calculate_standard_deviation` is True (default), also return a raster
showing the standard deviation at each pixel, inferred from the p5-p95
split (assuming normal distribution).

<a id="demeter.raster.usgs.topography"></a>

# demeter.raster.usgs.topography

Tools for fetching elevation data from USGS at 1/3 arc-second resolution:
https://data.usgs.gov/datacatalog/data/USGS:3a81321b-c153-416f-98b7-cc8e5f0e17c3

**Example**:

  
```python
raster, transform, crs = fetch_and_merge_rasters("path/to/boundaries.geojson")
```

<a id="demeter.raster.usgs.topography.fetch_and_merge_rasters"></a>

#### fetch\_and\_merge\_rasters

```python
def fetch_and_merge_rasters(
        geometries: Union[str, geopandas.GeoDataFrame, geopandas.GeoSeries],
        *,
        crop: bool = True,
        dst_path: Optional[str] = None) -> Union[str, Raster]
```

Fetch 1/3 arc-second resolution elevation data for the given geometries
from USGS. If the geometries span multiple 1 degree x 1 degree tiles, fetch
all the necessary tiles and stitch them together.

If `crop` is True (the default), crop the output raster to the given
geometries.

If `dst_path` is provided, save the output raster to the given path. Use
this for large geometries that don't fit in memory.

<a id="demeter.raster.usgs.topography.fetch_rasters"></a>

#### fetch\_rasters

```python
def fetch_rasters(
    geometries: Union[str, geopandas.GeoDataFrame, geopandas.GeoSeries]
) -> Iterable[str]
```

Fetch all the 1 degree x 1 degree tiles from USGS that overlap with the
given geometries. Yield the path to each downloaded raster.

<a id="demeter.raster.usgs.hydrography"></a>

# demeter.raster.usgs.hydrography

Tools for fetching hydrography data from USGS:
https://www.usgs.gov/national-hydrography/access-national-hydrography-products

**Example**:

  
```python
catchments = fetch_and_merge_rasters("cat.tif", "path/to/boundaries.geojson")
pixels, transform, crs = catchments.raster

# USGS hydrography zip archives include a DBF file with pixel counts, which we
# parse and return as a dict:
pixel_counts = catchments.counts
catchment_areas_in_m2 = {
    catchment_id: pixel_count * 100
    for catchment_id, pixel_count in pixel_counts.items()
}
```

<a id="demeter.raster.usgs.hydrography.USGSHydrographyRaster"></a>

## USGSHydrographyRaster Objects

```python
@dataclass
class USGSHydrographyRaster(USGSHydrographyRasterBase)
```

A `Raster` instance containing the USGS hydrography raster, and a dict
mapping of pixel counts from the sidecar DBF file.

<a id="demeter.raster.usgs.hydrography.USGSHydrographyRasterOnDisk"></a>

## USGSHydrographyRasterOnDisk Objects

```python
@dataclass
class USGSHydrographyRasterOnDisk(USGSHydrographyRasterBase)
```

Same as `USGSHydrographyRaster`, but the raster is stored on disk.

<a id="demeter.raster.usgs.hydrography.fetch_and_merge_rasters"></a>

#### fetch\_and\_merge\_rasters

```python
def fetch_and_merge_rasters(
    raster_filename: Union[str, list[str]],
    geometries: Union[str, geopandas.GeoDataFrame, geopandas.GeoSeries],
    *,
    crop: bool = True,
    dst_path: Optional[str] = None
) -> Union[USGSHydrographyRasterBase, Iterable[USGSHydrographyRasterBase]]
```

Fetch the given raster (e.g. "cat.tif") from USGS for the given geometries.
If the geometries span multiple HU4 regions, fetch all the necessary
rasters and stitch them together.

Pass multiple raster filenames as a list to fetch multiple rasters for the
same geometries. For example:

```python
cat, fdr = fetch_and_merge_rasters(["cat.tif", "fdr.tif"], "path/to/boundaries.geojson")
```

If `crop` is True (the default), crop the output raster to the given
geometries.

If `dst_path` is provided, save the output raster to the given path. Use
this for large geometries that don't fit in memory.

<a id="demeter.raster.usgs.hydrography.fetch_rasters"></a>

#### fetch\_rasters

```python
def fetch_rasters(
    geometries: Union[str, geopandas.GeoDataFrame, geopandas.GeoSeries]
) -> Iterable[str]
```

Fetch all the raster archives that intersect with the given geometries.
Yield the local path to each downloaded zip archive.

<a id="demeter.raster.usgs.hydrography.find_hu4_codes"></a>

#### find\_hu4\_codes

```python
def find_hu4_codes(
    geometries: Union[str, geopandas.GeoDataFrame, geopandas.GeoSeries]
) -> Sequence[str]
```

Return the HU4 codes for the regions that intersect with the given
geometries.

USGS rasters are organized by 4-digit Hydrologic Unit (HU4). To know which
rasters to download, we need to identify which HU4 regions the input geometry
intersects with.

<a id="demeter.raster.usgs.hydrography.download_raster_archives"></a>

#### download\_raster\_archives

```python
def download_raster_archives(hu4_codes: Iterable[str]) -> Iterable[str]
```

Download the raster .zip files for the given HU4 codes.

<a id="demeter.raster.usgs.hydrography.raster_keys_by_hu4_code"></a>

#### raster\_keys\_by\_hu4\_code

```python
def raster_keys_by_hu4_code() -> dict[str, str]
```

Return a dict mapping HU4 codes to raster keys in S3.

<a id="demeter.raster.sentinel2.ndvi"></a>

# demeter.raster.sentinel2.ndvi

Tools for fetching Sentinel-2 rasters in the red and NIR bands, and using them
to calculate Normalized Difference Vegetation Index (NDVI) rasters.

**Example**:

  
```python
os.environ["COPERNICUS_AWS_ENDPOINT_URL"] = "https://eodata.dataspace.copernicus.eu/"
os.environ["COPERNICUS_AWS_ACCESS_KEY_ID"] = ...
os.environ["COPERNICUS_AWS_SECRET_ACCESS_KEY"] = ...

rasters = fetch_and_build_ndvi_rasters(
    "path/to/boundaries.geojson",
    year=2024,
    month=9,
    statistics=["mean", "min", "max", "stddev"],
)
```
  
  Sentinel-2 rasters are projected using the Universal Transverse Mercator (UTM)
  system. If the input geometries span multiple UTM zones, this function will
  return a separate raster for each zone. You can use them separately, or project
  to a common CRS if necessary. Note that projecting rasters involves resampling,
  which is lossy.

<a id="demeter.raster.sentinel2.ndvi.fetch_and_build_ndvi_rasters"></a>

#### fetch\_and\_build\_ndvi\_rasters

```python
def fetch_and_build_ndvi_rasters(
    geometries: Union[str, geopandas.GeoDataFrame, geopandas.GeoSeries],
    year: int,
    month: int,
    statistics: Optional[Collection[Literal["mean", "min", "max",
                                            "stddev"]]] = None,
    *,
    crop: bool = True,
    dst_path: Optional[str] = None
) -> Iterable[Union[NDVIRasters, NDVIRastersOnDisk]]
```

Download red and NIR reflectance rasters from Sentinel-2 for the given
geometries over the given month, use them to calculate NDVI, and merge the
NDVI rasters together per the requested statistic.

If `crop` is True (the default), crop the output raster to the given
geometries. If `crop` if False, the output raster will cover the extent of
the Sentinel-2 rasters intersecting with the given geometries.

If `dst_path` is given, NDVI rasters will be saved to that directory. Use
this for large geometries that don't fit in memory.

<a id="demeter.raster.sentinel2.ndvi.fetch_and_build_ndvi_rasters_from_keys"></a>

#### fetch\_and\_build\_ndvi\_rasters\_from\_keys

```python
def fetch_and_build_ndvi_rasters_from_keys(
    raster_keys: Iterable[str],
    statistics: Optional[Collection[Literal["mean", "min", "max",
                                            "stddev"]]] = None,
    crop_to: Optional[Union[geopandas.GeoDataFrame,
                            geopandas.GeoSeries]] = None,
    dst_path: Optional[str] = None
) -> Iterable[Union[NDVIRasters, NDVIRastersOnDisk]]
```

Download the given rasters, use them to calculate NDVI, and merge the NDVI
rasters together per the requested statistics.

The given raster keys should be for red, NIR, and SCL bands. Red and NIR are
needed to calculate NDVI, and SCL is used to mask out clouds.

<a id="demeter.raster.sentinel2.ndvi.build_ndvi_rasters_for_crs"></a>

#### build\_ndvi\_rasters\_for\_crs

```python
def build_ndvi_rasters_for_crs(
        crs: str,
        raster_paths: Iterable[str],
        statistics: Optional[Collection[Literal["mean", "min", "max",
                                                "stddev"]]] = None,
        crop_to: Optional[Union[geopandas.GeoDataFrame,
                                geopandas.GeoSeries]] = None,
        dst_path: Optional[str] = None
) -> Union[NDVIRasters, NDVIRastersOnDisk]
```

Build NDVI rasters for each datatake, then merge them together according to
the requested statistic.

Input rasters should all be in the given CRS, and sorted by datatake so we
can process them lazily. The output raster will also in the same CRS as the
input rasters.

<a id="demeter.raster.sentinel2.ndvi.build_ndvi_raster_for_datatake"></a>

#### build\_ndvi\_raster\_for\_datatake

```python
def build_ndvi_raster_for_datatake(
    datatake_timestamp: str,
    raster_paths: Iterable[str],
    crop_to: Optional[Union[geopandas.GeoDataFrame,
                            geopandas.GeoSeries]] = None
) -> Raster
```

From the given rasters from the same datatake, calculate and return an NDVI
raster.

<a id="demeter.raster.sentinel2.ndvi.build_and_save_ndvi_raster_for_datatake"></a>

#### build\_and\_save\_ndvi\_raster\_for\_datatake

```python
def build_and_save_ndvi_raster_for_datatake(output_directory: str,
                                            datatake_timestamp: str, *args,
                                            **kwargs) -> str
```

Same as `build_ndvi_raster_for_datatake`, but also save the NDVI raster to
a file in the given directory and return its path.

<a id="demeter.raster.sentinel2.ndvi.merge_and_crop_rasters"></a>

#### merge\_and\_crop\_rasters

```python
def merge_and_crop_rasters(
    raster_paths: Sequence[str],
    crop_to: Optional[Union[geopandas.GeoDataFrame,
                            geopandas.GeoSeries]] = None
) -> Raster
```

Merge rasters, optionally cropping to the given geometries.

Note that Sentinel-2 tiles are 100km x 100km, but the rasters are buffered
to 110km x 110km, so there is some overlap between rasters at tile
boundaries. Values from adjacent tiles should be identical within this
boundary region - if not, log a warning.

<a id="demeter.raster.sentinel2.ndvi.extract_surface_reflectance"></a>

#### extract\_surface\_reflectance

```python
def extract_surface_reflectance(pixels: numpy.ndarray) -> numpy.ma.MaskedArray
```

Sentinel-2 surface reflectance values are given in the 1-10000 range, with
0 as the nodata value. Scale reflectance to the 0-1 range, and add a nodata
mask.

<a id="demeter.raster.utils.mask"></a>

# demeter.raster.utils.mask

<a id="demeter.raster.utils.mask.mask"></a>

#### mask

```python
def mask(raster,
         shapes,
         *,
         crop: bool = False,
         dst_path: Optional[str] = None,
         **kwargs) -> Union[str, Raster]
```

Wraps `rasterio.mask.mask`, with the following differences:

- Can accept a `Raster` instance as well as a rasterio dataset.
- Returns a `Raster` instance instead of a (raster, transform) 2-tuple.
- Alternatively, writes the masked raster to disk if `dst_path` is given.
  Useful for large rasters that don't fit in memory.

<a id="demeter.raster.utils.merge"></a>

# demeter.raster.utils.merge

<a id="demeter.raster.utils.merge.merge"></a>

#### merge

```python
def merge(rasters,
          *,
          method: Union[Literal["first", "last", "min", "max", "sum", "count",
                                "mean"], Callable] = "first",
          bounds: Optional[tuple[float, float, float, float]] = None,
          allow_resampling: bool = True,
          dst_path: Optional[str] = None,
          **kwargs) -> Union[str, Raster]
```

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

<a id="demeter.raster.utils.merge.merge_variance"></a>

#### merge\_variance

```python
def merge_variance(rasters: Sequence,
                   mean: Union[str, Raster],
                   dst_path: Optional[str] = None,
                   **kwargs) -> Union[str, Raster]
```

Calculate the mean variance of rasters from the given mean.

<a id="demeter.raster.utils.merge.merge_stddev"></a>

#### merge\_stddev

```python
def merge_stddev(rasters: Sequence,
                 mean: Union[str, Raster],
                 dst_path: Optional[str] = None,
                 **kwargs) -> Union[str, Raster]
```

Calculate the mean standard deviation of rasters from the given mean.

<a id="demeter.raster.utils.merge.check_for_overlapping_pixels"></a>

#### check\_for\_overlapping\_pixels

```python
def check_for_overlapping_pixels(merged_data, new_data, merged_mask, new_mask,
                                 **kwargs)
```

When passed as the `method` argument to `rasterio.merge.merge`, this
function checks whether any two rasters have data for the same pixel.
If they do, it logs a warning.

<a id="demeter.raster.utils.reprojection"></a>

# demeter.raster.utils.reprojection

<a id="demeter.raster.utils.reprojection.reproject"></a>

#### reproject

```python
def reproject(raster: Union[str, Raster],
              crs: str,
              resampling_method: Literal[
                  "nearest",
                  "bilinear",
                  "cubic",
                  "cubic_spline",
                  "lanczos",
                  "average",
                  "mode",
                  "gauss",
                  "max",
                  "min",
                  "med",
                  "q1",
                  "q3",
                  "sum",
                  "rms",
              ],
              align_to_transform: Optional[rasterio.Affine] = None) -> Raster
```

Reproject a raster to a different coordinate reference system.

This is a lossy operation as it involves resampling. The resampling
algorithm to use depends on the nature of the data: `nearest` is a good
choice for categorical data, whereas `bilinear` or `average` might make
more sense for continuous data. See
https://rasterio.readthedocs.io/en/stable/api/rasterio.enums.html#rasterio.enums.Resampling
for the list of available resampling algorithms.

If `align_to_transform` is provided, align the output raster to the given
transform's pixel grid. This will perform up/downsampling if the given
transform's resolution doesn't match the input raster's resolution.

<a id="demeter.raster.utils.reprojection.align"></a>

#### align

```python
def align(
    raster: Union[str, Raster], to: Union[str,
                                          Raster], resampling_method: Literal[
                                              "nearest",
                                              "bilinear",
                                              "cubic",
                                              "cubic_spline",
                                              "lanczos",
                                              "average",
                                              "mode",
                                              "gauss",
                                              "max",
                                              "min",
                                              "med",
                                              "q1",
                                              "q3",
                                              "sum",
                                              "rms",
                                          ]
) -> Raster
```

Align a raster to another raster's grid.

<a id="demeter.raster.utils.reprojection.reproject_and_merge"></a>

#### reproject\_and\_merge

```python
def reproject_and_merge(rasters: Iterable[Union[str, Raster]],
                        crs: str,
                        resampling_method: Literal[
                            "nearest",
                            "bilinear",
                            "cubic",
                            "cubic_spline",
                            "lanczos",
                            "average",
                            "mode",
                            "gauss",
                            "max",
                            "min",
                            "med",
                            "q1",
                            "q3",
                            "sum",
                            "rms",
                        ],
                        merge_method: Union[Literal["first", "last", "min",
                                                    "max", "sum", "count",
                                                    "mean"],
                                            Callable] = "first",
                        align_to_transform: Optional[rasterio.Affine] = None,
                        **kwargs) -> Raster
```

Reproject multiple rasters to a common CRS, then merge them.

The `merge_method` argument specifies how to handle overlapping
pixels. Other arguments are passed to `merge`. Example:

```python
merged = reproject_and_merge(
    rasters,
    crs="EPSG:4326",
    resampling_method="average",  # how to resample when reprojecting
    merge_method="mean",          # how to merge overlapping pixels
)
```

<a id="demeter.raster.utils.reprojection.align_and_merge"></a>

#### align\_and\_merge

```python
def align_and_merge(rasters: Iterable[Union[str, Raster]],
                    to: Union[str, Raster],
                    resampling_method: Literal[
                        "nearest",
                        "bilinear",
                        "cubic",
                        "cubic_spline",
                        "lanczos",
                        "average",
                        "mode",
                        "gauss",
                        "max",
                        "min",
                        "med",
                        "q1",
                        "q3",
                        "sum",
                        "rms",
                    ],
                    merge_method: Union[Literal["first", "last", "min", "max",
                                                "sum", "count", "mean"],
                                        Callable] = "first",
                    **kwargs) -> Raster
```

Align multiple rasters to the given raster's grid, then merge them.

Keyword arguments are passed to `merge`.

<a id="demeter.vector.usda.ssurgo"></a>

# demeter.vector.usda.ssurgo

Tools for fetching Soil Survey (SSURGO) data from USDA:
https://www.nrcs.usda.gov/resources/data-and-reports/soil-survey-geographic-database-ssurgo

<a id="demeter.vector.usda.ssurgo.fetch_primary_soil_components"></a>

#### fetch\_primary\_soil\_components

```python
def fetch_primary_soil_components(geometries: Union[str,
                                                    geopandas.GeoDataFrame,
                                                    geopandas.GeoSeries],
                                  *,
                                  top_depth_cm: int = 0,
                                  bottom_depth_cm: int,
                                  crop: bool = True) -> geopandas.GeoDataFrame
```

Fetch all SSURGO map units that intersect with the given geometries. Return
a GeoDataFrame with the primary component of each map unit, along with a
depth-weighted average of that component's soil properties over the given
depth range.

**Example**:

  
```python
fetch_primary_soil_components("path/to/geometries.geojson", bottom_depth_cm=100)
```

