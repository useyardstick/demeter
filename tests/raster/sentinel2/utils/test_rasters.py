from demeter.raster.sentinel2.utils.rasters import SafeMetadata


def test_safe_metadata():
    metadata = SafeMetadata.from_filename(
        "Sentinel-2/MSI/L2A/2024/12/16/S2C_MSIL2A_20241216T184831_N9905_R070_T10SFG_20241216T221754.SAFE"
    )
    assert metadata.tile_id == "10SFG"
    assert metadata.datatake_timestamp == "20241216T184831"
    assert metadata.utm_zone == "10"
    assert metadata.crs == "EPSG:32610"
