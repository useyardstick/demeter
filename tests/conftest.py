import os
import re
from glob import glob
from tempfile import TemporaryDirectory

import boto3
import pytest
from moto import mock_aws

from demeter.raster import sentinel2, usgs


@pytest.fixture
def s3():
    with mock_aws():
        yield boto3.client("s3")


@pytest.fixture(autouse=True)
def copernicus_s3_credentials(monkeypatch):
    if sentinel2.ndvi._SAVE_TEST_FIXTURES:
        if not os.environ.get("COPERNICUS_AWS_ENDPOINT_URL"):
            raise Exception(
                "Set COPERNICUS_AWS_ENDPOINT_URL and credentials to save test fixtures"
            )
    else:
        monkeypatch.setenv("COPERNICUS_AWS_ACCESS_KEY_ID", "key")
        monkeypatch.setenv("COPERNICUS_AWS_SECRET_ACCESS_KEY", "secret")


@pytest.fixture(scope="session")
def copernicus_s3():
    if sentinel2.ndvi._SAVE_TEST_FIXTURES:
        yield None
    else:
        # Mock Copernicus S3, but allow requests to the USGS bucket in AWS.
        # TODO: Mock AWS S3 as well.
        with mock_aws(
            config={
                "core": {
                    "mock_credentials": False,
                    "passthrough": {
                        "urls": [
                            re.escape(
                                f"https://{usgs.constants.S3_BUCKET_NAME}.s3.amazonaws.com"
                            )
                        ],
                    },
                    "reset_boto3_session": True,
                    "service_whitelist": ["s3"],
                },
            }
        ):
            s3 = boto3.client("s3", endpoint_url="https://eodata.mock-copernicus.eu")
            s3.create_bucket(Bucket=sentinel2.constants.S3_BUCKET_NAME)
            yield s3


@pytest.fixture(scope="session")
def sentinel2_rasters_in_s3(copernicus_s3):
    """
    These are real rasters, cropped to cover only the input geometries (plus a
    small buffer) to keep file size small.

    To regenerate these fixtures, set `_SAVE_TEST_FIXTURES` to True and rerun
    the tests.
    """
    if sentinel2.ndvi._SAVE_TEST_FIXTURES:
        return

    fixtures_dir = "tests/raster/fixtures/sentinel2/eodata/"

    for path in glob(os.path.join(fixtures_dir, "**/*"), recursive=True):
        _, ext = os.path.splitext(path)
        if ext not in {".safe", ".jp2"}:
            continue

        key = os.path.relpath(path, fixtures_dir)
        copernicus_s3.upload_file(path, sentinel2.constants.S3_BUCKET_NAME, key)


@pytest.fixture(autouse=True)
def sentinel2_cache_directory(monkeypatch):
    if sentinel2.ndvi._SAVE_TEST_FIXTURES:
        yield ".sentinel2_cache"
    else:
        with TemporaryDirectory() as tmpdir:
            monkeypatch.setenv("SENTINEL2_CACHED_RASTER_FILES_DIRECTORY", tmpdir)
            yield tmpdir


@pytest.fixture(autouse=True)
def polaris_cache_directory(monkeypatch):
    with TemporaryDirectory() as tmpdir:
        monkeypatch.setenv("POLARIS_CACHED_RASTER_FILES_DIRECTORY", tmpdir)
        yield tmpdir
