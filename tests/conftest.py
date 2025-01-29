import os
import re
from glob import glob
from tempfile import TemporaryDirectory

import boto3
import pytest
import responses
from moto import mock_aws
from moto.core.models import override_responses_real_send
from responses import _recorder

from demeter.raster import sentinel2, usgs


@pytest.fixture(scope="session")
def save_test_fixtures():
    return "SAVE_TEST_FIXTURES" in os.environ


@pytest.fixture
def recorder(save_test_fixtures):
    if save_test_fixtures:
        with _recorder.Recorder() as recorder:
            yield recorder
    else:
        yield None


@pytest.fixture
def requests_mock(recorder):
    # https://github.com/getmoto/moto/blob/master/docs/docs/faq.rst#how-can-i-mock-my-own-http-requests-using-the-responses-module
    try:
        with responses.RequestsMock() as mock:
            override_responses_real_send(mock)
            if recorder:
                # Allow unmocked requests to pass through to the recorder:
                mock._real_send = recorder.unbound_on_send()
                mock.passthru_prefixes = ("http",)
            yield mock
    finally:
        override_responses_real_send(None)


@pytest.fixture
def record_or_replay_requests(recorder, replay_requests):
    if recorder:
        # Yield a function to specify the fixture path. Once the test finishes,
        # dump the recorded responses to that path.
        path = None

        def record(fixture_path):
            nonlocal path
            path = fixture_path

        yield record
        assert path
        recorder.dump_to_file(path)
    else:
        yield replay_requests


@pytest.fixture
def replay_requests(requests_mock):
    def replay(fixture_path):
        requests_mock._add_from_file(file_path=fixture_path)

    return replay


@pytest.fixture
def s3(requests_mock):
    with mock_aws():
        yield boto3.client("s3")


@pytest.fixture(autouse=True)
def copernicus_s3_credentials(save_test_fixtures, monkeypatch):
    if save_test_fixtures:
        if not os.environ.get("COPERNICUS_AWS_ENDPOINT_URL"):
            raise Exception(
                "Set COPERNICUS_AWS_ENDPOINT_URL and credentials to save test fixtures"
            )
    else:
        monkeypatch.setenv(
            "COPERNICUS_AWS_ENDPOINT_URL", "https://eodata.mock-copernicus.eu/"
        )
        monkeypatch.setenv("COPERNICUS_AWS_ACCESS_KEY_ID", "key")
        monkeypatch.setenv("COPERNICUS_AWS_SECRET_ACCESS_KEY", "secret")


@pytest.fixture
def copernicus_s3(save_test_fixtures, requests_mock):
    if save_test_fixtures:
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


@pytest.fixture
def sentinel2_rasters_in_s3(save_test_fixtures, copernicus_s3):
    """
    These are real rasters, cropped to cover only the input geometries (plus a
    small buffer) to keep file size small.

    To regenerate these fixtures, set the `SAVE_TEST_FIXTURES` environment
    variable and rerun the tests.
    """
    if save_test_fixtures:
        return

    fixtures_dir = "tests/raster/fixtures/sentinel2/eodata/"

    for path in glob(os.path.join(fixtures_dir, "**/*"), recursive=True):
        _, ext = os.path.splitext(path)
        if ext not in {".safe", ".jp2"}:
            continue

        key = os.path.relpath(path, fixtures_dir)
        copernicus_s3.upload_file(path, sentinel2.constants.S3_BUCKET_NAME, key)


@pytest.fixture(autouse=True)
def sentinel2_cache_directory(save_test_fixtures, monkeypatch):
    if save_test_fixtures:
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
