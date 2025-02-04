import os
import re
import shutil
from glob import glob
from tempfile import TemporaryDirectory

import boto3
import pytest
import rasterio
import rasterio.mask
import responses
from moto import mock_aws
from moto.core.models import override_responses_real_send
from responses._recorder import Recorder

from demeter.raster import polaris, sentinel2, usgs


@pytest.fixture(scope="session")
def save_test_fixtures():
    return "SAVE_TEST_FIXTURES" in os.environ


@pytest.fixture
def recorder(save_test_fixtures):
    if save_test_fixtures:
        with Recorder() as recorder:
            yield recorder
    else:
        yield None


@pytest.fixture
def requests_mock(recorder):
    # The moto library for mocking AWS services uses `responses` under the
    # hood. Configure it to pass through requests to our `RequestsMock`:
    # https://github.com/getmoto/moto/blob/master/docs/docs/faq.rst#how-can-i-mock-my-own-http-requests-using-the-responses-module
    try:
        with responses.RequestsMock() as mock:
            override_responses_real_send(mock)

            # If we're recording responses, pass unmocked requests through to
            # the recorder:
            if recorder:
                recorder_send = recorder.unbound_on_send()
                real_send = mock._real_send

                # Don't record TIF files. They are huge, and we have a separate
                # mechanism for saving them as fixtures.
                def record_unless_tiff(adapter, request, **kwargs):
                    content_type = request.headers.get("Content-Type", "")
                    url = request.url
                    if content_type.startswith("image/tiff") or url.endswith(".tif"):
                        return real_send(adapter, request, **kwargs)
                    return recorder_send(adapter, request, **kwargs)

                mock._real_send = record_unless_tiff
                mock.passthru_prefixes = ("http",)

            yield mock

    finally:
        override_responses_real_send(None)


@pytest.fixture
def recorded_responses_fixture_path(test_function_name):
    return os.path.join(
        "tests/fixtures/recorded_responses", f"{test_function_name}.yaml"
    )


@pytest.fixture
def record_or_replay_requests(recorder, requests_mock, recorded_responses_fixture_path):
    if recorder:
        yield
        recorder.dump_to_file(recorded_responses_fixture_path)
    else:
        requests_mock._add_from_file(file_path=recorded_responses_fixture_path)
        yield


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
def use_sentinel2_fixtures(
    save_test_fixtures,
    sentinel2_cache_directory,
    sentinel2_fixture_directory,
    copernicus_s3,
):
    """
    These are real rasters, cropped to cover only the input geometries (plus a
    small buffer) to keep file size small.

    To regenerate these fixtures, set the `SAVE_TEST_FIXTURES` environment
    variable and rerun the tests.
    """
    # If we're not saving new test fixtures, upload the existing ones to our
    # mock Copernicus S3:
    if not save_test_fixtures:
        for path in glob(
            os.path.join(sentinel2_fixture_directory, "**/*"), recursive=True
        ):
            _, ext = os.path.splitext(path)
            if ext not in {".safe", ".jp2"}:
                continue

            key = os.path.relpath(path, sentinel2_fixture_directory)
            copernicus_s3.upload_file(path, sentinel2.constants.S3_BUCKET_NAME, key)

    # Yield a function to specify the geometries to crop test fixtures to:
    geometries = None

    def _save_sentinel2_fixtures(crop_to):
        nonlocal geometries
        geometries = crop_to

    yield _save_sentinel2_fixtures

    # If we're saving new test fixtures, crop the downloaded rasters to the
    # test geometries and save them to the fixtures directory:
    if save_test_fixtures:
        safe_paths = glob(
            os.path.join(sentinel2_cache_directory, "**/*.safe"), recursive=True
        )
        raster_paths = glob(
            os.path.join(sentinel2_cache_directory, "**/*.jp2"), recursive=True
        )
        for safe_path in safe_paths:
            _copy_test_fixture(
                safe_path,
                sentinel2_cache_directory,
                sentinel2_fixture_directory,
            )
        for raster_path in raster_paths:
            _copy_test_fixture(
                raster_path,
                sentinel2_cache_directory,
                sentinel2_fixture_directory,
                crop_to=geometries,
            )


@pytest.fixture(autouse=True)
def sentinel2_cache_directory(save_test_fixtures, monkeypatch):
    with TemporaryDirectory() as tmpdir:
        monkeypatch.setenv("SENTINEL2_CACHED_RASTER_FILES_DIRECTORY", tmpdir)
        yield tmpdir


@pytest.fixture
def sentinel2_fixture_directory(test_function_name):
    return os.path.join("tests/raster/fixtures/sentinel2", test_function_name)


@pytest.fixture
def use_polaris_fixtures(
    save_test_fixtures,
    requests_mock,
    polaris_cache_directory,
    polaris_fixture_directory,
):
    polaris_url_pattern = re.compile(re.escape(polaris.BASE_URL))

    if save_test_fixtures:
        requests_mock.add_passthru(polaris_url_pattern)
    else:
        requests_mock.add_callback(
            "GET",
            polaris_url_pattern,
            callback=_mock_polaris_callback(polaris_fixture_directory),
        )

    # Yield a function to specify the geometries to crop test fixtures to:
    geometries = None

    def _save_polaris_fixtures(crop_to):
        nonlocal geometries
        geometries = crop_to

    yield _save_polaris_fixtures

    # If we're saving new test fixtures, crop the downloaded rasters to the
    # test geometries and save them to the fixtures directory:
    if save_test_fixtures:
        raster_paths = glob(
            os.path.join(polaris_cache_directory, "**/*.tif"), recursive=True
        )
        for raster_path in raster_paths:
            _copy_test_fixture(
                raster_path,
                polaris_cache_directory,
                polaris_fixture_directory,
                crop_to=geometries,
            )


def _mock_polaris_callback(polaris_fixture_directory):
    def _callback(request):
        request_path = request.url.removeprefix(polaris.BASE_URL)
        fixture_path = os.path.join(
            polaris_fixture_directory, request_path.removeprefix("/")
        )
        print(f"Returning test fixture: {fixture_path}")
        with open(fixture_path, "rb") as file:
            return 200, {"Content-Type": "image/tiff"}, file.read()

    return _callback


@pytest.fixture(autouse=True)
def polaris_cache_directory(monkeypatch):
    with TemporaryDirectory() as tmpdir:
        monkeypatch.setenv("POLARIS_CACHED_RASTER_FILES_DIRECTORY", tmpdir)
        yield tmpdir


@pytest.fixture
def polaris_fixture_directory(test_function_name):
    return os.path.join("tests/raster/fixtures/polaris", test_function_name)


@pytest.fixture
def test_function_name(request):
    return f"{request.module.__name__}.{request.function.__qualname__}"


def _copy_test_fixture(raster_path, cache_directory, fixture_directory, crop_to=None):
    """
    Copy downloaded rasters to the test fixtures directory.

    To keep file sizes small, crop rasters to the input geometries, plus a
    small buffer.
    """
    relative_path = os.path.relpath(raster_path, cache_directory)
    fixture_path = os.path.join(fixture_directory, relative_path)
    os.makedirs(os.path.dirname(fixture_path), exist_ok=True)

    if crop_to is None:
        shutil.copyfile(raster_path, fixture_path)
    else:
        with rasterio.open(raster_path) as src:
            profile = src.profile
            crs = src.crs

            # Add a small buffer to ensure we don't crop out any data:
            if crs.is_geographic:
                crop_to = crop_to.to_crs("EPSG:5070")
            else:
                assert crs.linear_units == "metre"
                crop_to = crop_to.to_crs(crs)
            buffered = crop_to.geometry.buffer(250).to_crs(crs)

            # Mask the raster to the buffered geometry:
            array, transform = rasterio.mask.mask(src, buffered, all_touched=True)

        assert transform == profile["transform"]

        # Save the masked raster as a test fixture:
        with rasterio.open(
            fixture_path,
            mode="w",
            quality=100,
            reversible="YES",
            **profile,
        ) as dst:
            dst.write(array)

    print(f"Saved test fixture to {fixture_path}")
