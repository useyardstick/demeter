import os
from functools import wraps
from tempfile import TemporaryDirectory

import pytest
import responses

from demeter.raster.sentinel2.ndvi import _SAVE_TEST_FIXTURES

SEARCH_RESPONSES_FIXTURE_DIR = "tests/raster/fixtures/sentinel2/search_responses/"


def record_or_replay_sentinel2_search_responses(fixture_name):
    """
    If `_SAVE_TEST_FIXTURES` is True, records responses from the Sentinel-2
    OData search API.
    """
    if _SAVE_TEST_FIXTURES:
        from responses import _recorder

        def decorator(test_fn):
            fixture_path = os.path.join(
                SEARCH_RESPONSES_FIXTURE_DIR, f"{fixture_name}.yaml"
            )
            return _recorder.record(file_path=fixture_path)(test_fn)

        return decorator

    return replay_sentinel2_search_responses(fixture_name)


def replay_sentinel2_search_responses(fixture_name):
    """
    Replay Sentinel-2 OData API search responses, to avoid hitting the real API
    in tests.
    """

    def decorator(test_fn):
        fixture_path = os.path.join(
            SEARCH_RESPONSES_FIXTURE_DIR, f"{fixture_name}.yaml"
        )

        def wrapper(fn):
            @wraps(fn)
            @responses.activate
            def wrapped(*args, **kwargs):
                responses._add_from_file(file_path=fixture_path)
                return test_fn(*args, **kwargs)

            return wrapped

        return wrapper(test_fn)

    return decorator


@pytest.fixture(autouse=True)
def cache_directory(monkeypatch):
    if _SAVE_TEST_FIXTURES:
        yield ".sentinel2_cache"
    else:
        with TemporaryDirectory() as tmpdir:
            monkeypatch.setenv("SENTINEL2_CACHED_RASTER_FILES_DIRECTORY", tmpdir)
            yield tmpdir
