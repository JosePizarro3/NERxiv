import os

import pytest

from ragxiv.logger import log_storage

if os.getenv("_PYTEST_RAISE", "0") != "0":

    @pytest.hookimpl(tryfirst=True)
    def pytest_exception_interact(call):
        raise call.excinfo.value

    @pytest.hookimpl(tryfirst=True)
    def pytest_internalerror(excinfo):
        raise excinfo.value


@pytest.fixture(autouse=True)
def cleared_log_storage():
    """Fixture to clear the log storage before each test."""
    log_storage.clear()
    yield log_storage
