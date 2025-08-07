import os
from pathlib import Path

import h5py
import pytest

from nerxiv.logger import log_storage

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


def hdf5_test_file(tmp_path: str, text: str = "Some scientific content here.") -> str:
    """Creates a minimal HDF5 file with expected structure."""
    file_path = Path(tmp_path) / "1234.5678.hdf5"
    with h5py.File(file_path, "w") as f:
        paper_id = "1234.5678"
        grp = f.create_group(f"{paper_id}/arxiv_paper")
        grp.create_dataset("text", data=text.encode("utf-8"))
    return file_path
