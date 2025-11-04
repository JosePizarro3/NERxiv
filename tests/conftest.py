import os
from pathlib import Path

import h5py
import pytest
from langchain_core.documents import Document

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


def hdf5_test_file(tmp_path: str, text: str = "Some scientific content here.") -> Path:
    """Creates a minimal HDF5 file with expected structure."""
    file_path = Path(tmp_path) / "1234.5678.hdf5"
    with h5py.File(file_path, "w") as f:
        paper_id = "1234.5678"
        grp = f.create_group(f"{paper_id}/arxiv_paper")
        grp.create_dataset("text", data=text.encode("utf-8"))
    return file_path


# Mock classes for testing
class SimpleChunker:
    def __init__(self, text: str = "", **kwargs):
        if not text:
            raise ValueError("text required")
        self.text = text

    def chunk_text(self):
        # naive split by sentences
        return [
            Document(page_content=s.strip(), metadata={"source": "test"})
            for s in self.text.split(".")
            if s.strip()
        ]


class SimpleRetriever:
    def __init__(self, query: str = "", **kwargs):
        if not query:
            raise ValueError("query required")
        self.query = query

    def get_relevant_chunks(self, chunks):
        # return first n chunks joined
        items = [c.page_content for c in chunks][:2]
        return "\n\n".join(items)


class SimpleGenerator:
    def __init__(self, text: str = "", **kwargs):
        if not text:
            raise ValueError("text required")
        self.text = text

    def generate(self, prompt: str = ""):
        return f"GENERATED ANSWER based on: {self.text[:30]}..."
