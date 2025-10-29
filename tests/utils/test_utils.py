from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from nerxiv.utils import (
    answer_to_dict,
    clean_description,
    files_to_subfolder_answer,
    filter_material_formula_predicate,
    filter_only_dmft_predicate,
)


@pytest.mark.parametrize(
    "answer, result",
    [
        ("invalid JSON string", []),
        (
            '[{"key": "value"}, {"key2": "value2"}]',
            [{"key": "value"}, {"key2": "value2"}],
        ),
    ],
)
def test_answer_to_dict(answer: str, result: list[dict]):
    """Tests the `answer_to_dict` function."""
    assert answer_to_dict(answer) == result


def test_clean_description():
    """Tests the `clean_description` function."""
    description = "\n  This is a   test\n   description.  "
    cleaned_description = clean_description(description)
    assert cleaned_description == "This is a test description."


@pytest.mark.parametrize(
    "answer, expected",
    [
        ("model", True),
        ("something_else", False),
    ],
)
def test_material_formula_predicate(answer, expected):
    """Tests the `filter_material_formula_predicate` function."""
    assert filter_material_formula_predicate(answer) == expected


@pytest.mark.parametrize(
    "answer, expected",
    [
        ("True", False),
        ("False", True),
        ("model", True),
    ],
)
def test_only_dmft_predicate(answer, expected):
    """Tests the `filter_only_dmft_predicate` function."""
    assert filter_only_dmft_predicate(answer) == expected


def test_files_to_subfolder_answer(tmp_path):
    """Tests `files_to_subfolder_answer` logic without real file moves."""
    # create fake file
    fake_file = tmp_path / "test.hdf5"
    fake_file.touch()

    # fake HDF5 structure
    mock_run_group = {
        "subfolder": {"answer": MagicMock(**{"__getitem__.return_value": b"True"})}
    }
    mock_h5_file = MagicMock()
    mock_h5_file.__enter__.return_value = {
        "raw_llm_answers": {"run_0000": mock_run_group}
    }
    mock_h5_file.__exit__.return_value = None

    with (
        patch("nerxiv.utils.utils.Path.rglob", return_value=[fake_file]),
        patch("nerxiv.utils.utils.h5py.File", return_value=mock_h5_file),
        patch.object(Path, "mkdir") as mock_mkdir,
        patch.object(Path, "rename") as mock_rename,
    ):
        files_to_subfolder_answer(path=tmp_path, run="run_0000")

        mock_mkdir.assert_called_once()
        mock_rename.assert_called_once()
