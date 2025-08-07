import pytest

from nerxiv.utils import answer_to_dict


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
