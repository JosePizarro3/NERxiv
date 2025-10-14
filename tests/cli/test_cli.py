from nerxiv.cli.cli import parse_llm_option_to_args


def test_parse_llm_option_to_args():
    assert parse_llm_option_to_args(
        (
            "temperature=0.2",
            "mirostat=None",
            "format=''",
            "base_url=https://api.openai.com/v1",
        )
    ) == {
        "temperature": 0.2,
        "mirostat": None,
        "format": "",
        "base_url": "https://api.openai.com/v1",
    }
