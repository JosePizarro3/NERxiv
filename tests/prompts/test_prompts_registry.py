import pytest

from nerxiv.prompts import PROMPT_REGISTRY


def test_PROMPT_REGISTRY():
    registered_queries = set(PROMPT_REGISTRY.keys())
    expected_queries = {"material_formula"}
    if registered_queries != expected_queries:
        pytest.fail(
            f"`PROMPT_REGISTRY` query keys have changed.\n"
            f"Registered: {registered_queries}\n"
            f"Expected: {expected_queries}\n"
            ">>> Update this test if you have added or removed entries from the registry."
        )
