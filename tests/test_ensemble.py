"""Tests for the ensemble module."""

import json
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel, Field

from nerxiv.ensemble import (
    average_json_results,
    extract_json_from_text,
    run_ensemble_prompts,
    run_single_llm_prompt,
)
from nerxiv.prompts.prompts import Prompt, StructuredPrompt


def test_extract_json_from_text():
    """Test JSON extraction from various text formats."""
    # Test with markdown code block
    text1 = 'Some text\n```json\n{"key": "value"}\n```\nMore text'
    result1 = extract_json_from_text(text1)
    assert result1 == {"key": "value"}

    # Test with raw JSON
    text2 = 'Prefix {"key": "value"} suffix'
    result2 = extract_json_from_text(text2)
    assert result2 == {"key": "value"}

    # Test with pure JSON
    text3 = '{"key": "value"}'
    result3 = extract_json_from_text(text3)
    assert result3 == {"key": "value"}

    # Test with invalid JSON
    text4 = "No JSON here"
    result4 = extract_json_from_text(text4)
    assert result4 is None


def test_run_single_llm_prompt_mocked():
    """Test running a single LLM prompt with mocking."""
    with patch("nerxiv.ensemble.LLMGenerator") as mock_gen_cls:
        # Mock generator
        mock_gen = MagicMock()
        mock_gen.generate.return_value = '{"result": "test"}'
        mock_gen_cls.return_value = mock_gen

        # Create a simple prompt
        prompt = Prompt(
            expert="Test Expert",
            main_instruction="extract information",
        )

        # Run single prompt
        run_id, answer = run_single_llm_prompt(
            prompt=prompt,
            text="test text",
            model="test-model",
            temperature=0.5,
            run_id=1,
        )

        assert run_id == 1
        assert answer == '{"result": "test"}'
        mock_gen_cls.assert_called_once()
        mock_gen.generate.assert_called_once()


def test_average_json_results_single():
    """Test averaging with a single result."""
    results = [{"key": "value"}]
    averaged = average_json_results(results)
    assert averaged == {"key": "value"}


def test_average_json_results_empty():
    """Test averaging with empty results."""
    results = []
    averaged = average_json_results(results)
    assert averaged == {}


def test_average_json_results_multiple_mocked():
    """Test averaging multiple JSON results with mocked LLM."""
    with patch("nerxiv.ensemble.LLMGenerator") as mock_gen_cls:
        # Mock generator for averaging
        mock_gen = MagicMock()
        mock_gen.generate.return_value = '{"key": "consensus_value"}'
        mock_gen_cls.return_value = mock_gen

        results = [
            {"key": "value1"},
            {"key": "value2"},
            {"key": "value1"},
        ]

        averaged = average_json_results(results)
        assert averaged == {"key": "consensus_value"}
        mock_gen_cls.assert_called_once()


def test_run_ensemble_prompts_basic_mocked():
    """Test basic ensemble prompting with mocking."""
    with patch("nerxiv.ensemble.LLMGenerator") as mock_gen_cls:
        # Mock generator to return different results
        mock_gen = MagicMock()
        call_count = [0]

        def side_effect_generate(prompt):
            call_count[0] += 1
            return f"Answer {call_count[0]}"

        mock_gen.generate.side_effect = side_effect_generate
        mock_gen_cls.return_value = mock_gen

        # Create a simple prompt
        prompt = Prompt(
            expert="Test Expert",
            main_instruction="extract information",
        )

        # Run ensemble with 3 runs
        combined, averaged_json = run_ensemble_prompts(
            prompt=prompt,
            text="test text",
            n_runs=3,
            models=["model1"],
            temperatures=[0.2, 0.5],
            parallel=False,  # Sequential for predictable testing
        )

        # Check that combined answer contains all runs
        assert "Run 1:" in combined
        assert "Run 2:" in combined
        assert "Run 3:" in combined

        # For non-StructuredPrompt, averaged_json should be None
        assert averaged_json is None


def test_run_ensemble_prompts_structured_mocked():
    """Test ensemble prompting with StructuredPrompt and mocking."""

    class TestSchema(BaseModel):
        value: str = Field(description="A test value")

    with patch("nerxiv.ensemble.LLMGenerator") as mock_gen_cls:
        # Mock generator to return JSON results
        mock_gen = MagicMock()
        call_count = [0]

        def side_effect_generate(prompt):
            call_count[0] += 1
            if call_count[0] <= 3:  # First 3 calls are ensemble runs
                return f'{{"value": "test{call_count[0]}"}}'
            else:  # 4th call is averaging
                return '{"value": "consensus"}'

        mock_gen.generate.side_effect = side_effect_generate
        mock_gen_cls.return_value = mock_gen

        # Create a StructuredPrompt
        prompt = StructuredPrompt(
            expert="Test Expert",
            output_schema=TestSchema,
            target_fields=["value"],
        )

        # Run ensemble with 3 runs
        combined, averaged_json = run_ensemble_prompts(
            prompt=prompt,
            text="test text",
            n_runs=3,
            models=["model1"],
            temperatures=[0.5],
            parallel=False,  # Sequential for predictable testing
        )

        # Check that we got an averaged JSON result
        assert averaged_json is not None
        assert averaged_json == {"value": "consensus"}
        assert "Run 1:" in combined
