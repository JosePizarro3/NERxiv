"""Integration test for ensemble CLI functionality."""

import tempfile
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from nerxiv.cli.cli import cli
from tests.conftest import hdf5_test_file


def test_prompt_ensemble_cli_integration():
    """Test the prompt_ensemble CLI command with mocked LLM."""
    runner = CliRunner()

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a test HDF5 file
        test_file = hdf5_test_file(
            tmpdir,
            text="This is a test paper about density functional theory calculations using PBE functional.",
        )

        # Mock the LLM to avoid actual API calls
        with patch("nerxiv.ensemble.LLMGenerator") as mock_gen_cls:
            mock_gen = MagicMock()
            call_count = [0]

            def side_effect_generate(prompt):
                call_count[0] += 1
                # First N calls are ensemble runs
                if call_count[0] <= 3:
                    return '{"DFT": {"xc_functional": "PBE"}}'
                # Last call is averaging
                else:
                    return '{"DFT": {"xc_functional": "PBE"}}'

            mock_gen.generate.side_effect = side_effect_generate
            mock_gen_cls.return_value = mock_gen

            # Mock the retriever as well
            with patch("nerxiv.cli.run_prompt.CustomRetriever") as mock_retriever_cls:
                mock_retriever = MagicMock()
                mock_retriever.get_relevant_chunks.return_value = (
                    "Test retrieved text about DFT."
                )
                mock_retriever_cls.return_value = mock_retriever

                # Run the CLI command
                result = runner.invoke(
                    cli,
                    [
                        "prompt_ensemble",
                        "--file-path",
                        str(test_file),
                        "--query",
                        "dft",
                        "--n-ensemble-runs",
                        "3",
                        "--no-parallel",  # Sequential for predictable testing
                        "--ensemble-chunk-size",
                        "2000",
                    ],
                )

                # Check that the command succeeded
                assert result.exit_code == 0
                assert "Processed arXiv paper" in result.output
                assert "with ensemble" in result.output


def test_prompt_ensemble_cli_help():
    """Test that the prompt_ensemble command help works."""
    runner = CliRunner()
    result = runner.invoke(cli, ["prompt_ensemble", "--help"])

    assert result.exit_code == 0
    assert "ensemble prompting" in result.output
    assert "--n-ensemble-runs" in result.output
    assert "--ensemble-model" in result.output
    assert "--averaging-temperature" in result.output
