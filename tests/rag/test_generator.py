from unittest.mock import MagicMock, patch

from nerxiv.rag import LLMGenerator


def test_llm_generator_generate_mocked():
    """Tests the `_check_tokens_limit` and `generate` methods of the `LLMGenerator` class."""
    # Mock OllamaLLM + AutoTokenizer
    with patch("nerxiv.rag.generator.OllamaLLM") as mock_llm_cls:
        # --- Mock the LLM ---
        mock_llm = MagicMock()
        mock_llm.model = "deepseek-r1"
        mock_llm.invoke.return_value = "Mocked response"
        mock_llm_cls.return_value = mock_llm

        # Generates a mocked prompt and answer from the LLM
        generator = LLMGenerator(model="deepseek-r1", text="mock input")
        prompt = "Extract all computational methods."
        assert generator.generate(prompt=prompt) == "Mocked response"
