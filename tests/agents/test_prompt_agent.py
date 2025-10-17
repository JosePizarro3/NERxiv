"""Tests for the PromptAgent class."""

import pytest

from nerxiv.agents.prompt_agent import PromptAgent


class FakeChunker:
    """Minimal chunker for testing."""
    
    def __init__(self, text: str = "", **kwargs):
        if not text:
            raise ValueError("text required")
        self.text = text
    
    def chunk_text(self):
        """Split by sentences."""
        sentences = [s.strip() for s in self.text.split(".") if s.strip()]
        # Return fake Document objects
        return [type("Doc", (), {"page_content": s})() for s in sentences]


class FakeRetriever:
    """Minimal retriever for testing."""
    
    def __init__(self, query: str = "", **kwargs):
        if not query:
            raise ValueError("query required")
        self.query = query
    
    def get_relevant_chunks(self, chunks, n_top_chunks=5):
        """Return first n chunks."""
        items = [c.page_content for c in chunks][:n_top_chunks]
        return "\n\n".join(items)


class FakeGenerator:
    """Minimal generator for testing."""
    
    def __init__(self, text: str = "", **kwargs):
        if not text:
            raise ValueError("text required")
        self.text = text
    
    def generate(self, prompt: str = "", **kwargs):
        """Return mock JSON response for material_formula_structured."""
        # Simulate a structured response
        return """```json
{
    "ChemicalFormulation": {
        "iupac": "SrVO3"
    }
}
```"""


def test_prompt_agent_initialization():
    """Test that PromptAgent initializes correctly with valid query."""
    # This will fail without proper dependencies, so skip if imports fail
    try:
        agent = PromptAgent(
            query_name="material_formula",
            chunker="Chunker",
            retriever_model="all-MiniLM-L6-v2",
            llm_model="test-model",
        )
        assert agent.query_name == "material_formula"
        assert agent.retriever_model == "all-MiniLM-L6-v2"
        assert agent.llm_model == "test-model"
    except (ImportError, ModuleNotFoundError):
        pytest.skip("Optional dependencies not available")


def test_prompt_agent_invalid_query():
    """Test that PromptAgent raises error for invalid query name."""
    try:
        with pytest.raises(ValueError, match="Unknown query"):
            PromptAgent(
                query_name="nonexistent_query",
                chunker="Chunker",
            )
    except (ImportError, ModuleNotFoundError):
        pytest.skip("Optional dependencies not available")


def test_prompt_agent_invalid_chunker():
    """Test that PromptAgent raises error for invalid chunker."""
    try:
        with pytest.raises(KeyError, match="Unknown chunker"):
            PromptAgent(
                query_name="material_formula",
                chunker="NonexistentChunker",
            )
    except (ImportError, ModuleNotFoundError):
        pytest.skip("Optional dependencies not available")


def test_prompt_agent_empty_text():
    """Test that run() raises error for empty text."""
    try:
        agent = PromptAgent(
            query_name="material_formula",
            chunker="Chunker",
        )
        with pytest.raises(ValueError, match="Input text cannot be empty"):
            agent.run(text="")
    except (ImportError, ModuleNotFoundError):
        pytest.skip("Optional dependencies not available")


def test_prompt_agent_parse_json_with_code_block():
    """Test JSON parsing from markdown code blocks."""
    try:
        agent = PromptAgent(
            query_name="material_formula_structured",
            chunker="Chunker",
        )
        
        answer = """```json
{
    "iupac": "Fe2O3"
}
```"""
        
        parsed, error = agent._parse_and_validate(answer)
        
        # Should successfully parse
        assert error is None
        assert parsed is not None
        assert parsed.get("iupac") == "Fe2O3"
        
    except (ImportError, ModuleNotFoundError):
        pytest.skip("Optional dependencies not available")


def test_prompt_agent_parse_json_without_code_block():
    """Test JSON parsing without markdown code blocks."""
    try:
        agent = PromptAgent(
            query_name="material_formula_structured",
            chunker="Chunker",
        )
        
        answer = '{"iupac": "SrVO3"}'
        
        parsed, error = agent._parse_and_validate(answer)
        
        # Should successfully parse
        assert error is None
        assert parsed is not None
        assert parsed.get("iupac") == "SrVO3"
        
    except (ImportError, ModuleNotFoundError):
        pytest.skip("Optional dependencies not available")


def test_prompt_agent_parse_invalid_json():
    """Test JSON parsing with invalid JSON."""
    try:
        agent = PromptAgent(
            query_name="material_formula_structured",
            chunker="Chunker",
        )
        
        answer = "This is not JSON at all"
        
        parsed, error = agent._parse_and_validate(answer)
        
        # Should fail to parse
        assert parsed is None
        assert error is not None
        assert "No JSON found" in error
        
    except (ImportError, ModuleNotFoundError):
        pytest.skip("Optional dependencies not available")


def test_prompt_agent_parse_list_output():
    """Test JSON parsing with list of items."""
    try:
        agent = PromptAgent(
            query_name="material_formula_structured",
            chunker="Chunker",
        )
        
        answer = """```json
[
    {"iupac": "Fe2O3"},
    {"iupac": "Fe2O3.25"}
]
```"""
        
        parsed, error = agent._parse_and_validate(answer)
        
        # Should successfully parse list
        assert error is None
        assert parsed is not None
        assert isinstance(parsed, list)
        assert len(parsed) == 2
        assert parsed[0].get("iupac") == "Fe2O3"
        assert parsed[1].get("iupac") == "Fe2O3.25"
        
    except (ImportError, ModuleNotFoundError):
        pytest.skip("Optional dependencies not available")
