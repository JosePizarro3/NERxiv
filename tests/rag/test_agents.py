from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import h5py
import pytest
from langchain_core.documents import Document

from nerxiv.datamodel.crystal_structure import ChemicalFormulation
from nerxiv.prompts.prompts import Prompt, StructuredPrompt
from nerxiv.rag import RAGExtractorAgent
from tests.conftest import SimpleChunker, SimpleGenerator, SimpleRetriever


class TestRAGExtractorAgent:
    """Test suite for RAGExtractorAgent class."""

    def test_init(self):
        """Test RAGExtractorAgent initialization."""
        agent = RAGExtractorAgent(
            chunker=SimpleChunker,
            retriever=SimpleRetriever,
            generator=SimpleGenerator,
            chunker_params={"param1": "value1"},
            retriever_params={"query": "test query"},
            generator_params={"model": "test-model"},
        )
        assert agent.chunker == SimpleChunker
        assert agent.retriever == SimpleRetriever
        assert agent.generator == SimpleGenerator
        assert agent.chunker_params == {"param1": "value1"}
        assert agent.retriever_params == {"query": "test query"}
        assert agent.generator_params == {"model": "test-model"}

    def test_obj_name_with_class(self):
        """Test _obj_name method with a class."""
        agent = RAGExtractorAgent(
            chunker=SimpleChunker,
            retriever=SimpleRetriever,
            generator=SimpleGenerator,
        )
        assert agent._obj_name(SimpleChunker) == "SimpleChunker"

    def test_obj_name_with_instance(self):
        """Test _obj_name method with an instance."""
        agent = RAGExtractorAgent(
            chunker=SimpleChunker,
            retriever=SimpleRetriever,
            generator=SimpleGenerator,
        )
        instance = SimpleChunker(text="test")
        assert agent._obj_name(instance) == "SimpleChunker"

    def test_instantiate_with_class(self):
        """Test _instantiate method with a class."""
        agent = RAGExtractorAgent(
            chunker=SimpleChunker,
            retriever=SimpleRetriever,
            generator=SimpleGenerator,
        )
        instance = agent._instantiate(SimpleChunker, {"text": "test text"})
        assert isinstance(instance, SimpleChunker)
        assert instance.text == "test text"

    def test_instantiate_with_instance(self):
        """Test _instantiate method with an already instantiated object."""
        agent = RAGExtractorAgent(
            chunker=SimpleChunker,
            retriever=SimpleRetriever,
            generator=SimpleGenerator,
        )
        pre_instance = SimpleChunker(text="test")
        result = agent._instantiate(pre_instance, {})
        assert result is pre_instance

    def test_parse_json_with_code_block(self):
        """Test parse method with JSON in markdown code block."""
        agent = RAGExtractorAgent(
            chunker=SimpleChunker,
            retriever=SimpleRetriever,
            generator=SimpleGenerator,
        )
        answer = '```json\n{"key": "value", "number": 42}\n```'
        result = agent.parse(answer)
        assert result == {"key": "value", "number": 42}

    def test_parse_json_without_code_block(self):
        """Test parse method with JSON without code block."""
        agent = RAGExtractorAgent(
            chunker=SimpleChunker,
            retriever=SimpleRetriever,
            generator=SimpleGenerator,
        )
        answer = '{"key": "value", "number": 42}'
        result = agent.parse(answer)
        assert result == {"key": "value", "number": 42}

    def test_parse_json_with_array(self):
        """Test parse method with JSON array."""
        agent = RAGExtractorAgent(
            chunker=SimpleChunker,
            retriever=SimpleRetriever,
            generator=SimpleGenerator,
        )
        answer = '[{"key": "value"}, {"key": "value2"}]'
        result = agent.parse(answer)
        assert result == [{"key": "value"}, {"key": "value2"}]

    def test_parse_invalid_json(self):
        """Test parse method with invalid JSON."""
        agent = RAGExtractorAgent(
            chunker=SimpleChunker,
            retriever=SimpleRetriever,
            generator=SimpleGenerator,
        )
        answer = "This is not JSON"
        result = agent.parse(answer)
        assert result is None

    def test_run_without_file(self):
        """Test run method without file parameter."""
        agent = RAGExtractorAgent(
            chunker=SimpleChunker,
            retriever=SimpleRetriever,
            generator=SimpleGenerator,
        )
        prompt = Prompt(expert="test", main_instruction="test instruction")
        result = agent.run(file=None, text="test", prompt=prompt)
        assert result is None

    def test_run_without_text(self):
        """Test run method without text parameter."""
        agent = RAGExtractorAgent(
            chunker=SimpleChunker,
            retriever=SimpleRetriever,
            generator=SimpleGenerator,
        )
        prompt = Prompt(expert="test", main_instruction="test instruction")
        with h5py.File("/tmp/test_no_text.hdf5", "w") as f:
            result = agent.run(file=f, text="", prompt=prompt)
            assert result is None

    def test_run_without_prompt(self):
        """Test run method without prompt parameter."""
        agent = RAGExtractorAgent(
            chunker=SimpleChunker,
            retriever=SimpleRetriever,
            generator=SimpleGenerator,
        )
        with h5py.File("/tmp/test_no_prompt.hdf5", "w") as f:
            result = agent.run(file=f, text="test", prompt=None)
            assert result is None

    def test_run_without_query_in_retriever_params(self):
        """Test run method without query in retriever_params."""
        agent = RAGExtractorAgent(
            chunker=SimpleChunker,
            retriever=SimpleRetriever,
            generator=SimpleGenerator,
            retriever_params={},  # No query
        )
        prompt = Prompt(expert="test", main_instruction="test instruction")
        with h5py.File("/tmp/test_no_query.hdf5", "w") as f:
            result = agent.run(file=f, text="test", prompt=prompt)
            assert result is None

    def test_run_basic_flow(self, tmp_path):
        """Test run method with basic flow."""
        agent = RAGExtractorAgent(
            chunker=SimpleChunker,
            retriever=SimpleRetriever,
            generator=SimpleGenerator,
            retriever_params={"query": "test query", "query_name": "test_query"},
        )
        prompt = Prompt(expert="test", main_instruction="extract information")
        text = "This is sentence one. This is sentence two. This is sentence three."

        file_path = tmp_path / "test_basic.hdf5"
        with h5py.File(file_path, "w") as f:
            agent.run(file=f, text=text, prompt=prompt)

            # Verify HDF5 structure was created
            assert "rag_extraction" in f
            assert "chunks_cache" in f["rag_extraction"]
            assert "retrieval_cache" in f["rag_extraction"]
            assert "raw_llm_answers" in f["rag_extraction"]
            assert "test_query" in f["rag_extraction/raw_llm_answers"]

    def test_run_with_caching(self, tmp_path):
        """Test run method with caching - second run should reuse cached data."""
        agent = RAGExtractorAgent(
            chunker=SimpleChunker,
            retriever=SimpleRetriever,
            generator=SimpleGenerator,
            retriever_params={"query": "test query", "query_name": "test_query"},
        )
        prompt = Prompt(expert="test", main_instruction="extract information")
        text = "This is sentence one. This is sentence two."

        file_path = tmp_path / "test_caching.hdf5"

        # First run - creates cache
        with h5py.File(file_path, "w") as f:
            agent.run(file=f, text=text, prompt=prompt)
            chunks_cache_keys_first = list(f["rag_extraction/chunks_cache"].keys())
            retrieval_cache_keys_first = list(
                f["rag_extraction/retrieval_cache"].keys()
            )

        # Second run - should reuse cache
        with h5py.File(file_path, "a") as f:
            agent.run(file=f, text=text, prompt=prompt)
            chunks_cache_keys_second = list(f["rag_extraction/chunks_cache"].keys())
            retrieval_cache_keys_second = list(
                f["rag_extraction/retrieval_cache"].keys()
            )

            # Cache keys should be the same (no new caches created)
            assert chunks_cache_keys_first == chunks_cache_keys_second
            assert retrieval_cache_keys_first == retrieval_cache_keys_second

            # But we should have two runs recorded
            assert len(f["rag_extraction/raw_llm_answers/test_query"].keys()) == 2

    def test_run_with_structured_prompt_valid(self, tmp_path):
        """Test run method with StructuredPrompt and valid JSON response."""

        # Mock the generator to return valid JSON
        class MockStructuredGenerator:
            def __init__(self, text: str = "", **kwargs):
                self.text = text

            def generate(self, prompt: str = ""):
                return """```json
{
  "ChemicalFormulation": {
    "iupac": "H2O",
    "reduced": "H2O"
  }
}
```"""

        agent = RAGExtractorAgent(
            chunker=SimpleChunker,
            retriever=SimpleRetriever,
            generator=MockStructuredGenerator,
            retriever_params={"query": "test query", "query_name": "test_query"},
        )
        prompt = StructuredPrompt(
            expert="test",
            output_schema=ChemicalFormulation,
            target_fields=["iupac", "reduced"],
        )
        text = "Water has the formula H2O."

        file_path = tmp_path / "test_structured.hdf5"
        with h5py.File(file_path, "w") as f:
            result = agent.run(file=f, text=text, prompt=prompt)
            # Structured prompt should not return result on success (returns None)
            assert result is None

    def test_run_with_structured_prompt_invalid_json(self, tmp_path):
        """Test run method with StructuredPrompt and invalid JSON response."""

        class MockBadGenerator:
            def __init__(self, text: str = "", **kwargs):
                self.text = text

            def generate(self, prompt: str = ""):
                return "This is not valid JSON"

        agent = RAGExtractorAgent(
            chunker=SimpleChunker,
            retriever=SimpleRetriever,
            generator=MockBadGenerator,
            retriever_params={"query": "test query", "query_name": "test_query"},
        )
        prompt = StructuredPrompt(
            expert="test",
            output_schema=ChemicalFormulation,
            target_fields=["iupac", "reduced"],
        )
        text = "Some text."

        file_path = tmp_path / "test_invalid_json.hdf5"
        with h5py.File(file_path, "w") as f:
            result = agent.run(file=f, text=text, prompt=prompt)
            assert result is None

    def test_get_chunks_new_chunking(self, tmp_path):
        """Test _get_chunks method with new chunking."""
        agent = RAGExtractorAgent(
            chunker=SimpleChunker,
            retriever=SimpleRetriever,
            generator=SimpleGenerator,
        )

        file_path = tmp_path / "test_get_chunks.hdf5"
        with h5py.File(file_path, "w") as f:
            cached_chunks_group = f.create_group("test_chunks")
            chunks = agent._get_chunks(
                chunker_hash="test_hash",
                text="Sentence one. Sentence two.",
                chunker_name="SimpleChunker",
                cached_chunks_group=cached_chunks_group,
                global_time=0.0,
            )

            assert len(chunks) == 2
            assert all(isinstance(c, Document) for c in chunks)
            assert "chunker" in cached_chunks_group.attrs
            assert "chunker_params" in cached_chunks_group.attrs

    def test_run_multiple_queries(self, tmp_path):
        """Test run method with multiple different queries."""
        agent1 = RAGExtractorAgent(
            chunker=SimpleChunker,
            retriever=SimpleRetriever,
            generator=SimpleGenerator,
            retriever_params={"query": "query 1", "query_name": "query_1"},
        )
        agent2 = RAGExtractorAgent(
            chunker=SimpleChunker,
            retriever=SimpleRetriever,
            generator=SimpleGenerator,
            retriever_params={"query": "query 2", "query_name": "query_2"},
        )
        prompt = Prompt(expert="test", main_instruction="extract information")
        text = "Test text."

        file_path = tmp_path / "test_multiple_queries.hdf5"

        # Run with first query
        with h5py.File(file_path, "w") as f:
            agent1.run(file=f, text=text, prompt=prompt)

        # Run with second query
        with h5py.File(file_path, "a") as f:
            agent2.run(file=f, text=text, prompt=prompt)
            # Should have two query groups
            assert "query_1" in f["rag_extraction/raw_llm_answers"]
            assert "query_2" in f["rag_extraction/raw_llm_answers"]
