import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import h5py
import pytest

from nerxiv.cli.run_prompt import run_prompt_paper


@pytest.fixture
def mock_hdf5_file(tmp_path):
    """Creates a test HDF5 file with expected structure."""
    file_path = tmp_path / "test_paper.hdf5"
    with h5py.File(file_path, "w") as f:
        paper_id = "test_paper"
        grp = f.create_group(f"{paper_id}/arxiv_paper")
        test_text = "This is a sample scientific text about quantum mechanics and materials science. It discusses various properties and applications."
        grp.create_dataset("text", data=test_text.encode("utf-8"))
    return file_path


@pytest.fixture
def mock_prompt():
    """Create a mock prompt object."""
    prompt = MagicMock()
    prompt.build.return_value = "Test prompt"
    return prompt


@pytest.fixture
def mock_retriever():
    """Create a mock retriever."""
    retriever = MagicMock()
    retriever.get_relevant_chunks.return_value = "Retrieved text content"
    return retriever


@pytest.fixture
def mock_generator():
    """Create a mock generator."""
    generator = MagicMock()
    generator.generate.return_value = "Generated answer"
    return generator


class TestRunPromptPaperChunkerHash:
    """Tests for chunker hash functionality in run_prompt_paper."""

    @patch("nerxiv.cli.run_prompt.LLMGenerator")
    @patch("nerxiv.cli.run_prompt.CustomRetriever")
    def test_chunker_hash_stored_in_hdf5(
        self, mock_retriever_cls, mock_generator_cls, mock_hdf5_file, mock_prompt
    ):
        """Test that chunker hash and retriever hash are stored in HDF5."""
        mock_retriever_cls.return_value = MagicMock(
            get_relevant_chunks=MagicMock(return_value="Retrieved text")
        )
        mock_generator_cls.return_value = MagicMock(
            generate=MagicMock(return_value="Generated answer")
        )

        # Run the function
        run_prompt_paper(
            paper=mock_hdf5_file,
            chunker="Chunker",
            retriever_model="test-model",
            n_top_chunks=5,
            model="test-llm",
            retriever_query="test query",
            prompt=mock_prompt,
            query="test_query",
        )

        # Check that hashes were stored
        with h5py.File(mock_hdf5_file, "r") as f:
            run_group = f["raw_llm_answers/test_query/run_0000"]
            assert "chunker_hash" in run_group.attrs
            assert len(run_group.attrs["chunker_hash"]) == 64  # SHA256 hex length
            assert "retriever_hash" in run_group.attrs
            assert len(run_group.attrs["retriever_hash"]) == 64  # SHA256 hex length

            # Check that caches were created
            assert "chunks_cache" in f
            assert len(list(f["chunks_cache"].keys())) == 1
            assert "retrieval_cache" in f
            assert len(list(f["retrieval_cache"].keys())) == 1
            
            # Check that chunks and top_k_chunks are NOT stored in run group anymore
            assert "chunks" not in run_group

    @patch("nerxiv.cli.run_prompt.LLMGenerator")
    @patch("nerxiv.cli.run_prompt.CustomRetriever")
    def test_chunks_reused_on_second_run(
        self, mock_retriever_cls, mock_generator_cls, mock_hdf5_file, mock_prompt
    ):
        """Test that chunks are reused when hash matches."""
        mock_retriever_cls.return_value = MagicMock(
            get_relevant_chunks=MagicMock(return_value="Retrieved text")
        )
        mock_generator_cls.return_value = MagicMock(
            generate=MagicMock(return_value="Generated answer")
        )

        # First run - creates chunks
        run_prompt_paper(
            paper=mock_hdf5_file,
            chunker="Chunker",
            retriever_model="test-model",
            n_top_chunks=5,
            model="test-llm",
            retriever_query="test query",
            prompt=mock_prompt,
            query="test_query",
        )

        # Second run - should reuse chunks
        run_prompt_paper(
            paper=mock_hdf5_file,
            chunker="Chunker",
            retriever_model="test-model",
            n_top_chunks=5,
            model="test-llm",
            retriever_query="test query",
            prompt=mock_prompt,
            query="test_query2",  # Different query but same chunking
        )

        # Verify that only one entry exists in chunks_cache (chunks were reused)
        with h5py.File(mock_hdf5_file, "r") as f:
            assert len(list(f["chunks_cache"].keys())) == 1

            # Verify both runs reference the same hash
            hash1 = f["raw_llm_answers/test_query/run_0000"].attrs["chunker_hash"]
            hash2 = f["raw_llm_answers/test_query2/run_0000"].attrs["chunker_hash"]
            assert hash1 == hash2

    @patch("nerxiv.cli.run_prompt.LLMGenerator")
    @patch("nerxiv.cli.run_prompt.CustomRetriever")
    def test_different_params_create_new_chunks(
        self, mock_retriever_cls, mock_generator_cls, mock_hdf5_file, mock_prompt
    ):
        """Test that different chunker parameters create new chunks."""
        mock_retriever_cls.return_value = MagicMock(
            get_relevant_chunks=MagicMock(return_value="Retrieved text")
        )
        mock_generator_cls.return_value = MagicMock(
            generate=MagicMock(return_value="Generated answer")
        )

        # First run with default params
        run_prompt_paper(
            paper=mock_hdf5_file,
            chunker="Chunker",
            retriever_model="test-model",
            n_top_chunks=5,
            model="test-llm",
            retriever_query="test query",
            prompt=mock_prompt,
            query="test_query",
        )

        # Second run with different chunk size
        run_prompt_paper(
            paper=mock_hdf5_file,
            chunker="Chunker",
            retriever_model="test-model",
            n_top_chunks=5,
            model="test-llm",
            retriever_query="test query",
            prompt=mock_prompt,
            query="test_query2",
            chunk_size=500,  # Different chunk size
        )

        # Verify that two entries exist in chunks_cache
        with h5py.File(mock_hdf5_file, "r") as f:
            assert len(list(f["chunks_cache"].keys())) == 2

            # Verify runs have different hashes
            hash1 = f["raw_llm_answers/test_query/run_0000"].attrs["chunker_hash"]
            hash2 = f["raw_llm_answers/test_query2/run_0000"].attrs["chunker_hash"]
            assert hash1 != hash2

    @patch("nerxiv.cli.run_prompt.LLMGenerator")
    @patch("nerxiv.cli.run_prompt.CustomRetriever")
    @patch("nerxiv.chunker.get_spacy_model")
    def test_different_chunker_creates_new_chunks(
        self, mock_spacy, mock_retriever_cls, mock_generator_cls, mock_hdf5_file, mock_prompt
    ):
        """Test that different chunker types create new chunks."""
        # Mock spacy for SemanticChunker
        mock_sent = MagicMock()
        mock_sent.text = "This is a sample scientific text."
        mock_doc = MagicMock()
        mock_doc.sents = [mock_sent]
        mock_nlp = MagicMock()
        mock_nlp.return_value = mock_doc
        mock_spacy.return_value = mock_nlp

        mock_retriever_cls.return_value = MagicMock(
            get_relevant_chunks=MagicMock(return_value="Retrieved text")
        )
        mock_generator_cls.return_value = MagicMock(
            generate=MagicMock(return_value="Generated answer")
        )

        # First run with Chunker
        run_prompt_paper(
            paper=mock_hdf5_file,
            chunker="Chunker",
            retriever_model="test-model",
            n_top_chunks=5,
            model="test-llm",
            retriever_query="test query",
            prompt=mock_prompt,
            query="test_query",
        )

        # Second run with SemanticChunker
        run_prompt_paper(
            paper=mock_hdf5_file,
            chunker="SemanticChunker",
            retriever_model="test-model",
            n_top_chunks=5,
            model="test-llm",
            retriever_query="test query",
            prompt=mock_prompt,
            query="test_query2",
        )

        # Verify that two entries exist in chunks_cache
        with h5py.File(mock_hdf5_file, "r") as f:
            assert len(list(f["chunks_cache"].keys())) == 2

            # Verify runs have different hashes
            hash1 = f["raw_llm_answers/test_query/run_0000"].attrs["chunker_hash"]
            hash2 = f["raw_llm_answers/test_query2/run_0000"].attrs["chunker_hash"]
            assert hash1 != hash2

    @patch("nerxiv.cli.run_prompt.LLMGenerator")
    @patch("nerxiv.cli.run_prompt.CustomRetriever")
    def test_chunks_cache_contains_metadata(
        self, mock_retriever_cls, mock_generator_cls, mock_hdf5_file, mock_prompt
    ):
        """Test that chunks_cache stores metadata correctly."""
        mock_retriever_cls.return_value = MagicMock(
            get_relevant_chunks=MagicMock(return_value="Retrieved text")
        )
        mock_generator_cls.return_value = MagicMock(
            generate=MagicMock(return_value="Generated answer")
        )

        # Run with specific parameters
        run_prompt_paper(
            paper=mock_hdf5_file,
            chunker="Chunker",
            retriever_model="test-model",
            n_top_chunks=5,
            model="test-llm",
            retriever_query="test query",
            prompt=mock_prompt,
            query="test_query",
            chunk_size=500,
            chunk_overlap=100,
        )

        # Verify metadata in chunks_cache
        with h5py.File(mock_hdf5_file, "r") as f:
            chunks_cache = f["chunks_cache"]
            cache_key = list(chunks_cache.keys())[0]
            cached_group = chunks_cache[cache_key]

            assert cached_group.attrs["chunker"] == "Chunker"
            assert "n_chunks" in cached_group.attrs
            assert cached_group.attrs["chunker_hash"] == cache_key
            
            # Check that params are stored as JSON
            params = json.loads(cached_group.attrs["chunker_params"])
            assert params["chunk_size"] == 500
            assert params["chunk_overlap"] == 100

    @patch("nerxiv.cli.run_prompt.LLMGenerator")
    @patch("nerxiv.cli.run_prompt.CustomRetriever")
    def test_retriever_cache_reuse(
        self, mock_retriever_cls, mock_generator_cls, mock_hdf5_file, mock_prompt
    ):
        """Test that retrieval results are reused when retriever config matches."""
        mock_retriever_cls.return_value = MagicMock(
            get_relevant_chunks=MagicMock(return_value="Retrieved text")
        )
        mock_generator_cls.return_value = MagicMock(
            generate=MagicMock(return_value="Generated answer")
        )

        # First run - creates chunks and performs retrieval
        run_prompt_paper(
            paper=mock_hdf5_file,
            chunker="Chunker",
            retriever_model="test-model",
            n_top_chunks=5,
            model="test-llm",
            retriever_query="test query",
            prompt=mock_prompt,
            query="test_query",
        )

        # Second run - same chunker and retriever params, should reuse both caches
        run_prompt_paper(
            paper=mock_hdf5_file,
            chunker="Chunker",
            retriever_model="test-model",
            n_top_chunks=5,
            model="test-llm-2",  # Different model, but same retriever
            retriever_query="test query",
            prompt=mock_prompt,
            query="test_query2",
        )

        # Verify that only one entry exists in both caches (both were reused)
        with h5py.File(mock_hdf5_file, "r") as f:
            assert len(list(f["chunks_cache"].keys())) == 1
            assert len(list(f["retrieval_cache"].keys())) == 1

            # Verify both runs reference the same hashes
            hash1_chunker = f["raw_llm_answers/test_query/run_0000"].attrs["chunker_hash"]
            hash2_chunker = f["raw_llm_answers/test_query2/run_0000"].attrs["chunker_hash"]
            assert hash1_chunker == hash2_chunker

            hash1_retriever = f["raw_llm_answers/test_query/run_0000"].attrs["retriever_hash"]
            hash2_retriever = f["raw_llm_answers/test_query2/run_0000"].attrs["retriever_hash"]
            assert hash1_retriever == hash2_retriever

    @patch("nerxiv.cli.run_prompt.LLMGenerator")
    @patch("nerxiv.cli.run_prompt.CustomRetriever")
    def test_different_retriever_params_create_new_retrieval(
        self, mock_retriever_cls, mock_generator_cls, mock_hdf5_file, mock_prompt
    ):
        """Test that different retriever parameters create new retrieval results."""
        mock_retriever_cls.return_value = MagicMock(
            get_relevant_chunks=MagicMock(return_value="Retrieved text")
        )
        mock_generator_cls.return_value = MagicMock(
            generate=MagicMock(return_value="Generated answer")
        )

        # First run with n_top_chunks=5
        run_prompt_paper(
            paper=mock_hdf5_file,
            chunker="Chunker",
            retriever_model="test-model",
            n_top_chunks=5,
            model="test-llm",
            retriever_query="test query",
            prompt=mock_prompt,
            query="test_query",
        )

        # Second run with n_top_chunks=10 (different retriever param)
        run_prompt_paper(
            paper=mock_hdf5_file,
            chunker="Chunker",
            retriever_model="test-model",
            n_top_chunks=10,  # Different
            model="test-llm",
            retriever_query="test query",
            prompt=mock_prompt,
            query="test_query2",
        )

        # Verify that one chunker cache but two retrieval caches exist
        with h5py.File(mock_hdf5_file, "r") as f:
            assert len(list(f["chunks_cache"].keys())) == 1  # Same chunks
            assert len(list(f["retrieval_cache"].keys())) == 2  # Different retrieval

            # Verify runs have same chunker hash but different retriever hash
            hash1_chunker = f["raw_llm_answers/test_query/run_0000"].attrs["chunker_hash"]
            hash2_chunker = f["raw_llm_answers/test_query2/run_0000"].attrs["chunker_hash"]
            assert hash1_chunker == hash2_chunker

            hash1_retriever = f["raw_llm_answers/test_query/run_0000"].attrs["retriever_hash"]
            hash2_retriever = f["raw_llm_answers/test_query2/run_0000"].attrs["retriever_hash"]
            assert hash1_retriever != hash2_retriever
            assert params["chunk_overlap"] == 100
