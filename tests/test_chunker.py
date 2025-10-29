from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from nerxiv.chunker import (
    AdvancedSemanticChunker,
    Chunker,
    SemanticChunker,
    compute_chunker_hash,
    compute_retriever_hash,
)


class TestChunker:
    def test_chunker_raises_without_text(self):
        """Tests that the `Chunker` raises a ValueError when initialized without text."""
        with pytest.raises(ValueError, match="`text` is required for chunking."):
            Chunker()

    @pytest.mark.parametrize(
        "text, chunk_size, chunk_overlap, result",
        [
            (
                "We perform first-principles calculations using Density Functional Theory to investigate "
                "the electronic structure of the layered compound. The exchange-correlation functional is "
                "treated within the Generalized Gradient Approximation.",
                50,
                1,
                [
                    "We perform first-principles calculations using",
                    "Density Functional Theory to investigate the",
                    "electronic structure of the layered compound. The",
                    "exchange-correlation functional is treated within",
                    "the Generalized Gradient Approximation.",
                ],
            ),
            (
                "We perform first-principles calculations using Density Functional Theory to investigate "
                "the electronic structure of the layered compound. The exchange-correlation functional is "
                "treated within the Generalized Gradient Approximation.",
                100,
                1,
                [
                    "We perform first-principles calculations using Density Functional Theory to investigate the",
                    "electronic structure of the layered compound. The exchange-correlation functional is treated within",
                    "the Generalized Gradient Approximation.",
                ],
            ),
            (
                "We perform first-principles calculations using Density Functional Theory to investigate "
                "the electronic structure of the layered compound. The exchange-correlation functional is "
                "treated within the Generalized Gradient Approximation.",
                50,
                10,
                [
                    "We perform first-principles calculations using",
                    "using Density Functional Theory to investigate",
                    "the electronic structure of the layered compound.",
                    "compound. The exchange-correlation functional is",
                    "is treated within the Generalized Gradient",
                    "Gradient Approximation.",
                ],
            ),
        ],
    )
    def test_chunk_text(
        self, text: str, chunk_size: int, chunk_overlap: int, result: list[str] | None
    ):
        """Tests the `chunk_text` method of the `Chunker` class."""
        chunks = Chunker(text=text).chunk_text(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        assert len(chunks) == len(result)
        for i, chunk in enumerate(chunks):
            assert chunk.page_content == result[i]


class TestSemanticChunker:
    @patch("nerxiv.chunker.get_spacy_model")
    def test_chunk_text(self, mock_get_spacy):
        # Create sentence mocks
        mock_sent1 = MagicMock()
        mock_sent1.text = "Sentence one."
        mock_sent2 = MagicMock()
        mock_sent2.text = "Sentence two."

        # Create a mock NLP doc that has .sents
        mock_doc = MagicMock()
        mock_doc.sents = [mock_sent1, mock_sent2]

        # Make the NLP model callable, returning the mock doc
        mock_nlp_instance = MagicMock()
        mock_nlp_instance.return_value = mock_doc
        mock_get_spacy.return_value = mock_nlp_instance

        text = "Dummy text."
        chunker = SemanticChunker(text=text)
        chunks = chunker.chunk_text()

        assert isinstance(chunks, list)
        assert all(isinstance(c, Document) for c in chunks)
        assert chunks[0].page_content == "Sentence one."
        assert chunks[1].page_content == "Sentence two."
        assert chunks[0].metadata["source"] == "nerxiv.chunker.SemanticChunker"


class TestAdvancedSemanticChunker:
    @patch("nerxiv.chunker.get_sentence_model")
    @patch("nerxiv.chunker.get_spacy_model")
    def test_chunk_text(self, mock_get_spacy, mock_get_sentence_model):
        # Mock spacy model
        mock_sent1 = MagicMock()
        mock_sent1.text = "Sentence one."
        mock_sent2 = MagicMock()
        mock_sent2.text = "Sentence two."

        mock_doc = MagicMock()
        mock_doc.sents = [mock_sent1, mock_sent2]

        mock_nlp_instance = MagicMock()
        mock_nlp_instance.return_value = mock_doc  # NLP(text) returns doc
        mock_get_spacy.return_value = mock_nlp_instance

        # Mock sentence transformer model
        mock_model = MagicMock()
        # One embedding per sentence
        mock_model.encode.return_value = [[0.1, 0.2], [0.3, 0.4]]
        mock_get_sentence_model.return_value = mock_model

        text = "Sentence one. Sentence two."
        chunker = AdvancedSemanticChunker(text=text)
        chunks = chunker.chunk_text(n_chunks=2)

        assert isinstance(chunks, list)
        assert len(chunks) > 0
        assert all(isinstance(c, Document) for c in chunks)
        for c in chunks:
            assert c.metadata["source"] == "nerxiv.chunker.AdvancedSemanticChunker"
        # Ensure SentenceTransformer.encode was called
        mock_model.encode.assert_called_once()

    @patch("nerxiv.chunker.get_sentence_model")
    @patch("nerxiv.chunker.get_spacy_model")
    def test_chunk_text_fewer_sentences_than_n_clusters(
        self, mock_get_spacy, mock_get_sentence_model
    ):
        # Mock spacy model
        mock_sent1 = MagicMock()
        mock_sent1.text = "Only one sentence."

        mock_doc = MagicMock()
        mock_doc.sents = [mock_sent1]

        mock_nlp_instance = MagicMock()
        mock_nlp_instance.return_value = mock_doc  # NLP(text) returns doc
        mock_get_spacy.return_value = mock_nlp_instance

        # Mock sentence transformer model
        mock_model = MagicMock()
        # One embedding per sentence
        mock_model.encode.return_value = [[0.1, 0.2]]
        mock_get_sentence_model.return_value = mock_model

        text = "Only one sentence."
        chunker = AdvancedSemanticChunker(text=text)
        chunks = chunker.chunk_text(n_chunks=5)

        assert len(chunks) == 1  # Only one sentence available
        assert chunks[0].page_content == "Only one sentence."


class TestComputeChunkerHash:
    """Tests for the compute_chunker_hash function."""

    def test_hash_is_consistent(self):
        """Tests that the same inputs produce the same hash."""
        text = "This is a test text."
        chunker_name = "Chunker"
        chunker_params = {"chunk_size": 1000, "chunk_overlap": 200}

        hash1 = compute_chunker_hash(text, chunker_name, chunker_params)
        hash2 = compute_chunker_hash(text, chunker_name, chunker_params)

        assert hash1 == hash2

    def test_different_text_produces_different_hash(self):
        """Tests that different texts produce different hashes."""
        chunker_name = "Chunker"
        chunker_params = {"chunk_size": 1000, "chunk_overlap": 200}

        hash1 = compute_chunker_hash("Text A", chunker_name, chunker_params)
        hash2 = compute_chunker_hash("Text B", chunker_name, chunker_params)

        assert hash1 != hash2

    def test_different_chunker_produces_different_hash(self):
        """Tests that different chunkers produce different hashes."""
        text = "This is a test text."
        chunker_params = {}

        hash1 = compute_chunker_hash(text, "Chunker", chunker_params)
        hash2 = compute_chunker_hash(text, "SemanticChunker", chunker_params)

        assert hash1 != hash2

    def test_different_params_produce_different_hash(self):
        """Tests that different parameters produce different hashes."""
        text = "This is a test text."
        chunker_name = "Chunker"

        hash1 = compute_chunker_hash(
            text, chunker_name, {"chunk_size": 1000, "chunk_overlap": 200}
        )
        hash2 = compute_chunker_hash(
            text, chunker_name, {"chunk_size": 500, "chunk_overlap": 100}
        )

        assert hash1 != hash2

    def test_hash_with_no_params(self):
        """Tests that hash works with no parameters (SemanticChunker case)."""
        text = "This is a test text."
        chunker_name = "SemanticChunker"

        hash1 = compute_chunker_hash(text, chunker_name, None)
        hash2 = compute_chunker_hash(text, chunker_name, {})

        # None and {} should produce the same hash
        assert hash1 == hash2

    def test_hash_format(self):
        """Tests that the hash is in the expected format (SHA256 hex)."""
        text = "This is a test text."
        chunker_name = "Chunker"
        chunker_params = {"chunk_size": 1000}

        hash_value = compute_chunker_hash(text, chunker_name, chunker_params)

        # SHA256 produces a 64-character hexadecimal string
        assert len(hash_value) == 64
        assert all(c in "0123456789abcdef" for c in hash_value)


class TestComputeRetrieverHash:
    """Tests for the compute_retriever_hash function."""

    def test_hash_is_consistent(self):
        """Tests that the same inputs produce the same hash."""
        chunker_hash = "abc123"
        retriever_model = "all-MiniLM-L6-v2"
        retriever_query = "test query"
        n_top_chunks = 5

        hash1 = compute_retriever_hash(
            chunker_hash, retriever_model, retriever_query, n_top_chunks
        )
        hash2 = compute_retriever_hash(
            chunker_hash, retriever_model, retriever_query, n_top_chunks
        )

        assert hash1 == hash2

    def test_different_chunker_hash_produces_different_hash(self):
        """Tests that different chunker hashes produce different retriever hashes."""
        retriever_model = "all-MiniLM-L6-v2"
        retriever_query = "test query"
        n_top_chunks = 5

        hash1 = compute_retriever_hash(
            "hash1", retriever_model, retriever_query, n_top_chunks
        )
        hash2 = compute_retriever_hash(
            "hash2", retriever_model, retriever_query, n_top_chunks
        )

        assert hash1 != hash2

    def test_different_retriever_model_produces_different_hash(self):
        """Tests that different retriever models produce different hashes."""
        chunker_hash = "abc123"
        retriever_query = "test query"
        n_top_chunks = 5

        hash1 = compute_retriever_hash(
            chunker_hash, "model1", retriever_query, n_top_chunks
        )
        hash2 = compute_retriever_hash(
            chunker_hash, "model2", retriever_query, n_top_chunks
        )

        assert hash1 != hash2

    def test_different_query_produces_different_hash(self):
        """Tests that different queries produce different hashes."""
        chunker_hash = "abc123"
        retriever_model = "all-MiniLM-L6-v2"
        n_top_chunks = 5

        hash1 = compute_retriever_hash(
            chunker_hash, retriever_model, "query1", n_top_chunks
        )
        hash2 = compute_retriever_hash(
            chunker_hash, retriever_model, "query2", n_top_chunks
        )

        assert hash1 != hash2

    def test_different_n_top_chunks_produces_different_hash(self):
        """Tests that different n_top_chunks values produce different hashes."""
        chunker_hash = "abc123"
        retriever_model = "all-MiniLM-L6-v2"
        retriever_query = "test query"

        hash1 = compute_retriever_hash(
            chunker_hash, retriever_model, retriever_query, 5
        )
        hash2 = compute_retriever_hash(
            chunker_hash, retriever_model, retriever_query, 10
        )

        assert hash1 != hash2

    def test_hash_format(self):
        """Tests that the hash is in the expected format (SHA256 hex)."""
        chunker_hash = "abc123"
        retriever_model = "all-MiniLM-L6-v2"
        retriever_query = "test query"
        n_top_chunks = 5

        hash_value = compute_retriever_hash(
            chunker_hash, retriever_model, retriever_query, n_top_chunks
        )

        # SHA256 produces a 64-character hexadecimal string
        assert len(hash_value) == 64
        assert all(c in "0123456789abcdef" for c in hash_value)
