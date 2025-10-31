from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from nerxiv.chunker import AdvancedSemanticChunker, Chunker, SemanticChunker


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
        chunks = Chunker(
            text=text, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        ).chunk_text()
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
        chunker = AdvancedSemanticChunker(text=text, n_chunks=2)
        chunks = chunker.chunk_text()

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
        chunker = AdvancedSemanticChunker(text=text, n_chunks=5)
        chunks = chunker.chunk_text()

        assert len(chunks) == 1  # Only one sentence available
        assert chunks[0].page_content == "Only one sentence."
