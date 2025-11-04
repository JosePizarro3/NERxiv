from unittest.mock import MagicMock, patch

import pytest
import torch
from langchain_core.documents import Document

from nerxiv.rag import CustomRetriever, LangChainRetriever
from nerxiv.rag.retriever import CustomRetriever as CustomRetrieverForTest


class TestRetrieverBase:
    """Test suite for Retriever base class."""

    def test_retriever_requires_query(self):
        """Test that Retriever requires a query parameter."""
        # Cannot instantiate abstract class directly, use CustomRetriever instead
        with pytest.raises(ValueError, match="`query` is required"):
            with patch("nerxiv.rag.retriever.SentenceTransformer"):
                CustomRetrieverForTest(query=None)

    def test_retriever_default_values(self):
        """Test default values for Retriever."""
        with patch("nerxiv.rag.retriever.SentenceTransformer"):
            retriever = CustomRetriever(query="test query")
            assert retriever.model_name == "all-MiniLM-L6-v2"
            assert retriever.n_top_chunks == 5
            assert retriever.query == "test query"

    def test_retriever_custom_values(self):
        """Test custom values for Retriever."""
        with patch("nerxiv.rag.retriever.SentenceTransformer"):
            retriever = CustomRetriever(
                query="test query",
                model="custom-model",
                n_top_chunks=10,
            )
            assert retriever.model_name == "custom-model"
            assert retriever.n_top_chunks == 10


class TestCustomRetriever:
    """Test suite for CustomRetriever class."""

    def test_custom_retriever_mocked(self):
        """Tests the `get_relevant_chunks` method of the `CustomRetriever` class."""
        with patch("nerxiv.rag.retriever.SentenceTransformer") as mock_model:
            mock_instance = mock_model.return_value

            # Fake embeddings: query (1 x dim) and chunks (N x dim)
            def fake_encode(x, convert_to_tensor=False):
                if isinstance(x, str):
                    return torch.ones(1, 384)  # query embedding
                return torch.ones(len(x), 384)  # chunk embeddings

            mock_instance.encode.side_effect = fake_encode

            # Chunks to be ranked
            chunks = [
                Document(page_content="This text mentions DFT."),
                Document(page_content="A DMFT mention."),
                Document(page_content="No mention of any methodology."),
            ]

            # Mock the query on relevant chunks
            query = "What methods were used?"
            result = CustomRetriever(query=query).get_relevant_chunks(chunks=chunks)
            assert isinstance(result, str)
            splitted_result = result.split("\n\n")
            assert "DMFT" in splitted_result[0]
            assert "DFT" in splitted_result[1]

    def test_custom_retriever_empty_chunks(self):
        """Test CustomRetriever with empty chunks list."""
        with patch("nerxiv.rag.retriever.SentenceTransformer") as mock_model:
            mock_instance = mock_model.return_value
            mock_instance.encode.return_value = torch.ones(1, 384)

            retriever = CustomRetriever(query="test query")
            result = retriever.get_relevant_chunks(chunks=[])
            assert result == []

    def test_custom_retriever_fewer_chunks_than_top_n(self):
        """Test CustomRetriever when there are fewer chunks than n_top_chunks."""
        with patch("nerxiv.rag.retriever.SentenceTransformer") as mock_model:
            mock_instance = mock_model.return_value

            def fake_encode(x, convert_to_tensor=False):
                if isinstance(x, str):
                    return torch.ones(1, 384)
                return torch.ones(len(x), 384)

            mock_instance.encode.side_effect = fake_encode

            chunks = [
                Document(page_content="Chunk 1"),
                Document(page_content="Chunk 2"),
            ]

            retriever = CustomRetriever(query="test", n_top_chunks=10)
            result = retriever.get_relevant_chunks(chunks=chunks)
            # Should return all available chunks
            assert isinstance(result, str)
            assert "Chunk 1" in result or "Chunk 2" in result


class TestLangChainRetriever:
    """Test suite for LangChainRetriever class."""

    def test_langchain_retriever_mocked(self):
        """Tests the `get_relevant_chunks` method of the `LangChainRetriever` class."""
        with (
            patch("nerxiv.rag.retriever.HuggingFaceEmbeddings") as mock_embed_cls,
            patch("nerxiv.rag.retriever.InMemoryVectorStore") as mock_store_cls,
        ):
            # Mock embeddings
            mock_embed = MagicMock()
            mock_embed_cls.return_value = mock_embed

            # Mock vector store
            mock_store = MagicMock()
            mock_store_cls.return_value = mock_store

            # Simulate return of `similarity_search_with_score`
            mock_store.similarity_search_with_score.return_value = [
                (Document(page_content="We used DMFT."), 0.95),
                (Document(page_content="We also applied DFT."), 0.93),
            ]

            # Chunks to be ranked
            chunks = [
                Document(page_content="This text mentions DFT."),
                Document(page_content="A DMFT mention."),
                Document(page_content="No method mentioned here."),
            ]

            # Mock the query on relevant chunks
            query = "What methods were used?"
            result = LangChainRetriever(
                query=query, n_top_chunks=2
            ).get_relevant_chunks(chunks=chunks)
            assert isinstance(result, str)
            splitted_result = result.split("\n\n")
            assert "DMFT" in splitted_result[0]
            assert "DFT" in splitted_result[1]

    def test_langchain_retriever_n_top_chunks(self):
        """Test that LangChainRetriever respects n_top_chunks parameter."""
        with (
            patch("nerxiv.rag.retriever.HuggingFaceEmbeddings") as mock_embed_cls,
            patch("nerxiv.rag.retriever.InMemoryVectorStore") as mock_store_cls,
        ):
            mock_embed = MagicMock()
            mock_embed_cls.return_value = mock_embed

            mock_store = MagicMock()
            mock_store_cls.return_value = mock_store

            # Return 3 chunks
            mock_store.similarity_search_with_score.return_value = [
                (Document(page_content="Chunk 1"), 0.95),
                (Document(page_content="Chunk 2"), 0.90),
                (Document(page_content="Chunk 3"), 0.85),
            ]

            chunks = [
                Document(page_content="Chunk 1"),
                Document(page_content="Chunk 2"),
                Document(page_content="Chunk 3"),
            ]

            retriever = LangChainRetriever(query="test", n_top_chunks=3)
            result = retriever.get_relevant_chunks(chunks=chunks)

            # Verify that similarity_search_with_score was called with correct k
            mock_store.similarity_search_with_score.assert_called_once_with("test", k=3)
            assert isinstance(result, str)
            assert "Chunk 1" in result
            assert "Chunk 2" in result
            assert "Chunk 3" in result

    def test_langchain_retriever_adds_documents_to_store(self):
        """Test that LangChainRetriever adds documents to vector store."""
        with (
            patch("nerxiv.rag.retriever.HuggingFaceEmbeddings") as mock_embed_cls,
            patch("nerxiv.rag.retriever.InMemoryVectorStore") as mock_store_cls,
        ):
            mock_embed = MagicMock()
            mock_embed_cls.return_value = mock_embed

            mock_store = MagicMock()
            mock_store_cls.return_value = mock_store
            mock_store.similarity_search_with_score.return_value = [
                (Document(page_content="Test"), 0.95)
            ]

            chunks = [Document(page_content="Test chunk")]

            retriever = LangChainRetriever(query="test")
            retriever.get_relevant_chunks(chunks=chunks)

            # Verify add_documents was called
            mock_store.add_documents.assert_called_once_with(documents=chunks)
