from unittest.mock import MagicMock, patch

import torch
from langchain_core.documents import Document

from nerxiv.rag import CustomRetriever, LangChainRetriever


def test_custom_retriever_mocked():
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


def test_langchain_retriever_mocked():
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
        result = LangChainRetriever(query=query).get_relevant_chunks(
            chunks=chunks, n_top_chunks=2
        )
        assert isinstance(result, str)
        splitted_result = result.split("\n\n")
        assert "DMFT" in splitted_result[0]
        assert "DFT" in splitted_result[1]
