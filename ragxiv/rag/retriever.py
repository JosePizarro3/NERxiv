from abc import ABC, abstractmethod

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer, util

from ragxiv.logger import logger


# TODO add measure of performance
# TODO check other model_name
class Retriever(ABC):
    """
    Abstract base class for retrieving relevant chunks of text from a list of documents. This class
    is designed to be inherited from and implemented by specific retriever classes.
    """

    def __init__(self, model: str = "all-MiniLM-L6-v2", query: str = "", **kwargs):
        self.logger = kwargs.get("logger", logger)
        self.model_name = model
        self.query = query

    @abstractmethod
    def get_relevant_chunks(self, chunks: list[Document] = [], n_top_chunks: int = 5):
        """Find the most relevant chunks describing methods."""
        pass


class CustomRetriever(Retriever):
    """
    A custom retriever class that uses the `SentenceTransformer` model to retrieve relevant chunks of text
    from a list of documents.
    """

    def __init__(self, model: str = "all-MiniLM-L6-v2", query: str = "", **kwargs):
        super().__init__(model, query, **kwargs)
        self.model = SentenceTransformer(self.model_name)
        self.logger.info(f"Loaded SentenceTransformer model: {self.model_name}")

    def get_relevant_chunks(
        self, chunks: list[Document] = [], n_top_chunks: int = 5
    ) -> str:
        """
        Retrieves the most relevant chunks of text from a list of documents using the `SentenceTransformer` model.

        Args:
            chunks (list[Document], optional): The chunks to be ranked. Defaults to [].
            n_top_chunks (int, optional): The number of top chunks to be returned. Defaults to 5.

        Returns:
            str: The top `n_top_chunks` chunks joined in a single string with the highest similarity score with respect to the query.
        """
        if not chunks:
            self.logger.warning("No chunks provided.")
            return []
        chunks = [chunk.page_content for chunk in chunks]

        # Converting `self.query` and `chunks` to embeddings
        query_embeddings = self.model.encode(self.query, convert_to_tensor=True)
        chunk_embeddings = self.model.encode(chunks, convert_to_tensor=True)

        # TODO check other similarities
        similarities = util.pytorch_cos_sim(query_embeddings, chunk_embeddings).squeeze(
            0
        )
        sorted_similarities = similarities.sort(descending=True)

        # Get the top `n_top_chunks` chunks with the highest similarity score with respect to the query
        top_chunks = [chunks[i] for i in sorted_similarities.indices[:n_top_chunks]]
        self.logger.info(
            f"Top {n_top_chunks} chunks retrieved with similarities of {sorted_similarities.values[:n_top_chunks]}"
        )
        return "\n\n".join(top_chunk for top_chunk in top_chunks)


class LangChainRetriever(Retriever):
    def __init__(self, model: str = "all-MiniLM-L6-v2", query: str = "", **kwargs):
        super().__init__(model, query, **kwargs)

        self.embeddings = HuggingFaceEmbeddings(model_name=self.model_name)
        self.logger.info(f"Loaded `HuggingFaceEmbeddings` model: {self.model_name}")

    def get_relevant_chunks(self, chunks: list[Document] = [], n_top_chunks=5) -> str:
        """
        Retrieves the most relevant chunks of text from a list of documents using the `HuggingFaceEmbeddings` model.

        Args:
            chunks (list[Document], optional): The chunks to be ranked. Defaults to [].
            n_top_chunks (int, optional): The number of top chunks to be returned. Defaults to 5.

        Returns:
            str: The top `n_top_chunks` chunks joined in a single string with the highest similarity score with respect to the query.
        """
        vector_store = InMemoryVectorStore(self.embeddings)
        _ = vector_store.add_documents(documents=chunks)
        results = vector_store.similarity_search_with_score(self.query, k=n_top_chunks)
        top_chunks, scores = (
            [r[0].page_content for r in results],
            [r[1] for r in results],
        )
        self.logger.info(
            f"Top {n_top_chunks} chunks retrieved with similarities of {scores}"
        )
        return "\n\n".join(top_chunk for top_chunk in top_chunks)
