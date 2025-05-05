from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ragxiv.logger import logger


class Chunker:
    """
    Chunk text into smaller parts for processing and avoiding the token limit of an LLM model.
    """

    def __init__(self, text: str = "", **kwargs):
        if not text:
            raise ValueError("Text is required for chunking.")
        self.text = text
        self.logger = kwargs.get("logger", logger)

    def chunk_text(
        self, chunk_size: int = 1000, chunk_overlap: int = 200
    ) -> list[Document]:
        """
        Chunk the text into smaller parts.
        This is done to avoid exceeding the token limit of the LLM.

        Args:
            chunk_size (int, optional): The size of each chunk. Defaults to 1000.
            chunk_overlap (int, optional): The overlap between chunks. Defaults to 200.

        Returns:
            list[Document]: The list of chunks as `Document` objects.
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=True
        )

        # ! we define a list of `Document` objects in LangChain to use the `split_documents(pages)` method
        pages = [
            Document(
                page_content=self.text, metadata={"source": "TextExtractor.get_text()"}
            )
        ]
        chunks = text_splitter.split_documents(pages)
        self.logger.info(f"Text chunked into {len(chunks)} parts")
        return chunks


class SemanticChunker:
    """https://python.langchain.com/docs/how_to/semantic-chunker/"""

    # TODO implement
    pass
