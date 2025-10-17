"""RAG extractor agent that composes chunker, retriever and generator.

This class provides a small orchestration layer around the existing
Chunker, Retriever and LLMGenerator implementations so they can be used
together as a single agent for extraction/QA tasks.

The agent is intentionally thin: it accepts classes or pre-instantiated
objects for each component to make testing and dependency injection easy.
"""
from typing import Any, Callable, List, Optional, Type, Union

from nerxiv.logger import logger

# typing for duck-typed components
ComponentType = Union[Type, object]


class RAGExtractorAgent:
    """Compose Chunker -> Retriever -> Generator into a single agent.

    Usage patterns supported:
    - Provide component classes: the agent will instantiate them when needed.
    - Provide component instances: the agent will use them directly.

    Constructor accepts optional kwargs dicts that are forwarded to the
    respective components on instantiation.
    """

    def __init__(
        self,
        chunker: ComponentType,
        retriever: ComponentType,
        generator: ComponentType,
        chunker_kwargs: Optional[dict] = None,
        retriever_kwargs: Optional[dict] = None,
        generator_kwargs: Optional[dict] = None,
    ):
        self.chunker = chunker
        self.retriever = retriever
        self.generator = generator
        self.chunker_kwargs = chunker_kwargs or {}
        self.retriever_kwargs = retriever_kwargs or {}
        self.generator_kwargs = generator_kwargs or {}
        self.logger = logger

    def _instantiate(self, comp: ComponentType, required_kwargs: dict) -> Any:
        """Instantiate component if it's a class, otherwise return the instance.

        The method merges required_kwargs with the preconfigured kwargs for
        that component (used by the caller).
        """
        if isinstance(comp, type):
            # comp is a class; instantiate with kwargs
            return comp(**required_kwargs)
        # assume comp is already an instance
        return comp

    def extract(
        self,
        text: str,
        query: str,
        n_top_chunks: int = 5,
        prompt_template: str = "Context:\n{context}\n\nQuestion:\n{query}\n\nAnswer:",
        regex: str = r"\n\nAnswer\: *",
        del_regex: str = r"\n\nAnswer\: *",
    ) -> dict:
        """Run the full RAG flow and return the result payload.

        Returns a dict containing:
        - chunks: list of chunk objects returned by the chunker
        - retrieved: joined string of top-retrieved chunks
        - prompt: the prompt sent to the generator
        - answer: the generated answer string
        """
        if not text:
            raise ValueError("`text` is required for extraction")
        if not query:
            raise ValueError("`query` is required for extraction")

        # 1) Chunk the input text
        # chunker classes expect `text` in their constructor; if the caller
        # supplied an instance we assume it's already bound to the right text.
        chunker_obj = self._instantiate(self.chunker, {**self.chunker_kwargs, "text": text})
        if not hasattr(chunker_obj, "chunk_text"):
            raise TypeError("chunker must provide a `chunk_text()` method")
        chunks = chunker_obj.chunk_text()
        self.logger.info(f"RAG agent: produced {len(chunks) if chunks else 0} chunks")

        # 2) Retrieve relevant chunks
        # many retriever classes require `query` on init; prefer passing it
        retriever_obj = self._instantiate(self.retriever, {**self.retriever_kwargs, "query": query})
        if not hasattr(retriever_obj, "get_relevant_chunks"):
            raise TypeError("retriever must provide a `get_relevant_chunks()` method")
        retrieved = retriever_obj.get_relevant_chunks(chunks, n_top_chunks=n_top_chunks)
        self.logger.info("RAG agent: retrieval finished")

        # 3) Generate answer using the retrieved context
        # generator classes in this repo expect `text` during construction; use the retrieved text as context
        generator_obj = self._instantiate(self.generator, {**self.generator_kwargs, "text": retrieved})
        if not hasattr(generator_obj, "generate"):
            raise TypeError("generator must provide a `generate()` method")

        prompt = prompt_template.format(context=retrieved, query=query)
        answer = generator_obj.generate(prompt=prompt, regex=regex, del_regex=del_regex)
        self.logger.info("RAG agent: generation finished")

        return {
            "chunks": chunks,
            "retrieved": retrieved,
            "prompt": prompt,
            "answer": answer,
        }
