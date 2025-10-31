import datetime
import json
import time
from typing import Any

import h5py
from langchain_core.documents import Document

from nerxiv.agents.base import BaseAgent
from nerxiv.logger import logger
from nerxiv.prompts.prompts import BasePrompt
from nerxiv.utils.caching import compute_chunker_hash, compute_retriever_hash


class RAGExtractorAgent(BaseAgent):
    def __init__(
        self,
        chunker: type | object,
        retriever: type | object,
        generator: type | object,
        **kwargs,
    ):
        self.chunker = chunker
        self.retriever = retriever
        self.generator = generator

        self.chunker_params = kwargs.get("chunker_params", {})
        self.retriever_params = kwargs.get("retriever_params", {})
        self.generator_params = kwargs.get("generator_params", {})

        self.logger = kwargs.get("logger", logger)

    def _obj_name(self, obj: type | object) -> str:
        """Get the class name of `obj`, whether it's a class or an instance."""
        if isinstance(obj, type):
            return obj.__name__
        return obj.__class__.__name__

    def _instantiate(self, component: type | object, required_kwargs: dict) -> Any:
        """Instantiate `component` if it's a class, otherwise return the instance.

        The method merges `required_kwargs` with the preconfigured kwargs for
        that `component` (used by the caller).
        """
        if isinstance(component, type):
            # component is a class; instantiate with kwargs
            return component(**required_kwargs)
        # assume component is already an instance
        return component

    def run(
        self,
        file: h5py.File | None = None,
        text: str = "",
        retriever_model: str = "all-MiniLM-L6-v2",
        n_top_chunks: int = 5,
        model: str = "gpt-oss:20b",
        prompt: BasePrompt | None = None,
    ):
        # initial checks
        if not file:
            self.logger.critical("`file` is required for RAGExtractorAgent")
            return None
        if not text:
            self.logger.critical("`text` is required for RAGExtractorAgent")
            return None
        if not prompt:
            self.logger.critical("`prompt` is required for RAGExtractorAgent")
            return None
        query = self.retriever_params.get("query")
        if not query:
            self.logger.critical(
                "`retriever_params` must include a 'query' key for RAGExtractorAgent"
            )
            return None

        # Create group to store RAG pipeline
        global_time = time.time()
        rag_group = file.require_group("rag_extraction")
        rag_group.attrs["stat_time"] = datetime.datetime.now().isoformat()

        ### Chunking
        chunker_name = self._obj_name(self.chunker)

        # Use caching to compute chunker hash and avoid re-chunking if already done
        chunker_hash = compute_chunker_hash(
            text=text,
            chunker_name=chunker_name,
            chunker_params=self.chunker_params,
        )
        chunks_cache_group = rag_group.require_group("chunks_cache")
        if chunker_hash in chunks_cache_group:  # reuse existing chunks
            logger.info(f"Reusing chunks from cache with hash {chunker_hash}")
            cached_chunks_group = chunks_cache_group[chunker_hash]
            chunks = []
            for key in cached_chunks_group.keys():
                chunks.append(
                    Document(
                        page_content=cached_chunks_group[key][()].decode("utf-8"),
                        metadata={"source": f"nerxiv.chunker.{chunker_name}"},
                    )
                )
        else:  # perform new chunking
            logger.info(f"Performing new chunking with hash {chunker_hash}")
            chunker = self._instantiate(
                self.chunker, {**self.chunker_params, "text": text}
            )
            chunks = chunker.chunk_text()
            # Store chunks in cache
            cached_chunks_group = chunks_cache_group.create_group(chunker_hash)
            cached_chunks_group.attrs["chunker"] = f"nerxiv.chunker.{chunker_name}"
            cached_chunks_group.attrs["chunker_params"] = json.dumps(
                self.chunker_params
            )
            cached_chunks_group.attrs["run_time"] = time.time() - global_time
            for i, chunk in enumerate(chunks):
                cached_chunks_group.create_dataset(
                    f"chunk_{i:04d}", data=chunk.encode("utf-8")
                )

        ### Retrieval
        start_time = time.time()
        retriever_name = self._obj_name(self.retriever)

        # Use caching to compute retriever hash and avoid re-retrieving if already done
        retriever_hash = compute_retriever_hash(
            chunker_hash=chunker_hash, retriever_params=self.retriever_params
        )
        retrieval_cache_group = rag_group.require_group("retrieval_cache")
        if retriever_hash in retrieval_cache_group:  # reuse existing retrieval
            logger.info(
                f"Reusing retrieval results from cache with hash {retriever_hash}"
            )
            cached_retrieval_group = retrieval_cache_group[retriever_hash]
            text = cached_retrieval_group["retrieved_text"][()].decode("utf-8")
        else:  # perform new retrieval
            logger.info(f"Performing new retrieval with hash {retriever_hash}")
            retriever = self._instantiate(
                self.retriever, {**self.retriever_params, "model": retriever_model}
            )
            text = retriever.get_relevant_chunks(
                chunks=chunks,
                n_top_chunks=n_top_chunks,
            )

            # Store retrieval results in cache
            cached_retrieval_group = retrieval_cache_group.create_group(retriever_hash)
            cached_retrieval_group.attrs["retriever"] = (
                f"nerxiv.rag.retriever.{retriever_name}"
            )
            cached_retrieval_group.attrs["chunker_hash"] = chunker_hash
            cached_retrieval_group.attrs["retriever_model"] = retriever_model
            cached_retrieval_group.attrs["retriever_params"] = json.dumps(
                self.retriever_params
            )
            cached_retrieval_group.attrs["n_top_chunks"] = n_top_chunks
            cached_retrieval_group.attrs["retriever_hash"] = retriever_hash
            cached_retrieval_group.create_dataset(
                "retrieved_text", data=text.encode("utf-8")
            )

        ### Generation
        start_time = time.time()
        generator = self._instantiate(
            self.generator, {"model": model, "text": text, **self.generator_params}
        )
        built_prompt = prompt.build(text=text)
        answer = generator.generate(prompt=built_prompt)

        # Store raw answer in HDF5
        raw_answer_group = rag_group.require_group("raw_llm_answers")
        # Define group for the `query` (e.g., raw_llm_answers/filter_material_formula)
        query_group = raw_answer_group.require_group(query)
        # Define group for the run ID (e.g., raw_llm_answers/filter_material_formula/run_0000)
        existing_runs = list(query_group.keys())
        run_id = f"run_{len(existing_runs):04d}"  # Auto-increment run ID
        run_group = query_group.create_group(run_id)
        # Store general metainformation
        run_group.attrs["model"] = model
        # Store prompt and answer
        run_group.create_dataset("prompt", data=built_prompt.encode("utf-8"))
        run_group.create_dataset("answer", data=answer.encode("utf-8"))
        # Store references to cached data instead of duplicating
        run_group.attrs["chunker_hash"] = chunker_hash
        run_group.attrs["retriever_hash"] = retriever_hash
        # Store elapsed time and timestamp of the run
        run_group.attrs["elapsed_time"] = time.time() - start_time

        # Store total RAG pipeline time
        paper_time = time.time() - global_time
        rag_group.attrs["elapsed_time"] = paper_time
        logger.info(f"Prompting completed for {file} in {paper_time:.2f} seconds.")

        ### Return structured result
