import datetime
import json
import re
import time
from abc import ABC, abstractmethod
from typing import Any

import h5py
from langchain_core.documents import Document

from nerxiv.logger import logger
from nerxiv.prompts.prompts import BasePrompt, StructuredPrompt
from nerxiv.utils.caching import compute_chunker_hash, compute_retriever_hash


class BaseAgent(ABC):
    """Abstract base class for extraction agents.

    All agents should implement the `run` method which executes the
    extraction workflow and returns structured results.
    """

    @abstractmethod
    def run(self, text: str, prompt: BasePrompt | None, **kwargs) -> None:
        """Execute the extraction workflow.

        Args:
            text (str): Input text to process.
            prompt (BasePrompt | None): Prompt template for LLM.
            **kwargs (dict): Additional parameters specific to the agent

        Returns:
            Dictionary containing extraction results
        """
        pass


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
        """
        Gets the class name of an object or an object instance.

        Args:
            obj (type | object): The object or class to get the name of.

        Returns:
            str: The class name of the object or the name of the class itself.
        """
        if isinstance(obj, type):
            return obj.__name__
        return obj.__class__.__name__

    def _instantiate(self, component: type | object, required_kwargs: dict) -> Any:
        """I
        nstantiate `component` if it's a class, otherwise return the instance.

        The method merges `required_kwargs` with the preconfigured kwargs for
        that `component` (used by the caller).
        """
        if isinstance(component, type):
            # component is a class; instantiate with kwargs
            return component(**required_kwargs)
        # assume component is already an instance
        return component

    def _get_chunks(
        self,
        chunker_hash: str,
        text: str,
        chunker_name: str,
        cached_chunks_group: h5py.Group,
        global_time: float,
    ) -> list[Document]:
        """
        Gets the chunks when the chunker class needs to be instantiated (not read from cache).

        Args:
            chunker_hash (str): The chunker hash.
            text (str): The text to be chunked.
            chunker_name (str): The name of the chunker.
            cached_chunks_group (h5py.Group): The HDF5 group to store cached chunks.
            global_time (float): The global start time.

        Returns:
            list[Document]: The list of chunks.
        """
        self.logger.info(f"Performing new chunking with hash {chunker_hash}")
        chunker = self._instantiate(self.chunker, {**self.chunker_params, "text": text})
        chunks = chunker.chunk_text()
        # Store chunks in cache
        cached_chunks_group.attrs["chunker"] = f"nerxiv.chunker.{chunker_name}"
        cached_chunks_group.attrs["chunker_params"] = json.dumps(self.chunker_params)
        cached_chunks_group.attrs["run_time"] = time.time() - global_time
        for i, chunk in enumerate(chunks):
            cached_chunks_group.create_dataset(
                f"chunk_{i:04d}", data=chunk.page_content.encode("utf-8")
            )
        return chunks

    def parse(self, answer: str) -> dict[str, Any] | None:
        """
        Parse JSON from LLM answer if the prompt is of `StructuredPrompt` type. This method
        attempts to extract JSON from markdown code blocks (```json...```) and if successful,
        return the parsed data. If no code blocks are found, it tries to find JSON patterns
        directly in the text.

        Args:
            answer (str): Raw LLM output string.

        Returns:
            dict[str, Any] | None: Parsed JSON data as a dictionary, or None if parsing fails.
        """
        try:
            # Try to extract JSON from markdown code block
            json_match = re.search(
                r"```json\s*\n(.*?)\n\s*```", answer, re.DOTALL | re.IGNORECASE
            )

            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to find JSON without code blocks
                # Look for content between { and } or [ and ]
                json_match = re.search(r"(\{.*\}|\[.*\])", answer, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                else:
                    self.logger.error("No JSON found in answer")
                    return None

            # Parse JSON
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON decode error: {e}")
            return None
        return data

    def run(
        self,
        file: h5py.File | None = None,
        text: str = "",
        prompt: BasePrompt | None = None,
    ) -> None:
        """
        Runs the RAG extraction pipeline: chunking, retrieval, and generation.
        Chunking and retrieval results are cached in the provided HDF5 file to avoid redundant computations.
        If the prompt is of type `StructuredPrompt`, the generated answer is parsed into structured data.

        Args:
            file (h5py.File | None, optional): The file were to store the metainformation. Defaults to None.
            text (str, optional): The text to process. Defaults to "".
            prompt (BasePrompt | None, optional): The prompt used for the LLM prompting. Defaults to None.
        """
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
            self.logger.info(f"Reusing chunks from cache with hash {chunker_hash}")
            cached_chunks_group = chunks_cache_group[chunker_hash]
            if len(cached_chunks_group.keys()) == 0:
                chunks = self._get_chunks(
                    chunker_hash,
                    text,
                    chunker_name,
                    cached_chunks_group,
                    global_time,
                )
            else:
                chunks = []
                for key in cached_chunks_group.keys():
                    chunks.append(
                        Document(
                            page_content=cached_chunks_group[key][()].decode("utf-8"),
                            metadata={"source": f"nerxiv.chunker.{chunker_name}"},
                        )
                    )
        else:  # perform new chunking
            chunks = self._get_chunks(
                chunker_hash,
                text,
                chunker_name,
                chunks_cache_group.create_group(chunker_hash),
                global_time,
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
            self.logger.info(
                f"Reusing retrieval results from cache with hash {retriever_hash}"
            )
            cached_retrieval_group = retrieval_cache_group[retriever_hash]
            text = cached_retrieval_group["retrieved_text"][()].decode("utf-8")
        else:  # perform new retrieval
            self.logger.info(f"Performing new retrieval with hash {retriever_hash}")
            retriever = self._instantiate(self.retriever, {**self.retriever_params})
            text = retriever.get_relevant_chunks(chunks=chunks)

            # Store retrieval results in cache
            cached_retrieval_group = retrieval_cache_group.create_group(retriever_hash)
            cached_retrieval_group.attrs["retriever"] = (
                f"nerxiv.rag.retriever.{retriever_name}"
            )
            cached_retrieval_group.attrs["chunker_hash"] = chunker_hash
            cached_retrieval_group.attrs["retriever_hash"] = retriever_hash
            cached_retrieval_group.attrs["retriever_params"] = json.dumps(
                self.retriever_params
            )
            cached_retrieval_group.create_dataset(
                "retrieved_text", data=text.encode("utf-8")
            )

        ### Generation
        start_time = time.time()
        generator = self._instantiate(
            self.generator, {"text": text, **self.generator_params}
        )
        built_prompt = prompt.build(text=text)
        answer = generator.generate(prompt=built_prompt)

        # Store raw answer in HDF5
        raw_answer_group = rag_group.require_group("raw_llm_answers")
        # Define group for the `query` (e.g., raw_llm_answers/filter_material_formula)
        query_group = raw_answer_group.require_group(
            self.retriever_params.get("query_name")
        )
        # Define group for the run ID (e.g., raw_llm_answers/filter_material_formula/run_0000)
        existing_runs = list(query_group.keys())
        run_id = f"run_{len(existing_runs):04d}"  # Auto-increment run ID
        run_group = query_group.create_group(run_id)
        # Store general metainformation
        run_group.attrs["model"] = self.generator_params.get("model", "gpt-oss:20b")
        # Store prompt and answer
        run_group.create_dataset("prompt", data=built_prompt.encode("utf-8"))
        run_group.create_dataset("answer", data=answer.encode("utf-8"))
        # Store references to cached data instead of duplicating
        run_group.attrs["chunker_hash"] = chunker_hash
        run_group.attrs["retriever_hash"] = retriever_hash
        # Store elapsed time and timestamp of the run
        run_group.attrs["elapsed_time"] = time.time() - start_time
        run_group.attrs["timestamp"] = datetime.datetime.now().isoformat()

        # Store total RAG pipeline time
        paper_time = time.time() - global_time
        rag_group.attrs["elapsed_time"] = paper_time
        self.logger.info(f"Prompting completed for {file} in {paper_time:.2f} seconds.")

        ### Return structured result
        # Parse and validate output for structured prompts
        if isinstance(prompt, StructuredPrompt):
            data = self.parse(answer=answer)
            if data is None:
                self.logger.error("Failed to parse LLM answer.")
                return None
            try:
                schema = prompt.output_schema
                data_fields = data[self._obj_name(schema)]
                filled_schema = schema(**data_fields)
                self.logger.info(f"Schema={filled_schema}")
            except Exception as e:
                self.logger.error(f"Validation error: {e}")
                return None
