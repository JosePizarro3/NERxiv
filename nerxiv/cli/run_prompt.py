import datetime
import json
import time
from pathlib import Path
from typing import TYPE_CHECKING

import h5py
from langchain_core.documents import Document

from nerxiv.chunker import _CHUNKER_MAP, Chunker
from nerxiv.ensemble import run_ensemble_prompts
from nerxiv.logger import logger
from nerxiv.prompts.prompts import BasePrompt, StructuredPrompt
from nerxiv.rag import CustomRetriever, LLMGenerator
from nerxiv.utils.caching import compute_chunker_hash, compute_retriever_hash

if TYPE_CHECKING:
    from structlog._config import BoundLoggerLazyProxy


def run_prompt_paper(
    paper: Path,
    chunker: str = "Chunker",
    retriever_model: str = "all-MiniLM-L6-v2",
    n_top_chunks: int = 5,
    model: str = "gpt-oss:20b",
    retriever_query: str = "",
    prompt: BasePrompt | None = None,
    query: str = "filter_material_formula",
    paper_time: float = 0.0,
    logger: "BoundLoggerLazyProxy" = logger,
    **kwargs,
) -> float:
    """Runs the prompt based on `retriever_query` and `template` on a given `paper`.

    Args:
        paper (Path): Path to the HDF5 file containing the paper data.
        chunker (str, optional): The chunker class to use for chunking the text. Defaults to `Chunker`.
        retriever_model (str, optional): The model used in the retriever. Defaults to "all-MiniLM-L6-v2".
        n_top_chunks (int, optional): The number of top chunks to retrieve. Defaults to 5.
        model (_type_, optional): The model used in the generator. Defaults to "gpt-oss:20b".
        retriever_query (str, optional): The query used in the retriever. This is set using `query` and the `QUERY_REGISTRY`. Defaults to "".
        prompt (BasePrompt, optional): The prompt used in the generator. This is set using `query` and the `QUERY_REGISTRY`.. Defaults to None.
        query (str, optional): The query used for retrieval and generation. See the registry in PROMPT_REGISTRY. Defaults to "filter_material_formula".
        paper_time (float, optional): The starting time of this paper prompting. Defaults to 0.0.
        logger (BoundLoggerLazyProxy, optional): The logger to log messages. Defaults to logger.
        chunk_size (int, optional): For Chunker, the size of each chunk. Defaults to 1000. Passed via kwargs.
        chunk_overlap (int, optional): For Chunker, the overlap between chunks. Defaults to 200. Passed via kwargs.
        n_chunks (int, optional): For AdvancedSemanticChunker, the number of chunks. Defaults to 10. Passed via kwargs.

    Returns:
        float: The time taken to run the prompt on the paper in seconds.

    Note:
        This function implements a two-level caching mechanism:

        1. **Chunking cache** (chunks_cache/): Stores chunks indexed by chunker hash.
           If the same text is chunked with the same parameters, cached chunks are reused.

        2. **Retrieval cache** (retrieval_cache/): Stores top-k chunks indexed by retriever hash.
           If the same chunks are retrieved with the same retriever parameters, cached results are reused.

        Run metadata only stores references (hashes) to the cached data, avoiding duplication.
        Additional kwargs are passed to LLMGenerator.
    """
    # Initial error handling
    if not paper.exists():
        logger.error(f"File {paper} does not exist.")
        return 0.0
    if not paper.name.endswith(".hdf5"):
        logger.error(f"File {paper} is not an HDF5 file.")
        return 0.0
    if not retriever_query or not prompt:
        logger.error("`retriever_query` and `prompt` must be provided.")
        return 0.0

    # Writing prompting results to the HDF5 of the paper
    with h5py.File(paper, "a") as f:
        arxiv_id = f.filename.split("/")[-1].replace(".hdf5", "")
        text = f[arxiv_id]["arxiv_paper"]["text"][()].decode("utf-8")

        # Extract chunker-specific parameters from kwargs
        # Default parameters for each chunker type
        chunker_params = {}
        if chunker == "Chunker":
            chunker_params = {
                "chunk_size": kwargs.pop("chunk_size", 1000),
                "chunk_overlap": kwargs.pop("chunk_overlap", 200),
            }
        elif chunker == "AdvancedSemanticChunker":
            chunker_params = {"n_chunks": kwargs.pop("n_chunks", 10)}
        # SemanticChunker has no parameters

        # Compute hash for this chunking configuration
        chunker_hash = compute_chunker_hash(
            text=text, chunker_name=chunker, chunker_params=chunker_params
        )
        logger.info(f"Chunker hash: {chunker_hash}")

        # Check if chunks with this hash already exist in a global cache
        chunks_cache_group = f.require_group("chunks_cache")

        if chunker_hash in chunks_cache_group:
            # Reuse existing chunks
            logger.info(f"Reusing chunks from cache with hash {chunker_hash}")
            cached_chunks_group = chunks_cache_group[chunker_hash]
            n_chunks = cached_chunks_group.attrs["n_chunks"]
            chunks = []
            for i in range(n_chunks):
                chunk_content = cached_chunks_group[f"chunk_{i:04d}"][()].decode(
                    "utf-8"
                )
                chunk_source = cached_chunks_group.attrs.get(
                    "chunker", f"nerxiv.chunker.{chunker}"
                )
                chunks.append(
                    Document(
                        page_content=chunk_content, metadata={"source": chunk_source}
                    )
                )
        else:
            # Perform new chunking
            logger.info(f"Performing new chunking with hash {chunker_hash}")
            chunker_cls = _CHUNKER_MAP.get(chunker, Chunker)(text=text)
            if chunker in ("Chunker", "AdvancedSemanticChunker"):
                chunks = chunker_cls.chunk_text(**chunker_params)
            else:
                chunks = chunker_cls.chunk_text()

            # Store chunks in cache
            cached_chunks_group = chunks_cache_group.create_group(chunker_hash)
            cached_chunks_group.attrs["chunker"] = f"nerxiv.chunker.{chunker}"
            cached_chunks_group.attrs["n_chunks"] = len(chunks)
            cached_chunks_group.attrs["chunker_hash"] = chunker_hash
            # Store chunker parameters as JSON string for readability
            cached_chunks_group.attrs["chunker_params"] = json.dumps(chunker_params)
            for i, chunk in enumerate(chunks):
                cached_chunks_group.create_dataset(
                    f"chunk_{i:04d}", data=chunk.page_content.encode("utf-8")
                )

        # Retrieval with caching
        retriever_hash = compute_retriever_hash(
            chunker_hash=chunker_hash,
            retriever_model=retriever_model,
            retriever_query=retriever_query,
            n_top_chunks=n_top_chunks,
        )
        logger.info(f"Retriever hash: {retriever_hash}")

        # Check if retrieval results are cached
        retrieval_cache_group = f.require_group("retrieval_cache")

        if retriever_hash in retrieval_cache_group:
            # Reuse cached retrieval results
            logger.info(
                f"Reusing retrieval results from cache with hash {retriever_hash}"
            )
            cached_retrieval_group = retrieval_cache_group[retriever_hash]
            text = cached_retrieval_group["retrieved_text"][()].decode("utf-8")
        else:
            # Perform new retrieval
            logger.info(f"Performing new retrieval with hash {retriever_hash}")
            retriever = CustomRetriever(
                model=retriever_model, query=retriever_query, logger=logger
            )
            text = retriever.get_relevant_chunks(
                chunks=chunks,
                n_top_chunks=n_top_chunks,
            )

            # Store retrieval results in cache
            cached_retrieval_group = retrieval_cache_group.create_group(retriever_hash)
            cached_retrieval_group.attrs["chunker_hash"] = chunker_hash
            cached_retrieval_group.attrs["retriever_model"] = retriever_model
            cached_retrieval_group.attrs["retriever_query"] = retriever_query
            cached_retrieval_group.attrs["n_top_chunks"] = n_top_chunks
            cached_retrieval_group.attrs["retriever_hash"] = retriever_hash
            cached_retrieval_group.create_dataset(
                "retrieved_text", data=text.encode("utf-8")
            )

        # Generation
        generator = LLMGenerator(model=model, text=text, logger=logger, **kwargs)
        built_prompt = prompt.build(text=text)
        answer = generator.generate(prompt=built_prompt)

        # Store raw answer in HDF5
        raw_answer_group = f.require_group("raw_llm_answers")
        # Define group for the `query` (e.g., raw_llm_answers/filter_material_formula)
        query_group = raw_answer_group.require_group(query)
        # Define group for the run ID (e.g., raw_llm_answers/filter_material_formula/run_0000)
        existing_runs = list(query_group.keys())
        run_id = f"run_{len(existing_runs):04d}"  # Auto-increment run ID
        run_group = query_group.create_group(run_id)
        # Store general metainformation
        run_group.attrs["model"] = model
        run_group.attrs["query"] = query
        run_group.attrs["timestamp"] = datetime.datetime.now().isoformat()
        # Store prompt and answer
        run_group.create_dataset("prompt", data=built_prompt.encode("utf-8"))
        run_group.create_dataset("answer", data=answer.encode("utf-8"))

        # Store references to cached data instead of duplicating
        run_group.attrs["chunker_hash"] = chunker_hash
        run_group.attrs["retriever_hash"] = retriever_hash

        paper_time = time.time() - paper_time
        run_group.attrs["elapsed_time"] = paper_time
    return paper_time


def run_ensemble_prompt_paper(
    paper: Path,
    chunker: str = "Chunker",
    retriever_model: str = "all-MiniLM-L6-v2",
    n_top_chunks: int = 5,
    model: str = "gpt-oss:20b",
    retriever_query: str = "",
    prompt: BasePrompt | None = None,
    query: str = "filter_material_formula",
    paper_time: float = 0.0,
    logger: "BoundLoggerLazyProxy" = logger,
    n_ensemble_runs: int = 5,
    ensemble_models: list[str] | None = None,
    ensemble_temperatures: list[float] | None = None,
    ensemble_chunk_sizes: list[int] | None = None,
    ensemble_parallel: bool = True,
    averaging_model: str = "gpt-oss:20b",
    averaging_temperature: float = 0.2,
    **kwargs,
) -> float:
    """Runs ensemble prompting with multiple configurations to minimize hallucinations.

    This function runs the same prompt multiple times with variations (different models,
    temperatures, and chunk sizes) and then averages the results using an LLM. This approach
    is designed to reduce hallucinations in LLM outputs, particularly for StructuredPrompts.

    Args:
        paper (Path): Path to the HDF5 file containing the paper data.
        chunker (str, optional): The chunker class to use for chunking the text. Defaults to `Chunker`.
        retriever_model (str, optional): The model used in the retriever. Defaults to "all-MiniLM-L6-v2".
        n_top_chunks (int, optional): The number of top chunks to retrieve. Defaults to 5.
        model (str, optional): The default model used in the generator. Defaults to "gpt-oss:20b".
        retriever_query (str, optional): The query used in the retriever.
        prompt (BasePrompt, optional): The prompt used in the generator.
        query (str, optional): The query used for retrieval and generation. Defaults to "filter_material_formula".
        paper_time (float, optional): The starting time of this paper prompting. Defaults to 0.0.
        logger (BoundLoggerLazyProxy, optional): The logger to log messages. Defaults to logger.
        n_ensemble_runs (int, optional): Number of ensemble runs. Defaults to 5.
        ensemble_models (list[str] | None, optional): List of models to use in ensemble. Defaults to [model].
        ensemble_temperatures (list[float] | None, optional): List of temperatures. Defaults to [0.2, 0.5, 0.7].
        ensemble_chunk_sizes (list[int] | None, optional): List of chunk sizes (minimum 2000). Defaults to [2000, 3000, 4000].
        ensemble_parallel (bool, optional): Whether to run in parallel. Defaults to True.
        averaging_model (str, optional): Model for averaging results. Defaults to "gpt-oss:20b".
        averaging_temperature (float, optional): Temperature for averaging. Defaults to 0.2.
        **kwargs: Additional arguments for LLMGenerator.

    Returns:
        float: The time taken to run the ensemble prompts on the paper in seconds.
    """
    # Initial error handling
    if not paper.exists():
        logger.error(f"File {paper} does not exist.")
        return 0.0
    if not paper.name.endswith(".hdf5"):
        logger.error(f"File {paper} is not an HDF5 file.")
        return 0.0
    if not retriever_query or not prompt:
        logger.error("`retriever_query` and `prompt` must be provided.")
        return 0.0

    # Set default ensemble configurations
    if ensemble_models is None:
        ensemble_models = [model]
    if ensemble_temperatures is None:
        ensemble_temperatures = [0.2, 0.5, 0.7]
    if ensemble_chunk_sizes is None:
        # Default chunk sizes with minimum of 2000
        ensemble_chunk_sizes = [2000, 3000, 4000]

    # Validate chunk sizes
    for chunk_size in ensemble_chunk_sizes:
        if chunk_size < 2000:
            logger.warning(
                f"Chunk size {chunk_size} is below minimum 2000, adjusting to 2000"
            )
            ensemble_chunk_sizes[ensemble_chunk_sizes.index(chunk_size)] = 2000

    # Writing prompting results to the HDF5 of the paper
    with h5py.File(paper, "a") as f:
        arxiv_id = f.filename.split("/")[-1].replace(".hdf5", "")
        text = f[arxiv_id]["arxiv_paper"]["text"][()].decode("utf-8")

        # Generate multiple chunk sets with different chunk sizes
        chunks_cache_group = f.require_group("chunks_cache")
        chunks_list = []

        for chunk_size in ensemble_chunk_sizes:
            chunker_params = {
                "chunk_size": chunk_size,
                "chunk_overlap": kwargs.get("chunk_overlap", 200),
            }

            # Compute hash for this chunking configuration
            chunker_hash = compute_chunker_hash(
                text=text, chunker_name=chunker, chunker_params=chunker_params
            )

            # Check cache or create new chunks
            if chunker_hash in chunks_cache_group:
                logger.info(f"Reusing chunks from cache with hash {chunker_hash}")
                cached_chunks_group = chunks_cache_group[chunker_hash]
                n_chunks = cached_chunks_group.attrs["n_chunks"]
                chunks = []
                for i in range(n_chunks):
                    chunk_content = cached_chunks_group[f"chunk_{i:04d}"][()].decode(
                        "utf-8"
                    )
                    chunk_source = cached_chunks_group.attrs.get(
                        "chunker", f"nerxiv.chunker.{chunker}"
                    )
                    chunks.append(
                        Document(
                            page_content=chunk_content,
                            metadata={"source": chunk_source},
                        )
                    )
            else:
                logger.info(f"Creating new chunks with size {chunk_size}")
                chunker_cls = _CHUNKER_MAP.get(chunker, Chunker)(text=text)
                chunks = chunker_cls.chunk_text(**chunker_params)

                # Store chunks in cache
                cached_chunks_group = chunks_cache_group.create_group(chunker_hash)
                cached_chunks_group.attrs["chunker"] = f"nerxiv.chunker.{chunker}"
                cached_chunks_group.attrs["n_chunks"] = len(chunks)
                cached_chunks_group.attrs["chunker_hash"] = chunker_hash
                cached_chunks_group.attrs["chunker_params"] = json.dumps(chunker_params)
                for i, chunk in enumerate(chunks):
                    cached_chunks_group.create_dataset(
                        f"chunk_{i:04d}", data=chunk.page_content.encode("utf-8")
                    )

            chunks_list.append(chunks)

        # Run retrieval on each chunk set and create retrieved text list
        retrieved_texts = []
        retrieval_cache_group = f.require_group("retrieval_cache")

        for i, chunks in enumerate(chunks_list):
            chunk_size = ensemble_chunk_sizes[i]
            chunker_params = {
                "chunk_size": chunk_size,
                "chunk_overlap": kwargs.get("chunk_overlap", 200),
            }
            chunker_hash = compute_chunker_hash(
                text=text, chunker_name=chunker, chunker_params=chunker_params
            )

            retriever_hash = compute_retriever_hash(
                chunker_hash=chunker_hash,
                retriever_model=retriever_model,
                retriever_query=retriever_query,
                n_top_chunks=n_top_chunks,
            )

            if retriever_hash in retrieval_cache_group:
                logger.info(
                    f"Reusing retrieval results from cache with hash {retriever_hash}"
                )
                cached_retrieval_group = retrieval_cache_group[retriever_hash]
                retrieved_text = cached_retrieval_group["retrieved_text"][()].decode(
                    "utf-8"
                )
            else:
                logger.info(f"Performing new retrieval for chunk size {chunk_size}")
                retriever = CustomRetriever(
                    model=retriever_model, query=retriever_query, logger=logger
                )
                retrieved_text = retriever.get_relevant_chunks(
                    chunks=chunks,
                    n_top_chunks=n_top_chunks,
                )

                # Store retrieval results in cache
                cached_retrieval_group = retrieval_cache_group.create_group(
                    retriever_hash
                )
                cached_retrieval_group.attrs["chunker_hash"] = chunker_hash
                cached_retrieval_group.attrs["retriever_model"] = retriever_model
                cached_retrieval_group.attrs["retriever_query"] = retriever_query
                cached_retrieval_group.attrs["n_top_chunks"] = n_top_chunks
                cached_retrieval_group.attrs["retriever_hash"] = retriever_hash
                cached_retrieval_group.create_dataset(
                    "retrieved_text", data=retrieved_text.encode("utf-8")
                )

            retrieved_texts.append(retrieved_text)

        # Use the first retrieved text as the base, but we'll pass different ones during ensemble
        # For simplicity, we cycle through retrieved_texts in the ensemble
        base_text = retrieved_texts[0]

        # Prepare chunks_list with different retrieved texts as Documents
        chunks_for_ensemble = [
            [Document(page_content=text, metadata={"source": "retrieval"})]
            for text in retrieved_texts
        ]

        # Run ensemble prompts
        combined_answer, averaged_json = run_ensemble_prompts(
            prompt=prompt,
            text=base_text,
            n_runs=n_ensemble_runs,
            models=ensemble_models,
            temperatures=ensemble_temperatures,
            chunks_list=chunks_for_ensemble,
            parallel=ensemble_parallel,
            averaging_model=averaging_model,
            averaging_temperature=averaging_temperature,
            model=model,
            **kwargs,
        )

        # Store results in HDF5
        raw_answer_group = f.require_group("raw_llm_answers")
        query_group = raw_answer_group.require_group(query)
        existing_runs = list(query_group.keys())
        run_id = f"run_{len(existing_runs):04d}"
        run_group = query_group.create_group(run_id)

        # Store metadata
        run_group.attrs["model"] = str(ensemble_models)
        run_group.attrs["query"] = query
        run_group.attrs["timestamp"] = datetime.datetime.now().isoformat()
        run_group.attrs["ensemble_mode"] = True
        run_group.attrs["n_ensemble_runs"] = n_ensemble_runs
        run_group.attrs["ensemble_temperatures"] = str(ensemble_temperatures)
        run_group.attrs["ensemble_chunk_sizes"] = str(ensemble_chunk_sizes)
        run_group.attrs["averaging_model"] = averaging_model
        run_group.attrs["averaging_temperature"] = averaging_temperature

        # Store combined answer (all runs)
        run_group.create_dataset("answer", data=combined_answer.encode("utf-8"))

        # Store averaged JSON if available (for StructuredPrompts)
        if averaged_json and isinstance(prompt, StructuredPrompt):
            averaged_json_str = json.dumps(averaged_json, indent=2)
            run_group.create_dataset(
                "averaged_json", data=averaged_json_str.encode("utf-8")
            )

        paper_time = time.time() - paper_time
        run_group.attrs["elapsed_time"] = paper_time

    return paper_time
