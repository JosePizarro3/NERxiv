import datetime
import time
from pathlib import Path
from typing import TYPE_CHECKING

import h5py

from nerxiv.chunker import _CHUNKER_MAP, Chunker
from nerxiv.logger import logger
from nerxiv.prompts.prompts import BasePrompt
from nerxiv.rag import CustomRetriever, LLMGenerator

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
    query: str = "material_formula",
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
        query (str, optional): The query used for retrieval and generation. See the registry in PROMPT_REGISTRY. Defaults to "material_formula".
        paper_time (float, optional): The starting time of this paper prompting. Defaults to 0.0.
        logger (BoundLoggerLazyProxy, optional): The logger to log messages. Defaults to logger.

    Returns:
        float: The time taken to run the prompt on the paper in seconds.
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

        # Chunking text
        chunker_cls = _CHUNKER_MAP.get(chunker, Chunker)(text=text)
        chunks = chunker_cls.chunk_text()

        # Retrieval
        retriever = CustomRetriever(
            model=retriever_model, query=retriever_query, logger=logger
        )
        text = retriever.get_relevant_chunks(
            chunks=chunks,
            n_top_chunks=n_top_chunks,
        )

        # Generation
        generator = LLMGenerator(model=model, text=text, logger=logger, **kwargs)
        built_prompt = prompt.build(text=text)
        answer = generator.generate(prompt=built_prompt)

        # Store raw answer in HDF5
        raw_answer_group = f.require_group("raw_llm_answers")
        # Auto-increment run ID
        existing_runs = list(raw_answer_group.keys())
        run_id = f"run_{len(existing_runs):04d}"
        run_group = raw_answer_group.create_group(run_id)
        # Store run metadata and answer
        run_group.attrs["retriever_model"] = retriever_model
        run_group.attrs["model"] = model
        run_group.attrs["n_top_chunks"] = n_top_chunks
        run_group.attrs["query"] = query
        run_group.attrs["timestamp"] = datetime.datetime.now().isoformat()
        query_group = run_group.require_group(query)
        query_group.create_dataset(
            "retriever_query", data=retriever_query.encode("utf-8")
        )
        query_group.create_dataset("prompt", data=built_prompt.encode("utf-8"))
        query_group.create_dataset("answer", data=answer.encode("utf-8"))
        # Store chunks and top-k chunks
        chunks_group = query_group.require_group("chunks")
        chunks_group.attrs["n_chunks"] = len(chunks)
        for i, chunk in enumerate(chunks):
            chunks_group.create_dataset(
                f"chunk_{i:04d}", data=chunk.page_content.encode("utf-8")
            )
            chunks_group.attrs["chunker"] = chunk.metadata.get("source")

        top_k_chunks = text.split("\n\n")
        chunks_group.attrs["n_top_k_chunks"] = len(top_k_chunks)
        for i, top_k_chunk in enumerate(top_k_chunks):
            chunks_group.create_dataset(
                f"top_k_chunk_{i:04d}", data=top_k_chunk.encode("utf-8")
            )

        paper_time = time.time() - paper_time
        run_group.attrs["elapsed_time"] = paper_time
    return paper_time
