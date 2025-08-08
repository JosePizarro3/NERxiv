import datetime
import time
from pathlib import Path
from typing import TYPE_CHECKING

import h5py

from nerxiv.chunker import Chunker
from nerxiv.logger import logger
from nerxiv.rag import CustomRetriever, LLMGenerator

if TYPE_CHECKING:
    from structlog._config import BoundLoggerLazyProxy

    from nerxiv.prompts.prompts import Prompt


def run_prompt_paper(
    paper: Path,
    retriever_model: str = "all-MiniLM-L6-v2",
    n_top_chunks: int = 5,
    model: str = "gpt-oss:20b",
    retriever_query: str = "",
    prompt: Prompt | None = None,
    query: str = "material_formula",
    paper_time: float = 0.0,
    logger: "BoundLoggerLazyProxy" = logger,
) -> float:
    """Runs the prompt based on `retriever_query` and `template` on a given `paper`.

    Args:
        paper (Path): Path to the HDF5 file containing the paper data.
        retriever_model (str, optional): The model used in the retriever. Defaults to "all-MiniLM-L6-v2".
        n_top_chunks (int, optional): The number of top chunks to retrieve. Defaults to 5.
        model (_type_, optional): The model used in the generator. Defaults to "gpt-oss:20b".
        retriever_query (str, optional): The query used in the retriever. This is set using `query` and the `QUERY_REGISTRY`. Defaults to "".
        prompt (Prompt, optional): The prompt used in the generator. This is set using `query` and the `QUERY_REGISTRY`.. Defaults to None.
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
        chunker = Chunker(text=text)
        chunks = chunker.chunk_text()

        # Retrieval
        retriever = CustomRetriever(
            model=retriever_model, query=retriever_query, logger=logger
        )
        text = retriever.get_relevant_chunks(
            chunks=chunks,
            n_top_chunks=n_top_chunks,
        )

        # Generation
        generator = LLMGenerator(model=model, text=text, logger=logger)
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

        # Move hdf5 files to a model subfolder
        if answer == "model":
            target_dir = paper.parent / "model"
            target_dir.mkdir(exist_ok=True)
            target_path = target_dir / paper.name
            paper.rename(target_path)

        paper_time = time.time() - paper_time
        run_group.attrs["elapsed_time"] = paper_time
    return paper_time
