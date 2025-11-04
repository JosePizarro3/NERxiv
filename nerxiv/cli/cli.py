import re
import time
from pathlib import Path

import click
import h5py

from nerxiv.chunker import _CHUNKER_MAP
from nerxiv.logger import logger
from nerxiv.prompts import PROMPT_REGISTRY
from nerxiv.rag import CustomRetriever, LLMGenerator, RAGExtractorAgent


def parse_llm_option_to_args(llm_option: tuple[str]) -> dict:
    """
    Parses a list of key=value strings from `llm_option` into a dictionary.

    Example:
        ("temperature=0.7", "num_ctx=8192", "reasoning=true", "base_url=https://api.openai.com/v1")
        -> {"temperature": 0.7, "num_ctx": 8192, "reasoning": True, "base_url": "https://api.openai.com/v1"}


    Args:
        llm_option (list[str]): List of key=value strings.

    Returns:
        dict: Dictionary of parsed key-value pairs.
    """
    llm_kwargs = {}
    for option in llm_option:
        if "=" not in option:
            click.echo(f"Invalid --llm-option format: {option}. Use key=value.")
            continue
        key, value = option.split("=", 1)
        value = value.strip()
        try:
            # Attempt to cast to int/float/bool if possible
            if value.lower() in {"true", "false"}:
                value = value.lower() == "true"
            elif re.fullmatch(r"[-+]?\d*\.\d+", value):
                value = float(value)
            elif re.fullmatch(r"\d+", value):
                value = int(value)
            elif value.lower() == "none":
                value = None
            elif value in {"''", '""'}:
                value = ""
        except Exception:
            continue
        llm_kwargs[key] = value
    return llm_kwargs


def get_chunker_cls(chunker_name: str, logger) -> type | None:
    """Get the chunker class from its name.

    Args:
        chunker_name (str): The name of the chunker class.
        logger: Logger to log messages.

    Returns:
        type | None: The chunker class.
    """
    if chunker_name not in _CHUNKER_MAP:
        available = list(_CHUNKER_MAP.keys())
        logger.error(
            f"Unknown chunker: '{chunker_name}'. Available chunkers: {available}"
        )
        return None
    return _CHUNKER_MAP[chunker_name]


@click.group(help="Entry point to run `nerxiv` CLI commands.")
def cli():
    pass


@cli.command(
    name="prompt",
    help="Prompts the LLM with the text from the HDF5 file and stores the raw answer.",
)
@click.option(
    "--file-path",
    "-path",
    type=str,
    required=True,
    multiple=True,
    help="""
    The path to the HDF5 file or files used to prompt the LLM.
    """,
)
@click.option(
    "--chunker",
    "-ch",
    type=str,
    default="Chunker",
    required=False,
    help="""
    (Optional) The chunker class to use for chunking the text. Defaults to `Chunker`.
    Options are: `Chunker`, `SemanticChunker`, `AdvancedSemanticChunker`.
    """,
)
@click.option(
    "--retriever-model",
    "-rm",
    type=str,
    default="all-MiniLM-L6-v2",
    required=False,
    help="""
    (Optional) The model used in the retriever. Defaults to "all-MiniLM-L6-v2".
    """,
)
@click.option(
    "--n-top-chunks",
    "-ntc",
    type=int,
    default=5,
    required=False,
    help="""
    (Optional) The number of top chunks to retrieve. Defaults to 5.
    """,
)
@click.option(
    "--model",
    "-m",
    type=str,
    default="gpt-oss:20b",
    required=False,
    help="""
    (Optional) The model used in the generator. Defaults to "gpt-oss:20b".
    """,
)
@click.option(
    "--query",
    "-q",
    type=str,
    default="filter_material_formula",
    required=False,
    help="""
    (Optional) The query used for retrieval and generation. See the registry PROMPT_REGISTRY. Defaults to "filter_material_formula".
    """,
)
@click.option(
    "--llm-option",
    "-llmo",
    multiple=True,
    type=str,
    required=False,
    help="""
    (Optional) key=value pairs for OllamaLLM parameters (e.g. -llmo temperature=0.2 -llmo top_p=0.9).
    """,
)
@click.option(
    "--chunker-option",
    "-cho",
    multiple=True,
    type=str,
    required=False,
    help="""
    (Optional) key=value pairs for chunker parameters.
    Examples: -cho chunk_size=500 -cho chunk_overlap=100 for Chunker,
    or -cho n_chunks=15 for AdvancedSemanticChunker.
    """,
)
def prompt(
    file_path,
    chunker,
    retriever_model,
    n_top_chunks,
    model,
    query,
    llm_option,
    chunker_option,
):
    start_time = time.time()

    # Get prompt and retriever query from registry
    if query not in PROMPT_REGISTRY:
        click.echo(
            f"Query '{query}' not found in registry. Available queries are: {list(PROMPT_REGISTRY.keys())}"
        )
        return
    entry = PROMPT_REGISTRY[query]
    retriever_query = entry.retriever_query
    prompt = entry.prompt

    # Parse key=value options into dict and pass everything to dictionaries for parameters of Chunker, Retriever and Generator
    chunker_params = parse_llm_option_to_args(chunker_option)
    retriever_params = {
        "model": retriever_model,
        "n_top_chunks": n_top_chunks,
        "query": retriever_query,
        "query_name": query,
    }
    llm_kwargs = parse_llm_option_to_args(llm_option)
    generator_params = {
        "model": model,
        **llm_kwargs,
    }

    # Get chunker class
    chunker_cls = get_chunker_cls(chunker_name=chunker, logger=logger)

    # Create RAGExtractorAgent and run for each paper specified in `file_path`
    agent = RAGExtractorAgent(
        chunker=chunker_cls,
        retriever=CustomRetriever,
        generator=LLMGenerator,
        chunker_params=chunker_params,
        retriever_params=retriever_params,
        generator_params=generator_params,
        logger=logger,
    )
    for file in file_path:
        paper = Path(file)
        with h5py.File(paper, "a") as f:
            arxiv_id = f.filename.split("/")[-1].replace(".hdf5", "")
            text = f[arxiv_id]["arxiv_paper"]["text"][()].decode("utf-8")
            agent.run(file=f, text=text, prompt=prompt)

    logger.info(
        f"Processed {len(file_path)} arXiv papers in {time.time() - start_time:.2f} seconds\n\n"
    )


# @cli.command(
#     name="prompt_all",
#     help="Prompts the LLM with the text from all the HDF5 file and stores the raw answer.",
# )
# @click.option(
#     "--data-path",
#     "-path",
#     type=str,
#     default="./data",
#     required=False,
#     help="""
#     (Optional) The path to folder containing all the HDF5 file used to prompt the LLM.
#     """,
# )
# @click.option(
#     "--chunker",
#     "-ch",
#     type=str,
#     default="Chunker",
#     required=False,
#     help="""
#     (Optional) The chunker class to use for chunking the text. Defaults to `Chunker`.
#     Options are: `Chunker`, `SemanticChunker`, `AdvancedSemanticChunker`.
#     """,
# )
# @click.option(
#     "--retriever-model",
#     "-rm",
#     type=str,
#     default="all-MiniLM-L6-v2",
#     required=False,
#     help="""
#     (Optional) The model used in the retriever. Defaults to "all-MiniLM-L6-v2".
#     """,
# )
# @click.option(
#     "--n-top-chunks",
#     "-ntc",
#     type=int,
#     default=5,
#     required=False,
#     help="""
#     (Optional) The number of top chunks to retrieve. Defaults to 5.
#     """,
# )
# @click.option(
#     "--model",
#     "-m",
#     type=str,
#     default="gpt-oss:20b",
#     required=False,
#     help="""
#     (Optional) The model used in the generator. Defaults to "gpt-oss:20b".
#     """,
# )
# @click.option(
#     "--query",
#     "-q",
#     type=str,
#     default="filter_material_formula",
#     required=False,
#     help="""
#     (Optional) The query used for retrieval and generation. See the registry in PROMPT_REGISTRY. Defaults to "filter_material_formula".
#     """,
# )
# @click.option(
#     "--llm-option",
#     "-llmo",
#     multiple=True,
#     type=str,
#     required=False,
#     help="""
#     (Optional) key=value pairs for OllamaLLM parameters (e.g. -llmo temperature=0.2 -llmo top_p=0.9).
#     """,
# )
# @click.option(
#     "--chunker-option",
#     "-cho",
#     multiple=True,
#     type=str,
#     required=False,
#     help="""
#     (Optional) key=value pairs for chunker parameters.
#     Examples: -cho chunk_size=500 -cho chunk_overlap=100 for Chunker,
#     or -cho n_chunks=15 for AdvancedSemanticChunker.
#     """,
# )
# def prompt_all(
#     data_path,
#     chunker,
#     retriever_model,
#     n_top_chunks,
#     model,
#     query,
#     llm_option,
#     chunker_option,
# ):
#     start_time = time.time()
#     paper_time = start_time

#     if query not in PROMPT_REGISTRY:
#         click.echo(
#             f"Query '{query}' not found in registry. Available queries are: {list(PROMPT_REGISTRY.keys())}"
#         )
#         return
#     entry = PROMPT_REGISTRY[query]
#     retriever_query = entry.retriever_query
#     prompt = entry.prompt

#     # Parse key=value options into dict
#     llm_kwargs = parse_llm_option_to_args(llm_option)
#     chunker_kwargs = parse_llm_option_to_args(chunker_option)

#     # list all papers `{data_path}/*.hdf5`
#     papers = list(Path(data_path).rglob("*.hdf5"))
#     for paper in papers:
#         paper_time = run_prompt_paper(
#             paper=paper,
#             chunker=chunker,
#             retriever_model=retriever_model,
#             n_top_chunks=n_top_chunks,
#             model=model,
#             retriever_query=retriever_query,
#             prompt=prompt,
#             query=query,
#             paper_time=paper_time,
#             logger=logger,
#             **chunker_kwargs,
#             **llm_kwargs,
#         )

#     elapsed_time = time.time() - start_time
#     click.echo(f"Processed arXiv papers in {elapsed_time:.2f} seconds\n\n")
