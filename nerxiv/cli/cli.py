import re
import time
from pathlib import Path

import click

from nerxiv.logger import logger
from nerxiv.prompts import PROMPT_REGISTRY

from .run_prompt import run_prompt_paper


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


@click.group(help="Entry point to run `pyrxiv` CLI commands.")
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
    help="""
    The path to the HDF5 file used to prompt the LLM.
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
    default="material_formula",
    required=False,
    help="""
    (Optional) The query used for retrieval and generation. See the registry PROMPT_REGISTRY. Defaults to "material_formula".
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
def prompt(file_path, chunker, retriever_model, n_top_chunks, model, query, llm_option):
    start_time = time.time()

    if query not in PROMPT_REGISTRY:
        click.echo(
            f"Query '{query}' not found in registry. Available queries are: {list(PROMPT_REGISTRY.keys())}"
        )
        return
    entry = PROMPT_REGISTRY[query]
    retriever_query = entry.retriever_query
    prompt = entry.prompt

    # Parse key=value options into dict
    llm_kwargs = parse_llm_option_to_args(llm_option)

    # Transform to Path and get the hdf5 data
    paper = Path(file_path)
    paper_time = run_prompt_paper(
        paper=paper,
        chunker=chunker,
        retriever_model=retriever_model,
        n_top_chunks=n_top_chunks,
        model=model,
        retriever_query=retriever_query,
        prompt=prompt,
        query=query,
        paper_time=start_time,
        logger=logger,
        **llm_kwargs,
    )
    click.echo(f"Processed arXiv papers in {paper_time:.2f} seconds\n\n")


@cli.command(
    name="prompt_all",
    help="Prompts the LLM with the text from all the HDF5 file and stores the raw answer.",
)
@click.option(
    "--data-path",
    "-path",
    type=str,
    default="./data",
    required=False,
    help="""
    (Optional) The path to folder containing all the HDF5 file used to prompt the LLM.
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
    default="material_formula",
    required=False,
    help="""
    (Optional) The query used for retrieval and generation. See the registry in PROMPT_REGISTRY. Defaults to "material_formula".
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
def prompt_all(
    data_path, chunker, retriever_model, n_top_chunks, model, query, llm_option
):
    start_time = time.time()
    paper_time = start_time

    if query not in PROMPT_REGISTRY:
        click.echo(
            f"Query '{query}' not found in registry. Available queries are: {list(PROMPT_REGISTRY.keys())}"
        )
        return
    entry = PROMPT_REGISTRY[query]
    retriever_query = entry.retriever_query
    prompt = entry.prompt

    # Parse key=value options into dict
    llm_kwargs = parse_llm_option_to_args(llm_option)

    # list all papers `{data_path}/*.hdf5`
    papers = list(Path(data_path).rglob("*.hdf5"))
    for paper in papers:
        paper_time = run_prompt_paper(
            paper=paper,
            chunker=chunker,
            retriever_model=retriever_model,
            n_top_chunks=n_top_chunks,
            model=model,
            retriever_query=retriever_query,
            prompt=prompt,
            query=query,
            paper_time=paper_time,
            logger=logger,
            **llm_kwargs,
        )

    elapsed_time = time.time() - start_time
    click.echo(f"Processed arXiv papers in {elapsed_time:.2f} seconds\n\n")
