import time
from pathlib import Path

import click

from nerxiv.logger import logger
from nerxiv.prompts import QUERY_REGISTRY

from .prompt import run_prompt_paper


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
    default="material",
    required=False,
    help="""
    (Optional) The query used for retrieval and generation. See the registry in `nerxiv/prompts/__init__.py`. Defaults to "material".
    """,
)
def prompt(file_path, retriever_model, n_top_chunks, model, query):
    start_time = time.time()

    if query not in QUERY_REGISTRY:
        click.echo(
            f"Query '{query}' not found in registry. Available queries are: {list(QUERY_REGISTRY.keys())}"
        )
        return
    retriever_query, template = QUERY_REGISTRY.get(query)

    # Transform to Path and get the hdf5 data
    paper = Path(file_path)
    paper_time = run_prompt_paper(
        paper=paper,
        retriever_model=retriever_model,
        n_top_chunks=n_top_chunks,
        model=model,
        retriever_query=retriever_query,
        template=template,
        query=query,
        paper_time=start_time,
        logger=logger,
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
    default="material",
    required=False,
    help="""
    (Optional) The query used for retrieval and generation. See the registry in `nerxiv/prompts/__init__.py`. Defaults to "material".
    """,
)
def prompt_all(data_path, retriever_model, n_top_chunks, model, query):
    start_time = time.time()
    paper_time = start_time

    if query not in QUERY_REGISTRY:
        click.echo(
            f"Query '{query}' not found in registry. Available queries are: {list(QUERY_REGISTRY.keys())}"
        )
        return
    retriever_query, template = QUERY_REGISTRY.get(query)

    # list all papers `{data_path}/*.hdf5`
    papers = list(Path(data_path).rglob("*.hdf5"))
    for paper in papers:
        paper_time = run_prompt_paper(
            paper=paper,
            retriever_model=retriever_model,
            n_top_chunks=n_top_chunks,
            model=model,
            retriever_query=retriever_query,
            template=template,
            query=query,
            paper_time=paper_time,
            logger=logger,
        )

    elapsed_time = time.time() - start_time
    click.echo(f"Processed arXiv papers in {elapsed_time:.2f} seconds\n\n")
