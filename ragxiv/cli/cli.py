import time
from pathlib import Path

import click
import h5py

from ragxiv.chunker import Chunker
from ragxiv.logger import logger
from ragxiv.prompts import QUERY_REGISTRY, prompt
from ragxiv.rag import CustomRetriever, LLMGenerator


@click.group(help="Entry point to run `pyrxiv` CLI commands.")
def cli():
    pass


@cli.command(
    name="process",
    help="Searchs papers in arXiv for a specified category and downloads them in a specified path.",
)
@click.option(
    "--data-path",
    "-path",
    type=str,
    default="data",
    required=False,
    help="""
    (Optional) The path containing the HDF5 files to be processed. Defaults to "data".
    """,
)
@click.option(
    "--retriever-model",
    "-rm",
    type=str,
    default="all-MiniLM-L6-v2",
    required=False,
    help="""
    (Optional) The model to use for the retriever. Defaults to "all-MiniLM-L6-v2".
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
    default="llama3.1:70b",
    required=False,
    help="""
    (Optional) The model to use for the generator. Defaults to "llama3.1:70b".
    """,
)
@click.option(
    "--query",
    "-q",
    type=str,
    default="material",
    required=False,
    help="""
    (Optional) The query to use for the retriever and generation. See the registry under `ragxiv/prompts/__init__.py`.
    Defaults to "material".
    """,
)
def process(data_path, retriever_model, n_top_chunks, model, query):
    start_time = time.time()

    if query not in QUERY_REGISTRY:
        click.echo(
            f"Query '{query}' not found in registry. Available queries are: {list(QUERY_REGISTRY.keys())}"
        )
        return
    retriever_query, template = QUERY_REGISTRY.get(query)

    # list all papers `data/*.hdf5`
    papers = list(Path(data_path).rglob("*.hdf5"))
    for paper in papers:
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
            answer = generator.generate(prompt=prompt(template, text=text))

    elapsed_time = time.time() - start_time
    click.echo(f"Processed arXiv papers in {elapsed_time:.2f} seconds\n\n")
