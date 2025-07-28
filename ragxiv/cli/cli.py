import datetime
import time
from pathlib import Path

import click
import h5py

from ragxiv.chunker import Chunker
from ragxiv.logger import logger
from ragxiv.prompts import QUERY_REGISTRY
from ragxiv.prompts import prompt as prompt_template
from ragxiv.rag import CustomRetriever, LLMGenerator


@click.group(help="Entry point to run `pyrxiv` CLI commands.")
def cli():
    pass


@cli.command(
    name="prompt",
    help="Prompts the LLM with the text from the HDF5 file and stores the raw answer.",
)
@click.option(
    "--file-path",
    "-file",
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
    if not paper.exists():
        click.echo(f"File {file_path} does not exist.")
        return
    if not file_path.endswith(".hdf5"):
        click.echo(f"File {file_path} is not an HDF5 file.")
        return
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
        answer = generator.generate(prompt=prompt_template(template, text=text))

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
        query_group.create_dataset("template", data=template.encode("utf-8"))
        query_group.create_dataset("answer", data=answer.encode("utf-8"))

        # Move hdf5 files to a model subfolder
        if answer == "model":
            target_dir = paper.parent / "model"
            target_dir.mkdir(exist_ok=True)
            target_path = target_dir / paper.name
            paper.rename(target_path)

    elapsed_time = time.time() - start_time
    click.echo(f"Processed arXiv papers in {elapsed_time:.2f} seconds\n\n")


@cli.command(
    name="prompt_all",
    help="Prompts the LLM with the text from all the HDF5 file and stores the raw answer.",
)
@click.option(
    "--data-path",
    "-data",
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
def prompt_all(data_path, retriever_model, n_top_chunks, model, query):
    start_time = time.time()

    if query not in QUERY_REGISTRY:
        click.echo(
            f"Query '{query}' not found in registry. Available queries are: {list(QUERY_REGISTRY.keys())}"
        )
        return
    retriever_query, template = QUERY_REGISTRY.get(query)

    # list all papers `{data_path}/*.hdf5`
    papers = list(Path(data_path).rglob("*.hdf5"))
    for paper in papers:
        if not paper.exists():
            click.echo(f"File {paper} does not exist.")
            continue
        if not paper.name.endswith(".hdf5"):
            click.echo(f"File {paper} is not an HDF5 file.")
            continue

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
            answer = generator.generate(prompt=prompt_template(template, text=text))

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
            query_group.create_dataset("template", data=template.encode("utf-8"))
            query_group.create_dataset("answer", data=answer.encode("utf-8"))

            # Move hdf5 files to a model subfolder
            if answer == "model":
                target_dir = paper.parent / "model"
                target_dir.mkdir(exist_ok=True)
                target_path = target_dir / paper.name
                paper.rename(target_path)

    elapsed_time = time.time() - start_time
    click.echo(f"Processed arXiv papers in {elapsed_time:.2f} seconds\n\n")
