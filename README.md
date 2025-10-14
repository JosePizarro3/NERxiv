<h1 align="center">
  <picture>
    <source srcset="https://github.com/JosePizarro3/NERxiv/raw/main/docs/assets/nerxiv_logo_name.png">
    <img src="https://github.com/JosePizarro3/NERxiv/raw/main/docs/assets/nerxiv_logo_name.png"
         alt="NERxiv logo"
         style="width: 25rem">
  </picture>
</h1>


<h4 align="center">

![CI](https://github.com/JosePizarro3/NERxiv/actions/workflows/actions.yml/badge.svg)
[![Coverage Status](https://coveralls.io/repos/github/JosePizarro3/NERxiv/badge.svg?branch=main&nocache=1)](https://coveralls.io/github/JosePizarro3/NERxiv?branch=main)
[![License: PolyForm NC 1.0.0](https://img.shields.io/badge/license-PolyForm_NC_1.0.0-orange.svg)](LICENSE)
[![PyPI version](https://img.shields.io/pypi/v/nerxiv.svg)](https://img.shields.io/pypi/v/nerxiv.svg)
[![Python versions](https://img.shields.io/pypi/pyversions/nerxiv.svg)](https://img.shields.io/pypi/pyversions/nerxiv.svg)
<!--[![Commercial License Available](https://img.shields.io/badge/commercial-license-green.svg)](COMMERCIAL-LICENSE.md)-->

</h4>

# NERxiv

**N**amed **E**ntity **R**ecognition for ar**xiv** papers (**NERxiv**) is a Python wrapper tool for extracting **structured metadata** from scientific papers on [arXiv](https://arxiv.org) using **LLMs** and modern **retrieval-augmented generation (RAG)** techniques.

Visit the [documentation page](https://JosePizarro3.github.io/NERxiv/) to learn how to use this tool.

## What It Does

* Uses [`pyrxiv`](https://pypi.org/project/pyrxiv/) to fetch, download, and extract text from arXiv papers
* Chunks and embeds text with SentenceTransformers or LangChain to categorize papers content using local LLMs (via Ollama)
* Includes CLI tools and notebook tutorials for reproducible workflows

---

## Installation

Install the core package:
```bash
pip install nerxiv
```

## Running LLMs Locally

We recommend running your own models locally using [Ollama](https://ollama.com/download):
```bash
# Install Ollama (follow instructions on their website)
ollama pull <model-name>   # e.g., llama3, deepseek-r1, qwen3:30b

# Start the local server
ollama serve
```


---

# Development

To contribute to `NERxiv` or run it locally, follow these steps:


## Clone the Repository

```bash
git clone https://github.com/JosePizarro3/NERxiv.git
cd NERxiv
```

## Set Up a Virtual Environment

We recommend Python ≥ 3.10:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

## Install Dependencies

Use [`uv`](https://docs.astral.sh/uv/) (faster than pip) to install the package in editable mode with `dev` and `docu` extras:
```bash
pip install --upgrade pip
pip install uv
uv pip install -e .[dev,docu]
```

## Run tests

Use `pytest` with verbosity to run all tests:
```bash
python -m pytest -sv tests
```


To check code coverage:
```bash
python -m pytest --cov=nerxiv tests
```

### Code formatting and linting


We use [`Ruff`](https://docs.astral.sh/ruff/) for formatting and linting (configured via `pyproject.toml`).

Check linting issues:
```bash
ruff check .
```

Auto-format code:
```bash
ruff format . --check
```

Manually fix anything Ruff cannot handle automatically.

### Documentation writing

To view the documentation locally, make sure to have installed the extra `[docu]` packages:

```sh
uv pip install -e '[docu]'
```

**Note**: This command installs `mkdocs`, `mkdocs-material`, and other documentation-related dependencies.

The first time, build the server:

```sh
mkdocs build
```

Run the documentation server:

```sh
mkdocs serve
```

The output looks like:

```sh
INFO    -  Building documentation...
INFO    -  Cleaning site directory
INFO    -  [14:07:47] Watching paths for changes: 'docs', 'mkdocs.yml'
INFO    -  [14:07:47] Serving on http://127.0.0.1:8000/
```

Simply click on `http://127.0.0.1:8000/`. The changes in the `md` files of the documentation are immediately reflected when the files are saved (the local web will automatically refresh).
