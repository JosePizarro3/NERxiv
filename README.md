<h4 align="center">

![CI](https://github.com/JosePizarro3/RAGxiv/actions/workflows/actions.yml/badge.svg)
![Coverage](https://coveralls.io/repos/github/JosePizarro3/RAGxiv/badge.svg?branch=main)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
<!-- [![PyPI version](https://img.shields.io/pypi/v/ragxiv.svg)]
[![Python versions](https://img.shields.io/pypi/pyversions/ragxiv.svg)] -->

</h4>

# RAGxiv


**RAGxiv** is a Python package for extracting **structured metadata** from scientific papers on [arXiv](https://arxiv.org) using **LLMs** and modern **retrieval-augmented generation (RAG)** techniques.
While originally developed for the **Strongly Correlated Electron Systems** community in Condensed Matter Physics ([`cond-mat.str-el`](https://arxiv.org/list/cond-mat.str-el/recent)), it's designed to be flexible and applicable to **any arXiv category**.

## What It Does

* Uses [`pyrxiv`](https://pypi.org/project/pyrxiv/) to fetch, download, and extract text from arXiv papers
* Chunks and embeds text with SentenceTransformers or LangChain to categorize papers content using local LLMs (via Ollama)
* Includes CLI tools and notebook tutorials for reproducible workflows

---

## Installation

Install the core package:
```bash
pip install ragxiv
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

To contribute to `RAGxiv` or run it locally, follow these steps:


## Clone the Repository

```bash
git clone https://github.com/JosePizarro3/RAGxiv.git
cd RAGxiv
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
python -m pytest --cov=ragxiv tests
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
