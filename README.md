<h4 align="center">

[![CI Status](https://github.com/JosePizarro3/RAGxiv/actions/workflows/actions.yml/badge.svg)](https://github.com/JosePizarro3/RAGxiv/actions/workflows/actions.yml/badge.svg)
[![Coverage](https://coveralls.io/repos/github/JosePizarro3/RAGxiv/badge.svg?branch=main)](https://coveralls.io/repos/github/JosePizarro3/RAGxiv/badge.svg?branch=main)
<!-- [![PyPI versions](https://img.shields.io/pypi/v/ragxiv)](https://img.shields.io/pypi/v/ragxiv) -->
<!-- [![Python supported versions](https://img.shields.io/pypi/pyversions/ragxiv)](https://img.shields.io/pypi/pyversions/ragxiv) -->

</h4>

# RAGxiv

`RAGxiv` is a Python package used for extracting structured metadata information from arXiv papers. Its original application is for Strongly Correlated Electron Systems in Condensed Matter Physics, whose category is [`"cond-mat.str-el"`](https://arxiv.org/list/cond-mat.str-el/recent), but it can be used for any other category.

If you want to install it, do:
```sh
pip install ragxiv
```

In order to include the CLI functionalities and tutorials, you have to add the optional `[dev]` and `[docu]` dependencies when pip installing the package:
```sh
pip install ragxiv[dev,docu]
```

## Development

If you want to develop locally this package, clone the project and enter in the workspace folder:

```sh
git clone https://github.com/JosePizarro3/RAGxiv.git
cd RAGxiv
```

Create a virtual environment (you can use Python>3.10) in your workspace:

```sh
python3 -m venv .venv
source .venv/bin/activate
```

And pip install the package in editable mode. We recommend using `uv` for fast pip installation:
```sh
pip install --upgrade pip
pip install uv
uv pip install -e .[dev,docu]
```

### Run the tests

You can locally run the tests by doing:

```sh
python -m pytest -sv tests
```

where the `-s` and `-v` options toggle the output verbosity.

You can also generate a local coverage report:

```sh
python -m pytest --cov=ragxiv tests
```

### Run auto-formatting and linting

We use [Ruff](https://docs.astral.sh/ruff/) for formatting and linting the code following the rules specified in the `pyproject.toml`. You can run locally:

```sh
ruff check .
```

This will produce an output with the specific issues found. In order to auto-fix them, run:

```sh
ruff format . --check
```

If some issues are not possible to fix automatically, you will need to visit the file and fix them by hand.

### Documentation on Github pages

To view the documentation locally, make sure to have installed the extra packages (note that you can skip this step if you already installed the `[docu]` dependencies for the package in the step before):

```sh
uv pip install -e '[docu]'
```

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

## Main contributors

The main code developers are:

| Name                | E-mail                                                       |
| ------------------- | ------------------------------------------------------------ |
| Dr. Jose M. Pizarro | [jose.pizarro-blanco@bam.de](mailto:jose.pizarro-blanco@bam.de) |
