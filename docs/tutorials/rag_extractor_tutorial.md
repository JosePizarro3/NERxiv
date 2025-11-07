# Using the RAG Extractor Agent

This tutorial will guide you through using NERxiv's RAG (Retrieval-Augmented Generation) extractor agent to extract structured metadata from scientific papers. The RAG agent combines text chunking, semantic retrieval, and LLM-based generation to intelligently extract information in JSON format from arXiv papers.

The RAG extractor agent is a three-stage pipeline that:

1. **Chunks** the paper text into smaller, manageable pieces
2. **Retrieves** the most relevant chunks based on a query
3. **Generates** structured JSON answers using an LLM model

<div class="click-zoom">
    <label>
        <input type="checkbox">
        <img src="../../assets/drawio/rag_simplified.drawio.png" alt="NERxiv pipeline simplified." width="80%" title="Click to zoom in">
    </label>
</div>

!!! note "Prerequisites"
    - **Python ≥ 3.10** installed
    - A virtual environment with `nerxiv` installed
    - Downloaded and set up [Ollama](https://ollama.com/download) for running LLMs locally
    - At least one LLM model pulled: `ollama pull gpt-oss:20b` (or your preferred model)
    - An HDF5 file containing extracted paper text using [`pyrxiv`](https://github.com/JosePizarro3/pyrxiv) (see the [How to Use `pyrxiv`](https://github.com/JosePizarro3/pyrxiv/blob/main/docs/how_to_use_pyrxiv.md) documentation)


!!! example "Notebook example"
    We prepared a notebook example in [tutorials/rag_extractor_tutorial.ipynb](https://github.com/JosePizarro3/NERxiv/blob/main/tutorials/rag_extractor_tutorial.ipynb) following the same steps. For [`marimo`](https://marimo.io/) users, we also have the same tutorial in [tutorials/rag_extractor_tutorial_mo.py](https://github.com/JosePizarro3/NERxiv/blob/main/tutorials/rag_extractor_tutorial_mo.py)

## Installation and Setup

### Create an empty test directory

We will test the `nerxiv` functionalities in an empty directory. Open your terminal and type:
```bash
mkdir test_nerxiv
cd test_nerxiv/
```

??? question "Paths in Windows"
    The commands written here are all done in Ubuntu. For Windows, please, change the paths accordingly

### Create a Virtual Environment

We strongly recommend using a virtual environment to avoid conflicts with other packages.

**Using venv:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Using conda:**
```bash
conda create --name .venv python=3.10  # or any version 3.10 <= python <= 3.12
conda activate .venv
```

### Install the Package

`nerxiv` is part of the PyPI registry and can be installed via pip:
```bash
pip install --upgrade pip
pip install nerxiv
```

!!! tip "Faster Installation"
    For faster installation, you can use [`uv`](https://docs.astral.sh/uv/):
    ```bash
    pip install uv
    uv pip install nerxiv
    ```

In case of running the Jupyter notebook or Marimo tutorials, install the corresponding dependencies as well.

### Verify Installation

You can verify that the installation was successful by opening the terminal and typing:
```bash
nerxiv --help
```

If everything was successful, you will see the usage of the CLI:
```sh
Usage: nerxiv [OPTIONS] COMMAND [ARGS]...

  Entry point to run `nerxiv` CLI commands.

Options:
  --help  Show this message and exit.

Commands:
  prompt      Prompts the LLM with the text from the HDF5 file and stores...
  prompt_all  Prompts the LLM with the text from all the HDF5 file and...
```

### Ollama servers

Whenever you want to use the `RAGExtractorAgent` and run prompting, you need an Ollama server running constantly. This can be done by opening a new terminal window and running:

```bash
ollama serve
```

The `RAGExtractorAgent` will then take care of invoking the LLM and prompting it for results.

## Basic Usage

The simplest way to use the RAG extractor is through the CLI `prompt` command. Open a terminal and type:

```bash
nerxiv prompt --file-path /path/to/paper.hdf5
```

This will:

- Use the default `Chunker` to split the text
- Use the default retriever model (`all-MiniLM-L6-v2`)
- Retrieve the top 5 most relevant chunks
- Use the default LLM model (`gpt-oss:20b`)
- Execute the default query (`filter_material_formula`) to extract material formulas

## Understanding the Pipeline

### Step 1: Chunking

The chunker divides the paper text into smaller pieces. NERxiv provides three chunking strategies:

- **`Chunker`**: Fixed-size chunks with overlap (default: 1000 characters, 200 overlap)
- **`SemanticChunker`**: Sentence-level semantic chunking using [spaCy](https://pypi.org/project/spacy/)
- **`AdvancedSemanticChunker`**: KMeans-based clustering on sentence embeddings

Example with semantic chunking:

```bash
nerxiv prompt --file-path paper.hdf5 --chunker SemanticChunker
```

See [API References](../references/api.md#nerxiv.chunker) for more details on these classes.

### Step 2: Retrieval

The retriever uses a sentence transformer model to:

1. Encode the retrieval query and all chunks into embeddings. The retrieval query is defined in [`nerxiv.prompts.prompt_registry.py`](https://github.com/JosePizarro3/NERxiv/blob/main/nerxiv/prompts/prompts_registry.py) in the `PROMPT_REGISTRY` variable (see below)
2. Compute cosine similarity between the query and each chunk
3. Return the top-k most relevant chunks relative to the retrieval query

The default retriever model is `all-MiniLM-L6-v2` from SentenceTransformers, but you can specify others:

```bash
nerxiv prompt --file-path paper.hdf5 --retriever-model all-mpnet-base-v2
```

You can also adjust how many top-k chunks to retrieve:

```bash
nerxiv prompt --file-path paper.hdf5 --n-top-chunks 10
```

### Step 3: Generation

The LLM generator takes the retrieved chunks and answers your query using a carefully crafted prompt. The answer is structured according to the query type defined in the `PROMPT_REGISTRY`.

See [Programmatically Running the `RAGExtractorAgent`](#programmatically-running-the-ragextractoragent) for more details about the `PROMPT_REGISTRY` and generation.

## Using Different Queries

NERxiv comes with predefined queries in the `PROMPT_REGISTRY`. Each query has:

- A **retriever query**: Guides what content to retrieve
- A **prompt template**: Instructs the LLM on what to extract

Available queries include:

- `filter_material_formula`: Filters papers which are done over a real material chemical formula or a simplified model.
- `filter_only_dmft`: Filters papers by checking if Dynamical Mean-Field Theory (DMFT) methodology is used.
- `dft`: Returns structured Density Functional Theory (DFT) schema populated.


Example:

```bash
nerxiv prompt --file-path paper.hdf5 --query filter_only_dmft
```

All these are use-case examples which are probably not applicable to your case. If you learn how to define your own custom structured prompts, go to [Programmatically Running the `RAGExtractorAgent`](#programmatically-running-the-ragextractoragent) and [How-to: Create Custom Prompts](../howtos/create_custom_prompts.md).

## Configuring LLM Parameters

The LLM behavior is controled using `--llm-option` (or `-llmo`) flags. These are corresponding to the inputs of [`OllamaLLM`](https://python.langchain.com/api_reference/ollama/llms/langchain_ollama.llms.OllamaLLM.html) and are passed as `key=value` pairs:

```bash
nerxiv prompt \
  --file-path paper.hdf5 \
  --model llama3.1:70b \
  -llmo temperature=0.2 \
  -llmo top_p=0.9 \
  -llmo num_ctx=8192
```

Common LLM parameters:

- `temperature`: Controls randomness (0.0 = deterministic, 1.0 = creative)
- `top_p`: Nucleus sampling threshold
- `num_ctx`: Context window size

## Complete Example

Here's a complete example extracting material formulas with custom settings:

```bash
nerxiv prompt \
  --file-path ./data/papers/2502.12144v1.hdf5 \
  --chunker AdvancedSemanticChunker \
  --retriever-model all-mpnet-base-v2 \
  --n-top-chunks 8 \
  --model llama3.1:70b \
  --query dft \
  -llmo temperature=0.1 \
  -llmo num_ctx=16384
```

This command:

1. Uses advanced semantic chunking with KMeans clustering
2. Uses a more powerful retriever model
3. Retrieves the top 8 most relevant chunks
4. Uses the 70B parameter Llama model
5. Sets low temperature for consistent outputs
6. Expands the context window to 16K tokens

## Processing Multiple Papers

To process all papers in a directory, use the `prompt_all` command:

```bash
nerxiv prompt_all \
  --data-path /directory/containing/the/papers/ \
  --query dft \
  --model llama3.1:70b
```

This will process all `.hdf5` files in the specified directory with the same configuration.

## Output Storage

The RAG extractor stores results directly in the HDF5 file under the `rag_extraction` group. This group contains 3 sub-groups:

- `chunks_cache`: a local group database for the chunks
- `retrieval_cache`: a local group database for the retrieved top-k chunks
- `raw_llm_answers`: the generated LLM answers

Under the `raw_llm_answers` group, a new group is created with the name of the query/prompt you want to do. For example, if you run:

```bash
nerxiv prompt --file-path paper.hdf5 --query filter_only_dmft
```

This will create a group `filter_only_dmft` under `raw_llm_answers`. Then for each run of that specific prompt, we define a `run_XXXX`. The number of group goes from `run_0001` incrementing depending on how many times you run that prompt. The combined `pyrxiv`+`NERxiv` HDF5 groups diagram is:

```sh
arxiv_paper_hdf5
├── arxiv_id
│   └── arxiv_paper
│       ├── authors
│       ├── categories
│       └── text
└── rag_extraction
    ├── chunks_cache
    │       ├── b540959gnis... (hash)
    │       ├── ff48418kpd1...
    │       └── ...
    ├── retrieval_cache
    │       ├── p098na87bnb...
    │       ├── f5sn901nx01...
    │       └── ...
    └── raw_llm_answers
        ├── dft
        │   ├── run_0001
        │   │   ├── answer
        │   │   └── prompt
        │   ├── run_0002
        │   │   ├── answer
        │   │   └── prompt
        │   └── ...
        ├── another_query
        │   ├── run_0001
        │   │   ├── ...
        │   └── ...
        └── ...
```

You can inspect the results by opening the HDF5 file with any HDF5 viewer (e.g., [HDFView](https://www.hdfgroup.org/download-hdfview/)) or using Python:

```python
import h5py

with h5py.File("path/to/paper.hdf5", "r") as f:
    raw_llm_answers = f["rag_extraction"]["raw_llm_answers"]
    # List all runs
    runs = list(raw_llm_answers.keys())

    # Access the latest run
    latest_run = raw_llm_answers[runs[-1]]

    # Read the answer
    answer = latest_run["answer"][()].decode("utf-8")
    print(answer)
```


## Programmatically Running the `RAGExtractorAgent`

If instead, you want to run the `RAGExtractorAgent`, here we explain the steps needed to be done. You can also run and modify the [tutorials](#using-the-rag-extractor-agent) we mentioned at the beginning of this documentation page.

In your folder, create three files: `run_script.py`, `datamodel.py`, and `prompt_registry.py`.
- `run_script.py`: this script will contain the running calls for the agent and necessary logic behind it.
- `datamodel.py`: this module will contain the [pydantic](https://docs.pydantic.dev/latest/) model definitions needed to extract metadata from an arXiv paper and validate it.
- `prompt_registry.py`: this registry will contain the prompts needed to extract the workflow we are targetting in our paper.

For the sake of this example, we recommend choosing any recent paper from the [cond-mat.str-el](https://arxiv.org/list/cond-mat.str-el/recent) category on arXiv. You can search and download the corresponding HDF5 needed for NERxiv using `pyrxiv`:

```bash
pyrxiv search_and_download --save-hdf5 --category cond-mat.str-el --start-id "2505.21995v2" --n-papers 1
```

This will create a `data/` folder in your directory and store the corresponding PDF and HDF5 paper in there.

### Define a Data Model

We will simply try to extract Density Functional Theory (DFT) metadata in an oversimplified way. For this, we will create a pydantic model class `DFT` and add a couple of fields. In `datamodel.py`:

```python
from pydantic import Field

from nerxiv.datamodel.base_section import BaseSection


class DFT(BaseSection):
    """
    Section representing the Density Functional Theory (DFT) parameters used in the simulation
    of a material. This includes information about the computational code, exchange-correlation
    functional, basis set, pseudopotentials, cutoffs, k-point sampling, relativistic treatment,
    and spin-orbit coupling. Intended to capture the setup of DFT calculations as reported in
    computational materials science papers.
    """

    code_name: str | None = Field(
        None,
        description="""
        Name of the DFT software/code used. For example, 'VASP', 'Quantum ESPRESSO', 'FP-LMTO'.
        """,
    )

    code_version: str | None = Field(
        None,
        description="""
        Version of the DFT code. For example, '6.7', '7.3.2'.
        """,
    )
```

**Notes**:
- Using `BaseSection` from NERxiv is totally optional. You can instead directly use `BaseModel` from pydantic.
- Be descriptive without overexplaining what is each class and field.
- Add a default `None` for all fields to avoid problems when the LLM validates results. This is based on the fact that, 1) not all papers contain all metadata fields in their text, and 2) the agent might fail extracting some field.
- Adding examples in the description of fields is totally optional. But it helps the agent to format better the output.

### Adding Structured Prompts to the `PROMPT_REGISTRY`

With the `DFT` model defined in the previous section, we can now define the structured prompt we will use to extract structured metadata in JSON format and adapted to this model. In `prompt_registry.py`:

```python
from nerxiv.prompts.prompts import (
    PromptRegistryEntry,
    StructuredPrompt,
    PROMPT_REGISTRY
)
from .datamodel import DFT


new_entry = PromptRegistryEntry(
  retriever_query="""Identify all mentions of Density Functional Theory (DFT) calculations,
  defined as any description of electronic-structure computations within the Kohn-Sham
  formalism, including the chosen exchange-correlation functional, computational code, basis
  set, pseudopotential, convergence parameters, or spin treatment. Include any statements
  about how the DFT calculation was performed, validated, or referenced from prior work.""",
  prompt=StructuredPrompt(
      expert="Condensed Matter Physics",
      output_schema=DFT,
      target_fields=["all"],
      constraints=[
          "Return ONLY the requested JSON object without any additional text or explanation.",
          "If you do NOT find the value of a field in the text, do NOT make up a value. Leave it as null in the JSON output.",
          "Do NOT infer values of fields that are not explicitly mentioned in the text.",
          "Return the JSON as specified in the prompt. Do NOT make up a new JSON with different field names or structure.",
          "Ensure that all parsed values are of the correct data type as defined in the DFT schema.",
      ],
      examples=[],
  ),
)

PROMPT_REGISTRY["dft"] = new_entry
```

**Notes**:
- We created a new registry entry in the `PROMPT_REGISTRY`. You can add as many new entries as you want to extract metadata from a defined datamodel.
- We need to include a `retriever_query` to improve the extraction of the most relevant top-k chunks.
- `StructuredPrompt` contains some attributes that can be modified:
  - `expert`: a string containing the expertise expected by the LLM. This translated into "Act like an expert in \<expert\>".
  - `output_schema`: the pydantic model we want to target, e.g., `DFT`.
  - `target_fields`: the fields we want to extract from the pydantic model. If `all` is chosen, the LLM will attempt to extract all metadata fields defined in the pydantic class.
  - `constraints`: a list of instructions to constraint the behavior of the generated answer
  - `examples`: a list of examples; see [How-to: Create Custom Prompts](../howtos/create_custom_prompts.md) for more details about this attribute.


### Running `RAGExtractorAgent`

Both the datamodel and prompt registry defined above will help us running our agent to extract the desired information (in this example, two strings under `DFT`, `code_name` and `code_version`).

In `run_script.py`:

```python
from pathlib import Path

import h5py

from nerxiv.chunker import Chunker
from nerxiv.rag import CustomRetriever, LLMGenerator, RAGExtractorAgent

from .datamodel import DFT
from .prompt_registry import MOD_PROMPT_REGISTRY


query = "dft"
entry = MOD_PROMPT_REGISTRY[query]
prompt = entry.prompt


# Define dictionaries of parameters for chunking, retrieval, and generation
chunker_params = {
  "chunk_size": 2000,
  "chunk_overlap": 500,
}
retriever_params = {
  "retriever_query": entry.retriever_query,
  "model": "all-MiniLM-L6-v2",
  "n_top_chunks": 5,
  "query_name": query,
}
generator_params = {
  "temperature": 0.1,
  "model": "gpt-oss:20b",
}

# Create an instance of the `RAGExtractorAgent`
agent = RAGExtractorAgent(
    chunker=Chunker,
    retriever=CustomRetriever,
    generator=LLMGenerator,
    chunker_params=chunker_params,
    retriever_params=retriever_params,
    generator_params=generator_params,
)

# Run the agent for a specific HDF5 file as downloaded with pyrxiv
with h5py.File(Path("path_to_hdf5.hdf5"), "a") as f:
  arxiv_id = f.filename.split("/")[-1].replace(".hdf5", "")
  text = f[arxiv_id]["arxiv_paper"]["text"][()].decode("utf-8")
  agent.run(file=f, text=text, prompt=prompt)
```

This workflow will run the `RAGExtractorAgent`, extract the specific target fields for the specific output schema in the `MOD_PROMPT_REGISTRY` dictionary, and store the results in the HDF5 file containing the queried arXiv PDF information.

**Notes**:
- We used the normal `Chunker` in this example. Depending on the chunker you use, you will need to modify the `chunker_params` dictionary accordingly.
