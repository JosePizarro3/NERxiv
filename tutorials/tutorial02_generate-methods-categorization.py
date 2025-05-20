import marimo

__generated_with = "0.13.10"
app = marimo.App(width="medium")

with app.setup:
    # Initialization code that runs before all other cells
    import marimo as mo


@app.cell(hide_code=True)
def _():
    mo.md("""# RAGxiv tutorial 2 - LLMs for categorization of arXiv papers""")
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    This tutorial shows the functionalities to categorize a given arXiv paper into a list of the methodologies used.

    As an example for this tutorial, we will use the pdf found in `./tests/data/2502.12144v1.pdf`. This paper describes a computational calculation on a superconducting nickelate. It uses a plethora of methods, like DFT, DMFT, DΓA, etc.

    We will use a local LLM model, deployed using [`ollama`](https://ollama.com/). For this example, we will use `llama3.1:70b`, and `qwen3:32b`.
    """
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""We can extract and clean the text of the PDF to be categorized by running our `TextExtractor` functionality (see Tutorial 1 - Extracting text from arXiv papers).""")
    return


@app.cell
def _():
    from ragxiv.fetch_and_extract import TextExtractor


    extractor = TextExtractor()
    text = extractor.get_text(pdf_path="./tests/data/2502.12144v1.pdf", loader="pdfminer")
    text = extractor.delete_references(text=text)
    text = extractor.clean_text(text=text)
    text
    return (text,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""## Pre-requisites to run the LLM categorization""")
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    In order to prompt the LLM models, we recommend installing locally [`ollama`](https://ollama.com/). In a new terminal window, run:
    ```bash
    ollama serve
    ```

    This will launch the ollama server to use the downloaded LLM models. For downloading locally an LLM model, run:
    ```bash
    ollama pull <model-name>
    ```

    Then, you can use the downloaded model in the functionalities of this package.
    """
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""## Top-k chunks""")
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    Before prompting the LLM, we will:

    1. Chunk our text into smaller strings.
    2. Obtain the top chunks which correspond to text describing the methodology used in the paper.
    """
    )
    return


@app.cell
def _(text):
    from ragxiv.chunker import Chunker


    chunks = Chunker(text=text).chunk_text(chunk_size=800)
    return (chunks,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""This returns a list of chunks which is used for retrieving the most relevant ones. We can use two very similar implementations: `CustomRetriever` or `LangChainRetriever`.""")
    return


@app.cell
def _():
    from ragxiv.retriever import CustomRetriever, LangChainRetriever


    # We arbitrarily chose `CustomRetriever`. Its implementation in the next cells is the same as for `LangChainRetriever`
    categorizer = CustomRetriever()
    return (categorizer,)


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    /// note | Note

    Note that `CustomRetriever` is based on [`SentenceTransformer`](https://sbert.net/). Hence, we can load another model by passing an input `model` to `CustomRetriever`.

    The same occurs for `LangChainRetriever`.
    ///
    """
    )
    return


@app.cell
def _(categorizer, chunks):
    n_top_chunks = 5
    top_text = categorizer.get_relevant_chunks(chunks=chunks, n_top_chunks=n_top_chunks)
    return n_top_chunks, top_text


@app.cell
def _(n_top_chunks, top_text):
    mo.md(
        rf"""
    The top-{n_top_chunks} chunks are joined in a single string which will then be used to categorize the paper:

    ```txt
    {top_text}
    ```
    """
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""## LLM methodologies categorization""")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""Now, with `top_text`, we can try to categorize it by using our `LLMGenerator` and two different LLM models for benchmarking results.""")
    return


@app.cell
def _(top_text):
    from ragxiv.generator import LLMGenerator


    generator_llama = LLMGenerator(model="llama3.1:70b", text=top_text)
    generator_qwen = LLMGenerator(model="qwen3:32b", text=top_text)
    return generator_llama, generator_qwen


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    The workflow to categorize the paper will be:

    1. Categorize if the paper is "computational", "experimental", or "both". If the LLM is not sure of this category, it will return "none".
    2. Obtain the list of methods used in the paper.
    3. Clean the answer in case the LLM includes softwares or instrument names in the list of methods.
    """
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    We have a list of prompts in `ragxiv.prompts` which is useful to categorize the papers.

    You can modify such prompts at will.
    """
    )
    return


@app.cell
def _():
    from ragxiv.prompts import (
        EXP_OR_COMP_TEMPLATE,
        EXTRACT_METHODS_TEMPLATE,
        FILTER_METHODS_TEMPLATE,
        prompt,
    )
    return (
        EXP_OR_COMP_TEMPLATE,
        EXTRACT_METHODS_TEMPLATE,
        FILTER_METHODS_TEMPLATE,
        prompt,
    )


@app.cell(hide_code=True)
def _():
    ### Computational or Experimental
    return


@app.cell
def _(EXP_OR_COMP_TEMPLATE, generator_llama, prompt, top_text):
    answer_exp_or_comp_llama = generator_llama.generate(prompt=prompt(EXP_OR_COMP_TEMPLATE, text=top_text))
    return (answer_exp_or_comp_llama,)


@app.cell
def _(answer_exp_or_comp_llama):
    answer_exp_or_comp_llama
    return


@app.cell
def _(EXP_OR_COMP_TEMPLATE, generator_qwen, prompt, top_text):
    answer_exp_or_comp_qwen = generator_qwen.generate(prompt=prompt(EXP_OR_COMP_TEMPLATE, text=top_text))
    return (answer_exp_or_comp_qwen,)


@app.cell
def _(answer_exp_or_comp_qwen):
    answer_exp_or_comp_qwen
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    Both Llama3.1 and Qwen3 properly categorized the paper as a "computational" paper.

    In case the answer is not one of the expected ones, we must raise a `ValueError`.
    """
    )
    return


@app.cell
def _(answer_exp_or_comp_llama):
    # note that the same check should be performed for `answer_exp_or_comp_qwen`
    if answer_exp_or_comp_llama not in ["computational", "experimental", "both"]:
        raise ValueError(f"Answer is not valid. Expected one of ['computational', 'experimental', 'both'], but got\n\n{answer_exp_or_comp_llama}")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""### List of methods""")
    return


@app.cell
def _():
    mo.md(r"""We can know use the `answer_exp_or_comp` solution to help the LLM model to extract the list of methods from the text.""")
    return


@app.cell
def _(
    EXTRACT_METHODS_TEMPLATE,
    answer_exp_or_comp_llama,
    answer_exp_or_comp_qwen,
    generator_llama,
    generator_qwen,
    prompt,
    top_text,
):
    answer_methods_llama = generator_llama.generate(
        prompt=prompt(EXTRACT_METHODS_TEMPLATE, text=top_text, exp_or_comp=answer_exp_or_comp_llama)
    )

    answer_methods_qwen = generator_qwen.generate(
        prompt=prompt(EXTRACT_METHODS_TEMPLATE, text=top_text, exp_or_comp=answer_exp_or_comp_qwen)
    )
    return answer_methods_llama, answer_methods_qwen


@app.cell(hide_code=True)
def _():
    mo.md(r"""We can further filter the list in case a software name (e.g., "VASP") or an instrument name was wrongly categorized as an used method.""")
    return


@app.cell
def _(
    FILTER_METHODS_TEMPLATE,
    answer_methods_llama,
    answer_methods_qwen,
    generator_llama,
    generator_qwen,
    prompt,
):
    answer_filtered_methods_llama = generator_llama.generate(
        prompt=prompt(FILTER_METHODS_TEMPLATE, candidates=answer_methods_llama)
    )

    answer_filtered_methods_qwen = generator_qwen.generate(
        prompt=prompt(FILTER_METHODS_TEMPLATE, candidates=answer_methods_qwen)
    )
    return answer_filtered_methods_llama, answer_filtered_methods_qwen


@app.cell(hide_code=True)
def _(answer_filtered_methods_llama, answer_filtered_methods_qwen):
    mo.md(
        rf"""
    Filtered list of methods for:

    ---

    Llama3.1

    {answer_filtered_methods_llama}

    ---

    Qwen3

    {answer_filtered_methods_qwen}
    """
    )
    return


@app.cell
def _(answer_filtered_methods_llama, answer_filtered_methods_qwen):
    from ragxiv.generator import answer_to_dict


    answer_to_dict(answer_filtered_methods_llama)
    answer_to_dict(answer_filtered_methods_qwen)
    return


if __name__ == "__main__":
    app.run()
